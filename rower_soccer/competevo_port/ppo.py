"""PPO for the two-ant run-to-goal env, with CompetEvo's hyperparameters.

Stage 1 deliberately trains ONE policy that plays BOTH agents. Their runner holds
two independent learners and samples opponents from a checkpoint ring
(`multi_evo_agent_runner.py:377-461` -- port map section 4.3), which is what makes
competitive co-evolution work and is a later stage. Run-to-goal is symmetric
(mirror the world about x and agent 0 becomes agent 1), so a shared policy is a
legitimate smoke target: the two agents' transitions are simply two rows of the
same batch. `n_policies=2` is wired through for when the ring lands.

Faithful to their `Learner`/`DevLearner` where it is cheap:
  clip 0.2, gamma 0.995, GAE lambda 0.95, 10 optimizer epochs, minibatch 2048,
  Adam 5e-5 (policy) / 3e-4 (value), grad-clip 40, globally standardized
  advantages, actor MLP [128,128] tanh, critic MLP [512,256] tanh, diagonal
  Gaussian with a learned state-independent log_std initialized to 0
  (config/run-to-goal-ants-v0.yaml + custom/models/normal_actor.py).

Two of their quirks are NOT reproduced, both flagged in the port map:
  * their fixed-morph `Learner` never updates the critic (learner.py:218-229 is
    commented out), so its advantages come from a frozen random value net. We
    train the critic. Reproducing that bug is only worth it if a curve refuses
    to line up.
  * float64 everywhere (train.py:61-62). We are fp32, per the warp stack.
"""

import numpy as np
import torch
import torch.nn as nn

# THE REWARD THEY OPTIMIZE IS NOT THE ENV REWARD. Both runners wrap it in an
# exploration curriculum (`runner/multi_agent_runner.py:150-167`, and the evo
# runner's copy at 147-164):
#     r = alpha * dense + (1 - alpha) * parse,
#     alpha = max((termination_epoch - epoch) / termination_epoch, 0)
# with `use_exploration_curriculum: true, termination_epoch: 200` in
# `config/run-to-goal-ants-v0.yaml`. So early training sees ONLY the dense
# forward-progress reward -- the +/-1000 goal term fades IN, it is not there at
# the start -- and the numbers their TB logs show are this curriculum reward.
#
# Their "epoch" is `min_batch_size = 50,000` env steps per agent, so the schedule
# is expressed here in agent-steps rather than iterations: our iteration is a
# different size and tying alpha to an iteration counter would silently change
# the schedule.
CURRICULUM_STEPS = 200 * 50_000        # 10M agent-steps == their 200 epochs


class RunningNorm(nn.Module):
    """Their `lib/rl/core/running_norm.py`: an observation whitener whose
    statistics advance only when the module is in TRAINING mode, i.e. during the
    PPO update pass and not during sampling. Cadence matters -- normalizing with
    stats that moved mid-rollout changes what the ratio in the PPO objective is
    a ratio of -- so the odd-looking placement is deliberate and copied.

    `clip` was 10.0 here until 2026-08-12 and is 5.0 in theirs, which no call
    site of theirs ever overrides. It went unnoticed because every parity gate
    drives the ENV, and this lives in the policy. Measured under their epoch-107
    weights, 0.46% of control-observation components land beyond 5 sigma, so the
    two settings genuinely disagree on ~1 input in 200."""

    def __init__(self, dim, demean=True, destd=True, clip=5.0):
        super().__init__()
        self.register_buffer("n", torch.zeros(1))
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.demean, self.destd, self.clip = demean, destd, clip

    @torch.no_grad()
    def _update(self, x):
        x = x.reshape(-1, x.shape[-1]).double()
        m, v, k = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        n = self.n + k
        delta = m - self.mean.double()
        self.var.copy_(((self.var.double() * self.n + v * k
                         + delta ** 2 * self.n * k / n) / n).float())
        self.mean.copy_((self.mean.double() + delta * k / n).float())
        self.n.copy_(n)

    def forward(self, x):
        if self.training:
            self._update(x)
        if self.n.item() == 0:
            return x
        y = x - self.mean if self.demean else x
        if self.destd:
            y = y / (self.var.sqrt() + 1e-8)
        return y.clamp(-self.clip, self.clip) if self.clip else y


def _mlp(dim_in, hidden, act=nn.Tanh):
    layers, d = [], dim_in
    for h in hidden:
        layers += [nn.Linear(d, h), act()]
        d = h
    return nn.Sequential(*layers), d


class ActorCritic(nn.Module):
    """Their `NormalPolicy` + `NormalValue`, with a shared observation
    normalizer per network (theirs keeps one inside each)."""

    def __init__(self, obs_dim, act_dim, actor_hidden=(128, 128),
                 critic_hidden=(512, 256), log_std_init=0.0):
        super().__init__()
        self.pi_norm = RunningNorm(obs_dim)
        self.vf_norm = RunningNorm(obs_dim)
        self.pi, dpi = _mlp(obs_dim, actor_hidden)
        self.vf, dvf = _mlp(obs_dim, critic_hidden)
        self.action_net = nn.Linear(dpi, act_dim)
        self.value_net = nn.Linear(dvf, 1)
        # Their `init_fc_weights` (custom/utils/tools.py:19-21) on both output
        # heads: weights x0.1, bias 0. Not cosmetic -- it is why their iter-0
        # eval reward is ~499 and not ~440. A default-initialized head emits mean
        # actions around 0.1, which at gear 150 is real torque, so the untrained
        # ant drifts and loses forward reward; theirs starts almost still and
        # collects the +1 survive bonus for all 500 steps. Reproducing the init
        # is what makes the iter-0 number comparable to their measured one.
        for head in (self.action_net, self.value_net):
            head.weight.data.mul_(0.1)
            head.bias.data.mul_(0.0)
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

    @staticmethod
    def _clean(obs):
        # A diverged world must never be able to turn into a NaN action: a
        # network cannot emit a non-finite mean from a finite bounded input.
        return torch.nan_to_num(obs, nan=0.0, posinf=1e3,
                                neginf=-1e3).clamp(-1e3, 1e3)

    def dist(self, obs):
        mean = self.action_net(self.pi(self.pi_norm(self._clean(obs))))
        return torch.distributions.Normal(mean, self.log_std.exp())

    def mean_action(self, obs):
        """Their eval action: `DiagGaussian.mean_sample()` = the loc."""
        return self.action_net(self.pi(self.pi_norm(self._clean(obs))))

    def value(self, obs):
        return self.value_net(self.vf(self.vf_norm(self._clean(obs)))).squeeze(-1)

    @torch.no_grad()
    def act(self, obs):
        d = self.dist(obs)
        a = d.sample()
        return a, d.log_prob(a).sum(-1), self.value(obs)


class SelfPlayPPO:
    """One policy, both agents, one batched env.

    The rollout buffer is `[T, n_worlds, n_agents, ...]` and is flattened to
    `[T * n_worlds * n_agents, ...]` for the update, so each agent's transitions
    are ordinary samples. GAE runs per (world, agent) lane.
    """

    def __init__(self, env, ac, rollout_len=64, gamma=0.995, gae_lambda=0.95,
                 clip=0.2, epochs=10, minibatch_size=2048, policy_lr=5e-5,
                 value_lr=3e-4, max_grad_norm=40.0, value_l2=1e-3,
                 ent_coef=0.0, curriculum_steps=CURRICULUM_STEPS,
                 device="cuda"):
        self.env, self.ac = env, ac.to(device)
        self.T, self.N, self.A = rollout_len, env.n, env.n_agents
        self.gamma, self.lam, self.clip = gamma, gae_lambda, clip
        self.epochs, self.mb = epochs, minibatch_size
        self.ent_coef, self.max_grad_norm = ent_coef, max_grad_norm
        # None (or 0) => optimize the raw env reward parse + dense. Any positive
        # value runs their curriculum over that many agent-steps.
        self.curriculum_steps = curriculum_steps
        self.device = device
        # Their two optimizers with two learning rates (dev_learner.py:129-140);
        # the critic's weight decay is their `l2_reg` on the value net only.
        self.pi_params, self.vf_params = self._param_groups()
        self.value_l2 = value_l2
        self.opt_pi = torch.optim.Adam(self.pi_params, lr=policy_lr)
        # NOT `weight_decay=value_l2`. Theirs adds `sum(p^2) * l2_reg` to the
        # value LOSS (`dev_learner.py:177-178`), whose gradient is `2 * l2_reg *
        # p`, while Adam's non-decoupled weight_decay adds `l2_reg * p` -- so the
        # optimizer form is exactly half strength. Corrected 2026-08-12; see
        # M2E_VALIDATION section 8.
        self.opt_vf = torch.optim.Adam(self.vf_params, lr=value_lr)

        shape = (self.T, self.N, self.A)
        z = lambda *tail: torch.zeros(*shape, *tail, device=device)
        self.obs_buf = z(env.obs_dim)
        self.act_buf = z(env.act_dim)
        self.logp_buf, self.rew_buf, self.val_buf = z(), z(), z()
        # Their mask semantics (sample_worker:284-292): 0 ONLY on a true
        # termination. A truncated episode keeps mask=1, so GAE bootstraps across
        # the episode boundary. That is a bug in their code AND the behaviour the
        # reference curves were produced with; port map risk 7 says match it
        # first, fix it after the curves line up.
        self.mask_buf = z()
        self._obs = env.reset()
        self.total_steps = 0
        self.n_bad_grads = 0

    def _param_groups(self):
        """(policy, critic) parameters. Overridden by the dev trainer, whose
        actor is two named heads rather than one `pi`/`action_net` pair."""
        pi = [p for n, p in self.ac.named_parameters()
              if n.startswith(("pi", "action_net", "log_std"))]
        vf = [p for n, p in self.ac.named_parameters()
              if n.startswith(("vf", "value_net"))]
        return pi, vf

    def _logp_entropy(self, obs, act, slots=None):
        """(log pi(act|obs), entropy). Overridden by the dev trainer, whose
        actor has two heads and picks one per sample from the stage flag."""
        d = self.ac.dist(obs)
        return d.log_prob(act).sum(-1), d.entropy().sum(-1)

    def alpha(self):
        """Their curriculum weight on the dense reward: 1 at the start, linearly
        to 0 over `curriculum_steps` agent-steps, then pinned at 0."""
        if not self.curriculum_steps:
            return None
        return max(1.0 - self.total_steps / (self.A * self.curriculum_steps), 0.0)

    def collect(self):
        alpha = self.alpha()
        self.ep_fwd = 0.0
        for t in range(self.T):
            # .float(): the CPU backend runs float64 for the parity gate, the
            # networks are fp32 everywhere.
            flat = self._obs.reshape(-1, self.env.obs_dim).float()
            a, logp, v = self.ac.act(flat)
            a = a.reshape(self.N, self.A, self.env.act_dim)
            self.obs_buf[t] = self._obs.float()
            self.act_buf[t] = a
            self.logp_buf[t] = logp.reshape(self.N, self.A)
            self.val_buf[t] = v.reshape(self.N, self.A)
            self._obs, rew, done, info = self.env.step(a)
            if alpha is not None:
                rew = alpha * info["dense"] + (1.0 - alpha) * info["parse"]
            self.rew_buf[t] = rew.float()
            self.mask_buf[t] = (~info["terminated"]).float().unsqueeze(-1)
            # Mean forward-progress reward per agent-step. The honest early
            # training signal: episode RETURN is dominated by the +1/step survive
            # bonus, so a policy that learns to run but falls at t=200 scores
            # WORSE than one that stands still for 500 steps. This does not.
            self.ep_fwd += float(info["forward"].mean()) / self.T
        self.total_steps += self.T * self.N * self.A
        with torch.no_grad():
            last_v = self.ac.value(self._obs.reshape(-1, self.env.obs_dim).float())
        return self._gae(last_v.reshape(self.N, self.A))

    def _gae(self, last_v):
        adv = torch.zeros_like(self.rew_buf)
        gae = torch.zeros_like(last_v)
        nxt = last_v
        for t in reversed(range(self.T)):
            m = self.mask_buf[t]
            delta = self.rew_buf[t] + self.gamma * nxt * m - self.val_buf[t]
            gae = delta + self.gamma * self.lam * m * gae
            adv[t] = gae
            nxt = self.val_buf[t]
        ret = adv + self.val_buf
        # Their `estimate_advantages` standardizes over the WHOLE buffer.
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        return adv, ret

    def update(self, adv, ret):
        n = self.T * self.N * self.A
        obs = self.obs_buf.reshape(n, -1)
        act = self.act_buf.reshape(n, -1)
        logp_old = self.logp_buf.reshape(n)
        adv, ret = adv.reshape(n), ret.reshape(n)
        # 2h Option A: with one net per SLOT the flattening above destroys the
        # only thing that says which net owns a row, and the minibatch `perm`
        # then shuffles rows from both slots together. `obs_buf` is
        # [T, N, A, D] and A is the lane axis, so the slot of flat row r is
        # r % A. None for a shared-net policy, where every call below keeps its
        # original single-argument form.
        slot_all = (torch.arange(n, device=self.device) % self.A
                    if hasattr(self.ac, "n_slots") else None)
        self.ac.train()                      # RunningNorm advances here, only
        stats = {"pi_loss": 0.0, "vf_loss": 0.0, "kl": 0.0, "nb": 0}
        for _ in range(self.epochs):
            perm = torch.randperm(n, device=self.device)
            for s in range(0, n, self.mb):
                i = perm[s:s + self.mb]
                sl = None if slot_all is None else slot_all[i]
                logp, ent = self._logp_entropy(obs[i], act[i], slots=sl)
                ratio = (logp - logp_old[i]).exp()
                pi_loss = -torch.min(
                    ratio * adv[i],
                    ratio.clamp(1 - self.clip, 1 + self.clip) * adv[i]).mean()
                if self.ent_coef:
                    pi_loss = pi_loss - self.ent_coef * ent.mean()
                v_i = (self.ac.value(obs[i]) if sl is None
                       else self.ac.value_flat(obs[i], sl))
                vf_loss = (v_i - ret[i]).pow(2).mean()
                # Their explicit L2 penalty, in the loss where they put it.
                vf_total = vf_loss
                if self.value_l2:
                    vf_total = vf_total + self.value_l2 * sum(
                        p.pow(2).sum() for p in self.vf_params)
                self.opt_pi.zero_grad(set_to_none=True)
                self.opt_vf.zero_grad(set_to_none=True)
                (pi_loss + vf_total).backward()
                # `dev_learner.py:196` clips `policy_net.parameters()` ONLY --
                # the critic's gradient is never clipped. Clipping it too (which
                # this did until 2026-08-12) rescales the policy gradient by a
                # norm that includes the critic's, so the two nets were coupled
                # through the clip.
                gn = nn.utils.clip_grad_norm_(self.pi_params,
                                              self.max_grad_norm)
                # Theirs has no finiteness guard; keep ours, but measure the
                # critic's norm without scaling it (max_norm=inf never clips).
                gn_vf = nn.utils.clip_grad_norm_(self.vf_params, float("inf"))
                if not (torch.isfinite(gn) and torch.isfinite(gn_vf)):
                    self.n_bad_grads += 1
                    continue
                self.opt_pi.step()
                self.opt_vf.step()
                stats["pi_loss"] += float(pi_loss)
                stats["vf_loss"] += float(vf_loss)
                stats["kl"] += float((logp_old[i] - logp).mean())
                stats["nb"] += 1
        self.ac.eval()
        nb = max(stats.pop("nb"), 1)
        return {k: v / nb for k, v in stats.items()}

    def train_iter(self):
        adv, ret = self.collect()
        return self.update(adv, ret)


@torch.no_grad()
def evaluate(env, ac, max_steps=None, mean_action=True):
    """Their eval pass: full episodes, deterministic (mean) actions, win rate
    counted as wins / games with truncated draws IN the denominator
    (`sample`:369-372). Returns per-agent mean episode return and win rate.

    Reproduces their epoch-0 eval too: with an untrained net the mean action is
    ~0, so the return is dominated by the +1 survive bonus x 500 steps, which is
    where the port map's `iter-0 eval ~= 490-510, win rate 0.00` comes from.
    """
    ac.eval()
    # The dev env spends one extra step per episode on the design action, which
    # their `_elapsed_steps` does not count; without the +1 a stand-still policy
    # never closes an episode here and the eval reports zero games.
    max_steps = max_steps or (env.max_episode_steps
                              + int(getattr(env, "has_design_step", False)))
    obs = env.reset()
    env.reset_win_stats()
    rets, lens = [], []
    for _ in range(max_steps):
        flat = obs.reshape(-1, env.obs_dim).float()
        a = (ac.mean_action(flat) if mean_action else ac.act(flat)[0])
        obs, rew, done, info = env.step(a.reshape(env.n, env.n_agents, -1).to(env.dtype))
        if bool(done.any()):
            idx = done.nonzero(as_tuple=True)[0]
            rets.append(env.last_return[idx].float().cpu().numpy())
            lens.append(env.last_len[idx].float().cpu().numpy())
    if not rets:
        return {"ret": np.zeros(env.n_agents), "win_rate": env.win_rate(),
                "ep_len": 0.0, "games": env.games}
    rets = np.concatenate(rets, 0)
    return {"ret": rets.mean(0), "win_rate": env.win_rate(),
            "ep_len": float(np.concatenate(lens).mean()), "games": env.games}
