"""D1 unit 1f -- 2v2 self-play RL fine-tune in z-space (PIPELINE_V2 stage 6).

    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
    PYTHONPATH=. .venv/bin/python -m rower_soccer.warp_port.train_soccer2v2_warp \
        --run-name soccer2v2_1f_base --worlds 256 --minutes 720 \
        --init-from runs_v2/s5_c_all/best.pt --freeze-decoder

WHAT THIS IS (and what it deliberately is not)
----------------------------------------------
PIPELINE_V2 stage 6 is "self-play fine-tune: KL-to-BC + drill-prior mixture +
shaping rewards". This trainer runs it WITHOUT the KL-to-BC anchor, because the
human 2v2 demonstrations (stage 5) do not exist yet.

That is not a degraded stopgap, for two reasons, and both are why the run is
worth its GPU-hours now:

  1. It is the paper's own arrangement, minus one stage. Liu et al. 2022 used
     motion capture for the low-level controller and then PURE self-play RL for
     team play -- their team-play stage had no behavioural anchor either. What
     we are running IS that stage: a frozen z-space motor controller (ours came
     from drill RL rather than mocap, PIPELINE_V2's "stage 1 cut"), driven by a
     high-level policy that self-play optimises.
  2. It is the CONTROL that makes the human BC data measurable later. When the
     demos arrive, "did BC help?" is only answerable against a no-BC baseline
     trained in the same env with the same budget. This run is that baseline.
     Adding the KL-to-BC anchor later is a flag, not a rewrite.

THE Z-SPACE CONTRACT (the whole point; do not shortcut it)
-----------------------------------------------------------
The policy does NOT emit torques. It emits `z` (dim 16) and the FROZEN low-level
decoder turns `(proprio, z)` into torques -- the identical path
`gate_drill_priors.action_from_z` drives, and the identical decoder every drill
trained. `ActorCritic.dist` already computes exactly this
(`decoder(cat[proprio, z(obs)]) -> action_net`), so the z-space path is what you
get for free by using `ActorCritic` with `--freeze-decoder`; the gate measures
that equality bitwise rather than trusting this paragraph. A policy that emitted
raw torques would train, would score, and would transfer nothing: the drills,
the priors, and the eventual BC corpus all live in z.

WARM START (shoot -> football, task encoder INCLUDED)
------------------------------------------------------
`soccer2v2_env`'s first 13 task dims are `shoot`'s task block verbatim
(ball_ego6 + opp_goal_mid3 + post_l2 + post_r2), so a shoot checkpoint warm
starts more than the decoder. `ppo.load_pretrained` would drop
`task_enc.0` and `critic.0` on the shape mismatch (13 -> 34, 78 -> 99); this
module SPLICES them instead: the shoot columns land on the matching football
columns and the 21 new columns (own goal, team-mate, two opponents) start at
zero. At init the policy is therefore EXACTLY the shoot policy evaluated on
football observations.

Zero-init here is not PIPELINE_V2's zero-padding anti-pattern. That warning is
about a weight whose INPUT is always zero: its gradient is `delta * x = 0`, so
it never leaves its random init and then fires noise when the input goes live.
Here the input is live from step 1 -- the team-mate and opponent columns carry
real numbers -- so the zeroed weights get real gradient immediately. What is
zero is the weight, not the data.

DRILL-PRIOR MIXTURE (the paper's Eq. 5, adapted to a deterministic z)
----------------------------------------------------------------------
Eq. 5 regularises the football policy toward a mixture of the four distilled
drill priors. Our `z` is DETERMINISTIC (`LatentExtractor.z` is a projection; the
stochasticity lives in the action head, which is what keeps PPO's log-probs
exact), so the KL from a point mass to the mixture is, up to a constant that
does not depend on the policy, the mixture's negative log-density at that point:

    L_prior = -log sum_k alpha_k * N(z ; mu_k(o), sigma_k(o))

computed with `logsumexp`, alpha uniform by default. Minimising it pulls z into
the region the four skills actually occupy. `--w-prior 0` removes the term
entirely (an `if`, not a multiply by zero), which is the ablation.

TRUNCATION AND THE BOOTSTRAP (the bug this file exists to not have)
--------------------------------------------------------------------
D3_HANDOFF ("This inverts the port's problem") records the class: a fixed-T
batched sampler truncates every world at the rollout boundary, so GAE must
bootstrap `V(s_T)` at every cut that an episode-complete sampler never makes.
There are TWO kinds of cut here and `warp_port/ppo.py` gets the second one
wrong:

  * the rollout boundary (t = T-1): bootstrap `V(self._obs)`. ppo.py does this.
  * the match clock (`done`, every `match_seconds`): the env is reset INSIDE
    collect, so `val_buf[t+1]` is `V(the kickoff state of the NEXT match)`, not
    `V(s_T)`. ppo.py bootstraps that -- it discounts the wrong state into the
    last transition of every match. This trainer records `V(s_T)` before the
    reset (`boot_buf`) and bootstraps THAT, and stops the GAE recursion at the
    boundary so no advantage leaks across matches.

The time limit is a TRUNCATION, not a failure, so we bootstrap rather than cut
to zero -- the Transform2Act convention in D3_HANDOFF applies to a task whose
time limit is a genuine terminal; a 45 s slice of a match is not.

REWARD
------
The optimised objective is dm_soccer's own: +1 to the scoring team, -1 to the
conceding team, per player, and nothing else. Shaping (`--w-player-to-ball`,
`--w-ball-to-goal`) is multiplied by `env.shaping_scale`, which this trainer
anneals linearly to zero over `--shaping-anneal-steps`, so the sparse term is
what survives. Sizes are chosen so shaping is COMPARABLE to, never larger than,
a goal: at the defaults an end-to-end ball drive down a 26 m pitch is worth
~1.3 (about one goal) and chasing the ball at 0.5 m/s for a whole 45 s match is
worth ~0.9.

SELF-PLAY
---------
ONE shared policy drives all four slots. The env's kickoff is invariant under
the 180-degree mirror that swaps the teams and every observation is egocentric
(see `soccer2v2_env`'s TEAM SYMMETRY section), so neither side starts with an
advantage and the symmetric self-play match is fair by construction. A
checkpoint opponent pool is available behind `--opponent-pool` (default OFF);
when it is on, the away rows of the selected worlds are driven by a frozen past
snapshot and are MASKED OUT of the PPO update.

METRICS
-------
Mean fitness is identically zero in symmetric self-play (goal difference is
zero-sum over the four players), so it cannot rank anything. What is logged
instead, and only ever for COMPLETED matches:

    goals_per_match   home + away goals, i.e. is play happening at all
    goal_hist         share of worlds ending 0 / 1 / 2+ goals -- a rate with no
                      distribution behind it hides a wipeout artifact
    throw_ins, ball_dist, upright, diverged
"""

import argparse
import json
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn as nn

from rower_soccer.warp_port.ppo import (OBS_SANITY_LIMIT, ActorCritic,
                                        save_checkpoint, load_checkpoint,
                                        export_sb3_compatible)
from rower_soccer.warp_port.scene import BallSpec
from rower_soccer.warp_port.soccer2v2_env import WarpSoccer2v2Env, SoccerReward

PRIOR_SKILLS = ("follow", "dribble", "kick", "shoot")


# ---------------------------------------------------------------------------
# env
# ---------------------------------------------------------------------------
def make_env(args, num_worlds, seed, use_graph=True):
    reward = SoccerReward(w_goal=args.w_goal,
                          w_player_to_ball=args.w_player_to_ball,
                          w_ball_to_goal=args.w_ball_to_goal)
    return WarpSoccer2v2Env(
        num_worlds=num_worlds, seed=seed, use_graph=use_graph,
        creature_xml=args.creature_xml,
        ball=BallSpec(radius=args.ball_radius, mass=args.ball_mass),
        pitch_scale=args.pitch_scale, match_seconds=args.match_secs,
        spawn=args.spawn, ball_jitter=args.ball_jitter,
        reward=reward, energy_coef=args.energy_coef,
        smooth_coef=args.smooth_coef, use_gpu=not args.cpu,
        nconmax=args.nconmax, njmax=args.njmax)


# ---------------------------------------------------------------------------
# warm start
# ---------------------------------------------------------------------------
def _flat_source(path, device):
    from rower_soccer.warp_port.ppo import _flatten_checkpoint
    return _flatten_checkpoint(torch.load(path, map_location=device,
                                          weights_only=True))


def load_warm_start(ac, path, n_proprio, device="cpu", splice=True,
                    verbose=True):
    """Warm start the football policy from a drill checkpoint.

    Beyond `ppo.load_pretrained`: the two layers whose INPUT WIDTH changed are
    spliced column-wise instead of dropped, because the football task block is
    a strict superset-with-the-same-prefix of shoot's.

        task_enc.0.weight : [128, 13] -> [128, 34]   cols 0:13 copied, 13: zero
        critic.0.weight   : [256, 78] -> [256, 99]   layout [proprio | task],
                                                     so 0:65 and 65:78 copied,
                                                     78: zero

    The source's own widths are DERIVED from its tensors (T_src from
    task_enc.0, P_src = critic.0 in_features - T_src) rather than assumed, and
    P_src is asserted equal to this env's proprio width -- a decoder loaded
    against a different proprio layout is the silent-failure mode here.

    Returns a dict of counts + the list of spliced/skipped names, so a caller
    can print it and a gate can assert on it.
    """
    sd = _flat_source(path, device)
    own = ac.state_dict()
    skip_buffers = {"mlp_extractor.p_idx", "mlp_extractor.t_idx"}

    t_src = int(sd["mlp_extractor.task_enc.0.weight"].shape[1])
    p_src = int(sd["mlp_extractor.critic.0.weight"].shape[1]) - t_src
    if p_src != n_proprio:
        raise SystemExit(
            f"[warm-start] source proprio width {p_src} != env's {n_proprio}. "
            f"The decoder's entire input contract is proprio; loading it "
            f"against a different layout would train and transfer nothing.")
    t_dst = int(own["mlp_extractor.task_enc.0.weight"].shape[1])
    if t_src > t_dst:
        raise SystemExit(f"[warm-start] source task width {t_src} > env's {t_dst}")

    loaded, spliced, missing, shape_skip = [], [], [], []
    for k, v in own.items():
        if k in skip_buffers:
            continue
        if k not in sd:
            missing.append(k)
            continue
        s = sd[k].to(v.device)
        if s.shape == v.shape:
            v.copy_(s)
            loaded.append(k)
        elif splice and k == "mlp_extractor.task_enc.0.weight":
            v.zero_()
            v[:, :t_src].copy_(s)
            spliced.append(f"{k} {tuple(s.shape)}->{tuple(v.shape)}")
        elif splice and k == "mlp_extractor.critic.0.weight":
            v.zero_()
            v[:, :p_src].copy_(s[:, :p_src])
            v[:, p_src:p_src + t_src].copy_(s[:, p_src:])
            spliced.append(f"{k} {tuple(s.shape)}->{tuple(v.shape)}")
        else:
            shape_skip.append(f"{k} {tuple(s.shape)}->{tuple(v.shape)}")
    ac.load_state_dict(own)
    unexpected = [k for k in sd if k not in own and k not in skip_buffers]
    rep = dict(loaded=loaded, spliced=spliced, missing=missing,
               shape_skip=shape_skip, unexpected=unexpected,
               t_src=t_src, p_src=p_src, t_dst=t_dst)
    if verbose:
        print(f"[warm-start] {path}", flush=True)
        print(f"[warm-start] source proprio {p_src} task {t_src} -> "
              f"env proprio {n_proprio} task {t_dst}", flush=True)
        print(f"[warm-start] copied {len(loaded)} tensors verbatim "
              f"(decoder + action_net + proprio_enc + expert + z_proj "
              f"+ value_net + log_std)", flush=True)
        for s in spliced:
            print(f"[warm-start]   SPLICED: {s}", flush=True)
        for s in shape_skip:
            print(f"[warm-start]   re-init (shape): {s}", flush=True)
        for s in missing:
            print(f"[warm-start]   re-init (absent from source): {s}", flush=True)
        for s in unexpected:
            print(f"[warm-start]   unexpected in source, ignored: {s}", flush=True)
    return rep


# ---------------------------------------------------------------------------
# drill-prior mixture (paper Eq. 5)
# ---------------------------------------------------------------------------
class DrillPriorMixture(nn.Module):
    """-log sum_k alpha_k N(z ; mu_k(o), sigma_k(o)) over the 4 drill priors.

    Every prior is frozen and in eval mode; the only thing with a gradient is
    `z`, so the term regularises the high-level policy (proprio_enc, task_enc,
    expert, z_proj) and nothing else.

    Each prior sees the football observation through its OWN column map: the
    `task_cols` recorded in its checkpoint, applied to the env's task block.
    That is well-defined here only because soccer2v2's task block opens with
    shoot's 13 dims verbatim, whose first 6 are ball_ego -- exactly what
    `drill_prior.FOOTBALL_TASK_COLS` maps dribble and kick onto. follow's prior
    is proprio-only and takes no task input at all.
    """

    def __init__(self, prior_dir, skills, proprio_indices, task_indices,
                 device="cpu", alphas=None):
        super().__init__()
        from rower_soccer.warp_port.drill_prior import DrillPrior
        self.skills = list(skills)
        self.priors = nn.ModuleList()
        self.cols = []
        n_task_env = len(task_indices)
        for sk in self.skills:
            blob = torch.load(os.path.join(prior_dir, f"{sk}.pt"),
                              map_location=device, weights_only=False)
            if blob["n_proprio"] != len(proprio_indices):
                raise SystemExit(
                    f"[prior] {sk}: prior proprio {blob['n_proprio']} != env "
                    f"{len(proprio_indices)}")
            cols = list(blob["task_cols"])
            if cols and max(cols) >= n_task_env:
                raise SystemExit(f"[prior] {sk}: task col {max(cols)} outside "
                                 f"the env's {n_task_env}-wide task block")
            pr = DrillPrior(blob["n_proprio"], blob["n_task"],
                            z_dim=blob["z_dim"])
            pr.load_state_dict(blob["state_dict"])
            pr.eval()
            for p in pr.parameters():
                p.requires_grad_(False)
            self.priors.append(pr)
            self.cols.append(torch.as_tensor(cols, dtype=torch.long,
                                             device=device))
        a = (torch.full((len(self.skills),), 1.0 / len(self.skills))
             if alphas is None
             else torch.as_tensor(alphas, dtype=torch.float32))
        self.register_buffer("log_alpha", (a / a.sum()).log().to(device))
        self.register_buffer("p_idx", torch.as_tensor(proprio_indices,
                                                      dtype=torch.long,
                                                      device=device))
        self.register_buffer("t_idx", torch.as_tensor(task_indices,
                                                      dtype=torch.long,
                                                      device=device))
        self.to(device)

    def neg_log_prob(self, obs, z):
        """[B] the mixture's negative log-density at z. Gradient flows to z."""
        o = ActorCritic._clean(obs)
        prop = o.index_select(-1, self.p_idx)
        task_all = o.index_select(-1, self.t_idx)
        lp = []
        for k, pr in enumerate(self.priors):
            c = self.cols[k]
            task = task_all.index_select(-1, c) if c.numel() else None
            d = pr.dist(prop, task)
            lp.append(d.log_prob(z).sum(-1) + self.log_alpha[k])
        return -torch.logsumexp(torch.stack(lp, 0), 0)

    def responsibilities(self, obs, z):
        """[B, K] softmax over the per-prior log-densities: WHICH skill the
        policy currently looks like. Diagnostic only, no gradient."""
        with torch.no_grad():
            o = ActorCritic._clean(obs)
            prop = o.index_select(-1, self.p_idx)
            task_all = o.index_select(-1, self.t_idx)
            lp = []
            for k, pr in enumerate(self.priors):
                c = self.cols[k]
                task = task_all.index_select(-1, c) if c.numel() else None
                lp.append(pr.dist(prop, task).log_prob(z).sum(-1)
                          + self.log_alpha[k])
            return torch.softmax(torch.stack(lp, -1), -1)


# ---------------------------------------------------------------------------
# trainer
# ---------------------------------------------------------------------------
class SelfPlayPPO:
    """PPO over the FLATTENED (world, player) batch of a 2v2 match env.

    Differences from `ppo.PPOTrainer`, each of them load-bearing:
      * N is `env.n * env.n_agents`, not `env.n` -- every row is one player and
        the shared policy sees them all in one forward pass.
      * `V(s_T)` is recorded BEFORE the match-clock reset and bootstrapped
        there; the GAE recursion is cut at the boundary. See the module
        docstring.
      * optional drill-prior mixture term on z.
      * optional per-row mask so opponent-pool rows can be excluded from the
        update without being excluded from the rollout.
    """

    def __init__(self, env, ac, *, lr=3e-4, rollout_len=64, minibatches=8,
                 epochs=4, gamma=0.995, gae_lambda=0.95, clip=0.2,
                 ent_coef=0.003, vf_coef=0.5, max_grad_norm=0.5,
                 ent_floor=None, ent_ceil=0.0, ent_anneal_steps=0,
                 prior=None, w_prior=0.0, prior_anneal_steps=0,
                 device="cuda"):
        self.env, self.ac = env, ac.to(device)
        self.device = device
        self.A = env.n_agents
        self.T = rollout_len
        self.N = env.n * env.n_agents
        self.minibatches, self.epochs = minibatches, epochs
        self.gamma, self.lam, self.clip = gamma, gae_lambda, clip
        self.ent_coef_start = self.ent_coef = ent_coef
        self.ent_anneal_steps = ent_anneal_steps
        self.vf_coef, self.max_grad_norm = vf_coef, max_grad_norm
        self.ent_floor, self.ent_ceil = ent_floor, ent_ceil
        self.prior, self.w_prior_start = prior, w_prior
        self.w_prior = w_prior
        self.prior_anneal_steps = prior_anneal_steps
        self.opt = torch.optim.Adam([p for p in ac.parameters()
                                     if p.requires_grad], lr=lr)
        d = device
        T, N = self.T, self.N
        self.obs_buf = torch.zeros(T, N, env.obs_dim, device=d)
        self.act_buf = torch.zeros(T, N, env.act_dim, device=d)
        self.logp_buf = torch.zeros(T, N, device=d)
        self.rew_buf = torch.zeros(T, N, device=d)
        self.val_buf = torch.zeros(T, N, device=d)
        self.boot_buf = torch.zeros(T, N, device=d)   # V(s_T) at match ends
        self.done_buf = torch.zeros(T, device=d)      # one bool per step
        self.mask_buf = torch.ones(T, N, device=d)    # 1 = learner row
        self._obs = env.reset()
        self.total_steps = 0
        self.n_diverged = 0
        self.n_bad_grads = 0
        self.matches = 0
        self.last_match = None       # stats of the last COMPLETED match only
        # opponent pool (off unless enabled by the caller)
        self.pool = []
        self.pool_row = None         # bool [N]: rows driven by the opponent
        self.opp_ac = None
        row = torch.arange(N, device=d)
        self.row_world = row // self.A
        self.row_slot = row % self.A
        self.is_away = self.row_slot >= env.n_per_team

    # -- opponent pool ------------------------------------------------------
    def enable_pool(self, opp_ac, size, prob, gen):
        self.opp_ac = opp_ac.to(self.device).eval()
        for p in self.opp_ac.parameters():
            p.requires_grad_(False)
        self.pool_size, self.pool_prob, self.pool_gen = size, prob, gen

    def snapshot_pool(self):
        if self.opp_ac is None:
            return
        self.pool.append({k: v.detach().clone()
                          for k, v in self.ac.state_dict().items()})
        if len(self.pool) > self.pool_size:
            self.pool.pop(0)

    def _draw_opponent(self):
        """Pick the worlds whose AWAY slots a frozen past self will drive."""
        if self.opp_ac is None or not self.pool:
            self.pool_row = None
            return
        i = int(torch.randint(len(self.pool), (1,),
                              generator=self.pool_gen).item())
        self.opp_ac.load_state_dict(self.pool[i])
        w = (torch.rand(self.env.n, generator=self.pool_gen,
                        device=self.device) < self.pool_prob)
        self.pool_row = w[self.row_world] & self.is_away

    # -- rollout ------------------------------------------------------------
    def collect(self):
        env = self.env
        self._draw_opponent()
        self.boot_buf.zero_()
        self.mask_buf.fill_(1.0)
        for t in range(self.T):
            a, logp, v = self.ac.act(self._obs)
            if self.pool_row is not None and bool(self.pool_row.any()):
                with torch.no_grad():
                    ao = self.opp_ac.dist(self._obs.float()).sample()
                a = torch.where(self.pool_row.unsqueeze(-1), ao, a)
                self.mask_buf[t] = (~self.pool_row).float()
            obs2, rew, done = env.step(a.clamp(-1, 1))

            # ppo.collect's guard, verbatim in intent: large-but-FINITE
            # observations detonate the PPO ratio and NaN every gradient, and
            # isfinite() does not catch them.
            bad = (~torch.isfinite(obs2).all(dim=-1)
                   | (obs2.abs().amax(dim=-1) > OBS_SANITY_LIMIT))
            if bool(bad.any()):
                self.n_diverged += int(bad.sum())
                obs2 = torch.where(bad.unsqueeze(-1), torch.zeros_like(obs2),
                                   obs2)
                obs2 = torch.nan_to_num(obs2, nan=0.0, posinf=0.0, neginf=0.0)
                rew = torch.where(bad, torch.zeros_like(rew), rew)
                v = torch.where(bad, torch.zeros_like(v), v)
                self.mask_buf[t] = self.mask_buf[t] * (~bad).float()

            self.obs_buf[t] = self._obs
            self.act_buf[t] = a
            self.logp_buf[t] = logp
            self.rew_buf[t] = rew
            self.val_buf[t] = v
            self.done_buf[t] = float(done)
            if done:
                # The match clock is a TRUNCATION. V(s_T) must be read here,
                # from the final state -- after env.reset() the observation is
                # the next match's kickoff and its value is a different number.
                with torch.no_grad():
                    self.boot_buf[t] = self.ac.value(obs2)
                self.last_match = self._match_summary()
                self.matches += env.n
                obs2 = env.reset()
            self._obs = obs2
        self.total_steps += self.T * self.N
        with torch.no_grad():
            last_val = self.ac.value(self._obs)

        adv = torch.zeros_like(self.rew_buf)
        gae = torch.zeros(self.N, device=self.device)
        for t in reversed(range(self.T)):
            if self.done_buf[t] > 0:
                next_v, carry = self.boot_buf[t], torch.zeros_like(gae)
            else:
                next_v = last_val if t == self.T - 1 else self.val_buf[t + 1]
                carry = gae
            delta = self.rew_buf[t] + self.gamma * next_v - self.val_buf[t]
            gae = delta + self.gamma * self.lam * carry
            adv[t] = gae
        return adv, adv + self.val_buf

    def _match_summary(self):
        """Per-world goal totals of the match that just ended. Called BEFORE
        the reset, so `env.score` still holds this match."""
        env = self.env
        tot = (env.score[:, 0] + env.score[:, 1])
        st = env.match_stats()
        st["goals_per_match"] = float(tot.mean())
        st["p_0_goals"] = float((tot == 0).float().mean())
        st["p_1_goal"] = float((tot == 1).float().mean())
        st["p_2plus_goals"] = float((tot >= 2).float().mean())
        st["worlds"] = int(env.n)
        return st

    # -- update -------------------------------------------------------------
    def update(self, adv, ret, critic_only=False):
        B = self.T * self.N
        obs = self.obs_buf.reshape(B, -1)
        act = self.act_buf.reshape(B, -1)
        logp_old = self.logp_buf.reshape(B)
        adv_f = adv.reshape(B)
        ret_f = ret.reshape(B)
        mask = self.mask_buf.reshape(B)
        live = mask.nonzero(as_tuple=True)[0]
        # Normalise over the rows that will actually be used; including masked
        # rows in the mean/std would let a frozen opponent shift the learner's
        # advantage scale.
        a_live = adv_f[live]
        adv_f = (adv_f - a_live.mean()) / (a_live.std() + 1e-8)

        idx = live[torch.randperm(live.numel(), device=self.device)]
        mb = max(1, idx.numel() // self.minibatches)
        stats = {}
        for _ in range(self.epochs):
            for i in range(self.minibatches):
                j = idx[i * mb:(i + 1) * mb]
                if j.numel() == 0:
                    continue
                o = obs[j]
                d = self.ac.dist(o)
                logp = d.log_prob(act[j]).sum(-1)
                ratio = (logp - logp_old[j]).exp()
                pg = -torch.min(
                    ratio * adv_f[j],
                    ratio.clamp(1 - self.clip, 1 + self.clip)
                    * adv_f[j]).mean()
                v = self.ac.value(o)
                vloss = ((v - ret_f[j]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                if critic_only:
                    # Critic-only warmup. The shoot value head predicts 21.95
                    # here against actual returns of ~0.4 (measured), because
                    # shoot's reward scale is nothing like dm_soccer's +/-1.
                    # A miscalibrated critic makes every early advantage a
                    # readout of its own bias, and those updates would be spent
                    # damaging the one thing the warm start bought. So: hold
                    # the policy exactly still until the critic is calibrated.
                    loss = self.vf_coef * vloss
                else:
                    loss = pg + self.vf_coef * vloss - self.ent_coef * ent
                if not critic_only and self.prior is not None and self.w_prior > 0:
                    z = self.ac.z(o)
                    pen = self.prior.neg_log_prob(o, z).mean()
                    loss = loss + self.w_prior * pen
                    stats["prior_nll"] = float(pen.detach())
                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                gnorm = nn.utils.clip_grad_norm_(
                    [p for p in self.ac.parameters() if p.requires_grad],
                    self.max_grad_norm)
                if not torch.isfinite(gnorm):
                    self.n_bad_grads += 1
                    self.opt.zero_grad(set_to_none=True)
                    continue
                self.opt.step()
                if (not self.ac.state_dependent_std
                        and (self.ent_floor is not None
                             or self.ent_ceil is not None)
                        and self.ac.log_std.requires_grad):
                    with torch.no_grad():
                        self.ac.log_std.clamp_(min=self.ent_floor,
                                               max=self.ent_ceil)
                stats.update({"pg": float(pg), "vf": float(vloss),
                              "ent": float(ent), "std": float(d.stddev.mean()),
                              "grad_norm": float(gnorm)})
        return stats

    def train_iter(self, critic_only=False):
        if self.ent_anneal_steps > 0:
            f = min(1.0, self.total_steps / self.ent_anneal_steps)
            self.ent_coef = self.ent_coef_start * (1.0 - f)
        if self.prior_anneal_steps > 0:
            f = min(1.0, self.total_steps / self.prior_anneal_steps)
            self.w_prior = self.w_prior_start * (1.0 - f)
        adv, ret = self.collect()
        stats = self.update(adv, ret, critic_only=critic_only)
        if self.prior is not None:
            # WHICH drill the policy currently looks like. Logged even when
            # w_prior == 0 (the ablation) so the two runs are comparable on the
            # same axis: a run that drifts to uniform responsibilities has left
            # the drill manifold whether or not it was being pulled back.
            o = self.obs_buf[-1]
            with torch.no_grad():
                r = self.prior.responsibilities(o, self.ac.z(o)).mean(0)
            stats["resp"] = {k: round(float(v), 4)
                             for k, v in zip(self.prior.skills, r)}
        stats["critic_only"] = bool(critic_only)
        stats["ret_mean"] = float(ret.mean())
        stats["val_mean"] = float(self.val_buf.mean())
        stats["rew_mean"] = float(self.rew_buf.mean())
        stats["adv_abs"] = float(adv.abs().mean())
        stats["ent_coef"] = self.ent_coef
        stats["w_prior"] = self.w_prior
        return stats


# ---------------------------------------------------------------------------
# rendering
# ---------------------------------------------------------------------------
def make_eval(args, seed=7):
    """One-world env + the 1e probe's renderer, built ONCE and reused.

    Built once on purpose: every WarpSoccer2v2Env allocates its own GPU
    mujoco_warp Data and captures its own CUDA graph, and those do not come
    back with `torch.cuda.empty_cache()`. Rebuilding one per video would grow
    the footprint monotonically across an overnight run on a GPU we share.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    from rower_soccer.warp_port.probe_soccer2v2 import Soccer2v2Renderer
    env = make_env(args, num_worlds=1, seed=seed, use_graph=True)
    return env, Soccer2v2Renderer(env)


def make_bmw(args, seed=7):
    """`--video-worlds` env + renderer for the best/median/worst clip.

    Same build-once discipline as `make_eval`, and for the same reason: each
    env allocates its own mujoco_warp Data and captures its own CUDA graph,
    neither of which `empty_cache()` returns. One env held for the run is a
    fixed cost; one per video is a leak with a nice name.
    """
    os.environ.setdefault("MUJOCO_GL", "egl")
    from rower_soccer.warp_port.probe_soccer2v2 import Soccer2v2Renderer
    env = make_env(args, num_worlds=args.video_worlds, seed=seed,
                   use_graph=True)
    w, h = args.video_panel
    return env, Soccer2v2Renderer(env, width=w, height=h)


@torch.no_grad()
def render_clip(env, ren, ac, path, seconds=15.0, deterministic=True):
    """One world, deterministic (mean-action) play, rendered top-down.

    Deterministic on purpose: at the entropy ceiling the action std is a large
    fraction of the +/-1 action range, and a sampled clip shows the exploration
    noise rather than the policy.
    """
    import imageio
    obs = env.reset()
    steps = int(seconds / 0.025)
    frames = []
    for _ in range(steps):
        d = ac.dist(obs.float())
        a = (d.mean if deterministic else d.sample()).clamp(-1, 1)
        obs, _, done = env.step(a)
        frames.append(ren.frame(env, 0, "topdown"))
        if done:
            break
    with imageio.get_writer(path, fps=40, quality=7) as wr:
        for f in frames:
            wr.append_data(f)
    return path, env.match_stats()


def render_best_median_worst(env, ren, ac, path, rank_key="goals",
                             deterministic=True, seed=0):
    """One full match across `env.n` worlds; film the best, median and worst.

    Ranked by GOALS, not by reward. Under symmetric self-play the team reward
    is zero-sum and one shared policy plays both sides, so per-world reward is
    ~0 by construction and would rank noise. Goals (tie-broken by how far the
    ball actually travelled) is what separates a lively match from a dead one,
    which is the thing a human watching the clip wants to see.

    Reuses `eval_soccer2v2`: the same rollout-then-render-from-qpos path the
    final evaluation uses, so the in-loop clip and the end-of-run evaluation
    cannot drift apart. Recording qpos rather than frames is what makes three
    panels affordable -- frames for a whole match across every world would be
    tens of GB.
    """
    from rower_soccer.warp_port.eval_soccer2v2 import render_grid, run_matches

    rows, qpos = run_matches(env, ac, 1, deterministic=deterministic,
                             record=True, seed=seed)
    order = sorted(range(len(rows)),
                   key=lambda i: (rows[i][rank_key], rows[i]["ball_path"]))
    picks = []
    for label, k in (("worst", 0), ("median", len(order) // 2),
                     ("best", len(order) - 1)):
        r = rows[order[k]]
        picks.append({"match": 0, "world": r["world"], "rank": k,
                      "title": f"{label}  ({r['home']:.0f}-{r['away']:.0f})",
                      "sub": f"ball {r['ball_path']:.0f} m   "
                             f"upright {r['upright']:.2f}"})
    render_grid(env, qpos, picks, path, panel=(ren.width, ren.height),
                cols=3, ren=ren, pip=False)
    n = float(len(rows))
    return path, {
        "goals_per_match": sum(r["goals"] for r in rows) / n,
        "upright": sum(r["upright"] for r in rows) / n,
        "ball_path": sum(r["ball_path"] for r in rows) / n,
        "throw_ins": sum(r["throw_ins"] for r in rows) / n,
        "worlds": len(rows),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--run-name", required=True)
    p.add_argument("--worlds", type=int, default=256,
                   help="4 creatures per world; 256 worlds = 1024 agent rows")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--nconmax", type=int, default=256)
    p.add_argument("--njmax", type=int, default=2048)

    # -- budget -------------------------------------------------------------
    p.add_argument("--steps", type=int, default=2_000_000_000)
    p.add_argument("--iters", type=int, default=0, help="0 = unlimited")
    p.add_argument("--minutes", type=float, default=0.0, help="0 = unlimited")

    # -- PPO ----------------------------------------------------------------
    p.add_argument("--rollout", type=int, default=64)
    p.add_argument("--minibatches", type=int, default=8)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995,
                   help="control dt is 0.025 s, so 0.99 is a 2.5 s horizon -- "
                        "too short to connect a shot to the goal it scores. "
                        "0.995 is 5 s.")
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.003)
    p.add_argument("--ent-floor", type=float, default=-2.5)
    p.add_argument("--ent-ceil", type=float, default=0.0)
    p.add_argument("--ent-anneal-steps", type=int, default=0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)

    # -- policy -------------------------------------------------------------
    p.add_argument("--z-dim", type=int, default=16)
    p.add_argument("--state-dependent-std", action="store_true")
    p.add_argument("--init-from", default="runs_v2/s5_c_all/best.pt",
                   help="drill checkpoint to warm start from. shoot is the "
                        "right one: soccer2v2's first 13 task dims ARE shoot's "
                        "task block, so its task encoder transfers too.")
    p.add_argument("--no-splice", action="store_true",
                   help="ablation: drop task_enc.0/critic.0 on the width "
                        "mismatch (ppo.load_pretrained's behaviour) instead of "
                        "splicing shoot's columns in")
    p.add_argument("--freeze-decoder", action="store_true", default=True)
    p.add_argument("--no-freeze-decoder", dest="freeze_decoder",
                   action="store_false",
                   help="ablation only. Unfreezing the decoder breaks the "
                        "z-space contract the drills and the future BC corpus "
                        "share.")
    p.add_argument("--freeze-log-std", action="store_true")
    p.add_argument("--keep-value-head", action="store_true",
                   help="keep shoot's value_net output layer. Off by default: "
                        "MEASURED, it predicts 21.95 on football states whose "
                        "actual returns are ~0.4, because shoot's reward scale "
                        "(goal bonus 5, strike term) is nothing like "
                        "dm_soccer's +/-1. The critic TRUNK is kept either way "
                        "-- only the last linear layer is zeroed.")
    p.add_argument("--critic-warmup-iters", type=int, default=10,
                   help="iterations at the start that update ONLY the critic. "
                        "The policy is held exactly still while the value "
                        "function calibrates to the new reward scale, so the "
                        "warm start is not spent on advantages that are just "
                        "the critic's own bias.")

    # -- drill-prior mixture (paper Eq. 5) ----------------------------------
    p.add_argument("--w-prior", type=float, default=0.001,
                   help="weight on -log p_mixture(z). 0 removes the term "
                        "entirely (the ablation). See the module docstring for "
                        "why this is Eq. 5 with a deterministic z. 0.001 was "
                        "SIZED, not guessed: on a real warm-started rollout "
                        "||grad prior_nll|| = 989 against ||grad pg|| = 3.10, "
                        "so 0.001 makes the regulariser ~0.32x the policy "
                        "gradient -- a constraint, not a whisper, and not a "
                        "second objective. 0.003 would make it 0.96x.")
    p.add_argument("--no-prior", action="store_true",
                   help="do not even LOAD the priors (skips the diagnostic "
                        "too). For the ablation you want --w-prior 0, which "
                        "removes the loss term but keeps the diagnostic.")
    p.add_argument("--prior-dir", default="runs_v2/_priors")
    p.add_argument("--prior-skills", nargs="*", default=list(PRIOR_SKILLS))
    p.add_argument("--prior-anneal-steps", type=int, default=0,
                   help="linearly decay --w-prior to 0 over this many env "
                        "steps (0 = constant)")

    # -- env / reward -------------------------------------------------------
    p.add_argument("--creature-xml", default="creature_configs/ant.xml")
    p.add_argument("--match-secs", type=float, default=45.0,
                   help="dm_soccer's match length, as game/match.py uses")
    p.add_argument("--pitch-scale", type=float, default=0.3125)
    p.add_argument("--ball-radius", type=float, default=0.15)
    p.add_argument("--ball-mass", type=float, default=0.045)
    p.add_argument("--spawn", default="mirror", choices=("mirror", "uniform"))
    p.add_argument("--ball-jitter", type=float, default=0.0)
    p.add_argument("--w-goal", type=float, default=1.0)
    p.add_argument("--w-player-to-ball", type=float, default=0.0005,
                   help="rate term. MEASURED on a warm-started rollout: the "
                        "shoot-initialised ant closes on the ball at ~2 m/s "
                        "mean, so 0.0005 is worth ~0.9 over a 45 s match, i.e. "
                        "about one goal. (0.002, the first guess, measured 3.5 "
                        "per match -- 3.5x a goal, which would have made the "
                        "shaping the task.)")
    p.add_argument("--w-ball-to-goal", type=float, default=0.05,
                   help="potential term: driving the ball the full 26 m of "
                        "pitch is worth ~1.3, i.e. about one goal")
    p.add_argument("--shaping-anneal-steps", type=int, default=200_000_000,
                   help="linearly take env.shaping_scale 1 -> 0 over this many "
                        "env steps, so the SPARSE goal reward is what the run "
                        "ends up optimising")
    p.add_argument("--energy-coef", type=float, default=0.0)
    p.add_argument("--smooth-coef", type=float, default=0.0)

    # -- opponent pool (default OFF) ----------------------------------------
    p.add_argument("--opponent-pool", action="store_true",
                   help="UNTESTED AT LENGTH. Drive the away slots of a random "
                        "share of worlds with a frozen past checkpoint and "
                        "mask those rows out of the update.")
    p.add_argument("--pool-size", type=int, default=5)
    p.add_argument("--pool-prob", type=float, default=0.3)
    p.add_argument("--pool-every", type=int, default=200,
                   help="iterations between snapshots into the pool")

    # -- plumbing -----------------------------------------------------------
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--ckpt-secs", type=float, default=900.0)
    p.add_argument("--video-secs", type=float, default=900.0,
                   help="0 disables in-loop video")
    p.add_argument("--video-worlds", type=int, default=32,
                   help="worlds in the video rollout -- sampled MORE widely "
                        "than you might expect, because world count is nearly "
                        "free here. Measured on this box under contention: "
                        "32 worlds / 800x600 panels cost 60.6 s per event and "
                        "8 worlds / 480x360 cost 49.3 s. Cutting worlds 4x "
                        "barely moved it, because a match is 1,800 SEQUENTIAL "
                        "env steps whatever the batch width, and the GPU "
                        "absorbs the width. So the lever on cost is the "
                        "CADENCE (--video-secs), not this; and since more "
                        "worlds buy a better-sampled best/median/worst for "
                        "almost nothing, take them.")
    p.add_argument("--video-panel", type=int, nargs=2, default=(640, 480),
                   metavar=("W", "H"))
    p.add_argument("--video-rank", default="goals",
                   help="per-world key to rank best/median/worst on. NOT "
                        "reward: self-play reward is zero-sum and ~0 per world")
    p.add_argument("--video-mode", default="bmw", choices=("bmw", "single"),
                   help="bmw = best/median/worst over --video-worlds; single "
                        "= the old one-world clip")
    p.add_argument("--stop-file", default="",
                   help="touch this path to end the run cleanly at the next "
                        "iteration boundary. NEVER kill a CUDA process under "
                        "MPS -- killing one client can corrupt the live "
                        "others, which has destroyed two runs on this project. "
                        "Without this flag the only way to stop a run early is "
                        "the dangerous one, which is why it exists.")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="creature-soccer")
    p.add_argument("--clip-secs", type=float, default=15.0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--gcs-bucket", default="",
                   help="'' disables the remote backup")
    return p


def run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    run_dir = os.path.join("runs_v2", args.run_name)
    # `train.log` is excluded: the documented launch redirects the process's
    # stdout into the run directory, which the shell creates BEFORE python
    # starts -- so a fresh run would otherwise refuse itself.
    existing = [f for f in (os.listdir(run_dir) if os.path.isdir(run_dir) else [])
                if f != "train.log"]
    if existing and not args.resume:
        raise SystemExit(f"{run_dir} exists and is non-empty ({existing[:4]}). "
                         f"Pass --resume or pick a different --run-name.")
    os.makedirs(os.path.join(run_dir, "videos"), exist_ok=True)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()
    config = {**vars(args), "git_sha": sha, "backend": "mujoco_warp",
              "task": "soccer2v2", "unit": "D1-1f",
              "kl_to_bc": False,
              "kl_to_bc_note": "stage 5 demos do not exist yet; this run is "
                               "the no-BC control that makes them measurable"}
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=1)

    env = make_env(args, num_worlds=args.worlds, seed=args.seed)
    dev = str(env.device)
    ac = ActorCritic(env.obs_dim, env.act_dim,
                     proprio_indices=env.proprio_indices.tolist(),
                     task_indices=env.task_indices.tolist(),
                     z_dim=args.z_dim,
                     state_dependent_std=args.state_dependent_std).to(dev)
    warm = None
    if args.init_from:
        warm = load_warm_start(ac, args.init_from, env.n_proprio, device=dev,
                               splice=not args.no_splice)

    if warm is not None and not args.keep_value_head:
        with torch.no_grad():
            ac.value_net.weight.zero_()
            ac.value_net.bias.zero_()
        print("[setup] value head ZEROED (critic trunk kept); shoot's head "
              "predicts ~22 where football returns are ~0.4", flush=True)

    if args.freeze_decoder:
        frozen = 0
        for mod in (ac.mlp_extractor.decoder, ac.action_net):
            for prm in mod.parameters():
                prm.requires_grad_(False)
                frozen += prm.numel()
        if args.freeze_log_std:
            ac.log_std.requires_grad_(False)
            frozen += ac.log_std.numel()
        live = sum(p.numel() for p in ac.parameters() if p.requires_grad)
        print(f"[setup] decoder + action head FROZEN: {frozen:,} params held, "
              f"{live:,} trainable (the high-level z policy + critic)",
              flush=True)

    # Built even at --w-prior 0: the mixture responsibilities are the
    # diagnostic that makes the ON and OFF runs comparable. Only the LOSS is
    # gated on w_prior (an `if`, not a multiply by zero).
    prior = None
    if not args.no_prior:
        prior = DrillPriorMixture(args.prior_dir, args.prior_skills,
                                  env.proprio_indices.tolist(),
                                  env.task_indices.tolist(), device=dev)
        print(f"[setup] drill-prior mixture over {args.prior_skills} "
              f"(uniform alpha), w_prior={args.w_prior}"
              f"{' -- DIAGNOSTIC ONLY, not in the loss' if args.w_prior <= 0 else ''}",
              flush=True)

    trainer = SelfPlayPPO(
        env, ac, lr=args.lr, rollout_len=args.rollout,
        minibatches=args.minibatches, epochs=args.epochs, gamma=args.gamma,
        gae_lambda=args.gae_lambda, clip=args.clip, ent_coef=args.ent_coef,
        vf_coef=args.vf_coef, max_grad_norm=args.max_grad_norm,
        ent_floor=args.ent_floor, ent_ceil=args.ent_ceil,
        ent_anneal_steps=args.ent_anneal_steps, prior=prior,
        w_prior=args.w_prior, prior_anneal_steps=args.prior_anneal_steps,
        device=dev)

    if args.opponent_pool:
        opp = ActorCritic(env.obs_dim, env.act_dim,
                          proprio_indices=env.proprio_indices.tolist(),
                          task_indices=env.task_indices.tolist(),
                          z_dim=args.z_dim,
                          state_dependent_std=args.state_dependent_std).to(dev)
        g = torch.Generator(device=dev).manual_seed(args.seed + 991)
        trainer.enable_pool(opp, args.pool_size, args.pool_prob, g)
        trainer.snapshot_pool()
        print(f"[setup] opponent pool ON: size {args.pool_size}, "
              f"{args.pool_prob:.0%} of worlds, snapshot every "
              f"{args.pool_every} iters", flush=True)

    ckpt_path = os.path.join(run_dir, "checkpoint.pt")
    latest_path = os.path.join(run_dir, "latest.pt")
    log_path = os.path.join(run_dir, "log.json")
    start_steps = 0
    rows_prior = []
    if args.resume and os.path.exists(ckpt_path):
        start_steps = load_checkpoint(trainer, ckpt_path)
        # Carry the existing log forward. Without this the resumed run opens
        # log.json fresh and the previous run's curve is gone the first time it
        # writes -- which is exactly what happened resuming the 2B-step run: a
        # 560 kB, 763-point history became 1.7 kB. The series was recoverable
        # from train.log only because the monitor line happens to carry
        # goals/match; that is luck, not a design.
        if os.path.exists(log_path):
            try:
                with open(log_path) as fh:
                    prev = json.load(fh)
                rows_prior = prev.get("iters", prev) if isinstance(prev, dict) else prev
                rows_prior = [r for r in rows_prior if isinstance(r, dict)]
            except Exception as e:
                print(f"[setup] WARNING: could not read {log_path} ({e}); "
                      f"the previous curve will NOT be carried forward",
                      flush=True)
        print(f"[setup] resumed at step {start_steps:,}, "
              f"carrying {len(rows_prior)} logged rows forward", flush=True)

    print(f"[setup] worlds={env.n} agents/world={env.n_agents} "
          f"rows={trainer.N} obs={env.obs_dim} act={env.act_dim} "
          f"proprio={env.n_proprio} task={env.task_dim} "
          f"match_steps={env.episode_steps} "
          f"steps/iter={trainer.T * trainer.N:,}", flush=True)

    log = {"config": config, "warm_start": None if warm is None else
           {k: (v if not isinstance(v, list) else v)
            for k, v in warm.items() if k != "loaded"},
           "warm_start_loaded_n": None if warm is None else len(warm["loaded"]),
           "iters": list(rows_prior)}

    wb = None
    if args.wandb:
        try:
            import wandb
            # id = run name so a resume REATTACHES to the same wandb run
            # instead of opening a second one beside it.
            wb = wandb.init(project=args.wandb_project, name=args.run_name,
                            id=args.run_name, resume="allow", config=config)
            print(f"[setup] wandb: {wb.url}", flush=True)
        except Exception as e:                          # noqa: BLE001
            print(f"[setup] wandb DISABLED ({e!r}) -- training continues",
                  flush=True)
            wb = None

    def flush_log():
        tmp = log_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(log, f)
        os.replace(tmp, log_path)

    eval_pair = None
    bmw_pair = None
    t0 = time.perf_counter()
    deadline = t0 + args.minutes * 60.0 if args.minutes > 0 else float("inf")
    last_ckpt = last_video = t0
    last_steps, last_t = start_steps, t0
    it = 0
    while trainer.total_steps < args.steps and time.perf_counter() < deadline:
        if args.stop_file and os.path.exists(args.stop_file):
            print(f"[setup] stop file {args.stop_file} seen -- finishing "
                  f"cleanly at iter {it}", flush=True)
            break
        if args.iters and it >= args.iters:
            break
        if args.shaping_anneal_steps > 0:
            env.shaping_scale = max(0.0, 1.0 - trainer.total_steps
                                    / args.shaping_anneal_steps)
        stats = trainer.train_iter(critic_only=it < args.critic_warmup_iters)
        it += 1
        now = time.perf_counter()
        if args.opponent_pool and it % args.pool_every == 0:
            trainer.snapshot_pool()
        if it % args.log_every == 0 or it == 1:
            fps = (trainer.total_steps - last_steps) / max(1e-9, now - last_t)
            last_steps, last_t = trainer.total_steps, now
            rec = {"iter": it, "step": trainer.total_steps,
                   "wall_min": (now - t0) / 60.0, "fps": fps,
                   "shaping_scale": float(env.shaping_scale),
                   "diverged_obs": trainer.n_diverged,
                   "diverged_sim": env.n_diverged,
                   "bad_grads": trainer.n_bad_grads,
                   "matches": trainer.matches, **stats}
            if trainer.last_match is not None:
                rec["match"] = trainer.last_match
            log["iters"].append(rec)
            flush_log()
            m = trainer.last_match or {}
            if wb is not None:
                flat = {f"train/{k}": v for k, v in stats.items()
                        if isinstance(v, (int, float))}
                flat.update({f"match/{k}": v for k, v in m.items()
                             if isinstance(v, (int, float))})
                flat.update({"train/fps": fps,
                             "train/shaping_scale": env.shaping_scale,
                             "train/matches": trainer.matches,
                             "train/diverged": trainer.n_diverged})
                wb.log(flat, step=trainer.total_steps)
            print(f"[monitor] it={it} step={trainer.total_steps:,} "
                  f"fps={fps:,.0f} rew={stats['rew_mean']:+.5f} "
                  f"pg={stats.get('pg', float('nan')):+.4f} "
                  f"vf={stats.get('vf', float('nan')):.4f} "
                  f"std={stats.get('std', float('nan')):.3f} "
                  f"V={stats.get('val_mean', float('nan')):.2f}/"
                  f"{stats.get('ret_mean', float('nan')):.2f} "
                  f"prior_nll={stats.get('prior_nll', float('nan')):.2f} "
                  f"resp={stats.get('resp', {})} "
                  f"matches={trainer.matches} "
                  f"goals/match={m.get('goals_per_match', float('nan')):.3f} "
                  f"p0={m.get('p_0_goals', float('nan')):.2f} "
                  f"upright={m.get('upright', float('nan')):.2f} "
                  f"shap={env.shaping_scale:.2f} "
                  f"div={trainer.n_diverged}/{env.n_diverged}", flush=True)
        if now - last_ckpt >= args.ckpt_secs:
            last_ckpt = now
            save_checkpoint(trainer, ckpt_path)
            export_sb3_compatible(ac, latest_path)
            print(f"[monitor] checkpoint at {trainer.total_steps:,}", flush=True)
            if args.gcs_bucket:
                from rower_soccer.warp_port.gcs import sync_async
                for pth in (ckpt_path, latest_path, log_path):
                    sync_async(pth, args.gcs_bucket, args.run_name)
        if args.video_secs > 0 and now - last_video >= args.video_secs:
            last_video = now
            vp = os.path.join(run_dir, "videos",
                              f"step_{trainer.total_steps:012d}.mp4")
            try:
                tv = time.perf_counter()
                if args.video_mode == "bmw":
                    if bmw_pair is None:
                        bmw_pair = make_bmw(args)
                    _, vst = render_best_median_worst(
                        *bmw_pair, ac, vp, rank_key=args.video_rank,
                        seed=int(trainer.total_steps % 100000))
                else:
                    if eval_pair is None:
                        eval_pair = make_eval(args)
                    _, vst = render_clip(*eval_pair, ac, vp,
                                         seconds=args.clip_secs)
                dtv = time.perf_counter() - tv
                # Cost is PRINTED, not estimated: the whole point of the
                # cadence knob is that someone can see what it is spending.
                print(f"[monitor] video {vp} {vst} "
                      f"cost={dtv:.1f}s ({100 * dtv / args.video_secs:.1f}% "
                      f"of the {args.video_secs:.0f}s cadence)", flush=True)
                if wb is not None:
                    import wandb
                    wb.log({"video/match": wandb.Video(vp, format="mp4"),
                            "video/cost_s": dtv,
                            **{f"video/{k}": v for k, v in vst.items()
                               if isinstance(v, (int, float))}},
                           step=trainer.total_steps)
            except Exception as e:                     # noqa: BLE001
                print(f"[monitor] video FAILED: {e!r}", flush=True)

    save_checkpoint(trainer, ckpt_path)
    export_sb3_compatible(ac, os.path.join(run_dir, "final.pt"))
    export_sb3_compatible(ac, latest_path)
    flush_log()
    print(f"[setup] done: {it} iters, {trainer.total_steps:,} env steps, "
          f"{(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    return trainer


def main():
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
