"""Compact GPU PPO for Warp-batched envs.

Reuses rower_soccer.models.latent_policy.LatentExtractor so weights are
interchangeable with the SB3 CPU path (same module names; export helpers
below). Everything (rollout buffer, GAE, updates) stays on-GPU.

Exploration option: AR(1)-correlated action noise in z-space is emulated by
an auxiliary smoothness penalty plus an entropy floor; the paper's NPMP prior
motivates temporally-correlated exploration for gait discovery.
"""

import numpy as np
import torch
import torch.nn as nn

from rower_soccer.models.latent_policy import LatentExtractor

# A world whose |obs| exceeds this is diverging, not merely energetic. Everything in
# the observation is bounded by real geometry -- the pitch is 48 m, the ball peaks
# around 30 m/s, touch is divided by 10000 -- so anything past ~100 is unphysical and
# 1e3 is a generous ceiling. Measured on the run that crashed: |obs| reached 8,168
# while staying perfectly finite, which is precisely what an isfinite() check misses.
OBS_SANITY_LIMIT = 1.0e3


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, proprio_indices, task_indices,
                 z_dim=16, log_std_init=0.0, state_dependent_std=False,
                 log_std_min=-4.0, log_std_max=0.0):
        super().__init__()
        self.mlp_extractor = LatentExtractor(
            proprio_indices=proprio_indices, task_indices=task_indices, z_dim=z_dim)
        self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, act_dim)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # State-dependent std (gSDE-flavored): the exploration noise is a learned
        # function of the actor latent, so the policy can go QUIET near the ball for
        # fine control and stay noisy in open field. The global-parameter form below
        # cannot do that -- one std for every state -- which is why dribble's precise
        # nudges were drowned by std=0.30. Clamp range [log_std_min, log_std_max] is
        # wide and LOW at the bottom (exp(-4)=0.018) precisely to permit fine control;
        # the trainer's ent_floor is NOT applied to it (that floor is what forbade
        # low std in the first place).
        self.state_dependent_std = state_dependent_std
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        if state_dependent_std:
            self.log_std_net = nn.Linear(self.mlp_extractor.latent_dim_pi, act_dim)
            # Start state-INDEPENDENT (zero weights) at a MODERATE std, then let the
            # policy learn to modulate -- a gentle start, not a jolt. Bias -1.0 => std
            # 0.37, near where follow_base trained (0.30); a fresh head at the default
            # log_std_init=0 would be std 1.0, which would swamp the warm-started
            # locomotion with noise before the head learns to quiet down.
            nn.init.zeros_(self.log_std_net.weight)
            nn.init.constant_(self.log_std_net.bias, -1.0)
        else:
            self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

    def _log_std(self, lat):
        if self.state_dependent_std:
            return self.log_std_net(lat).clamp(self.log_std_min, self.log_std_max)
        return self.log_std

    @staticmethod
    def _clean(obs):
        """Sanitize observations at the network boundary. This is THE guarantee that
        the policy never emits a NaN mean and crashes Normal(): a network cannot
        produce a non-finite output from a finite, bounded input.

        Every upstream guard (env._sanitize on qvel/qpos, ppo.collect on obs2) has a
        seam -- a contact impulse spikes the accelerometer to inf/NaN in a single
        substep while integrated velocity is still moderate, so it slips past a
        velocity check, and clamp() does not repair a NaN (clamp(NaN)=NaN). Those
        guards still matter (they stop garbage propagating in the sim), but this is
        the one that makes the crash impossible. Normal obs live in ~[-50, 50]
        (accelerometer is already scaled /100 and clamped), so +/-100 never touches
        real data and only bounds divergence garbage.
        """
        return torch.nan_to_num(obs, nan=0.0, posinf=100.0, neginf=-100.0).clamp(-100.0, 100.0)

    def dist(self, obs):
        lat = self.mlp_extractor.forward_actor(self._clean(obs))
        mean = self.action_net(lat)
        return torch.distributions.Normal(mean, self._log_std(lat).exp())

    def value(self, obs):
        return self.value_net(self.mlp_extractor.forward_critic(self._clean(obs))).squeeze(-1)

    @torch.no_grad()
    def act(self, obs):
        d = self.dist(obs)
        a = d.sample()
        return a, d.log_prob(a).sum(-1), self.value(obs)

    def z(self, obs):
        return self.mlp_extractor.z(self._clean(obs))


class SimpleActorCritic(nn.Module):
    """Plain-MLP baseline: obs -> action, with NO latent bottleneck and NO shared
    decoder. Same public interface as ActorCritic (dist/value/act/_clean), so it
    drops straight into PPOTrainer.

    This is the control experiment for the central architectural question we never
    tested: is the policy->z->decoder structure itself what handicaps dribble? If
    this plain policy learns a task ActorCritic cannot, the bottleneck/decoder is the
    culprit; if it also cannot, the task/body is, not our design. Capacity is matched
    to ActorCritic's actor+critic (two 256-wide ELU trunks) for a fair comparison.
    """
    def __init__(self, obs_dim, act_dim, hidden=256, log_std_init=0.0, **_ignored):
        super().__init__()
        self.state_dependent_std = False
        def trunk():
            return nn.Sequential(nn.Linear(obs_dim, hidden), nn.ELU(),
                                 nn.Linear(hidden, hidden), nn.ELU())
        self.pi, self.vf = trunk(), trunk()
        self.action_net = nn.Linear(hidden, act_dim)
        self.value_net = nn.Linear(hidden, 1)
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

    def dist(self, obs):
        lat = self.pi(ActorCritic._clean(obs))
        return torch.distributions.Normal(self.action_net(lat), self.log_std.exp())

    def value(self, obs):
        return self.value_net(self.vf(ActorCritic._clean(obs))).squeeze(-1)

    @torch.no_grad()
    def act(self, obs):
        d = self.dist(obs)
        a = d.sample()
        return a, d.log_prob(a).sum(-1), self.value(obs)


class PPOTrainer:
    def __init__(self, env, ac: ActorCritic, lr=3e-4, rollout_len=64,
                 minibatches=8, epochs=4, gamma=0.99, gae_lambda=0.95,
                 clip=0.2, ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5,
                 z_smooth_coef=0.0, ent_floor=None, ent_ceil=None, device="cuda",
                 ent_anneal_steps=0, distributed=False):
        # distributed: data-parallel PPO (DD-PPO). Each rank runs its own env +
        # a policy replica; gradients are all-reduced (averaged) before every
        # optimizer step, so replicas stay bit-identical and the effective batch
        # is world_size x the per-rank batch. See train_worm_fetch_ddp.py.
        self.distributed = distributed
        self.env, self.ac = env, ac.to(device)
        self.opt = torch.optim.Adam(ac.parameters(), lr=lr)
        self.T, self.N = rollout_len, env.n
        self.minibatches, self.epochs = minibatches, epochs
        self.gamma, self.lam, self.clip = gamma, gae_lambda, clip
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        # Linearly anneal the entropy bonus ent_coef -> 0 over the first
        # ent_anneal_steps env-steps (0 = off, constant coef). This is the fix for
        # the follow_v5 collapse: once the policy converges, the fixed entropy bonus
        # became the loudest term left and inflated std back to the ceiling, drowning
        # the policy in its own exploration noise. Decaying it removes that pressure
        # exactly when it stops being useful.
        self.ent_coef_start = ent_coef
        self.ent_anneal_steps = ent_anneal_steps
        self.max_grad_norm = max_grad_norm
        self.z_smooth_coef = z_smooth_coef
        self.ent_floor = ent_floor  # min log_std, e.g. -1.5
        # max log_std. Actions are clamped to [-1, 1] in collect(), so a std
        # much above ~1 samples almost entirely into the clamp and explores at
        # random. Without a ceiling log_std drifts up unboundedly.
        self.ent_ceil = ent_ceil
        self.device = device
        d = device
        self.obs_buf = torch.zeros(self.T, self.N, env.obs_dim, device=d)
        self.act_buf = torch.zeros(self.T, self.N, env.act_dim, device=d)
        self.logp_buf = torch.zeros(self.T, self.N, device=d)
        self.rew_buf = torch.zeros(self.T, self.N, device=d)
        self.val_buf = torch.zeros(self.T, self.N, device=d)
        self.done_buf = torch.zeros(self.T, self.N, device=d)
        self._obs = env.reset()
        self.total_steps = 0
        # world-steps whose physics diverged (non-finite OR |obs| > OBS_SANITY_LIMIT);
        # see collect(). Should stay at 0 or very near it -- a climbing count means the
        # contact model is wrong and the run is training on garbage.
        self.n_diverged = 0
        # minibatch updates dropped because the gradient was non-finite. The last line
        # of defence; if this is nonzero the guards above are leaking.
        self.n_bad_grads = 0

    def collect(self):
        ep_returns = []
        for t in range(self.T):
            a, logp, v = self.ac.act(self._obs)
            obs2, rew, done = self.env.step(a.clamp(-1, 1))

            # A single diverged world out of 2048 must not kill an 800M-step run.
            #
            # mujoco_warp occasionally blows a contact up: the ball (45 g, priority=1,
            # so its solref governs contact with a 22 kg creature segment) gets driven
            # deep into a geom and explosively ejected -- it leaves at 20-30 m/s off a
            # 0.9 m/s worm, which is the solver injecting energy. Rarely, that state
            # goes non-finite. It then propagates: obs -> network -> NaN action mean ->
            # ValueError out of Normal(), and the process dies. This killed
            # dribble_paper_v5 (17.7M steps) and dribble_paper_v6 (106M).
            #
            # scene.py's ball solref is tuned to make this rare; this makes it
            # survivable. Zero the offending worlds' observations and rewards so they
            # contribute nothing to the update, and count them. A steadily climbing
            # count means the physics is wrong, not merely unlucky -- so it is logged,
            # not silently swallowed.
            # NOT just isfinite. dribble_paper_v7 died at 149M with diverged=0: the
            # poison was large-but-FINITE. A diverging contact produced |obs| = 8,168
            # -- perfectly finite, ~80x anything physical (the pitch is 48 m; ball
            # speed peaks ~30 m/s) -- which detonates exp(logp - logp_old) in the PPO
            # ratio, yields a NaN gradient, and then clip_grad_norm_ SCALES EVERY
            # GRADIENT BY NaN (it clips, it does not sanitise). Weights poisoned,
            # action mean NaN, process dead, with the obs guard never firing.
            bad = ~torch.isfinite(obs2).all(dim=-1) | (obs2.abs().amax(dim=-1) > OBS_SANITY_LIMIT)
            if bad.any():
                self.n_diverged += int(bad.sum())
                obs2 = torch.where(bad.unsqueeze(-1), torch.zeros_like(obs2), obs2)
                obs2 = torch.nan_to_num(obs2, nan=0.0, posinf=0.0, neginf=0.0)
                rew = torch.where(bad, torch.zeros_like(rew), rew)
                v = torch.where(bad, torch.zeros_like(v), v)

            self.obs_buf[t] = self._obs
            self.act_buf[t] = a
            self.logp_buf[t] = logp
            self.rew_buf[t] = rew
            self.val_buf[t] = v
            self.done_buf[t] = float(done)
            if done:  # world-synchronized episode end
                ep_returns.append(float(rew.mean()))
                obs2 = self.env.reset()
            self._obs = obs2
        self.total_steps += self.T * self.N
        with torch.no_grad():
            last_val = self.ac.value(self._obs)
        # GAE (time-limit truncation: bootstrap through resets)
        adv = torch.zeros_like(self.rew_buf)
        gae = torch.zeros(self.N, device=self.device)
        for t in reversed(range(self.T)):
            nonterminal = 1.0  # truncation, not failure -> always bootstrap
            next_v = last_val if t == self.T - 1 else self.val_buf[t + 1]
            delta = self.rew_buf[t] + self.gamma * next_v * nonterminal - self.val_buf[t]
            gae = delta + self.gamma * self.lam * nonterminal * gae
            # reset advantage accumulation across episode boundary
            if self.done_buf[t, 0] > 0:
                gae = delta
            adv[t] = gae
        ret = adv + self.val_buf
        return adv, ret

    def update(self, adv, ret):
        B = self.T * self.N
        obs = self.obs_buf.reshape(B, -1)
        act = self.act_buf.reshape(B, -1)
        logp_old = self.logp_buf.reshape(B)
        adv_f = adv.reshape(B)
        ret_f = ret.reshape(B)
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
        idx = torch.randperm(B, device=self.device)
        mb = B // self.minibatches
        stats = {}
        for _ in range(self.epochs):
            for i in range(self.minibatches):
                j = idx[i * mb:(i + 1) * mb]
                d = self.ac.dist(obs[j])
                logp = d.log_prob(act[j]).sum(-1)
                ratio = (logp - logp_old[j]).exp()
                pg = -torch.min(
                    ratio * adv_f[j],
                    ratio.clamp(1 - self.clip, 1 + self.clip) * adv_f[j]).mean()
                v = self.ac.value(obs[j])
                vloss = ((v - ret_f[j]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pg + self.vf_coef * vloss - self.ent_coef * ent
                if self.z_smooth_coef > 0:
                    z = self.ac.z(obs[j])
                    loss = loss + self.z_smooth_coef * (z ** 2).mean()
                self.opt.zero_grad()
                loss.backward()
                # DD-PPO: average this minibatch's gradients across all ranks
                # BEFORE the finite-check + step. Averaging first means a NaN on
                # ANY rank poisons the reduced grad on ALL ranks, so every rank
                # takes the same skip decision below and the replicas never
                # diverge. Comm is one all-reduce of a ~1-2M-param net per
                # minibatch -- a few MB, cheap relative to the rollout.
                if self.distributed:
                    import torch.distributed as dist
                    ws = dist.get_world_size()
                    for p in self.ac.parameters():
                        if p.grad is not None:
                            # SUM then divide: works on both nccl and gloo
                            # (ReduceOp.AVG is nccl-only).
                            dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
                            p.grad.div_(ws)
                # clip_grad_norm_ CLIPS, it does not SANITISE: given one NaN gradient
                # the total norm is NaN and every gradient is scaled by NaN, which
                # poisons the weights permanently. It returns the pre-clip norm, so
                # check it and drop the update rather than commit a NaN.
                gnorm = nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                if not torch.isfinite(gnorm):
                    self.n_bad_grads += 1
                    self.opt.zero_grad(set_to_none=True)
                    continue
                self.opt.step()
                # Only the global-parameter std is clamped here. State-dependent std
                # has no global param -- its bounds live in ActorCritic._log_std
                # (log_std_min/max), deliberately wide so fine control is allowed.
                if (not self.ac.state_dependent_std
                        and (self.ent_floor is not None or self.ent_ceil is not None)):
                    with torch.no_grad():
                        self.ac.log_std.clamp_(min=self.ent_floor, max=self.ent_ceil)
                # d.stddev works for both the global param and the state-dependent
                # head -- it is the actual std of this minibatch's distribution.
                stats = {"pg": float(pg), "vf": float(vloss), "ent": float(ent),
                         "std": float(d.stddev.mean())}
        return stats

    def train_iter(self):
        if self.ent_anneal_steps > 0:
            frac = min(1.0, self.total_steps / self.ent_anneal_steps)
            self.ent_coef = self.ent_coef_start * (1.0 - frac)
        adv, ret = self.collect()
        stats = self.update(adv, ret)
        stats["ep_rew_env_mean"] = float(self.rew_buf.mean() * self.env.episode_steps)
        stats["ent_coef"] = self.ent_coef
        return stats


def save_checkpoint(trainer: "PPOTrainer", path):
    """Atomic full checkpoint (model + optimizer + step count). Overwrites:
    writes to <path>.tmp then os.replace, so a kill mid-write never corrupts
    the previous checkpoint and disk usage stays at one file."""
    import os
    tmp = path + ".tmp"
    torch.save({
        "ac": trainer.ac.state_dict(),
        "opt": trainer.opt.state_dict(),
        "total_steps": trainer.total_steps,
    }, tmp)
    os.replace(tmp, path)


def load_checkpoint(trainer: "PPOTrainer", path):
    sd = torch.load(path, map_location=trainer.device, weights_only=True)
    trainer.ac.load_state_dict(sd["ac"])
    trainer.opt.load_state_dict(sd["opt"])
    trainer.total_steps = int(sd["total_steps"])
    return trainer.total_steps


def export_sb3_compatible(ac: ActorCritic, path):
    """Save weights loadable into LatentActorCriticPolicy (CPU eval path)."""
    # The plain baseline (SimpleActorCritic) has no mlp_extractor, so the structured
    # SB3 export does not apply -- save its flat state_dict instead. It is never
    # warm-started FROM or loaded into an SB3 policy; this just needs to not crash and
    # to be resumable.
    if not hasattr(ac, "mlp_extractor"):
        torch.save({"plain_state_dict": ac.state_dict()}, path)
        return
    out = {
        "mlp_extractor": ac.mlp_extractor.state_dict(),
        "action_net": ac.action_net.state_dict(),
        "value_net": ac.value_net.state_dict(),
    }
    if ac.state_dependent_std:
        out["log_std_net"] = ac.log_std_net.state_dict()
    else:
        out["log_std"] = ac.log_std.detach().cpu()
    torch.save(out, path)


def _flatten_checkpoint(sd):
    """Accept either a resume checkpoint ({'ac': ...}) or an
    export_sb3_compatible export, and return one flat ActorCritic state_dict."""
    if "ac" in sd:
        return sd["ac"]
    flat = {f"mlp_extractor.{k}": v for k, v in sd["mlp_extractor"].items()}
    flat.update({f"action_net.{k}": v for k, v in sd["action_net"].items()})
    flat.update({f"value_net.{k}": v for k, v in sd["value_net"].items()})
    if "log_std_net" in sd:
        flat.update({f"log_std_net.{k}": v for k, v in sd["log_std_net"].items()})
    if "log_std" in sd:
        flat["log_std"] = sd["log_std"]
    return flat


def load_pretrained(ac: ActorCritic, path, device="cpu", verbose=True):
    """Warm-start one drill's ActorCritic from another drill's checkpoint.

    Copies every parameter whose shape matches and skips the rest, because the
    two drills differ only in their task-observation width:

      transfers : proprio_enc, expert, z_proj, decoder, action_net, log_std,
                  value_net  -- crucially the DECODER, which is the motor skill
                  (it sees proprio + z only, never task obs, so it is
                  task-width-independent by construction)
      re-init   : task_enc.0 (follow's task obs is 4 wide, dribble's is 8) and
                  critic.0 (input is proprio+task, so it widens too)

    The index buffers p_idx/t_idx are never copied: they describe THIS env's
    observation layout, and overwriting them with the source env's would slice
    the wrong columns out of every observation -- silently, since the shapes of
    everything downstream would still line up.

    Returns (n_loaded, n_skipped).
    """
    sd = _flatten_checkpoint(torch.load(path, map_location=device, weights_only=True))
    own = ac.state_dict()
    skip_buffers = {"mlp_extractor.p_idx", "mlp_extractor.t_idx"}
    loaded, skipped = [], []
    for k, v in own.items():
        if k in skip_buffers or k not in sd:
            skipped.append(k)
            continue
        if sd[k].shape != v.shape:
            skipped.append(f"{k} {tuple(sd[k].shape)}->{tuple(v.shape)}")
            continue
        v.copy_(sd[k].to(v.device))
        loaded.append(k)
    ac.load_state_dict(own)
    if verbose:
        print(f"[warm-start] {path}", flush=True)
        print(f"[warm-start] loaded {len(loaded)} tensors "
              f"(incl. decoder + action_net = the low-level controller)", flush=True)
        for s in skipped:
            if not any(s.startswith(b) for b in skip_buffers):
                print(f"[warm-start]   re-init: {s}", flush=True)
    return len(loaded), len(skipped)


def load_into_sb3_policy(policy, path):
    sd = torch.load(path, map_location=policy.device, weights_only=True)
    policy.mlp_extractor.load_state_dict(sd["mlp_extractor"])
    policy.action_net.load_state_dict(sd["action_net"])
    policy.value_net.load_state_dict(sd["value_net"])
    with torch.no_grad():
        policy.log_std.copy_(sd["log_std"].to(policy.device))
