"""PIPELINE_V2 stage 3: distil each drill expert into a target-agnostic prior.

An expert is `pi_k(z | proprio, task_obs)` and needs an aim target. A **drill
prior** is `pi_hat_k(z | football_obs)`: the same skill, fitted by KL in
z-space, seeing only observations that exist in a football match -- no targets.
They are what the self-play fine-tune (stage 6) regularises against, the paper's
Eq. 5-6 mixture.

    PYTHONPATH=. MUJOCO_GL=osmesa .venv/bin/python \
        -m rower_soccer.warp_port.drill_prior --skill shoot \
        --ckpt runs_v2/s5_c_all/best.pt --out runs_v2/_priors/shoot.pt

--------------------------------------------------------------------------
Which observations survive into football, per skill
--------------------------------------------------------------------------
Read off dm_control's `locomotion/soccer/observables.py` against each drill's
own task block. A match gives every player: proprioception + kinematic sensors
+ `prev_action`; the ball as `ball_ego_pos` / `ball_ego_linvel` /
`ball_ego_angvel`; each other player as `ego_pos` / `ego_linvel` /
`ego_orientation` / `egocentric_end_effectors_xpos`; and eight egocentric arena
vectors including `opponent_goal_mid`, `opponent_goal_back_left` and
`opponent_goal_front_right`.

| skill   | task block                                            | survives |
|---------|-------------------------------------------------------|----------|
| follow  | target_ego3(3) + target_future3(3)                    | NOTHING  |
| dribble | ball_ego(6) + target(3) + target_future(3)            | 0:6      |
| kick    | ball_ego(6) + target(3) + cmd_dir(3) + deadline(2)    | 0:6      |
| shoot   | ball_ego(6) + goal_mid(3) + post_l(2) + post_r(2)     | ALL 13   |

Shoot keeps everything, and not by luck: `shoot_env`'s docstring records that
its goal observation was chosen as the game's `opponent_goal_mid` plus a corner
representation precisely "because this obs survives distillation into the shoot
prior and the prior is evaluated on GAME observations".

**Follow is the one to watch.** Its prior sees proprio alone, so it cannot be
goal-seeking -- a target-free follow prior is essentially "run". That is what a
paper-style prior should be (characteristic motion, not goal-seeking), but it is
also the largest information loss of the four and the most likely to collapse
into something degenerate. Gate it on video, hard.

--------------------------------------------------------------------------
Why a Gaussian prior and not a regression
--------------------------------------------------------------------------
The expert's `z` is deterministic (`LatentExtractor.z` is a plain projection;
the policy's stochasticity lives in the ACTION log_std, not in z). Fitting
`ẑ ≈ z` by MSE would be the obvious thing and is the wrong shape for what
stage 6 needs: Eq. 5's regulariser is a KL to a *distribution*. So the prior is
Normal(mu(obs), sigma) fitted by the NLL of the expert's z, which is MSE plus a
learned scale -- same fit, and the scale is exactly the "how confident is this
skill here" signal the mixture weighting wants.
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn

# Columns of each drill's TASK block that also exist in a football match.
# `follow` keeps none: an empty list means the prior is proprio-only.
FOOTBALL_TASK_COLS = {
    "follow": [],
    "dribble": list(range(6)),      # ball_ego
    "kick": list(range(6)),         # ball_ego
    "shoot": list(range(13)),       # ball_ego + goal_mid + both posts
}


class DrillPrior(nn.Module):
    """`pi_hat(z | football_obs)` as a diagonal Gaussian.

    Mirrors the expert's encoder path (`proprio_enc`, `task_enc`, `expert`,
    `z_proj`) so capacity is comparable and a failure to fit is a statement
    about the information available, not about the network being too small.
    A proprio-only prior (follow) simply has no `task_enc`.
    """

    def __init__(self, n_proprio, n_task, z_dim=16, enc_hidden=128,
                 expert_hidden=256, log_std_init=-1.0):
        super().__init__()
        self.n_proprio, self.n_task, self.z_dim = n_proprio, n_task, z_dim
        # Input and target standardisation, filled by `set_norm` from the
        # collected data. Without it the fit is badly conditioned: proprio
        # spans an accelerometer clamped to +/-50 next to touch sensors scaled
        # to ~1e-4, and the expert's z has sd ~10 while `log_std_init = -1`
        # means sigma 0.37 -- an NLL whose gradient on mu is scaled by
        # 1/sigma^2 ~ 7, which is what made the first fit stall at 63% of a
        # predict-the-mean baseline on the ONE skill whose prior sees exactly
        # what the expert sees.
        self.register_buffer("p_mu", torch.zeros(n_proprio))
        self.register_buffer("p_sd", torch.ones(n_proprio))
        self.register_buffer("t_mu", torch.zeros(max(n_task, 1)))
        self.register_buffer("t_sd", torch.ones(max(n_task, 1)))
        self.register_buffer("z_mu", torch.zeros(z_dim))
        self.register_buffer("z_sd", torch.ones(z_dim))

        def mlp(i, hs):
            layers, last = [], i
            for h in hs:
                layers += [nn.Linear(last, h), nn.ELU()]
                last = h
            return nn.Sequential(*layers)

        self.proprio_enc = mlp(n_proprio, [enc_hidden, enc_hidden])
        self.task_enc = mlp(n_task, [enc_hidden, enc_hidden]) if n_task else None
        trunk_in = enc_hidden * (2 if n_task else 1)
        self.expert = mlp(trunk_in, [expert_hidden])
        self.z_proj = nn.Linear(expert_hidden, z_dim)
        self.log_std = nn.Parameter(torch.full((z_dim,), float(log_std_init)))

    @torch.no_grad()
    def set_norm(self, prop, task, z):
        """Fit the standardisation from the collected data.

        Stored as buffers so a loaded prior needs no external statistics -- the
        checkpoint is self-contained and `dist()` takes RAW observations, the
        same ones the game will hand it.
        """
        eps = 1e-6
        self.p_mu.copy_(prop.mean(0))
        self.p_sd.copy_(prop.std(0).clamp(min=eps))
        if self.task_enc is not None:
            self.t_mu.copy_(task.mean(0))
            self.t_sd.copy_(task.std(0).clamp(min=eps))
        self.z_mu.copy_(z.mean(0))
        self.z_sd.copy_(z.std(0).clamp(min=eps))
        # sigma starts at the target's own scale, so the initial NLL gradient
        # is O(1) rather than O(1/sigma^2).
        self.log_std.data.fill_(0.0)

    def dist(self, prop, task=None):
        """RAW observations in, a Normal over RAW z out."""
        h = self.proprio_enc((prop - self.p_mu) / self.p_sd)
        if self.task_enc is not None:
            h = torch.cat([h, self.task_enc((task - self.t_mu) / self.t_sd)], -1)
        mu = self.z_proj(self.expert(h)) * self.z_sd + self.z_mu
        return torch.distributions.Normal(mu, self.log_std.exp() * self.z_sd)

    def mean_z(self, prop, task=None):
        return self.dist(prop, task).mean


@torch.no_grad()
def collect(env, ac, steps, device):
    """Roll the expert ON-POLICY, recording `(proprio, football_task, z)`.

    On-policy deliberately: the prior only has to be right on the states this
    skill actually visits, and fitting it on a mean-action rollout would give a
    narrower state distribution than the one stage 6 will query it from.
    """
    prop_i = torch.as_tensor(env.proprio_indices, device=device,
                             dtype=torch.long)
    task_i = torch.as_tensor(env.task_indices, device=device, dtype=torch.long)
    P, T, Z = [], [], []
    obs = env.reset()
    for _ in range(steps):
        o = obs.float()
        P.append(o.index_select(-1, prop_i).cpu())
        T.append(o.index_select(-1, task_i).cpu())
        Z.append(ac.z(o).cpu())
        a, _, _ = ac.act(o)
        obs = env.step(a.clamp(-1, 1))[0]
    return (torch.cat(P), torch.cat(T), torch.cat(Z))


def fit(prior, prop, task, z, epochs=40, batch=1024, lr=1e-3, device="cpu",
        holdout=0.1, log=print):
    """NLL of the expert's z under the prior. Returns the held-out history."""
    n = prop.shape[0]
    n_hold = max(1, int(n * holdout))
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    tr, ho = perm[n_hold:], perm[:n_hold]
    opt = torch.optim.Adam(prior.parameters(), lr=lr)
    prior.to(device)
    hist = []
    for ep in range(epochs):
        prior.train()
        order = tr[torch.randperm(tr.numel(), generator=g)]
        for i in range(0, order.numel(), batch):
            b = order[i:i + batch]
            d = prior.dist(prop[b].to(device),
                           task[b].to(device) if prior.task_enc else None)
            loss = -d.log_prob(z[b].to(device)).sum(-1).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        prior.eval()
        with torch.no_grad():
            d = prior.dist(prop[ho].to(device),
                           task[ho].to(device) if prior.task_enc else None)
            zt = z[ho].to(device)
            nll = float(-d.log_prob(zt).sum(-1).mean())
            rmse = float((d.mean - zt).pow(2).sum(-1).sqrt().mean())
        hist.append((ep, nll, rmse))
        if ep % 10 == 0 or ep == epochs - 1:
            log(f"  epoch {ep:3d}  holdout NLL {nll:9.3f}  ||dz|| {rmse:.4f}")
    return hist


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--skill", required=True, choices=sorted(FOOTBALL_TASK_COLS))
    p.add_argument("--ckpt", required=True, help="the trained expert")
    p.add_argument("--run-config", default=None,
                   help="config.json of the expert's run (defaults to the "
                        "one beside --ckpt)")
    p.add_argument("--out", required=True)
    p.add_argument("--worlds", type=int, default=256)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import json
    from rower_soccer.warp_port.ppo import ActorCritic, load_pretrained

    cfg_path = args.run_config or os.path.join(os.path.dirname(args.ckpt),
                                               "config.json")
    cfg = json.load(open(cfg_path))
    # Older runs predate flags that were added later, and a distillation is
    # exactly the moment you reach back for the oldest checkpoint in the tree.
    # These defaults must match each flag's own default in its trainer, so an
    # old config reconstructs the env that run actually used.
    for k, v in (("w_aim", 0.0), ("live_cmd_dir", False),
                 ("state_dependent_std", False), ("plain", False),
                 ("freeze_decoder", True), ("z_dim", 16)):
        cfg.setdefault(k, v)
    a = argparse.Namespace(**cfg)

    mk = {"follow": "train_follow_warp", "dribble": "train_dribble_warp",
          "kick": "train_kick_warp", "shoot": "train_shoot_warp"}[args.skill]
    # The four trainers do not agree on an env factory: kick and shoot expose
    # `make_env(args, num_worlds=, seed=, use_graph=)`, follow and dribble
    # expose `make_eval_env(args, num_worlds, seed)`. Resolve rather than
    # assume -- a wrong guess here is an AttributeError, which is the good
    # case, but only because nothing silently falls back to a default env.
    mod = __import__(f"rower_soccer.warp_port.{mk}",
                     fromlist=["make_env", "make_eval_env"])
    if hasattr(mod, "make_env"):
        env = mod.make_env(a, num_worlds=args.worlds, seed=args.seed,
                           use_graph=True)
    else:
        env = mod.make_eval_env(a, args.worlds, args.seed)
    dev = str(env.device)

    ac = ActorCritic(env.obs_dim, env.act_dim,
                     proprio_indices=env.proprio_indices.tolist(),
                     task_indices=env.task_indices.tolist(), z_dim=a.z_dim,
                     state_dependent_std=a.state_dependent_std).to(dev)
    load_pretrained(ac, args.ckpt, device=dev)
    ac.eval()

    cols = FOOTBALL_TASK_COLS[args.skill]
    print(f"[prior] {args.skill}: proprio {len(env.proprio_indices)} + "
          f"football task {len(cols)} of {len(env.task_indices)} "
          f"({'proprio-only' if not cols else 'cols ' + str(cols[0]) + ':' + str(cols[-1] + 1)})")

    prop, task_full, z = collect(env, ac, args.steps, dev)
    task = (task_full[:, cols] if cols
            else torch.zeros(task_full.shape[0], 0))
    print(f"[prior] collected {prop.shape[0]:,} states; "
          f"z mean |.| {z.abs().mean():.3f}, sd {z.std():.3f}")

    prior = DrillPrior(prop.shape[1], task.shape[1], z_dim=a.z_dim)
    prior.set_norm(prop, task, z)
    hist = fit(prior, prop, task, z, epochs=args.epochs, device=dev,
               lr=args.lr)

    # A prior that just predicts the mean z would be a plausible-looking
    # failure, so record what that baseline costs.
    zt = z.to(dev)
    base = float((zt.mean(0, keepdim=True) - zt).pow(2).sum(-1).sqrt().mean())
    print(f"[prior] final ||dz|| {hist[-1][2]:.4f} against a constant-mean "
          f"baseline of {base:.4f}  ({100 * hist[-1][2] / base:.1f}% of it)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({"skill": args.skill, "state_dict": prior.state_dict(),
                "n_proprio": prop.shape[1], "n_task": task.shape[1],
                "z_dim": a.z_dim, "task_cols": cols, "expert_ckpt": args.ckpt,
                "holdout_rmse": hist[-1][2], "constant_baseline": base},
               args.out)
    print(f"[prior] wrote {args.out}")


if __name__ == "__main__":
    main()
