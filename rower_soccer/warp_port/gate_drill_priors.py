"""Gate the drill priors: drive the frozen decoder from the PRIOR alone.

`drill_prior.py` reports how much of the expert's latent each prior recovers
(shoot 99.5%, follow 93.3%, dribble 90.4%, kick 89.8% of z variance). That says
the priors are FITTABLE. It does not say they are SKILLS -- a prior could
reconstruct z well on the expert's own state distribution and still produce
nothing recognisable when it is the thing choosing z, because the states it then
visits are its own.

So this closes the loop the way PIPELINE_V2 asks ("prior reproduces skill sans
target (video)"): the expert is removed entirely, the prior emits z from
football observations only, and the frozen shared decoder turns that into
torques. What comes out is what stage 6 would actually be regularising toward.

    PYTHONPATH=. MUJOCO_GL=osmesa .venv/bin/python \
        -m rower_soccer.warp_port.gate_drill_priors

Two things are measured, and the second is the one that matters:

  1. **Does it move like the skill?** Gait speed and torso height, prior-driven
     against expert-driven, in the same env. The two strike drills already
     differ visibly at the expert level -- kick converged to a 1.38 m/s flat
     crawl and shoot to a 2.61 m/s upright gait from the SAME frozen decoder --
     so a distillation that collapses them would show up here as four priors
     with one gait.

  2. **Are the four priors distinguishable from each other?** Pairwise z
     distance between priors evaluated on IDENTICAL states. If every prior maps
     the same observation to the same latent, they are one prior wearing four
     names, and the Eq. 5 mixture has nothing to mix. This is the negative
     control the residual numbers cannot provide.
"""

import argparse
import json
import os

import numpy as np
import torch

SKILLS = ("follow", "dribble", "kick", "shoot")
EXPERTS = {
    "follow": "runs_v2/follow_ant_final_frozen/best.pt",
    "dribble": "runs_v2/dribble_ant_v3/best.pt",
    "kick": "runs_v2/kick_ant_v14_prog/best.pt",
    "shoot": "runs_v2/s5_c_all/best.pt",
}
TRAINER = {"follow": "train_follow_warp", "dribble": "train_dribble_warp",
           "kick": "train_kick_warp", "shoot": "train_shoot_warp"}


def action_from_z(ac, obs, z):
    """The frozen low-level controller, driven by an EXTERNAL latent.

    `ActorCritic.dist` computes z itself from the expert; this is the same path
    with the expert removed -- decoder(proprio, z) -> action_net. Nothing about
    the controller changes, which is the point: the prior must work through the
    identical motor controller the expert used.
    """
    prop = ac._clean(obs).index_select(-1, ac.mlp_extractor.p_idx)
    lat = ac.mlp_extractor.decoder(torch.cat([prop, z], -1))
    return ac.action_net(lat)


def build(skill, worlds, seed):
    from rower_soccer.warp_port.drill_prior import DrillPrior
    from rower_soccer.warp_port.ppo import ActorCritic, load_pretrained

    ck = EXPERTS[skill]
    cfg = json.load(open(os.path.join(os.path.dirname(ck), "config.json")))
    for k, v in (("w_aim", 0.0), ("live_cmd_dir", False),
                 ("state_dependent_std", False), ("plain", False),
                 ("freeze_decoder", True), ("z_dim", 16)):
        cfg.setdefault(k, v)
    a = argparse.Namespace(**cfg)
    mod = __import__(f"rower_soccer.warp_port.{TRAINER[skill]}",
                     fromlist=["make_env", "make_eval_env"])
    env = (mod.make_env(a, num_worlds=worlds, seed=seed, use_graph=True)
           if hasattr(mod, "make_env")
           else mod.make_eval_env(a, worlds, seed))
    ac = ActorCritic(env.obs_dim, env.act_dim,
                     proprio_indices=env.proprio_indices.tolist(),
                     task_indices=env.task_indices.tolist(), z_dim=a.z_dim,
                     state_dependent_std=a.state_dependent_std).to(env.device)
    load_pretrained(ac, ck, device=str(env.device))
    ac.eval()

    blob = torch.load(f"runs_v2/_priors/{skill}.pt", map_location=env.device)
    prior = DrillPrior(blob["n_proprio"], blob["n_task"], z_dim=blob["z_dim"])
    prior.load_state_dict(blob["state_dict"])
    prior = prior.to(env.device).eval()
    return env, ac, prior, blob


@torch.no_grad()
def roll(env, ac, prior, blob, steps, driver):
    """`driver` is 'expert' or 'prior'. Returns per-step diagnostics."""
    dev = env.device
    p_i = torch.as_tensor(env.proprio_indices, device=dev, dtype=torch.long)
    t_i = torch.as_tensor(env.task_indices, device=dev, dtype=torch.long)
    cols = torch.as_tensor(blob["task_cols"], device=dev, dtype=torch.long)
    xy, z_hist = [], []
    obs = env.reset()
    for _ in range(steps):
        o = obs.float()
        if driver == "expert":
            a = ac.dist(o).mean
            z = ac.z(o)
        else:
            prop = o.index_select(-1, p_i)
            task = (o.index_select(-1, t_i).index_select(-1, cols)
                    if cols.numel() else None)
            z = prior.dist(prop, task).mean
            a = action_from_z(ac, o, z)
        z_hist.append(z.cpu())
        pos, _ = env._root_frames()
        xy.append(pos[:, :3].cpu().clone())
        obs = env.step(a.clamp(-1, 1))[0]
    X = torch.stack(xy)                       # [T, W, 3]
    d = (X[1:, :, :2] - X[:-1, :, :2]).norm(dim=-1)
    d = d[d < 0.5]                            # drop resets/respawns
    return {"speed": float(d.mean()) / 0.025, "z_torso": float(X[..., 2].mean()),
            "z": torch.cat(z_hist)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, default=64)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--skills", nargs="*", default=list(SKILLS))
    args = p.parse_args()

    rows, probe = {}, {}
    for sk in args.skills:
        env, ac, prior, blob = build(sk, args.worlds, seed=11)
        e = roll(env, ac, prior, blob, args.steps, "expert")
        r = roll(env, ac, prior, blob, args.steps, "prior")
        rows[sk] = (e, r, blob)
        # A fixed probe batch for the cross-prior comparison: the SAME states
        # for every prior, so any difference is the prior and not the states.
        obs = env.reset().float()
        probe[sk] = (obs.index_select(
            -1, torch.as_tensor(env.proprio_indices, device=env.device,
                                dtype=torch.long)).cpu(), blob["n_task"])
        del env, ac, prior

    print(f"\n{'skill':9s} {'fit resid':>10s} | {'expert m/s':>11s} "
          f"{'prior m/s':>10s} | {'expert z':>9s} {'prior z':>8s}")
    for sk in args.skills:
        e, r, blob = rows[sk]
        print(f"{sk:9s} {100 * blob['holdout_rmse'] / blob['constant_baseline']:9.1f}% "
              f"| {e['speed']:11.2f} {r['speed']:10.2f} "
              f"| {e['z_torso']:9.3f} {r['z_torso']:8.3f}")

    # ---- are the four priors actually different functions? ---------------
    # Proprio-only comparison, on one shared batch of states. A prior that
    # ignores its task input entirely would still show up as distinct here if
    # its proprio path differs; identical rows would mean distillation
    # collapsed them.
    print("\ncross-prior z distance on IDENTICAL proprio states "
          "(mean ||z_a - z_b||):")
    from rower_soccer.warp_port.drill_prior import DrillPrior
    shared = probe[args.skills[0]][0]
    zs = {}
    for sk in args.skills:
        blob = torch.load(f"runs_v2/_priors/{sk}.pt", map_location="cpu")
        pr = DrillPrior(blob["n_proprio"], blob["n_task"], z_dim=blob["z_dim"])
        pr.load_state_dict(blob["state_dict"])
        pr.eval()
        with torch.no_grad():
            t = (torch.zeros(shared.shape[0], blob["n_task"])
                 if blob["n_task"] else None)
            zs[sk] = pr.dist(shared, t).mean
    names = list(args.skills)
    print(f"    {'':9s}" + "".join(f"{n:>10s}" for n in names))
    for a_ in names:
        cells = "".join(
            f"{float((zs[a_] - zs[b]).norm(dim=-1).mean()):10.2f}"
            for b in names)
        print(f"    {a_:9s}{cells}")
    off = [float((zs[a_] - zs[b]).norm(dim=-1).mean())
           for i, a_ in enumerate(names) for b in names[i + 1:]]
    scale = float(torch.cat([zs[n] for n in names]).norm(dim=-1).mean())
    print(f"\n  min off-diagonal {min(off):.2f} against a mean ||z|| of "
          f"{scale:.2f} -- {'DISTINCT' if min(off) > 0.1 * scale else 'COLLAPSED'}")


if __name__ == "__main__":
    main()
