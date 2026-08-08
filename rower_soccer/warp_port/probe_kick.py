"""What kick's fitness cannot see: aim.

kick's fitness is `max over the segment of (ball_velocity . commanded_direction)`
-- a PROJECTION. Projection conflates two things a human watching the video keeps
separate:

    3.8 m/s dead on target        -> fitness 3.8
    5.4 m/s at 45 degrees off     -> fitness 3.8
    7.6 m/s at 60 degrees off     -> fitness 3.8

All three score identically. Only the first looks like a kick. So a rising
fitness curve is consistent with the policy learning to hit the ball HARDER
while aiming no better -- and hitting harder is the easier of the two.

This probe separates them: for every strike it records the ball's speed and the
angle between its velocity and the command at the moment the projection peaks.

    MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m rower_soccer.warp_port.probe_kick \
        --checkpoint runs_v2/kick_ant_v1/final.pt
"""

import argparse
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="runs_v2/kick_ant_v1/final.pt")
    p.add_argument("--creature-xml", default="creature_configs/ant.xml")
    p.add_argument("--worlds", type=int, default=256)
    p.add_argument("--seconds", type=float, default=30.0)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    import os
    os.environ.setdefault("MUJOCO_GL", "egl")
    from rower_soccer.warp_port.kick_env import WarpKickEnv
    from rower_soccer.tools.style import _build_policy

    env = WarpKickEnv(num_worlds=a.worlds, creature_xml=a.creature_xml,
                      seed=a.seed, episode_seconds=a.seconds)
    ac = _build_policy(env, a.checkpoint)
    print(f"[probe] obs={env.obs_dim} act={env.act_dim} "
          f"contact_dist={env.contact_dist:.2f}m speed_clip={env.speed_clip}", flush=True)

    obs = env.reset()
    n = env.n
    best = torch.full((n,), -1e9, device=env.device)   # peak projection this segment
    best_speed = torch.zeros(n, device=env.device)     # |v| at that moment
    best_cos = torch.zeros(n, device=env.device)       # cos(angle) at that moment
    rows = []
    steps = int(a.seconds / 0.025)
    prev_seg = env.n_segments.clone()

    for _ in range(steps):
        with torch.no_grad():
            act = ac.dist(obs.float()).mean.clamp(-1, 1)
        obs, _, done = env.step(act)

        v = env._ball_vel_xy()
        speed = torch.linalg.norm(v, dim=-1)
        proj = (v * env.cmd_dir).sum(-1)
        upd = env.touched & (proj > best)
        best = torch.where(upd, proj, best)
        best_speed = torch.where(upd, speed, best_speed)
        best_cos = torch.where(upd, proj / speed.clamp(min=1e-6), best_cos)

        # A segment just closed: harvest and reset the accumulators for it.
        closed = env.n_segments > prev_seg
        if bool(closed.any()):
            i = closed.nonzero(as_tuple=True)[0]
            keep = best[i] > -1e8
            if bool(keep.any()):
                j = i[keep]
                rows.append(np.stack([best[j].cpu().numpy(),
                                      best_speed[j].cpu().numpy(),
                                      np.clip(best_cos[j].cpu().numpy(), -1, 1)], 1))
            best[i] = -1e9
            best_speed[i] = 0.0
            best_cos[i] = 0.0
        prev_seg = env.n_segments.clone()
        if done:
            obs = env.reset()

    if not rows:
        raise SystemExit("no strikes recorded")
    r = np.concatenate(rows, 0)
    proj, speed, cos = r[:, 0], r[:, 1], r[:, 2]
    ang = np.degrees(np.arccos(cos))

    print(f"\n[probe] {len(proj)} strikes over {a.worlds} worlds x {a.seconds:.0f}s")
    print(f"{'':<26}{'median':>10}{'mean':>10}{'p10':>10}{'p90':>10}")
    print("-" * 66)
    for name, x in (("projection (= fitness)", proj), ("ball speed |v| m/s", speed),
                    ("aim error deg", ang)):
        print(f"{name:<26}{np.median(x):>10.2f}{x.mean():>10.2f}"
              f"{np.percentile(x,10):>10.2f}{np.percentile(x,90):>10.2f}")
    print("-" * 66)
    # The decomposition: how much of the projection is lost to aim?
    lost = 1.0 - proj.sum() / max(speed.sum(), 1e-9)
    print(f"fraction of ball speed lost to aim error: {lost:.1%}")
    for t in (15, 30, 45, 60, 90):
        print(f"  strikes within {t:>2} deg of command: {(ang <= t).mean():>5.1%}")


if __name__ == "__main__":
    main()
