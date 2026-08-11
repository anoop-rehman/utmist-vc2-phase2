"""How fast can this body ACTUALLY put the ball THERE?

Written for drill v4 (DRILL_V4_SPEC). The timed kick samples a pace
`v_pace ~ U(lo, hi)` and sets the deadline `T = d_spawn / v_pace`, clamped to
[0.5, 4] s; the segment ends at exactly T and the ball's distance to the target
at that instant is the whole reward. If the band's fast end is above what the
body can produce, those attempts are unreachable no matter what the policy does,
the arrival term is a constant, and the gradient over that part of the band is
FLAT -- the run trains for two days and learns nothing. So the band is measured
off the body before it is chosen.

Two quantities, and the second is the one that matters:

  1. **strike speed** -- peak |v_ball| between the creature first touching the
     ball and the segment ending. Nothing but the creature adds energy to the
     ball (floor and rolling friction only remove it), so that peak is the speed
     at contact-break.
  2. **pace** -- `d / t_reach(d)`, where `t_reach(d)` is the time from SEGMENT
     START until the ball is d metres from where it spawned. This is the
     quantity `v_pace` names, and it is much smaller than the strike speed for
     two reasons the strike speed hides: the creature has to WALK TO THE BALL
     first (1.5-3 m of approach at ~1 m/s), and rolling friction decelerates the
     ball hard (~4 m/s^2 -- 4 m/s dies inside 2 m). A band chosen from strike
     speed alone would be unreachable by a factor of several.

Defaults reproduce kick_ant_v3's env exactly (pitch, 0.15 m ball, point reward),
because a strike measured against a 0.35 m ball in a fenced arena is a
measurement of a different task.

    MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m rower_soccer.warp_port.probe_strike_speed \
        --checkpoint runs_v2/kick_ant_v3/best.pt --worlds 256 --min-strikes 400
"""

import argparse
import os

import numpy as np
import torch

CONTROL_DT = 0.025
# The target distances drill v4 samples from (--target-dist-range 3 6).
REACH_DISTS = (3.0, 4.0, 5.0, 6.0)


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="runs_v2/kick_ant_v3/best.pt")
    p.add_argument("--creature-xml", default="creature_configs/ant.xml")
    p.add_argument("--worlds", type=int, default=256)
    p.add_argument("--min-strikes", type=int, default=400)
    p.add_argument("--max-episodes", type=int, default=12)
    p.add_argument("--episode-secs", type=float, default=15.0)
    p.add_argument("--seed", type=int, default=11)
    # v3's env, verbatim.
    p.add_argument("--ball-radius", type=float, default=0.15)
    p.add_argument("--ball-mass", type=float, default=0.045)
    p.add_argument("--arena", default="pitch")
    p.add_argument("--pitch-scale", type=float, default=0.3125)
    p.add_argument("--reward-kind", default="point")
    p.add_argument("--target-dist-range", type=float, nargs=2, default=[3.0, 6.0])
    p.add_argument("--segment-secs-range", type=float, nargs=2, default=[2.0, 6.0])
    a = p.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    from rower_soccer.warp_port.kick_env import WarpKickEnv
    from rower_soccer.warp_port.scene import BallSpec
    from rower_soccer.tools.style import _build_policy

    env = WarpKickEnv(
        num_worlds=a.worlds, creature_xml=a.creature_xml, seed=a.seed,
        episode_seconds=a.episode_secs,
        ball=BallSpec(radius=a.ball_radius, mass=a.ball_mass),
        arena=a.arena, pitch_scale=a.pitch_scale, reward_kind=a.reward_kind,
        w_strike=0.1, w_arrive=3.0, w_upright=1.0,
        target_dist_range=tuple(a.target_dist_range),
        segment_seconds_range=tuple(a.segment_secs_range))
    ac = _build_policy(env, a.checkpoint)
    print(f"[probe] obs={env.obs_dim} act={env.act_dim} worlds={env.n} "
          f"contact_dist={env.contact_dist:.2f}m ball_r={a.ball_radius}", flush=True)

    n, dev = env.n, env.device
    NEVER = 1e9

    def z():
        return torch.zeros(n, device=dev)

    obs = env.reset()
    spawn_xy = env._ball_xy().clone()
    peak, t_touch = z(), torch.full((n,), NEVER, device=dev)
    seen = torch.zeros(n, dtype=torch.bool, device=dev)
    t_reach = torch.full((n, len(REACH_DISTS)), NEVER, device=dev)
    dists = torch.tensor(REACH_DISTS, device=dev).unsqueeze(0)
    rows = []
    prev_seg = env.n_segments.clone()
    steps = int(a.episode_secs / CONTROL_DT)

    def harvest(i):
        rows.append(np.concatenate([
            peak[i].unsqueeze(-1).cpu().numpy(),
            t_touch[i].unsqueeze(-1).cpu().numpy(),
            t_reach[i].cpu().numpy()], 1))

    def clear(i):
        peak[i] = 0.0
        t_touch[i] = NEVER
        seen[i] = False
        t_reach[i] = NEVER
        spawn_xy[i] = env._ball_xy()[i]

    for _ in range(a.max_episodes):
        for _ in range(steps):
            with torch.no_grad():
                act = ac.dist(obs.float()).mean.clamp(-1, 1)
            obs, _, done = env.step(act)

            # Harvest BEFORE reading this step's state. env.step() has already
            # closed and respawned any finished segment, so the ball position it
            # now reports belongs to the NEXT segment -- folding it in first
            # would score a respawn teleport as "travelled 3 m in 0 s".
            closed = env.n_segments > prev_seg
            if bool(closed.any()):
                i = closed.nonzero(as_tuple=True)[0]
                harvest(i)
                clear(i)
            prev_seg = env.n_segments.clone()

            t_now = env.seg_t * CONTROL_DT
            v = torch.linalg.norm(env._ball_vel_xy(), dim=-1)
            peak = torch.where(env.touched & (v > peak), v, peak)
            first = env.touched & ~seen
            t_touch = torch.where(first, t_now, t_touch)
            seen |= env.touched

            travelled = torch.linalg.norm(env._ball_xy() - spawn_xy, dim=-1)
            hit = (travelled.unsqueeze(-1) >= dists) & (t_reach >= NEVER)
            t_reach = torch.where(hit, t_now.unsqueeze(-1), t_reach)
            if done:
                break
        obs = env.reset()
        clear(torch.arange(n, device=dev))
        prev_seg = env.n_segments.clone()
        if sum(len(r) for r in rows) >= a.min_strikes:
            break

    r = np.concatenate(rows, 0)
    spd, t_t, reach = r[:, 0], r[:, 1], r[:, 2:]
    touched = t_t < NEVER / 2
    print(f"\n[probe] {len(spd)} segments ({touched.mean():.1%} with a strike), "
          f"checkpoint {a.checkpoint}")

    def line(name, x):
        if len(x) == 0:
            print(f"{name:<30}{'--- never ---':>63}")
            return
        print(f"{name:<30}{x.min():>9.2f}{np.percentile(x, 10):>9.2f}"
              f"{np.median(x):>9.2f}{x.mean():>9.2f}"
              f"{np.percentile(x, 90):>9.2f}{x.max():>9.2f}{len(x):>9d}")

    hdr = (f"{'':<30}{'min':>9}{'p10':>9}{'median':>9}{'mean':>9}"
           f"{'p90':>9}{'max':>9}{'n':>9}")
    print("\n== strike (what leaves the foot) ==")
    print(hdr)
    line("ball speed at break m/s", spd[touched])
    line("time to first touch s", t_t[touched])

    print("\n== pace = d / t_reach(d), from SEGMENT START ==")
    print("   (this is what v_pace means: approach + flight, not muzzle speed)")
    print(hdr)
    for k, d in enumerate(REACH_DISTS):
        tk = reach[:, k]
        ok = tk < NEVER / 2
        line(f"t to travel {d:.0f} m  s", tk[ok])
        line(f"  pace over {d:.0f} m  m/s", d / tk[ok])
        print(f"{'  reached ' + f'{d:.0f} m at all':<30}{ok.mean():>9.1%}")


if __name__ == "__main__":
    main()
