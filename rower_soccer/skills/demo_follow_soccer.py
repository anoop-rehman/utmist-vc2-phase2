"""WS3 gate demo — a SkillController-driven ant in the CPU soccer env.

Drives one ant through a scripted sequence of (skill, target) commands issued
MID-EPISODE, with no env reset between them, and reports how close it got to each
commanded point. This is the thing WS4's play server will do, with a human's
mouse in place of the schedule below.

    PYTHONPATH=<worktree> MUJOCO_GL=egl \
      python -m rower_soccer.skills.demo_follow_soccer --video /tmp/ws3.mp4

Checkpoints: `runs_v2/follow_ant_v1/best.pt` resolves against
`$VC2_CHECKPOINT_ROOT`, then the repo root, then the base checkout when running
in a git worktree (training writes to the main checkout's gitignored `runs_v2/`).
`--follow-model` overrides it, including with a `gs://` URI.

What "arrived" means: fitness `exp(-0.5 * dist)` is the drills' own metric
(Table S3), so the numbers printed here are directly comparable to the training
run's 0.997.
"""

import argparse
import os
import time

import numpy as np

# The default command schedule: (skill, offset, seconds).
#
# `offset` is RELATIVE to where the ant is when the command is issued, so every
# leg is the same length no matter where the previous one left it — an absolute
# schedule silently turns into "walk 14 m in 8 s" once the ant drifts, and then
# measures the ant's top speed rather than whether it steers. The ant's trained
# target speed range is 0.07-0.6 m/s (`follow_ant_v1`'s config), so a 4 m leg in
# 15 s is inside what it was optimised for.
#
# Two follow legs in opposite directions prove steering; the `idle` between them
# and the return to `follow` prove a mid-episode switch neither glitches nor
# leaves state behind; `scripted` proves the fallback chases the ball without
# being given a target.
DEFAULT_PLAN = [
    ("follow", (4.0, 0.0), 15.0),
    ("idle", None, 1.0),
    ("follow", (-2.0, 4.0), 15.0),
    ("scripted", None, 15.0),
    ("follow", (0.0, -4.0), 15.0),
]


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--creature", default="ant")
    p.add_argument("--follow-model", default=None,
                   help="override the registry's checkpoint for `follow`")
    p.add_argument("--action-mode", default="auto", choices=["auto", "mean", "noise"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target-clip", type=float, default=None,
                   help="metres; 0 disables the waypoint re-aim (default: registry)")
    p.add_argument("--match-physics-dt", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="run soccer at the drill's 0.0025 physics dt (default on)")
    p.add_argument("--verbose", action="store_true",
                   help="print the trajectory once per simulated second")
    p.add_argument("--video", default=None, help="optional top-down mp4")
    p.add_argument("--fps", type=int, default=40)
    args = p.parse_args()

    import torch

    from rower_soccer.skills import SkillController, DEFAULT_TARGET_CLIP
    from rower_soccer.skills.soccer import SoccerFrameSource, make_skill_soccer_env

    # A 69->8 MLP at 40 Hz is microseconds of arithmetic; torch's default
    # intra-op thread pool spends more time synchronising than computing and
    # fights the physics thread for cores. WS4's game loop should do the same.
    torch.set_num_threads(1)

    env = make_skill_soccer_env(home=(args.creature,), time_limit=1e6,
                                random_state=args.seed,
                                match_dt=args.match_physics_dt)
    src = SoccerFrameSource(env)

    if args.video:
        _add_topdown_camera(env)

    ctrl = SkillController(
        args.creature,
        action_mode=args.action_mode,
        seed=args.seed,
        name="home_0",
        target_clip=(DEFAULT_TARGET_CLIP if args.target_clip is None
                     else args.target_clip),
        checkpoints=({"follow": args.follow_model} if args.follow_model else None),
    )
    print(f"[demo] {ctrl.contract.describe()}  skills={ctrl.available_skills()}  "
          f"physics_dt={env.task.physics_timestep}  "
          f"control_dt={env.task.control_timestep}", flush=True)

    ts = env.reset()
    hz = int(round(1.0 / env.task.control_timestep))
    frames = []
    cam = _topdown_id(env) if args.video else None
    ok = True
    t_wall = time.time()
    n_steps = 0

    for skill, offset, secs in DEFAULT_PLAN:
        here = src.root_xy(0).copy()
        target = None if offset is None else tuple(here + np.asarray(offset))
        ctrl.set_command(skill, target_xy=target)

        def aim():
            if skill == "idle":
                return here                       # "target" is: do not move
            return np.asarray(target if target is not None else src.ball_xy(),
                              dtype=np.float64)

        d0 = float(np.linalg.norm(src.root_xy(0) - aim()))
        for i in range(int(round(secs * hz))):
            out = ctrl.act(src.frame(ts, 0))
            if not np.isfinite(out.action).all():
                raise SystemExit(f"[demo] non-finite action during '{skill}'")
            ts = env.step([out.action])
            n_steps += 1
            if args.verbose and i % hz == 0:
                f = src.frame(ts, 0)
                print(f"[demo]   t={i // hz:3d}s xy={np.round(src.root_xy(0), 2)} "
                      f"h={f.root_pos[2]:.3f} d={np.linalg.norm(src.root_xy(0) - aim()):5.2f} "
                      f"|tgt_ego|={np.linalg.norm(out.obs_vector[-4:-2]):5.2f}", flush=True)
            if cam is not None:
                frames.append(env.physics.render(camera_id=cam, width=640, height=480))
        d1 = float(np.linalg.norm(src.root_xy(0) - aim()))   # ball may have moved
        fit = float(np.exp(-0.5 * d1))

        if skill == "idle":
            passed = d1 < 0.25                     # zero torque must mean stay put
            verdict = "held" if passed else "DRIFTED"
        else:
            passed = d1 < d0 - 0.5
            verdict = "closer" if passed else "NO"
        ok = ok and passed
        print(f"[demo] {skill:9s} aim={_fmt(offset):>14s} {secs:5.1f}s  "
              f"dist {d0:6.2f} -> {d1:6.2f} m  fitness={fit:.3f}  [{verdict}]",
              flush=True)

    wall = time.time() - t_wall
    print(f"[demo] {n_steps} control steps in {wall:.1f}s "
          f"({n_steps / wall:.0f} steps/s, realtime is {hz})", flush=True)
    print(f"[demo] mode={ctrl.resolved_mode('follow')} "
          f"tick={ctrl.tick} switches were clean (no reset between commands)",
          flush=True)

    if args.video and frames:
        import imageio
        imageio.mimsave(args.video, frames, fps=args.fps)
        print(f"[demo] wrote {args.video} ({len(frames)} frames)", flush=True)

    print(f"[demo] GATE: {'PASS' if ok else 'FAIL'}", flush=True)
    raise SystemExit(0 if ok else 1)


def _fmt(t):
    return "ball" if t is None else f"+({t[0]:.1f}, {t[1]:.1f})"


def _add_topdown_camera(env, height=42.0, view_half=26.0):
    import math
    wb = env.task.arena.mjcf_model.worldbody
    wb.add("camera", name="ws3_topdown", pos=[0, 0, height],
           xyaxes=[1, 0, 0, 0, 1, 0],
           fovy=2.0 * math.degrees(math.atan(view_half / height)))
    # The pitch ships four 8192px shadowmaps (~90 ms/frame) purely cosmetic here.
    for light in env.task.arena.mjcf_model.find_all("light"):
        light.castshadow = "false"
    env.task.arena.mjcf_model.visual.quality.offsamples = 0


def _topdown_id(env):
    model = env.physics.model
    for i in range(model.ncam):
        name = model.camera(i).name
        if name and name.endswith("ws3_topdown"):
            return i
    raise RuntimeError("ws3_topdown camera not found")


if __name__ == "__main__":
    main()
