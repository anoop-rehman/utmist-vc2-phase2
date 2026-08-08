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

# The default command schedule: (skill, target_xy, seconds). Two follow targets
# in different directions prove steering; the `idle` in between and the return to
# `follow` prove that switching skills mid-episode neither glitches nor leaves
# state behind; `scripted` proves the no-command-needed fallback chases the ball.
DEFAULT_PLAN = [
    ("follow", (6.0, 6.0), 8.0),
    ("idle", None, 1.0),
    ("follow", (-6.0, 6.0), 8.0),
    ("scripted", None, 8.0),
    ("follow", (0.0, 0.0), 8.0),
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
    p.add_argument("--video", default=None, help="optional top-down mp4")
    p.add_argument("--fps", type=int, default=40)
    args = p.parse_args()

    from rower_soccer.skills import (MODE_AUTO, SkillController, DEFAULT_TARGET_CLIP)
    from rower_soccer.skills.soccer import SoccerFrameSource, make_skill_soccer_env

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

    for skill, target, secs in DEFAULT_PLAN:
        ctrl.set_command(skill, target_xy=target)
        aim = np.asarray(target if target is not None else src.ball_xy(),
                         dtype=np.float64)
        d0 = float(np.linalg.norm(src.root_xy(0) - aim))
        steps = int(round(secs * hz))
        for _ in range(steps):
            out = ctrl.act(src.frame(ts, 0))
            if not np.isfinite(out.action).all():
                raise SystemExit(f"[demo] non-finite action during '{skill}'")
            ts = env.step([out.action])
            n_steps += 1
            if cam is not None:
                frames.append(env.physics.render(camera_id=cam, width=640, height=480))
        # For `scripted` the aim moves with the ball; re-read it.
        aim = np.asarray(target if target is not None else src.ball_xy(),
                         dtype=np.float64)
        d1 = float(np.linalg.norm(src.root_xy(0) - aim))
        fit = float(np.exp(-0.5 * d1))
        verdict = "closer" if d1 < d0 - 0.25 else ("held" if skill == "idle" else "NO")
        if skill in ("follow", "scripted") and d1 >= d0 - 0.25:
            ok = False
        print(f"[demo] {skill:9s} target={_fmt(target):>16s} "
              f"{secs:4.1f}s  dist {d0:6.2f} -> {d1:6.2f} m  "
              f"fitness={fit:.3f}  [{verdict}]", flush=True)

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
    return "ball" if t is None else f"({t[0]:.1f}, {t[1]:.1f})"


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
