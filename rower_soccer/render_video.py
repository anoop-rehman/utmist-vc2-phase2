"""Offscreen video rendering for creature soccer episodes.

Usage:
    python -m rower_soccer.render_video --out videos/random_2v2.mp4 \
        [--policy random] [--seconds 20] [--camera auto] [--wh 1280 720]

Requires MUJOCO_GL=egl (or osmesa) in the environment.
"""

import argparse
import os

import imageio
import numpy as np


def list_cameras(physics):
    from dm_control.mujoco.wrapper import mjbindings  # noqa: F401
    n = physics.model.ncam
    names = []
    for i in range(n):
        names.append(physics.model.id2name(i, "camera") or f"cam{i}")
    return names


def render_episode(env, policy, writer, camera_id, wh, max_steps=None):
    """Steps env with policy(timestep) -> [actions], writing frames."""
    width, height = wh
    timestep = env.reset()
    control_dt = env.control_timestep()
    fps = int(round(1.0 / control_dt))
    steps = 0
    while not timestep.last():
        actions = policy(timestep)
        timestep = env.step(actions)
        frame = env.physics.render(camera_id=camera_id, width=width, height=height)
        writer.append_data(frame)
        steps += 1
        if max_steps and steps >= max_steps:
            break
    return steps, fps


def random_policy_factory(env):
    specs = env.action_spec()

    def policy(timestep):
        return [np.random.uniform(s.minimum, s.maximum, size=s.shape) for s in specs]

    return policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--teams", nargs=2, default=["rower,worm", "rower,worm"],
                        help="comma-separated creature kinds per team, e.g. rower,worm rower,worm")
    parser.add_argument("--camera", default="auto",
                        help="'auto' picks a wide camera, or a camera name/index")
    parser.add_argument("--wh", type=int, nargs=2, default=[1280, 720])
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--fps-div", type=int, default=1,
                        help="write every Nth frame (playback speedup)")
    args = parser.parse_args()

    from rower_soccer.envs.build import make_soccer_env

    env = make_soccer_env(home_team=tuple(args.teams[0].split(",")),
                          away_team=tuple(args.teams[1].split(",")),
                          time_limit=args.seconds)
    cams = list_cameras(env.physics)
    print(f"cameras: {cams}")
    if args.camera == "auto":
        # Prefer a pitch-wide camera if present, else camera 0.
        camera_id = 0
        for pref in ("top_down", "world", "tracking"):
            for i, name in enumerate(cams):
                if pref in (name or ""):
                    camera_id = i
                    break
    else:
        camera_id = int(args.camera) if args.camera.isdigit() else args.camera

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    control_dt = env.control_timestep()
    fps = int(round(1.0 / control_dt)) // max(1, args.fps_div)
    policy = random_policy_factory(env)
    with imageio.get_writer(args.out, fps=fps, quality=8) as writer:
        for ep in range(args.episodes):
            steps, _ = render_episode(env, policy, writer, camera_id, args.wh)
            print(f"episode {ep}: {steps} steps rendered")
    print(f"wrote {args.out} at {fps} fps (control dt {control_dt*1000:.1f} ms)")


if __name__ == "__main__":
    main()
