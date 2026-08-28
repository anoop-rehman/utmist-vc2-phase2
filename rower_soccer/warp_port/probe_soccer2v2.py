"""Look at the 2v2 env. Renders world 0 as a contact sheet and a short clip.

    MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m rower_soccer.warp_port.probe_soccer2v2 \
        --out runs/soccer2v2_1e --steps 120 --gpu

This project has twice shipped an env that was numerically fine and visually
wrong -- an oversized ball, an unscaled pitch -- both of which every metric in
the run happily reported as healthy. The env is not done until someone has
looked at a frame, so this is a deliverable, not a debugging aid.

Physics stays in the env's backend (warp or CPU MuJoCo). This module builds a
SEPARATE render-only MjModel from the same `build_soccer_scene` call, copies
qpos across and calls `mj_forward` for the picture only -- the same discipline
`warp_port/render.py` uses for the drills. The two models have identical qpos
layouts by construction (same builder, same arguments), so the picture is the
state that was simulated.
"""

import argparse
import os

import mujoco
import numpy as np
import torch

from rower_soccer.warp_port.scene import build_soccer_scene
from rower_soccer.warp_port.soccer2v2_env import WarpSoccer2v2Env, drill_ball


class Soccer2v2Renderer:
    def __init__(self, env, width=960, height=720, view_half=12.0,
                 cam_height=25.0):
        self.model, _, _ = build_soccer_scene(
            "creature_configs/ant.xml", n_players=env.n_agents,
            ball=env._ball_spec, pitch_scale=env._pitch_scale,
            topdown_cam=True, view_half=view_half, cam_height=cam_height)
        assert self.model.nq == env.model.nq, "render model qpos layout differs"
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        # Kept so callers that tile several panels can read the panel size off
        # the renderer instead of passing it a second time and letting the two
        # disagree.
        self.width, self.height = width, height
        self.topdown_id = int(mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "topdown"))
        # A broadcast-style free camera: behind the -y touchline, elevated,
        # looking at the centre spot -- the view `match.py` calls "broadcast".
        self.free = mujoco.MjvCamera()
        self.free.lookat[:] = [0.0, 0.0, 0.5]
        self.free.distance = 2.6 * env.pitch_half[1]
        self.free.elevation = -28.0
        self.free.azimuth = 90.0

    def frame(self, env, w=0, camera="topdown", lookat=None, distance=None):
        self.data.qpos[:] = env.qpos[w].detach().cpu().numpy()
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        if camera == "topdown":
            self.renderer.update_scene(self.data, camera=self.topdown_id)
        else:
            cam = mujoco.MjvCamera()
            cam.lookat[:] = self.free.lookat if lookat is None else lookat
            cam.distance = self.free.distance if distance is None else distance
            cam.elevation, cam.azimuth = self.free.elevation, self.free.azimuth
            self.renderer.update_scene(self.data, camera=cam)
        return self.renderer.render()


def contact_sheet(frames, cols=3, pad=6, bg=20):
    rows = int(np.ceil(len(frames) / cols))
    h, w, _ = frames[0].shape
    sheet = np.full((rows * h + (rows + 1) * pad,
                     cols * w + (cols + 1) * pad, 3), bg, np.uint8)
    for i, f in enumerate(frames):
        r, c = divmod(i, cols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        sheet[y:y + h, x:x + w] = f
    return sheet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/soccer2v2_1e")
    p.add_argument("--steps", type=int, default=120)
    p.add_argument("--worlds", type=int, default=2)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--spawn", default="mirror", choices=("mirror", "uniform"))
    p.add_argument("--action-scale", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.makedirs(args.out, exist_ok=True)

    env = WarpSoccer2v2Env(num_worlds=args.worlds, use_gpu=args.gpu,
                           seed=args.seed, spawn=args.spawn,
                           match_seconds=max(1.0, args.steps * 0.025),
                           ball=drill_ball())
    r = Soccer2v2Renderer(env)
    dev = env.device
    env.reset()
    g = torch.Generator(device=dev).manual_seed(args.seed)

    clip, picks, every = [], [], max(1, args.steps // 6)
    for t in range(args.steps):
        a = (torch.rand(env.n * env.n_agents, env.act_dim, generator=g,
                        device=dev) * 2 - 1) * args.action_scale
        env.step(a)
        clip.append(r.frame(env, 0, "topdown"))
        if t % every == 0 and len(picks) < 5:
            picks.append(r.frame(env, 0, "free" if len(picks) % 2 else "topdown"))

    # The sixth panel is a CLOSE-UP on the ball next to a creature. On a 30 x
    # 22.5 m pitch a 0.15 m ball is three pixels from overhead, which is not
    # evidence of anything -- and "the ball is the wrong size" is one of the two
    # bugs this render exists to catch, so it gets a frame where it is legible.
    near = min(range(env.n_agents),
               key=lambda k: float(torch.linalg.norm(
                   env.root_xy(k)[0] - env.ball_xy()[0])))
    b = env.ball_xyz()[0].detach().cpu().numpy()
    c = env.root_xy(near)[0].detach().cpu().numpy()
    mid = [(b[0] + c[0]) / 2, (b[1] + c[1]) / 2, 0.4]
    span = float(np.linalg.norm(b[:2] - c)) + 3.0
    picks.append(r.frame(env, 0, "free", lookat=mid, distance=span))

    import imageio
    sheet_path = os.path.join(args.out, "contact_sheet.png")
    imageio.imwrite(sheet_path, contact_sheet(picks))
    clip_path = os.path.join(args.out, "clip.mp4")
    with imageio.get_writer(clip_path, fps=40, quality=7) as wr:
        for f in clip:
            wr.append_data(f)

    xy = [[round(float(v), 2) for v in env.root_xy(k)[0].tolist()]
          for k in range(env.n_agents)]
    print(f"wrote {sheet_path} and {clip_path}")
    print(f"pitch half {env.pitch_half}, goal line {env.goal_x:.2f}, "
          f"goal half-width {env.goal_half_width:.2f}, "
          f"ball r {env.ball_radius}, creature spawn z {env.spawn_z}")
    print(f"final player xy {xy}, ball {[round(float(v), 2) for v in env.ball_xyz()[0].tolist()]}")
    print(f"stats {env.match_stats()}")


if __name__ == "__main__":
    main()
