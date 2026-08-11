"""Render a world of the batched run-to-goal env. Warp is ground truth.

Same contract as `warp_port/render.py`: physics never leaves Warp. This builds a
SEPARATE, render-only MjModel from the same MJCF, copies one world's qpos into
it, calls `mj_forward` to place the geoms, and takes a picture. No stepping, no
solver, no contacts -- what you see is exactly the state that was simulated.

The camera sits above the halfway line looking down the x axis, because the whole
task is a race along x between two goal lines at x = +/-4.
"""

import mujoco
import numpy as np
import torch

from rower_soccer.competevo_port.scene import build_run_to_goal_scene


class RunToGoalRenderer:
    def __init__(self, width=960, height=540, distance=9.0, elevation=-22.0,
                 azimuth=90.0, scene_kwargs=None):
        self.model, self.meta = build_run_to_goal_scene(**(scene_kwargs or {}))
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        self.cam = mujoco.MjvCamera()
        self.cam.distance = distance
        self.cam.elevation = elevation
        self.cam.azimuth = azimuth

    def frame(self, env, w=0):
        self.data.qpos[:] = env.qpos[w].detach().float().cpu().numpy()
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        # Track the midpoint of the two ants so both stay in frame as they race.
        torsos = [a.torso_body for a in self.meta.agents]
        self.cam.lookat[:] = self.data.xpos[torsos].mean(0)
        self.renderer.update_scene(self.data, camera=self.cam)
        return self.renderer.render()


@torch.no_grad()
def eval_video(env, ac, path, renderer=None, fps=40, world=0, max_steps=None):
    """One deterministic episode of world `world`, rendered to `path`.

    Deterministic = the distribution's mean, which is their eval action
    (`DiagGaussian.mean_sample`). Sampling here would show exploration noise, not
    the policy. Returns (per-agent episode return, episode length, winner index
    or None).
    """
    import imageio

    renderer = renderer or RunToGoalRenderer()
    obs = env.reset()
    frames, rew_sum, winner = [], np.zeros(env.n_agents), None
    steps = max_steps or env.max_episode_steps
    for t in range(steps):
        a = ac.mean_action(obs.reshape(-1, env.obs_dim)).clamp(-1, 1)
        obs, rew, done, info = env.step(a.reshape(env.n, env.n_agents, -1))
        rew_sum += rew[world].float().cpu().numpy()
        frames.append(renderer.frame(env, w=world))
        if bool(done[world]):
            win = info["winner"][world].cpu().numpy()
            winner = int(win.argmax()) if win.any() else None
            break
    with imageio.get_writer(path, fps=fps, quality=7) as wr:
        for f in frames:
            wr.append_data(f)
    return rew_sum, len(frames), winner
