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
    """Render a fixed window of world `world` to `path`, spanning as many
    episodes as fit (the env auto-resets), and return one record per finished
    episode: `(return per agent, length, winner index or None)`.

    A window rather than a single episode ON PURPOSE. Early in training an ant
    falls within a second or two, and a one-episode video is 90 frames of a
    creature toppling -- which tells a viewer nothing about whether the race is
    working. Several consecutive episodes show the actual behaviour distribution.

    Deterministic = the distribution's mean, which is their eval action
    (`DiagGaussian.mean_sample`). Sampling here would show exploration noise
    instead of the policy.
    """
    import imageio

    renderer = renderer or RunToGoalRenderer()
    obs = env.reset()
    frames, rew_sum, episodes, t0 = [], np.zeros(env.n_agents), [], 0
    steps = max_steps or env.max_episode_steps
    for t in range(steps):
        a = ac.mean_action(obs.reshape(-1, env.obs_dim).float()).clamp(-1, 1)
        obs, rew, done, info = env.step(a.reshape(env.n, env.n_agents, -1).to(env.dtype))
        rew_sum += rew[world].float().cpu().numpy()
        frames.append(renderer.frame(env, w=world))
        if bool(done[world]):
            win = info["winner"][world].cpu().numpy()
            episodes.append({"return": rew_sum.round(1).tolist(),
                             "length": t + 1 - t0,
                             "winner": int(win.argmax()) if win.any() else None})
            rew_sum, t0 = np.zeros(env.n_agents), t + 1
    with imageio.get_writer(path, fps=fps, quality=7) as wr:
        for f in frames:
            wr.append_data(f)
    return episodes, len(frames)
