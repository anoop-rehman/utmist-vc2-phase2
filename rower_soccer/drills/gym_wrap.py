"""Gymnasium wrapper for composer drill envs (SB3-compatible).

Flattens the observation dict in sorted-key order and records the layout so
the policy can slice proprio vs task features. Task observables are the keys
NOT prefixed by the walker name.
"""

import gymnasium as gym
import numpy as np


class DrillGymEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(self, env_factory, seed=None, camera="creature/floating",
                 render_wh=(640, 480)):
        self._env = env_factory(random_state=seed)
        self._camera = camera
        self._render_wh = render_wh

        ts = self._env.reset()
        keys = sorted(ts.observation.keys())
        self.obs_keys = keys
        sizes = {k: int(np.asarray(ts.observation[k]).size) for k in keys}
        self.obs_dim = sum(sizes.values())
        # layout: slices per key + proprio/task index lists
        self.layout = {}
        idx = 0
        for k in keys:
            self.layout[k] = (idx, idx + sizes[k])
            idx += sizes[k]
        self.task_indices = np.concatenate([
            np.arange(*self.layout[k]) for k in keys if "/" not in k]).astype(np.int64)
        self.proprio_indices = np.concatenate([
            np.arange(*self.layout[k]) for k in keys if "/" in k]).astype(np.int64)

        spec = self._env.action_spec()
        self.action_space = gym.spaces.Box(
            low=spec.minimum.astype(np.float32),
            high=spec.maximum.astype(np.float32), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

    def _flatten(self, obs_dict):
        return np.concatenate(
            [np.asarray(obs_dict[k], dtype=np.float32).ravel() for k in self.obs_keys])

    def reset(self, *, seed=None, options=None):
        ts = self._env.reset()
        return self._flatten(ts.observation), {}

    def step(self, action):
        ts = self._env.step(action)
        obs = self._flatten(ts.observation)
        reward = float(ts.reward or 0.0)
        terminated = ts.last() and self._env.task.should_terminate_episode(self._env.physics)
        truncated = ts.last() and not terminated
        return obs, reward, terminated, truncated, {}

    def render(self):
        w, h = self._render_wh
        return self._env.physics.render(camera_id=self._camera, width=w, height=h)
