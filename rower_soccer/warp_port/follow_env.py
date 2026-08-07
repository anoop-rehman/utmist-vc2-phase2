"""Batched GPU follow-drill env (worm) on the backend-agnostic WormEnv base.

Obs (proprio-first): proprio(29) | target_ego3(3) | target_ego_future3(3) = 35.
The target is 3-D (a ground target projected into the root frame). The scene is
the shared arena WITH a ball present; follow ignores/parks the ball.

See worm_env_base.WormEnv for the shared plumbing and FollowReward for the reward
(paper / velshape / progress modes).
"""
import numpy as np
import torch

from rower_soccer.warp_port.backend import WarpBackend
from rower_soccer.warp_port.worm_env_base import (WormEnv, MovingTargetMixin,
                                                  FollowReward)


class WarpFollowEnv(MovingTargetMixin, WormEnv):
    def __init__(self, num_worlds=2048,
                 creature_xml="creature_configs/three_seg_worm.xml",
                 episode_seconds=15.0, target_speed_range=(0.04, 0.34),
                 lookahead=1.0, reward_coef=0.5, bounds=10.0, device="cuda",
                 seed=0, use_graph=True, w_vel_shaping=0.0, reward_mode="paper",
                 progress_scale=2.0, settle_coef=0.5, arrival_radius=1.0,
                 arrival_bonus=0.5, spawn_dist_range=(1.76, 5.28),
                 nconmax=64, njmax=512, energy_coef=0.0, smooth_coef=0.0,
                 rew_clip=(-10.0, 10.0), reward=None, floor_half=5.0,
                 backend_cls=WarpBackend):
        self._lookahead = lookahead
        self._bounds = bounds
        self._speed_range = target_speed_range
        self.spawn_dist_range = spawn_dist_range
        reward = reward or FollowReward(
            mode=reward_mode, reward_coef=reward_coef, w_vel_shaping=w_vel_shaping,
            progress_scale=progress_scale, settle_coef=settle_coef,
            arrival_radius=arrival_radius, arrival_bonus=arrival_bonus)
        super().__init__(num_worlds=num_worlds, creature_xml=creature_xml,
                         episode_seconds=episode_seconds, device=device, seed=seed,
                         use_graph=use_graph, nconmax=nconmax, njmax=njmax,
                         reward=reward, floor_half=floor_half,
                         energy_coef=energy_coef, smooth_coef=smooth_coef,
                         rew_clip=rew_clip, backend_cls=backend_cls)

    def _task_dim(self):
        return 6  # target_ego3 (3) + target_ego_future3 (3)

    def _task_init(self):
        self._init_moving_target(self._lookahead, self._bounds, self._speed_range)

    def _task_obs(self):
        return self._target_obs3()

    def _reset_state(self):
        n = self.n
        yaw = self._rand(n) * (2 * np.pi)
        self._spawn_root(xy=torch.zeros(n, 2, device=self.device), yaw=yaw)
        # target: 1-3 body lengths out, random direction & speed.
        ang = self._rand(n) * (2 * np.pi)
        d0, d1 = self.spawn_dist_range
        dist = d0 + (d1 - d0) * self._rand(n)
        # Clamp into the arena (self.bounds is capped to the walls) so the target
        # never spawns outside; dribble already clamps its own spawn.
        self.target_xy = torch.stack([dist * torch.cos(ang),
                                      dist * torch.sin(ang)], -1).clamp(
            -self.bounds, self.bounds)
        vang = self._rand(n) * (2 * np.pi)
        s0, s1 = self.speed_range
        spd = s0 + (s1 - s0) * self._rand(n)
        self.target_vel = torch.stack([spd * torch.cos(vang),
                                       spd * torch.sin(vang)], -1)
        self._park_ball()

    def _park_ball(self, idx=slice(None)):
        # Follow ignores the ball; park it resting on the floor in a corner, clear
        # of both the worm's origin spawn and the arena walls. Parking it AT a wall
        # would interpenetrate it (ball radius), eject it, and trip the divergence
        # reset. Scales with the arena so it stays inside for any floor_half.
        p = max(self.ball_radius, self._floor_half - 1.0)
        self.qpos[idx, self.bq + 0] = p
        self.qpos[idx, self.bq + 1] = p
        self.qpos[idx, self.bq + 2] = self.ball_radius
        self.qpos[idx, self.bq + 3] = 1.0

    def _sanitize_task(self, idx):
        self._park_ball(idx)
