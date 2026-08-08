"""Batched GPU dribble-drill env (worm) on the backend-agnostic WormEnv base.

"The agent must keep the ball close to the moving target" (DeepMind 2021, Table
S2/S3). The ball is a real physics entity the creature contacts.

Obs (proprio-first, unified): proprio(29) | ball_ego(6) | target_ego3(3) |
target_ego_future3(3) = 41. ball_ego = ego pos(3) + ego linear vel(3), matching
the 2v2 game's ball obs so it survives distillation into the dribble prior. NOTE
ball_ego is now APPENDED after proprio (was prepended in the old sorted-key
layout); the task block is contiguous at [29, 41).

See worm_env_base.WormEnv for shared plumbing and DribbleReward (paper/progress).
"""
import numpy as np
import torch

from rower_soccer.warp_port.worm_env_base import (WormEnv, MovingTargetMixin,
                                                  DribbleReward)
from rower_soccer.warp_port.scene import BallSpec


class WarpDribbleEnv(MovingTargetMixin, WormEnv):
    def __init__(self, num_worlds=2048,
                 creature_xml="creature_configs/three_seg_worm.xml",
                 episode_seconds=15.0, target_speed_range=(0.04, 0.25),
                 lookahead=1.0, reward_coef=0.5, bounds=10.0, device=None,
                 seed=0, use_graph=True, target_dist_range=(2.0, 5.0),
                 ball_spawn_range=(1.5, 3.0), w_player_to_ball=0.1,
                 w_ball_to_target=0.3, reward_mode="paper", progress_scale=2.0,
                 approach_scale=0.5, ball: BallSpec = None, nconmax=64, njmax=512,
                 energy_coef=0.0, smooth_coef=0.0, rew_clip=(-10.0, 10.0),
                 fixed_start=False, target_cone=0.0, reward=None, floor_half=5.0,
                 use_gpu=True, backend_cls=None):
        self._lookahead = lookahead
        self._bounds = bounds
        self._speed_range = target_speed_range
        self.target_dist_range = target_dist_range
        self.ball_spawn_range = ball_spawn_range
        self._ball = ball
        # Public mutable knobs the trainer / eval set at runtime.
        self.shaping_scale = 1.0
        self.fixed_start = fixed_start
        self.target_cone = target_cone
        reward = reward or DribbleReward(
            mode=reward_mode, reward_coef=reward_coef,
            w_player_to_ball=w_player_to_ball, w_ball_to_target=w_ball_to_target,
            approach_scale=approach_scale, progress_scale=progress_scale)
        super().__init__(num_worlds=num_worlds, creature_xml=creature_xml,
                         episode_seconds=episode_seconds, use_gpu=use_gpu,
                         device=device, seed=seed, use_graph=use_graph,
                         nconmax=nconmax, njmax=njmax, reward=reward,
                         floor_half=floor_half, energy_coef=energy_coef,
                         smooth_coef=smooth_coef, rew_clip=rew_clip,
                         backend_cls=backend_cls)

    def _ball_spec(self):
        return self._ball or BallSpec()

    def _task_dim(self):
        return 12  # ball_ego(6) + target_ego3(3) + target_ego_future3(3)

    def _task_init(self):
        self._init_moving_target(self._lookahead, self._bounds, self._speed_range)

    def _task_obs(self):
        ball_ego = torch.cat([self._to_ego3(self._ball_xyz()),
                              self._vec_to_ego3(self._ball_vel_xyz())], -1)
        return torch.cat([ball_ego, self._target_obs3()], -1)

    def _reset_state(self):
        n = self.n
        # Curriculum stage 1 (--fixed-start): worm, ball, target COLINEAR along a
        # random theta so walking forward accidentally pushes the ball at the
        # target; theta is a world rotation invisible to the egocentric obs.
        theta = self._rand(n) * (2 * np.pi)
        yaw = theta if self.fixed_start else self._rand(n) * (2 * np.pi)
        self._spawn_root(xy=torch.zeros(n, 2, device=self.device), yaw=yaw)

        bang = theta if self.fixed_start else self._rand(n) * (2 * np.pi)
        b0, b1 = self.ball_spawn_range
        bdist = b0 + (b1 - b0) * self._rand(n)
        ball_xy = torch.stack([bdist * torch.cos(bang),
                               bdist * torch.sin(bang)], -1)
        self.qpos[:, self.bq + 0] = ball_xy[:, 0]
        self.qpos[:, self.bq + 1] = ball_xy[:, 1]
        self.qpos[:, self.bq + 2] = self.ball_radius
        self.qpos[:, self.bq + 3] = 1.0

        if self.fixed_start:
            offset = (self._rand(n) * 2.0 - 1.0) * self.target_cone
            ang = theta + offset
        else:
            ang = self._rand(n) * (2 * np.pi)
        d0, d1 = self.target_dist_range
        dist = d0 + (d1 - d0) * self._rand(n)
        tgt = ball_xy + torch.stack([dist * torch.cos(ang),
                                     dist * torch.sin(ang)], -1)
        self.target_xy = tgt.clamp(-self.bounds, self.bounds)
        vang = self._rand(n) * (2 * np.pi)
        s0, s1 = self.speed_range
        spd = s0 + (s1 - s0) * self._rand(n)
        self.target_vel = torch.stack([spd * torch.cos(vang),
                                       spd * torch.sin(vang)], -1)

    def _sanitize_task(self, idx):
        # Ball at rest 1 m out along +x, clear of the worm footprint.
        self.qpos[idx, self.bq + 0] = 1.0
        self.qpos[idx, self.bq + 1] = 0.0
        self.qpos[idx, self.bq + 2] = self.ball_radius
        self.qpos[idx, self.bq + 3] = 1.0
