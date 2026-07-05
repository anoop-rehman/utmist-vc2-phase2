"""Dribble drill (DeepMind 2021, supplementary Table S2/S3).

"The environment is similar to the 'follow' drill but the agent must keep
the ball close to the moving target."

Rewards (Table S3):
  - Ball Close to Target: exp(-0.5 * ||x_ball - x_target||)   [fitness]
  - Velocity Player to Ball (shaping)
  - Velocity Ball to Target (shaping)

Reuses FollowTask's target kinematics; adds a soccer ball. Shaping weights
are fixed v1 (the paper evolves them via PBT; we expose them as kwargs).
"""

import numpy as np
from dm_control import composer
from dm_control.composer.observation import observable
from dm_control.locomotion.soccer.soccer_ball import SoccerBall

from rower_soccer.drills.follow import FollowTask, _CONTROL_DT


class DribbleTask(FollowTask):
    def __init__(self, w_player_to_ball=0.1, w_ball_to_target=0.3,
                 target_speed_range=(0.1, 1.0), **kwargs):
        super().__init__(target_speed_range=target_speed_range, **kwargs)
        self._w_p2b = w_player_to_ball
        self._w_b2t = w_ball_to_target
        self._ball = SoccerBall()
        self._arena.add_free_entity(self._ball)

        def ball_ego(physics):
            pos = self._to_ego(physics, physics.bind(self._ball.root_body).xpos[:2])
            vel = physics.bind(self._ball.root_body).cvel[3:5]
            return np.concatenate([pos, vel]).astype(np.float32)

        self._task_observables["ball_ego"] = observable.Generic(ball_ego)
        self._task_observables["ball_ego"].enabled = True

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        # ball starts near the walker
        root_xy = physics.bind(self._walker.root_body).xpos[:2]
        angle = random_state.uniform(0, 2 * np.pi)
        dist = random_state.uniform(1.0, 3.0)
        ball_xy = root_xy + dist * np.array([np.cos(angle), np.sin(angle)])
        self._ball.set_pose(physics, position=np.array([*ball_xy, 0.5]))
        self._ball.set_velocity(physics, velocity=np.zeros(3),
                                angular_velocity=np.zeros(3))

    def _ball_xy(self, physics):
        return physics.bind(self._ball.root_body).xpos[:2].copy()

    def get_reward(self, physics):
        ball_xy = self._ball_xy(physics)
        ball_vel = physics.bind(self._ball.root_body).cvel[3:5]
        root = physics.bind(self._walker.root_body)
        root_xy, root_vel = root.xpos[:2], root.cvel[3:5]

        # fitness: ball close to target
        fitness = float(np.exp(-self._reward_coef * np.linalg.norm(ball_xy - self._target_xy)))

        # shaping: player velocity toward ball
        d_pb = ball_xy - root_xy
        n_pb = np.linalg.norm(d_pb)
        v_p2b = float(np.dot(root_vel, d_pb / n_pb)) if n_pb > 1e-6 else 0.0

        # shaping: ball velocity toward target
        d_bt = self._target_xy - ball_xy
        n_bt = np.linalg.norm(d_bt)
        v_b2t = float(np.dot(ball_vel, d_bt / n_bt)) if n_bt > 1e-6 else 0.0

        return fitness + self._w_p2b * max(v_p2b, 0.0) + self._w_b2t * max(v_b2t, 0.0)

    def get_fitness(self, physics):
        """Unshaped fitness for gating (Table S3 definition)."""
        ball_xy = self._ball_xy(physics)
        return float(np.exp(-self._reward_coef * np.linalg.norm(ball_xy - self._target_xy)))


def make_dribble_env(random_state=None, **task_kwargs):
    task = DribbleTask(**task_kwargs)
    return composer.Environment(
        task=task,
        time_limit=task._episode_seconds,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True)
