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
    def __init__(self, w_player_to_ball=0.15, w_ball_to_target=0.3,
                 target_speed_range=(0.03, 0.15),
                 ball_spawn_range=(1.5, 3.0),
                 target_dist_range=(2.0, 5.0), **kwargs):
        super().__init__(target_speed_range=target_speed_range, **kwargs)
        self._w_p2b = w_player_to_ball
        self._w_b2t = w_ball_to_target
        # dm_control's own 1-3 m. The 1.76 m worm's footprint is 0.82 m, so the
        # ball fits there; on the old 9.95 m worm (4.65 m footprint) it would
        # have spawned inside the creature.
        self._ball_spawn_range = ball_spawn_range
        # Target is anchored to the BALL, not the worm, so ||ball-target|| is
        # bounded directly rather than falling out of two independent draws.
        # Must stay in step with warp_port/dribble_env.py.
        self._target_dist_range = target_dist_range
        self._ball = SoccerBall()
        self._arena.add_free_entity(self._ball)

        def ball_ego(physics):
            # 3-D, egocentric, position + linear velocity -- i.e. exactly the 2v2
            # game's ball_ego_position (3) + ball_ego_linear_velocity (3), so the
            # distilled dribble prior can be evaluated on game observations. It was
            # 2-D, which the game could not have fed.
            #
            # Both position AND velocity are egocentric. This used to report the
            # ball's velocity in the WORLD frame alongside an egocentric position,
            # which breaks the egocentric invariance the whole observation is built
            # on -- the same ball, same relative motion, would look different
            # depending on which way the creature happened to be facing.
            # Must stay in step with warp_port/dribble_env.py.
            b = physics.bind(self._ball.root_body)
            pos = self._to_ego3(physics, np.array(b.xpos))
            vel = self._vec_to_ego3(physics, np.array(b.cvel[3:6]))
            return np.concatenate([pos, vel]).astype(np.float32)

        self._task_observables["ball_ego"] = observable.Generic(ball_ego)
        self._task_observables["ball_ego"].enabled = True

    def initialize_episode(self, physics, random_state):
        super().initialize_episode(physics, random_state)
        # ball starts outside the walker's own footprint.
        #
        # The np.array(...) cast is load-bearing, for the same reason FollowTask
        # casts: physics.bind() returns a SynchronizingArrayWrapper, and numpy
        # PROPAGATES that subclass through arithmetic. Without the cast, ball_xy
        # and then self._target_xy stay wrappers, and FollowTask.after_step's
        # bounce (`self._target_xy[i] = ...`) does item-assignment on one --
        # which routes into dm_control's physics binding and raises
        # AttributeError: 'SynchronizingArrayWrapper' object has no attribute
        # '_physics'.
        #
        # It only fires once the target first reaches the bounds, so a short
        # smoke test never sees it: both dribble runs trained happily for
        # 8-22M steps and then died on their first CPU eval video.
        root_xy = np.array(physics.bind(self._walker.root_body).xpos[:2])
        angle = random_state.uniform(0, 2 * np.pi)
        dist = random_state.uniform(*self._ball_spawn_range)
        ball_xy = root_xy + dist * np.array([np.cos(angle), np.sin(angle)])
        radius = float(physics.bind(self._ball.geom).size[0])
        self._ball.set_pose(physics, position=np.array([*ball_xy, radius]))
        self._ball.set_velocity(physics, velocity=np.zeros(3),
                                angular_velocity=np.zeros(3))
        # Re-anchor the target to the ball. super() placed it relative to the
        # WALKER (FollowTask's behaviour), which would leave it far enough from
        # the ball to sit in the flat-zero tail of exp(-c*d), where the drill has
        # no gradient at all.
        t_angle = random_state.uniform(0, 2 * np.pi)
        t_dist = random_state.uniform(*self._target_dist_range)
        self._target_xy = np.array(np.clip(
            ball_xy + t_dist * np.array([np.cos(t_angle), np.sin(t_angle)]),
            -self._bounds, self._bounds), dtype=np.float64)
        self._target.set_pose_xy(physics, self._target_xy, self._target_height)

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
