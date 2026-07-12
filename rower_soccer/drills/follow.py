"""Follow drill (DeepMind 2021 humanoid football, supplementary Table S2/S3).

"The agent must follow a moving target that moves at fixed velocity for a
short episode and in variable directions. The target velocity is randomized
at the start of the episode. The agent observes the current target and the
future position of the target."

Reward = fitness = exp(-coef * ||x_player - x_target||)   (paper: coef = 1/2,
distances at humanoid scale; coef configurable for our larger bodies).

The drill tasks are not part of open-source dm_soccer; implemented here as a
dm_control composer task on a flat floor.
"""

import numpy as np
from dm_control import composer, mjcf
from dm_control.composer.observation import observable
from dm_control.locomotion.arenas import floors

# Applies the project arena.xml monkey-patch (physics opts) at import time,
# keeping drill physics identical to the soccer env.
import rower_soccer.envs.build  # noqa: F401
from rower_soccer.envs.build import make_creature

_CONTROL_DT = 0.025
_PHYSICS_DT = 0.0025


class MovingTarget(composer.Entity):
    """Non-colliding visual target marker; attached as a free entity and moved
    kinematically via set_pose each control step."""

    def _build(self, radius=0.5, rgba=(1, 0.2, 0.2, 1)):
        self._mjcf_root = mjcf.RootElement(model="target")
        self._geom = self._mjcf_root.worldbody.add(
            "geom", name="target_geom", type="sphere", size=[radius],
            rgba=rgba, contype=0, conaffinity=0, mass=0.001, group=1)

    @property
    def mjcf_model(self):
        return self._mjcf_root

    def set_pose_xy(self, physics, xy, height):
        self.set_pose(physics, position=np.array([xy[0], xy[1], height]),
                      quaternion=np.array([1.0, 0.0, 0.0, 0.0]))
        self.set_velocity(physics, velocity=np.zeros(3), angular_velocity=np.zeros(3))


class FollowTask(composer.Task):
    """Single creature follows a moving target."""

    def __init__(self,
                 creature_kind="worm",
                 episode_seconds=15.0,
                 arena_size=(30.0, 30.0),
                 bounds=27.0,
                 target_speed_range=(0.25, 2.0),
                 spawn_dist_range=(2.0, 6.0),
                 direction_change_prob=0.0,   # per control step; 0 = constant velocity (v1)
                 lookahead_seconds=(1.0,),
                 reward_coef=0.5,
                 target_height=1.0):
        # These MUST track WarpFollowEnv's defaults: this task is the CPU
        # transfer/parity eval for the Warp-trained policy, so a mismatch in
        # target speed or spawn distance would show up as a phantom sim2sim gap.
        self._arena = floors.Floor(size=arena_size)
        self._walker = make_creature(creature_kind, "home")
        self._arena.add_free_entity(self._walker)
        self._target = MovingTarget()
        self._arena.add_free_entity(self._target)

        self._episode_seconds = episode_seconds
        self._speed_range = target_speed_range
        self._dir_change_prob = direction_change_prob
        self._lookahead = lookahead_seconds
        self._reward_coef = reward_coef
        self._target_height = target_height
        self._spawn_dist_range = spawn_dist_range
        # Explicit, not arena_size * 0.9: the floor is just ground to stand on
        # and can stay large, while the target's roaming box has to match the
        # Warp env's `bounds`.
        self._bounds = np.array([bounds, bounds], dtype=np.float64)
        self._target_xy = np.zeros(2)
        self._target_vel = np.zeros(2)

        self.set_timesteps(control_timestep=_CONTROL_DT, physics_timestep=_PHYSICS_DT)

        # --- task observables (egocentric target now + future) ---
        def target_ego_now(physics):
            return self._to_ego(physics, self._target_xy)

        def target_ego_future(physics):
            outs = []
            for dt in self._lookahead:
                outs.append(self._to_ego(physics, self._predict_target(dt)))
            return np.concatenate(outs)

        self._task_observables = {
            "target_ego": observable.Generic(target_ego_now),
            "target_ego_future": observable.Generic(target_ego_future),
        }
        for obs in self._task_observables.values():
            obs.enabled = True

        # enable walker proprioception
        for o in self._walker.observables.proprioception:
            o.enabled = True
        for o in [self._walker.observables.sensors_velocimeter,
                  self._walker.observables.sensors_gyro,
                  self._walker.observables.sensors_accelerometer]:
            o.enabled = True

    # --- target kinematics -------------------------------------------------
    def _predict_target(self, dt):
        xy = self._target_xy + self._target_vel * dt
        return np.clip(xy, -self._bounds, self._bounds)

    def _to_ego(self, physics, world_xy):
        root = physics.bind(self._walker.root_body)
        pos = np.array(root.xpos[:2])
        xmat = np.array(root.xmat).reshape(3, 3)
        fwd, left = xmat[:2, 0], xmat[:2, 1]
        d = world_xy - pos
        return np.array([np.dot(d, fwd), np.dot(d, left)], dtype=np.float32)

    # --- composer API ------------------------------------------------------
    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def initialize_episode(self, physics, random_state):
        # place walker at center-ish with random yaw
        self._walker.reinitialize_pose(physics, random_state)
        angle = random_state.uniform(0, 2 * np.pi)
        speed = random_state.uniform(*self._speed_range)
        self._target_vel = speed * np.array([np.cos(angle), np.sin(angle)])
        # target starts within a few meters of the walker
        start_angle = random_state.uniform(0, 2 * np.pi)
        start_dist = random_state.uniform(*self._spawn_dist_range)
        # np.array(...) casts are load-bearing: physics.bind views are
        # SynchronizingArrayWrapper, and numpy propagates the subclass through
        # arithmetic — item-assignment on a stale wrapper crashes workers.
        root_xy = np.array(physics.bind(self._walker.root_body).xpos[:2])
        self._target_xy = np.array(np.clip(
            root_xy + start_dist * np.array([np.cos(start_angle), np.sin(start_angle)]),
            -self._bounds, self._bounds), dtype=np.float64)
        self._target.set_pose_xy(physics, self._target_xy, self._target_height)
        self._rng = random_state

    def before_step(self, physics, action, random_state):
        self._walker.apply_action(physics, action, random_state)

    def after_step(self, physics, random_state):
        dt = self.control_timestep
        if self._dir_change_prob > 0 and random_state.rand() < self._dir_change_prob:
            angle = random_state.uniform(0, 2 * np.pi)
            self._target_vel = np.linalg.norm(self._target_vel) * np.array(
                [np.cos(angle), np.sin(angle)])
        self._target_xy = self._target_xy + self._target_vel * dt
        # bounce at bounds
        for i in range(2):
            if abs(self._target_xy[i]) > self._bounds[i]:
                self._target_xy[i] = np.sign(self._target_xy[i]) * self._bounds[i]
                self._target_vel[i] *= -1
        self._target.set_pose_xy(physics, self._target_xy, self._target_height)

    def get_reward(self, physics):
        root_xy = physics.bind(self._walker.root_body).xpos[:2]
        dist = float(np.linalg.norm(root_xy - self._target_xy))
        return float(np.exp(-self._reward_coef * dist))

    def should_terminate_episode(self, physics):
        return False

    def get_discount(self, physics):
        return 1.0


def make_follow_env(random_state=None, **task_kwargs):
    task = FollowTask(**task_kwargs)
    return composer.Environment(
        task=task,
        time_limit=task._episode_seconds,
        random_state=random_state,
        strip_singleton_obs_buffer_dim=True)
