"""Backend-agnostic worm env base + interchangeable task/reward blocks.

`WormEnv` owns everything shared across the follow / dribble / fetch drills: the
29-dim worm proprio observation, ego-frame geometry, divergence sanitizing, and
the reset/step/obs template flow. It talks to the simulator ONLY through a
`PhysicsBackend` (backend.py), so it contains no `mjw.`/`wp.` calls and a CPU
MuJoCo backend can be swapped in later.

A task = (task-obs block + reward + spawn):
  * `MovingTargetMixin` adds the shared follow/dribble moving-target machinery.
  * `RewardStrategy` subclasses are pluggable and read only the base's common
    state accessors, so a reward can attach to any task (`env.fitness()` delegates
    to the strategy).

Observation layout is unified to PROPRIO-FIRST for every task -- proprio at
[0, n_proprio), the task block appended after -- so `_obs = cat([proprio, task])`
and `proprio_indices`/`task_indices` are contiguous everywhere. The target is
represented in 3-D across all tasks (`_to_ego3`).
"""
import json

import mujoco
import numpy as np
import torch

from rower_soccer.warp_port.backend import WarpBackend
from rower_soccer.warp_port.scene import BallSpec, build_creature_scene

CONTROL_DT = 0.025    # worm-stack control rate (follow/dribble/decoder)
SUBSTEPS = 10         # physics dt 0.0025
# Keep targets clear of the arena walls: the floor spans +/-floor_half and the
# walls sit just beyond it, so the target bound is floor_half minus this margin.
TARGET_WALL_MARGIN = 0.5


def _arena_xml(floor_half):
    """The fetch arena, worm-scaled: plane + 4 inward-tilted walls + a target
    marker site. Shared by every task now that the scene defaults to the arena."""
    fh = floor_half
    wall = fh + 0.7
    return f"""
<mujoco model="worm_arena">
  <option cone="elliptic" timestep="0.0025"/>
  <visual><global offwidth="1024" offheight="1024"/></visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="256" height="256"
             rgb1=".2 .3 .4" rgb2=".1 .15 .2"/>
    <material name="grid" texture="grid" texrepeat="4 4" reflectance="0.1"/>
  </asset>
  <worldbody>
    <light name="sun" pos="0 0 12" dir="0 0 -1" diffuse="1 1 1" directional="true"/>
    <geom name="floor" type="plane" size="{fh} {fh} .5" material="grid"/>
    <geom name="wall_px" pos="-{wall} 0 .35" zaxis="1 0 1"  type="box" size=".5 {fh} .25" rgba=".5 .5 .55 1"/>
    <geom name="wall_nx" pos="{wall} 0 .35"  zaxis="-1 0 1" type="box" size=".5 {fh} .25" rgba=".5 .5 .55 1"/>
    <geom name="wall_py" pos="0 -{wall} .35" zaxis="0 1 1"  type="box" size="{fh} .5 .25" rgba=".5 .5 .55 1"/>
    <geom name="wall_ny" pos="0 {wall} .35"  zaxis="0 -1 1" type="box" size="{fh} .5 .25" rgba=".5 .5 .55 1"/>
    <site name="target" type="cylinder" size=".4 .01" pos="0 0 .011" rgba=".9 .2 .2 .6"/>
  </worldbody>
</mujoco>
"""


def fetch_ball():
    """The quadruped-fetch ball (r=.15, its friction), not the soccer ball."""
    return BallSpec(radius=0.15, mass=0.35, friction=(0.7, 0.005, 0.005),
                    solref=(0.01, 1.0))


# ----------------------------------------------------------------------------
# Reward strategies (pluggable). Each reads only the env's common state
# accessors, so any reward can attach to any task.
# ----------------------------------------------------------------------------
class RewardStrategy:
    def bind(self, env):
        """Called once after env init; cache anything task-invariant here."""

    def reset(self, env):
        """Called each env.reset() AFTER forward(); seed prev-distance buffers."""

    def __call__(self, env):
        raise NotImplementedError

    def fitness(self, env):
        raise NotImplementedError


class FollowReward(RewardStrategy):
    """exp(-c*dist) (`paper`), + velocity shaping (`velshape`), or potential-based
    progress with settle + arrival bonus (`progress`)."""

    def __init__(self, mode="paper", reward_coef=0.5, w_vel_shaping=0.0,
                 progress_scale=2.0, settle_coef=0.5, arrival_radius=1.0,
                 arrival_bonus=0.5):
        self.mode = mode
        self.reward_coef = reward_coef
        self.w_vel_shaping = w_vel_shaping
        self.progress_scale = progress_scale
        self.settle_coef = settle_coef
        self.arrival_radius = arrival_radius
        self.arrival_bonus = arrival_bonus
        self.prev_dist = None

    def reset(self, env):
        pos, _ = env._root_frames()
        self.prev_dist = torch.linalg.norm(pos[:, :2] - env.target_xy, dim=-1)

    def __call__(self, env):
        pos, _ = env._root_frames()
        d = env.target_xy - pos[:, :2]
        dist = torch.linalg.norm(d, dim=-1)
        if self.mode == "progress":
            progress = self.prev_dist - dist
            self.prev_dist = dist.detach()
            r = self.progress_scale * progress
            r = r + self.settle_coef * torch.exp(-self.reward_coef * dist)
            r = r + self.arrival_bonus * (dist < self.arrival_radius).float()
            return r
        r = torch.exp(-self.reward_coef * dist)
        if self.mode == "velshape" or self.w_vel_shaping > 0:
            v_to_t = (env._root_vel_xy()
                      * (d / dist.clamp(min=1e-6).unsqueeze(-1))).sum(-1)
            r = r + self.w_vel_shaping * v_to_t.clamp(min=0.0)
        return r

    def fitness(self, env):
        pos, _ = env._root_frames()
        dist = torch.linalg.norm(env.target_xy - pos[:, :2], dim=-1)
        return torch.exp(-self.reward_coef * dist)


class DribbleReward(RewardStrategy):
    """Table-S3 fitness exp(-c*||ball-target||) + two velocity shaping terms
    (`paper`, scaled by the mutable `env.shaping_scale`), or two telescoping
    potentials player->ball and ball->target (`progress`)."""

    def __init__(self, mode="paper", reward_coef=0.5, w_player_to_ball=0.1,
                 w_ball_to_target=0.3, approach_scale=0.5, progress_scale=2.0):
        self.mode = mode
        self.reward_coef = reward_coef
        self.w_p2b = w_player_to_ball
        self.w_b2t = w_ball_to_target
        self.approach_scale = approach_scale
        self.progress_scale = progress_scale
        self.prev_bt = None
        self.prev_pb = None

    def reset(self, env):
        pos, _ = env._root_frames()
        ball = env._ball_xy()
        self.prev_bt = torch.linalg.norm(ball - env.target_xy, dim=-1)
        self.prev_pb = torch.linalg.norm(ball - pos[:, :2], dim=-1)

    def __call__(self, env):
        ball_xy, ball_vel = env._ball_xy(), env._ball_vel_xy()
        pos, _ = env._root_frames()
        root_xy, root_vel = pos[:, :2], env._root_vel_xy()
        d_bt = env.target_xy - ball_xy
        dist_bt = torch.linalg.norm(d_bt, dim=-1)
        d_pb = ball_xy - root_xy
        dist_pb = torch.linalg.norm(d_pb, dim=-1)
        if self.mode == "progress":
            approach = self.prev_pb - dist_pb
            progress = self.prev_bt - dist_bt
            self.prev_pb = dist_pb.detach()
            self.prev_bt = dist_bt.detach()
            return (self.approach_scale * approach
                    + self.progress_scale * progress
                    + torch.exp(-self.reward_coef * dist_bt))
        fitness = torch.exp(-self.reward_coef * dist_bt)
        n_pb = dist_pb.clamp(min=1e-6)
        v_p2b = (root_vel * (d_pb / n_pb.unsqueeze(-1))).sum(-1)
        n_bt = dist_bt.clamp(min=1e-6)
        v_b2t = (ball_vel * (d_bt / n_bt.unsqueeze(-1))).sum(-1)
        shaping = (self.w_p2b * v_p2b.clamp(min=0.0)
                   + self.w_b2t * v_b2t.clamp(min=0.0))
        return fitness + env.shaping_scale * shaping

    def fitness(self, env):
        dist_bt = torch.linalg.norm(env.target_xy - env._ball_xy(), dim=-1)
        return torch.exp(-self.reward_coef * dist_bt)


class FetchReward(RewardStrategy):
    """Quadruped-fetch reward: upright * reach * (0.5 + 0.5*fetch), all linear
    tolerances. `upright` uses the env's labeled up-axis; margins use
    `env.arena_radius`."""

    def __init__(self, reach_bound=0.65, fetch_bound=0.4):
        self.reach_bound = reach_bound
        self.fetch_bound = fetch_bound

    @staticmethod
    def _linear_tolerance(d, bound, margin):
        return torch.clamp(1.0 - (d - bound) / margin, max=1.0).clamp(min=0.0)

    def _terms(self, env):
        pos, rot = env._root_frames()
        up_world_z = torch.einsum("nij,j->ni", rot, env.up_local)[:, 2]
        upright = ((1.0 + up_world_z) / 2.0).clamp(0.0, 1.0)
        ball = env.qpos[:, env.bq:env.bq + 2]
        reach = self._linear_tolerance(
            torch.linalg.norm(ball - pos[:, :2], dim=-1),
            self.reach_bound, env.arena_radius)
        fetch = self._linear_tolerance(
            torch.linalg.norm(ball - env.target_xy, dim=-1),
            self.fetch_bound, env.arena_radius)
        return upright, reach, fetch

    def __call__(self, env):
        upright, reach, fetch = self._terms(env)
        return upright * reach * (0.5 + 0.5 * fetch)

    def fitness(self, env):
        return self(env)


# ----------------------------------------------------------------------------
# The backend-agnostic base env.
# ----------------------------------------------------------------------------
class WormEnv:
    """Shared worm drill env. Subclasses fill the task hooks:
        _task_dim(), _task_init(), _task_obs(), _reset_state(), _sanitize_task(idx)
    and optionally override _ball_spec(), _base_xml(), _post_build_model(model),
    _update_task().
    """

    def __init__(self, num_worlds=2048,
                 creature_xml="creature_configs/three_seg_worm.xml",
                 episode_seconds=15.0, device="cuda", seed=0, use_graph=True,
                 nconmax=64, njmax=512, reward=None, backend_cls=WarpBackend,
                 floor_half=5.0, energy_coef=0.0, smooth_coef=0.0,
                 rew_clip=(-10.0, 10.0)):
        self.n = num_worlds
        self._floor_half = floor_half
        self.episode_steps = int(round(episode_seconds / CONTROL_DT))
        self.n_diverged = 0
        self.energy_coef = energy_coef
        self.smooth_coef = smooth_coef
        self.rew_clip = rew_clip

        # 1. scene (subclass hooks) -- arena + ball by default.
        self.model, self.meta = build_creature_scene(
            creature_xml, ball=self._ball_spec(), base_xml=self._base_xml())
        self._post_build_model(self.model)

        # 2. backend owns all mujoco_warp specifics.
        self.backend = backend_cls(self.model, num_worlds, SUBSTEPS,
                                   use_graph=use_graph, nconmax=nconmax,
                                   njmax=njmax, device=device)
        self.device = self.backend.device
        self.gen = torch.Generator(device=self.device).manual_seed(seed)
        self.qpos = self.backend.qpos
        self.qvel = self.backend.qvel
        self.ctrl = self.backend.ctrl
        self.xpos = self.backend.xpos
        self.xmat = self.backend.xmat
        self.sensordata = self.backend.sensordata

        # 3. proprio index plumbing (creature-generic; ball-joint aware).
        m = self.meta
        jq_idx = [i for start, n in m.joint_qpos for i in range(start, start + n)]
        jv_idx = [i for start, n in m.joint_qvel for i in range(start, start + n)]
        self.jq = torch.as_tensor(jq_idx, device=self.device, dtype=torch.long)
        self.jv = torch.as_tensor(jv_idx, device=self.device, dtype=torch.long)
        # `w` slot of any 4-wide (ball) creature joint quaternion -- must be reset
        # to the identity quat, else forward() normalizes 0/0 = NaN. Empty for the
        # hinge-only worm; kept for generality.
        self.ball_qw_idx = torch.as_tensor(
            [start for start, n in m.joint_qpos if n == 4],
            device=self.device, dtype=torch.long)
        self.body_ids = torch.as_tensor(m.body_ids, device=self.device)
        ss = m.sensor_slices
        touch_keys = sorted(k for k in ss if k.endswith("_touch"))
        self.sl_touch = [ss[k] for k in touch_keys]
        self.sl_vel, self.sl_gyro, self.sl_accel = (ss["torso_vel"],
                                                    ss["torso_gyro"],
                                                    ss["torso_accel"])
        self.bq, self.bv = m.ball_qpos, m.ball_qvel
        self.ball_radius, self.ball_body = m.ball_radius, m.ball_body
        self.rv = m.qvel_root
        # Default spawn height; fetch overrides via _task_init (labeled pose).
        self._spawn_z = m.spawn_z

        # 4. dims (all derived, proprio-first + contiguous).
        n_proprio = (3 * len(m.body_ids) + 1 + len(self.jq) + len(self.jv)
                     + 9 + len(touch_keys) + 3)
        self.obs_dim = n_proprio + self._task_dim()
        self.act_dim = m.nu
        self.proprio_indices = np.arange(0, n_proprio)
        self.task_indices = np.arange(n_proprio, self.obs_dim)
        self.prev_ctrl = torch.zeros(self.n, self.act_dim, device=self.device)
        self.t = 0

        # 5. task state + reward.
        self._task_init()
        if reward is None:
            raise ValueError("a RewardStrategy must be supplied by the subclass")
        self.reward = reward
        self.reward.bind(self)

        # 6. CUDA-graph capture already done in the backend ctor.

    # -- subclass hooks (defaults) -----------------------------------------
    def _ball_spec(self):
        return BallSpec()

    def _base_xml(self):
        return _arena_xml(self._floor_half)

    def _post_build_model(self, model):
        pass

    def _task_dim(self):
        raise NotImplementedError

    def _task_init(self):
        pass

    def _task_obs(self):
        raise NotImplementedError

    def _reset_state(self):
        raise NotImplementedError

    def _update_task(self):
        pass

    def _sanitize_task(self, idx):
        pass

    # -- random helpers -----------------------------------------------------
    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _randn(self, *shape):
        return torch.randn(*shape, generator=self.gen, device=self.device)

    # -- physics (delegates to the backend) --------------------------------
    def _physics_step(self):
        self.backend.step()

    def _forward(self):
        self.backend.forward()

    # -- frames / ego geometry (common state accessors) --------------------
    def _root_frames(self):
        rb = self.meta.root_body
        return self.xpos[:, rb, :], self.xmat[:, rb]

    def _root_vel_xy(self):
        qr = self.meta.qvel_root
        return self.qvel[:, qr:qr + 2]

    def _to_ego(self, world_xy):
        pos, rot = self._root_frames()
        fwd, left = rot[:, :2, 0], rot[:, :2, 1]
        d = world_xy - pos[:, :2]
        return torch.stack([(d * fwd).sum(-1), (d * left).sum(-1)], -1)

    def _to_ego3(self, world_xyz):
        """Rotate a world-frame POSITION (relative to root) into the root frame."""
        pos, rot = self._root_frames()
        return torch.einsum("nij,nj->ni", rot.transpose(1, 2), world_xyz - pos)

    def _vec_to_ego3(self, world_vec):
        """Rotate a world-frame VECTOR (no translation) into the root frame."""
        _, rot = self._root_frames()
        return torch.einsum("nij,nj->ni", rot.transpose(1, 2), world_vec)

    def _ball_xy(self):
        return self.qpos[:, self.bq:self.bq + 2]

    def _ball_vel_xy(self):
        return self.qvel[:, self.bv:self.bv + 2]

    def _ball_xyz(self):
        return self.qpos[:, self.bq:self.bq + 3]

    def _ball_vel_xyz(self):
        return self.qvel[:, self.bv:self.bv + 3]

    # -- spawn helper -------------------------------------------------------
    def _spawn_root(self, xy=None, yaw=None, quat=None):
        """Write the creature root freejoint (qpos already zeroed by reset()).
        `quat` = (w,x,y,z) components (each scalar/[n]); else `yaw` builds a
        yaw-about-z quat. Also restores any creature ball-joint identity quats."""
        qr = self.meta.qpos_root
        if xy is not None:
            self.qpos[:, qr + 0] = xy[:, 0]
            self.qpos[:, qr + 1] = xy[:, 1]
        self.qpos[:, qr + 2] = self._spawn_z
        if quat is not None:
            qw, qx, qy, qz = quat
            self.qpos[:, qr + 3] = qw
            self.qpos[:, qr + 4] = qx
            self.qpos[:, qr + 5] = qy
            self.qpos[:, qr + 6] = qz
        elif yaw is not None:
            self.qpos[:, qr + 3] = torch.cos(yaw / 2)
            self.qpos[:, qr + 6] = torch.sin(yaw / 2)
        else:
            self.qpos[:, qr + 3] = 1.0
        if self.ball_qw_idx.numel():
            self.qpos[:, self.ball_qw_idx] = 1.0

    # -- proprio obs block (29 dims for the worm; widths derived) ----------
    def _proprio_obs(self):
        n = self.n
        pos, rot = self._root_frames()
        bp = self.xpos[:, self.body_ids, :] - pos.unsqueeze(1)
        bodies_ego = torch.einsum("nij,nbj->nbi",
                                  rot.transpose(1, 2), bp).reshape(n, -1)
        touch = torch.cat([self.sensordata[:, s:s + d]
                           for s, d in self.sl_touch], -1) / 10000.0
        sv, sg, sa = (self.sensordata[:, s:s + d] for s, d in
                      (self.sl_vel, self.sl_gyro, self.sl_accel))
        # Accelerometer is the ONLY unbounded input (contact spikes ~5,700 m/s^2);
        # /100 + clamp is part of the obs contract any deployment body must apply.
        sa = (sa / 100.0).clamp(-50.0, 50.0)
        world_zaxis = rot.reshape(n, 9)[:, 6:9]
        return torch.cat([
            bodies_ego,                 # creature/bodies_pos       (3*nbody)
            pos[:, 2:3],                # creature/body_height      (1)
            self.qpos[:, self.jq],      # creature/joints_pos       (len jq)
            self.qvel[:, self.jv],      # creature/joints_vel       (len jv)
            sa, sg, sv,                 # accel, gyro, velocimeter  (9)
            touch,                      # creature/touch_sensors    (n touch)
            world_zaxis,                # creature/world_zaxis      (3)
        ], -1)

    # -- obs / reset / step templates --------------------------------------
    def _obs(self):
        return torch.cat([self._proprio_obs(), self._task_obs()], -1)

    def reset(self):
        self.qpos.zero_()
        self.qvel.zero_()
        self._reset_state()
        self.t = 0
        self.prev_ctrl = torch.zeros(self.n, self.act_dim, device=self.device)
        self._forward()
        self.reward.reset(self)   # after forward(): frames/ball positions valid
        return self._obs()

    def _sanitize(self):
        """Reset diverged worlds (NaN/inf or |qvel|>500) in place BEFORE
        obs/reward, so downstream sees only clean state."""
        bad = ((~torch.isfinite(self.qvel).all(-1))
               | (~torch.isfinite(self.qpos).all(-1))
               | (self.qvel.abs().amax(-1) > 500.0))
        if not bool(bad.any()):
            return
        self.n_diverged += int(bad.sum().item())
        idx = bad.nonzero(as_tuple=True)[0]
        qr = self.meta.qpos_root
        self.qvel[idx] = 0.0
        self.qpos[idx] = 0.0
        self.qpos[idx, qr + 2] = self._spawn_z
        self.qpos[idx, qr + 3] = 1.0   # identity quat (task may overwrite)
        if self.ball_qw_idx.numel():
            self.qpos[idx.unsqueeze(-1), self.ball_qw_idx.unsqueeze(0)] = 1.0
        self._sanitize_task(idx)
        self._forward()

    def _regularize(self, rew, a):
        if self.energy_coef > 0:
            rew = rew - self.energy_coef * (a ** 2).mean(-1)
        if self.smooth_coef > 0:
            rew = rew - self.smooth_coef * ((a - self.prev_ctrl) ** 2).mean(-1)
        self.prev_ctrl = a
        return rew.clamp(self.rew_clip[0], self.rew_clip[1])

    def step(self, actions):
        """actions: [n, nu] in [-1, 1]. Returns (obs, reward[n], done:bool)."""
        a = actions.clamp(-1.0, 1.0)
        self.ctrl.copy_(a)
        self._physics_step()
        self._sanitize()
        self._update_task()
        self.t += 1
        done = self.t >= self.episode_steps
        rew = self._regularize(self.reward(self), a)
        return self._obs(), rew, done

    def fitness(self):
        return self.reward.fitness(self)


class MovingTargetMixin:
    """Shared follow/dribble moving-target machinery. `target_xy` stays [n,2]
    (so play_server can assign it); the 3-D-ness lives in the obs projection."""

    def _init_moving_target(self, lookahead, bounds, speed_range):
        self.lookahead = lookahead
        # Cap the requested bounds to the arena interior so the target (its reset
        # spawn AND its per-step bounce/clamp, both keyed on self.bounds) always
        # stays inside the walls. self._floor_half is set by the base __init__.
        arena_bound = max(0.5, self._floor_half - TARGET_WALL_MARGIN)
        self.bounds = min(bounds, arena_bound)
        self.speed_range = speed_range
        self.target_xy = torch.zeros(self.n, 2, device=self.device)
        self.target_vel = torch.zeros(self.n, 2, device=self.device)

    def _update_task(self):
        self.target_xy = self.target_xy + self.target_vel * CONTROL_DT
        over = self.target_xy.abs() > self.bounds
        self.target_vel = torch.where(over, -self.target_vel, self.target_vel)
        self.target_xy = self.target_xy.clamp(-self.bounds, self.bounds)

    def _target_obs3(self):
        z = torch.zeros(self.n, 1, device=self.device)
        now3 = torch.cat([self.target_xy, z], -1)
        future = (self.target_xy + self.target_vel * self.lookahead).clamp(
            -self.bounds, self.bounds)
        fut3 = torch.cat([future, z], -1)
        return torch.cat([self._to_ego3(now3), self._to_ego3(fut3)], -1)
