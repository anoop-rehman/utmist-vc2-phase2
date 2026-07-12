"""Batched GPU dribble-drill env on MuJoCo Warp.

"The environment is similar to the 'follow' drill but the agent must keep the
ball close to the moving target." (DeepMind 2021, supplementary Table S2/S3.)

Unlike `follow`, the ball is a real physics entity the creature contacts, so the
scene carries a free-joint SoccerBall (see scene.py). The target remains a
kinematic abstraction computed in torch.

Observation layout replicates rower_soccer.drills.dribble via DrillGymEnv
exactly, which flattens the observation dict in SORTED KEY order. "ball_ego"
sorts before "creature/*", so the ball block lands at the FRONT -- dribble's obs
is not follow's obs with four numbers appended:

  ball_ego (4)                                            -> 0:4    [task]
  creature/absolute_root_mat (9), absolute_root_pos (3),
  bodies_pos (9), joints_pos (2), joints_vel (2),
  sensors_accelerometer (3), sensors_gyro (3),
  sensors_velocimeter (3), touch_sensors (3)              -> 4:41   [proprio]
  target_ego (2), target_ego_future (2)                   -> 41:45  [task]
                                                             = 45 dims

The proprio block is byte-identical to follow's, and stays contiguous, so the
decoder (proprio + z) transfers from a follow checkpoint unchanged. The task
indices are non-contiguous, which LatentExtractor handles -- it index_selects.
"""

import mujoco
import numpy as np
import torch
import warp as wp

import mujoco_warp as mjw

from rower_soccer.warp_port.scene import BallSpec, build_creature_ball_scene

CONTROL_DT = 0.025
SUBSTEPS = 10


class WarpDribbleEnv:
    def __init__(self, num_worlds=2048, creature_xml="creature_configs/three_seg_worm.xml",
                 episode_seconds=15.0, target_speed_range=(0.1, 1.0),
                 lookahead=1.0, reward_coef=0.5, bounds=27.0, device="cuda",
                 seed=0, use_graph=True,
                 target_dist_range=(2.0, 6.0), ball_spawn_range=(5.0, 8.0),
                 w_player_to_ball=0.1, w_ball_to_target=0.3,
                 reward_mode="paper", progress_scale=2.0, approach_scale=0.5,
                 ball: BallSpec = None, nconmax=64, njmax=512):
        self.n = num_worlds
        self.device = device
        self.episode_steps = int(round(episode_seconds / CONTROL_DT))
        self.speed_range = target_speed_range
        self.lookahead = lookahead
        self.reward_coef = reward_coef
        self.bounds = bounds
        # The target spawns relative to the BALL, not the creature. This matters
        # more than it looks. dm_control's drill spawns both relative to the
        # walker (ball 1-3 m, target 2-6 m), which for a 1.5 m humanoid leaves
        # them ~4 m apart. Do the same around a 9.95 m worm -- whose own
        # footprint is 4.65 m, so the ball has to start >=5 m out -- and ball and
        # target land ~13 m apart. exp(-0.5*13) = 0.0015: the fitness is flat
        # zero out there, the drill has no gradient at all, and no amount of
        # training fixes it. Anchoring the target to the ball keeps
        # ||ball-target|| in 2-6 m, i.e. the same spread dm_control actually gets.
        self.target_dist_range = target_dist_range
        # ...and the ball spawns just outside the creature's 4.65 m footprint,
        # so it is reachable but not interpenetrating the body at t=0.
        self.ball_spawn_range = ball_spawn_range
        self.w_p2b = w_player_to_ball
        self.w_b2t = w_ball_to_target
        # "paper": Table S3 -- exp(-c*||ball-target||) plus the two velocity
        # shaping terms. Those terms are velocity-based and therefore hackable
        # in the same way follow's `velshape` mode was.
        # "progress": potential-based, and it needs BOTH potentials. Rewarding
        # only the ball->target gap is a dead drill: until the creature touches
        # the ball that term is identically zero, so nothing ever rewards walking
        # over to the ball in the first place. The player->ball potential is what
        # gets it there. Both telescope, so neither can be farmed by oscillating.
        self.reward_mode = reward_mode
        self.progress_scale = progress_scale
        self.approach_scale = approach_scale
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.prev_bt = torch.zeros(num_worlds, device=device)
        self.prev_pb = torch.zeros(num_worlds, device=device)

        self.model, self.meta = build_creature_ball_scene(
            creature_xml, ball=ball or BallSpec())
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        self.wm = mjw.put_model(self.model)
        # Sized explicitly: put_data infers these from the initial MjData, where
        # nothing is in contact, so the buffers come out far too small once the
        # creature lands and the condim-6 ball starts touching things. On
        # overflow mujoco_warp drops constraints and the sim goes to NaN rather
        # than raising.
        self.wd = mjw.put_data(self.model, data, nworld=num_worlds,
                               nconmax=nconmax, njmax=njmax)

        self.qpos = wp.to_torch(self.wd.qpos)
        self.qvel = wp.to_torch(self.wd.qvel)
        self.ctrl = wp.to_torch(self.wd.ctrl)
        self.xpos = wp.to_torch(self.wd.xpos)
        self.xmat = wp.to_torch(self.wd.xmat).reshape(self.n, -1, 3, 3)
        self.sensordata = wp.to_torch(self.wd.sensordata)

        m = self.meta
        self.jq = torch.as_tensor(m.joint_qpos, device=device)
        self.jv = torch.as_tensor(m.joint_qvel, device=device)
        self.body_ids = torch.as_tensor(m.body_ids, device=device)
        ss = m.sensor_slices
        self.sl_touch = [ss[f"seg{i}_touch"] for i in range(3)]
        self.sl_vel = ss["torso_vel"]
        self.sl_gyro = ss["torso_gyro"]
        self.sl_accel = ss["torso_accel"]
        self.bq, self.bv = m.ball_qpos, m.ball_qvel
        self.ball_radius = m.ball_radius

        self.target_xy = torch.zeros(self.n, 2, device=device)
        self.target_vel = torch.zeros(self.n, 2, device=device)
        self.t = 0

        self.obs_dim = 45
        self.act_dim = m.nu
        # Sorted-key order: ball_ego first, then creature/*, then target_*.
        self.proprio_indices = np.arange(4, 41)
        self.task_indices = np.concatenate([np.arange(0, 4), np.arange(41, 45)])

        self._graph = None
        if use_graph:
            with wp.ScopedCapture() as cap:
                for _ in range(SUBSTEPS):
                    mjw.step(self.wm, self.wd)
            self._graph = cap.graph

    # ------------------------------------------------------------------
    def _physics_step(self):
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            for _ in range(SUBSTEPS):
                mjw.step(self.wm, self.wd)
        wp.synchronize_device()

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def reset(self):
        m = self.meta
        n, dev = self.n, self.device
        self.qvel.zero_()
        self.qpos.zero_()
        qr = m.qpos_root
        yaw = self._rand(n) * (2 * np.pi)
        self.qpos[:, qr + 2] = m.spawn_z
        self.qpos[:, qr + 3] = torch.cos(yaw / 2)
        self.qpos[:, qr + 6] = torch.sin(yaw / 2)

        # ball: random direction, just outside the body footprint
        bang = self._rand(n) * (2 * np.pi)
        b0, b1 = self.ball_spawn_range
        bdist = b0 + (b1 - b0) * self._rand(n)
        ball_xy = torch.stack([bdist * torch.cos(bang), bdist * torch.sin(bang)], -1)
        self.qpos[:, self.bq + 0] = ball_xy[:, 0]
        self.qpos[:, self.bq + 1] = ball_xy[:, 1]
        self.qpos[:, self.bq + 2] = self.ball_radius
        self.qpos[:, self.bq + 3] = 1.0  # unit quat

        # target: anchored to the BALL, so ||ball - target|| stays in a range
        # where exp(-c*d) still has a usable gradient (see __init__).
        ang = self._rand(n) * (2 * np.pi)
        d0, d1 = self.target_dist_range
        dist = d0 + (d1 - d0) * self._rand(n)
        self.target_xy = ball_xy + torch.stack(
            [dist * torch.cos(ang), dist * torch.sin(ang)], -1)
        self.target_xy = self.target_xy.clamp(-self.bounds, self.bounds)
        vang = self._rand(n) * (2 * np.pi)
        s0, s1 = self.speed_range
        spd = s0 + (s1 - s0) * self._rand(n)
        self.target_vel = torch.stack([spd * torch.cos(vang), spd * torch.sin(vang)], -1)

        self.t = 0
        self._forward()
        pos, _ = self._root_frames()
        self.prev_bt = torch.linalg.norm(self._ball_xy() - self.target_xy, dim=-1)
        self.prev_pb = torch.linalg.norm(self._ball_xy() - pos[:, :2], dim=-1)
        return self._obs()

    def _forward(self):
        mjw.forward(self.wm, self.wd)
        wp.synchronize_device()

    def step(self, actions):
        """actions: [n, nu] in [-1, 1]. Returns obs, reward, done(all-worlds)."""
        self.ctrl.copy_(actions.clamp(-1.0, 1.0))
        self._physics_step()
        self.target_xy = self.target_xy + self.target_vel * CONTROL_DT
        over = self.target_xy.abs() > self.bounds
        self.target_vel = torch.where(over, -self.target_vel, self.target_vel)
        self.target_xy = self.target_xy.clamp(-self.bounds, self.bounds)

        self.t += 1
        done = self.t >= self.episode_steps
        return self._obs(), self._reward(), done

    # ------------------------------------------------------------------
    def _root_frames(self):
        rb = self.meta.root_body
        return self.xpos[:, rb, :], self.xmat[:, rb]

    def _ball_xy(self):
        return self.qpos[:, self.bq:self.bq + 2]

    def _ball_vel_xy(self):
        return self.qvel[:, self.bv:self.bv + 2]

    def _to_ego(self, world_xy):
        pos, rot = self._root_frames()
        fwd, left = rot[:, :2, 0], rot[:, :2, 1]
        d = world_xy - pos[:, :2]
        return torch.stack([(d * fwd).sum(-1), (d * left).sum(-1)], -1)

    def _vec_to_ego(self, world_vec):
        """Rotate a world-frame vector (not a position) into the root frame."""
        _, rot = self._root_frames()
        fwd, left = rot[:, :2, 0], rot[:, :2, 1]
        return torch.stack([(world_vec * fwd).sum(-1),
                            (world_vec * left).sum(-1)], -1)

    def _reward(self):
        ball_xy, ball_vel = self._ball_xy(), self._ball_vel_xy()
        pos, _ = self._root_frames()
        root_xy = pos[:, :2]
        root_vel = self.qvel[:, self.meta.qvel_root:self.meta.qvel_root + 2]

        d_bt = self.target_xy - ball_xy
        dist_bt = torch.linalg.norm(d_bt, dim=-1)
        d_pb = ball_xy - root_xy
        dist_pb = torch.linalg.norm(d_pb, dim=-1)

        if self.reward_mode == "progress":
            # Two potentials, both telescoping (Ng et al. 1999), so oscillating
            # nets zero on each. The player->ball term is not optional: without
            # it nothing rewards walking to the ball, and until the creature
            # reaches the ball the ball->target term is identically zero -- the
            # drill would have no gradient anywhere and could never start.
            approach = self.prev_pb - dist_pb
            progress = self.prev_bt - dist_bt
            self.prev_pb = dist_pb.detach()
            self.prev_bt = dist_bt.detach()
            return (self.approach_scale * approach
                    + self.progress_scale * progress
                    + torch.exp(-self.reward_coef * dist_bt))

        # paper (Table S3): fitness + two velocity shaping terms
        fitness = torch.exp(-self.reward_coef * dist_bt)
        n_pb = dist_pb.clamp(min=1e-6)
        v_p2b = (root_vel * (d_pb / n_pb.unsqueeze(-1))).sum(-1)
        n_bt = dist_bt.clamp(min=1e-6)
        v_b2t = (ball_vel * (d_bt / n_bt.unsqueeze(-1))).sum(-1)
        return (fitness + self.w_p2b * v_p2b.clamp(min=0.0)
                + self.w_b2t * v_b2t.clamp(min=0.0))

    def fitness(self):
        """Unshaped Table-S3 fitness, for gating (mode-agnostic)."""
        dist_bt = torch.linalg.norm(self.target_xy - self._ball_xy(), dim=-1)
        return torch.exp(-self.reward_coef * dist_bt)

    def _obs(self):
        n = self.n
        pos, rot = self._root_frames()
        bp = self.xpos[:, self.body_ids, :] - pos.unsqueeze(1)
        bodies_ego = torch.einsum("nij,nbj->nbi", rot.transpose(1, 2), bp).reshape(n, -1)

        touch = torch.cat([self.sensordata[:, s:s + d] for s, d in self.sl_touch], -1) / 10000.0
        sv, sg, sa = (self.sensordata[:, s:s + d] for s, d in
                      (self.sl_vel, self.sl_gyro, self.sl_accel))

        ball_ego = torch.cat([self._to_ego(self._ball_xy()),
                              self._vec_to_ego(self._ball_vel_xy())], -1)
        tgt_now = self._to_ego(self.target_xy)
        future = (self.target_xy + self.target_vel * self.lookahead).clamp(
            -self.bounds, self.bounds)
        tgt_fut = self._to_ego(future)

        return torch.cat([
            ball_ego,                          # ball_ego            (sorts first)
            rot.reshape(n, 9),                 # creature/absolute_root_mat
            pos,                               # creature/absolute_root_pos
            bodies_ego,                        # creature/bodies_pos
            self.qpos[:, self.jq],             # creature/joints_pos
            self.qvel[:, self.jv],             # creature/joints_vel
            sa, sg, sv,                        # accelerometer, gyro, velocimeter
            touch,                             # touch_sensors
            tgt_now, tgt_fut,                  # target_ego, target_ego_future
        ], -1)
