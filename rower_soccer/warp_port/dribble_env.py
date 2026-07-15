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

  ball_ego (6) = ego pos (3) + ego linear vel (3)         -> 0:6    [task]
  creature/bodies_pos (9), body_height (1), joints_pos (2),
  joints_vel (2), sensors_accelerometer (3), sensors_gyro (3),
  sensors_velocimeter (3), touch_sensors (3),
  world_zaxis (3)                                         -> 6:35   [proprio]
  target_ego (2), target_ego_future (2)                   -> 35:39  [task]
                                                             = 39 dims

ball_ego is 3-D, matching the 2v2 game's ball_ego_position (3) +
ball_ego_linear_velocity (3). It has to be: the ball obs survives distillation
into the dribble prior, and the prior is evaluated on GAME observations.

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
                 episode_seconds=15.0, target_speed_range=(0.04, 0.25),
                 lookahead=1.0, reward_coef=0.5, bounds=10.0, device="cuda",
                 seed=0, use_graph=True,
                 target_dist_range=(2.0, 5.0), ball_spawn_range=(1.5, 3.0),
                 w_player_to_ball=0.1, w_ball_to_target=0.3,
                 reward_mode="paper", progress_scale=2.0, approach_scale=0.5,
                 ball: BallSpec = None, nconmax=64, njmax=512,
                 energy_coef=0.0, smooth_coef=0.0, rew_clip=(-10.0, 10.0)):
        self.n = num_worlds
        self.device = device
        self.episode_steps = int(round(episode_seconds / CONTROL_DT))
        self.speed_range = target_speed_range
        self.lookahead = lookahead
        self.reward_coef = reward_coef
        self.bounds = bounds
        # The target spawns relative to the BALL, not the creature. On the old
        # 9.95 m worm this was forced: its footprint was 4.65 m, so the ball had
        # to start >=5 m out, and anchoring the target to the worm as well left
        # ball and target ~13 m apart -- exp(-0.5*13) = 0.0015, a flat-zero
        # fitness with no gradient anywhere. The 1.76 m worm's footprint is only
        # 0.82 m so the pressure is off, but anchoring to the ball is still the
        # right call: it bounds ||ball-target|| directly instead of letting it
        # fall out of two independent draws.
        self.target_dist_range = target_dist_range
        # Ball spawn now matches dm_control's own 1-3 m, which the 0.82 m
        # footprint finally permits.
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
        # Regularizers (default OFF) + reward clip + divergence counter. See
        # follow_env for the rationale; identical mechanism here.
        self.energy_coef = energy_coef
        self.smooth_coef = smooth_coef
        self.rew_clip = rew_clip
        self.n_diverged = 0

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

        self.obs_dim = 39
        self.act_dim = m.nu
        self.prev_ctrl = torch.zeros(self.n, m.nu, device=device)
        # Sorted-key order: ball_ego first, then creature/*, then target_*.
        self.proprio_indices = np.arange(6, 35)
        self.task_indices = np.concatenate([np.arange(0, 6), np.arange(35, 39)])

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
        self.prev_ctrl = torch.zeros(self.n, self.act_dim, device=self.device)
        self._forward()
        pos, _ = self._root_frames()
        self.prev_bt = torch.linalg.norm(self._ball_xy() - self.target_xy, dim=-1)
        self.prev_pb = torch.linalg.norm(self._ball_xy() - pos[:, :2], dim=-1)
        return self._obs()

    def _forward(self):
        mjw.forward(self.wm, self.wd)
        wp.synchronize_device()

    def _sanitize(self):
        """Reset diverged worlds (creature AND ball) to rest, BEFORE obs/reward.
        See follow_env._sanitize for the full rationale -- this is the upstream fix
        for the dribble NaN deaths. The ball is the usual culprit here, so it is
        reset to a rest position outside the worm footprint."""
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
        self.qpos[idx, qr + 2] = self.meta.spawn_z
        self.qpos[idx, qr + 3] = 1.0
        # ball at rest, 1 m out along +x so it is not spawned inside the worm
        self.qpos[idx, self.bq + 0] = 1.0
        self.qpos[idx, self.bq + 2] = self.ball_radius
        self.qpos[idx, self.bq + 3] = 1.0
        self._forward()

    def step(self, actions):
        """actions: [n, nu] in [-1, 1]. Returns obs, reward, done(all-worlds)."""
        a = actions.clamp(-1.0, 1.0)
        self.ctrl.copy_(a)
        self._physics_step()
        self._sanitize()
        self.target_xy = self.target_xy + self.target_vel * CONTROL_DT
        over = self.target_xy.abs() > self.bounds
        self.target_vel = torch.where(over, -self.target_vel, self.target_vel)
        self.target_xy = self.target_xy.clamp(-self.bounds, self.bounds)

        self.t += 1
        done = self.t >= self.episode_steps
        rew = self._reward()
        if self.energy_coef > 0:
            rew = rew - self.energy_coef * (a ** 2).mean(-1)
        if self.smooth_coef > 0:
            rew = rew - self.smooth_coef * ((a - self.prev_ctrl) ** 2).mean(-1)
        self.prev_ctrl = a
        rew = rew.clamp(self.rew_clip[0], self.rew_clip[1])
        return self._obs(), rew, done

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

    # 3-D variants -- the ball obs uses these. See drills/dribble.py: the ball
    # observation survives distillation into the drill prior, and the prior is
    # evaluated on the game's 3-D ball_ego_position / ball_ego_linear_velocity.
    def _ball_xyz(self):
        return self.qpos[:, self.bq:self.bq + 3]

    def _ball_vel_xyz(self):
        return self.qvel[:, self.bv:self.bv + 3]

    def _to_ego3(self, world_xyz):
        pos, rot = self._root_frames()
        return torch.einsum("nij,nj->ni", rot.transpose(1, 2), world_xyz - pos)

    def _vec_to_ego3(self, world_vec):
        _, rot = self._root_frames()
        return torch.einsum("nij,nj->ni", rot.transpose(1, 2), world_vec)

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

        # Accelerometer scaled and clipped. It is the ONLY unbounded thing in the
        # observation: contact impacts spike it to ~5,700 m/s^2 while every other input
        # sits near 1 (bodies_pos 1.1, world_zaxis 1.0, joints_pos 2.1). That is not a
        # divergence, it is real physics -- and it is ruinous twice over:
        #
        #   1. It dominates the first layer, so the network is effectively conditioned
        #      on impact spikes.
        #   2. It killed dribble_paper_v7 at 149M steps. A ~6,000 input drives a huge
        #      action mean, which makes logp extreme, which makes the PPO ratio
        #      exp(logp - logp_old) overflow to inf, which makes the gradient NaN --
        #      and clip_grad_norm_ then scales EVERY gradient by NaN, because it clips
        #      rather than sanitises. The obs were finite the whole way down.
        #
        # touch already had exactly this treatment (/10000); accel never did. /100 puts
        # a hard impact at ~57, and the clamp bounds the tail. Any future body must
        # apply the same scaling at deployment -- it is part of the obs contract.
        sa = (sa / 100.0).clamp(-50.0, 50.0)


        ball_ego = torch.cat([self._to_ego3(self._ball_xyz()),
                              self._vec_to_ego3(self._ball_vel_xyz())], -1)
        tgt_now = self._to_ego(self.target_xy)
        future = (self.target_xy + self.target_vel * self.lookahead).clamp(
            -self.bounds, self.bounds)
        tgt_fut = self._to_ego(future)

        # world z-axis in the body frame == MuJoCo xmat[6:9]; see follow_env.
        world_zaxis = rot.reshape(n, 9)[:, 6:9]

        return torch.cat([
            ball_ego,                          # ball_ego  (sorts first)   (4)
            bodies_ego,                        # creature/bodies_pos       (9)
            pos[:, 2:3],                       # creature/body_height      (1)
            self.qpos[:, self.jq],             # creature/joints_pos       (2)
            self.qvel[:, self.jv],             # creature/joints_vel       (2)
            sa, sg, sv,                        # accel, gyro, velocimeter  (9)
            touch,                             # creature/touch_sensors    (3)
            world_zaxis,                       # creature/world_zaxis      (3)
            tgt_now, tgt_fut,                  # target_ego, _future       (4)
        ], -1)
