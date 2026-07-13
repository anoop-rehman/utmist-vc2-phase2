"""Batched GPU follow-drill env on MuJoCo Warp.

Observation layout replicates rower_soccer.drills.gym_wrap.DrillGymEnv
exactly (sorted-key order), so weights transfer between CPU and GPU
training and the dm_control env doubles as the transfer/parity eval:

  creature/absolute_root_mat (9), absolute_root_pos (3), bodies_pos (9),
  joints_pos (2), joints_vel (2), sensors_accelerometer (3),
  sensors_gyro (3), sensors_velocimeter (3), touch_sensors (3),
  target_ego (2), target_ego_future (2)                       -> 41 dims

Target kinematics/reward are computed in torch (targets are not physics
entities). Episodes are world-synchronized (global reset every
`episode_steps`), which keeps resets graph-friendly.
"""

import mujoco
import numpy as np
import torch
import warp as wp

import mujoco_warp as mjw

from rower_soccer.warp_port.scene import build_creature_scene

CONTROL_DT = 0.025
SUBSTEPS = 10  # physics dt 0.0025


class WarpFollowEnv:
    def __init__(self, num_worlds=2048, creature_xml="creature_configs/three_seg_worm.xml",
                 episode_seconds=15.0, target_speed_range=(0.04, 0.34),
                 lookahead=1.0, reward_coef=0.5, bounds=10.0, device="cuda",
                 seed=0, use_graph=True, w_vel_shaping=0.0,
                 reward_mode="paper", progress_scale=2.0, settle_coef=0.5,
                 arrival_radius=1.0, arrival_bonus=0.5,
                 spawn_dist_range=(1.76, 5.28)):
        # Calibrated to the 1.76 m / 22 kg worm (unity2mujoco --length-scale
        # 0.1768 --gear-scale 0.03), which is the body that can actually control
        # DeepMind's ball -- see docs/STAGE2_MULTITASK.md 0.5.
        #
        # target_speed_range: Froude-scale C's [0.1, 0.8], NOT the code default
        # [0.25, 2.0]. C ("warp_C_velshape_slowtgt") is the run that actually
        # learned to follow (transfer eval 445-495/600); the [0.25, 2.0] default
        # is the abandoned FAST target that earlier runs failed on -- the
        # "slowtgt" in C's name IS that finding. sqrt(0.1768) = 0.4205, so
        # [0.1, 0.8] -> [0.042, 0.336].
        #
        # What matters is the ratio of target speed to the creature's achievable
        # speed (probe_speed.py: 1.04-1.64 m/s, low end, chaotic toppling):
        #     C           0.8 / 2.83  = 0.28   comfortable margin, learned well
        #     corrected   0.34 / 1.04 = 0.32   matches C
        # A first attempt scaled the [0.25, 2.0] default instead, giving a cap of
        # 0.85 -- 82% of the worm's top speed. It has to catch AND hold the target
        # while turning and correcting, so that leaves no margin at all: the run
        # plateaued at 182/600 against 88/600 for doing nothing.
        #
        # drills/follow.py MUST stay in step: it is the CPU transfer/parity eval,
        # so a mismatch shows up as a phantom sim2sim gap.
        self.n = num_worlds
        self.device = device
        self.episode_steps = int(round(episode_seconds / CONTROL_DT))
        self.speed_range = target_speed_range
        self.lookahead = lookahead
        self.reward_coef = reward_coef
        self.w_vel_shaping = w_vel_shaping
        # reward_mode: "paper" (exp(-c d)), "velshape" (paper + velocity bonus,
        # hackable), "progress" (potential-based: reward closing distance;
        # dense everywhere and unhackable, Ng et al. 1999).
        self.reward_mode = reward_mode
        self.progress_scale = progress_scale
        self.settle_coef = settle_coef
        self.arrival_radius = arrival_radius
        self.arrival_bonus = arrival_bonus
        self.bounds = bounds
        self.spawn_dist_range = spawn_dist_range
        self.prev_dist = torch.zeros(num_worlds, device=device)
        self.gen = torch.Generator(device=device).manual_seed(seed)

        self.model, self.meta = build_creature_scene(creature_xml)
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        self.wm = mjw.put_model(self.model)
        self.wd = mjw.put_data(self.model, data, nworld=num_worlds)

        # torch views (zero-copy)
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

        # target state
        self.target_xy = torch.zeros(self.n, 2, device=device)
        self.target_vel = torch.zeros(self.n, 2, device=device)
        self.t = 0

        self.obs_dim = 41
        self.act_dim = m.nu
        # obs slices for the policy (matches DrillGymEnv layout)
        self.proprio_indices = np.arange(0, 37)
        self.task_indices = np.arange(37, 41)

        self._graph = None
        if use_graph:
            # capture SUBSTEPS physics steps into one CUDA graph
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

    def reset(self):
        m = self.meta
        self.qvel.zero_()
        self.qpos.zero_()
        qr = m.qpos_root
        # root pose: xy 0, spawn z, random yaw
        yaw = torch.rand(self.n, generator=self.gen, device=self.device) * (2 * np.pi)
        self.qpos[:, qr + 0] = 0.0
        self.qpos[:, qr + 1] = 0.0
        self.qpos[:, qr + 2] = m.spawn_z
        self.qpos[:, qr + 3] = torch.cos(yaw / 2)
        self.qpos[:, qr + 6] = torch.sin(yaw / 2)  # yaw about z
        # target init: 1-3 body lengths away, random direction & speed
        ang = torch.rand(self.n, generator=self.gen, device=self.device) * (2 * np.pi)
        d0, d1 = self.spawn_dist_range
        dist = d0 + (d1 - d0) * torch.rand(self.n, generator=self.gen, device=self.device)
        self.target_xy = torch.stack([dist * torch.cos(ang), dist * torch.sin(ang)], -1)
        vang = torch.rand(self.n, generator=self.gen, device=self.device) * (2 * np.pi)
        spd = self.speed_range[0] + (self.speed_range[1] - self.speed_range[0]) * \
            torch.rand(self.n, generator=self.gen, device=self.device)
        self.target_vel = torch.stack([spd * torch.cos(vang), spd * torch.sin(vang)], -1)
        self.t = 0
        self._forward()
        # seed prev_dist so the first progress delta is well-defined
        pos, _ = self._root_frames()
        self.prev_dist = torch.linalg.norm(pos[:, :2] - self.target_xy, dim=-1)
        return self._obs()

    def _forward(self):
        mjw.forward(self.wm, self.wd)
        wp.synchronize_device()

    def step(self, actions):
        """actions: [n, nu] in [-1, 1]. Returns obs, reward, done(all-worlds)."""
        self.ctrl.copy_(actions.clamp(-1.0, 1.0))
        self._physics_step()
        # target kinematics + bounce
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
        pos = self.xpos[:, rb, :]                    # [n,3]
        rot = self.xmat[:, rb]                       # [n,3,3]
        return pos, rot

    def _to_ego(self, world_xy):
        pos, rot = self._root_frames()
        fwd, left = rot[:, :2, 0], rot[:, :2, 1]     # world xy of body x/y axes
        d = world_xy - pos[:, :2]
        return torch.stack([(d * fwd).sum(-1), (d * left).sum(-1)], -1)

    def _reward(self):
        pos, _ = self._root_frames()
        d = self.target_xy - pos[:, :2]
        dist = torch.linalg.norm(d, dim=-1)

        if self.reward_mode == "progress":
            # potential-based: reward for closing the gap this step. Telescopes
            # to total distance reduced, so oscillating nets zero (unhackable);
            # dense at any range (no flat far-field). Plus a small proximity
            # "settle" term and an arrival bonus for staying on target.
            progress = self.prev_dist - dist
            self.prev_dist = dist.detach()
            r = self.progress_scale * progress
            r = r + self.settle_coef * torch.exp(-self.reward_coef * dist)
            r = r + self.arrival_bonus * (dist < self.arrival_radius).float()
            return r

        r = torch.exp(-self.reward_coef * dist)
        if self.reward_mode == "velshape" or self.w_vel_shaping > 0:
            # root velocity toward target (hackable — kept for comparison)
            vel_xy = self.qvel[:, self.meta.qvel_root:self.meta.qvel_root + 2]
            v_to_t = (vel_xy * (d / dist.clamp(min=1e-6).unsqueeze(-1))).sum(-1)
            r = r + self.w_vel_shaping * v_to_t.clamp(min=0.0)
        return r

    def _obs(self):
        n = self.n
        pos, rot = self._root_frames()
        # bodies_pos: each creature body rel root, in root frame
        bp = self.xpos[:, self.body_ids, :] - pos.unsqueeze(1)          # [n,3,3]
        bodies_ego = torch.einsum("nij,nbj->nbi", rot.transpose(1, 2), bp).reshape(n, -1)

        touch = torch.cat([self.sensordata[:, s:s + d] for s, d in self.sl_touch], -1) / 10000.0
        sv, sg, sa = (self.sensordata[:, s:s + d] for s, d in
                      (self.sl_vel, self.sl_gyro, self.sl_accel))

        tgt_now = self._to_ego(self.target_xy)
        future = (self.target_xy + self.target_vel * self.lookahead).clamp(-self.bounds, self.bounds)
        tgt_fut = self._to_ego(future)

        return torch.cat([
            rot.reshape(n, 9),                 # creature/absolute_root_mat
            pos,                               # creature/absolute_root_pos
            bodies_ego,                        # creature/bodies_pos
            self.qpos[:, self.jq],             # creature/joints_pos
            self.qvel[:, self.jv],             # creature/joints_vel
            sa, sg, sv,                        # accelerometer, gyro, velocimeter
            touch,                             # touch_sensors
            tgt_now, tgt_fut,                  # target_ego, target_ego_future
        ], -1)
