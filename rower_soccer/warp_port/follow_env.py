"""Batched GPU follow-drill env on MuJoCo Warp.

Observation layout replicates rower_soccer.drills.gym_wrap.DrillGymEnv
exactly (sorted-key order), so weights transfer between CPU and GPU
training and the dm_control env doubles as the transfer/parity eval:

  creature/bodies_pos (9), body_height (1), joints_pos (2), joints_vel (2),
  sensors_accelerometer (3), sensors_gyro (3), sensors_velocimeter (3),
  touch_sensors (3), world_zaxis (3)                          -> 0:29 [proprio]
  target_ego (2), target_ego_future (2)                       -> 29:33 [task]
                                                                 = 33 dims

NOTE the ordering is dm_control's sorted-key order, and "bodies_pos" sorts
BEFORE "body_height" ('i' < 'y'). Do not reorder to taste.

joints_pos/joints_vel widths above are for the stock 2-hinge worm (1 qpos +
1 dof per joint); they widen automatically for ball joints (4 qpos + 3 dof
each -- see scene.py's joint_qpos/joint_qvel and __init__ below), so obs_dim
is computed, not hardcoded. A ball-jointed creature therefore no longer
matches DrillGymEnv's fixed layout dim-for-dim -- CPU/GPU weight transfer and
the dm_control parity eval only hold for hinge-jointed bodies.

Proprio carries world_zaxis + body_height, NOT absolute root pos/mat: it is the
shared decoder's whole input contract, so it must hold only what exists in the
2v2 game and is invariant to pitch position/heading. See creature.py's
proprioception property for the full reasoning; the two must stay in lockstep.

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
                 spawn_dist_range=(1.76, 5.28), nconmax=64, njmax=512,
                 energy_coef=0.0, smooth_coef=0.0, rew_clip=(-10.0, 10.0)):
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
        # Regularizers (both default OFF): energy = -c*mean(a^2) discourages brute
        # thrust; smooth = -c*mean((a_t - a_{t-1})^2) is CAPS temporal smoothness,
        # penalising jerk (not speed). See docs. rew_clip bounds a diverged world's
        # reward spike (a blown contact can produce a 5e5 shaping reward) without
        # touching normal rewards, which live in ~[-1, 2].
        self.energy_coef = energy_coef
        self.smooth_coef = smooth_coef
        self.rew_clip = rew_clip
        self.prev_ctrl = torch.zeros(num_worlds, 0, device=device)  # sized after nu known
        self.n_diverged = 0
        self.gen = torch.Generator(device=device).manual_seed(seed)

        self.model, self.meta = build_creature_scene(creature_xml)
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        self.wm = mjw.put_model(self.model)
        # Size the contact/constraint buffers EXPLICITLY. put_data otherwise infers
        # them from the initial MjData, where the creature is still in the air and
        # nothing is touching -- so the buffers come out sized for a scene with no
        # contacts, and any overflow at runtime silently DROPS constraints (the
        # creature sinks or falls through). dribble_env has always done this;
        # follow_env never did. Measured with the trained policy: 145 contacts and
        # 19 efc rows at peak, so these are ~4x headroom.
        self.wd = mjw.put_data(self.model, data, nworld=num_worlds,
                               nconmax=nconmax, njmax=njmax)

        # torch views (zero-copy)
        self.qpos = wp.to_torch(self.wd.qpos)
        self.qvel = wp.to_torch(self.wd.qvel)
        self.ctrl = wp.to_torch(self.wd.ctrl)
        self.xpos = wp.to_torch(self.wd.xpos)
        self.xmat = wp.to_torch(self.wd.xmat).reshape(self.n, -1, 3, 3)
        self.sensordata = wp.to_torch(self.wd.sensordata)

        m = self.meta
        # Flatten (start, n) slices into per-column indices so a ball joint's
        # full 4-number quaternion / 3-number angular velocity land in the
        # observation, not just their first component. For an all-hinge
        # creature this reduces to one index per joint, unchanged from before.
        jq_idx = [i for start, n in m.joint_qpos for i in range(start, start + n)]
        jv_idx = [i for start, n in m.joint_qvel for i in range(start, start + n)]
        self.jq = torch.as_tensor(jq_idx, device=device, dtype=torch.long)
        self.jv = torch.as_tensor(jv_idx, device=device, dtype=torch.long)
        # Ball joints' qpos ("w" of their local quaternion, at slice start) --
        # zeroing qpos leaves these at [0,0,0,0], an invalid non-unit quaternion
        # (a hinge's qpos=0 is a valid rest angle, but a ball joint's is not).
        # mujoco_warp's kinematics normalizes it (q / |q|), so a bare zero quat
        # is 0/0 = NaN the instant forward() runs. reset() and _sanitize() must
        # set these back to the identity quat (w=1, x=y=z=0), same as the root
        # freejoint's quat already gets restored to.
        self.ball_qw_idx = torch.as_tensor(
            [start for start, n in m.joint_qpos if n == 4],
            device=device, dtype=torch.long)
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

        # 33 for the stock 2-hinge worm (bodies_pos 9, body_height 1, joints_pos
        # 2, joints_vel 2, accel/gyro/vel 9, touch 3, world_zaxis 3, target 4).
        # joints_pos/joints_vel widen automatically for ball joints (4 qpos +
        # 3 qvel each instead of 1+1), so obs_dim is derived, not hardcoded.
        n_joints_pos, n_joints_vel = len(self.jq), len(self.jv)
        self.obs_dim = 9 + 1 + n_joints_pos + n_joints_vel + 9 + 3 + 3 + 4
        self.act_dim = m.nu
        self.prev_ctrl = torch.zeros(self.n, m.nu, device=device)
        # obs slices for the policy (matches DrillGymEnv layout); the last 4
        # dims are always target_ego + target_ego_future, regardless of
        # joint dof widening.
        self.proprio_indices = np.arange(0, self.obs_dim - 4)
        self.task_indices = np.arange(self.obs_dim - 4, self.obs_dim)

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
        if self.ball_qw_idx.numel():
            self.qpos[:, self.ball_qw_idx] = 1.0  # ball joints: identity quat
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
        self.prev_ctrl = torch.zeros(self.n, self.act_dim, device=self.device)
        self._forward()
        # seed prev_dist so the first progress delta is well-defined
        pos, _ = self._root_frames()
        self.prev_dist = torch.linalg.norm(pos[:, :2] - self.target_xy, dim=-1)
        return self._obs()

    def _forward(self):
        mjw.forward(self.wm, self.wd)
        wp.synchronize_device()

    def _sanitize(self):
        """Reset any world whose PHYSICS has diverged, in place, BEFORE obs/reward.

        This is the real fix for the recurring NaN death. mujoco_warp occasionally
        blows a contact up; the world's qvel races to millions and then to inf/NaN.
        Catching it here -- at the source -- means the observation and reward
        computed just after are clean by construction, instead of leaking a
        finite-but-insane 25e6 obs / 5e5 reward downstream (which is what actually
        killed dribble_paper_v5/v6/v7). Downstream guards in ppo.collect stay as a
        second layer, but this stops the poison being created.

        qvel is bounded by real dynamics (max ~57 under random torque), so 500 is
        far above anything physical and far below the divergence scale.
        """
        bad = ((~torch.isfinite(self.qvel).all(-1))
               | (~torch.isfinite(self.qpos).all(-1))
               | (self.qvel.abs().amax(-1) > 500.0))
        if not bool(bad.any()):
            return
        self.n_diverged += int(bad.sum().item())
        idx = bad.nonzero(as_tuple=True)[0]
        qr = self.meta.qpos_root
        # In-place writes: qpos/qvel are zero-copy views into the Warp buffers, so
        # index-assignment must stay in place or the view is lost.
        self.qvel[idx] = 0.0
        self.qpos[idx] = 0.0
        self.qpos[idx, qr + 2] = self.meta.spawn_z
        self.qpos[idx, qr + 3] = 1.0  # identity quat (w=1)
        if self.ball_qw_idx.numel():
            self.qpos[idx.unsqueeze(-1), self.ball_qw_idx.unsqueeze(0)] = 1.0
        self._forward()

    def step(self, actions):
        """actions: [n, nu] in [-1, 1]. Returns obs, reward, done(all-worlds)."""
        a = actions.clamp(-1.0, 1.0)
        self.ctrl.copy_(a)
        self._physics_step()
        self._sanitize()
        # target kinematics + bounce
        self.target_xy = self.target_xy + self.target_vel * CONTROL_DT
        over = self.target_xy.abs() > self.bounds
        self.target_vel = torch.where(over, -self.target_vel, self.target_vel)
        self.target_xy = self.target_xy.clamp(-self.bounds, self.bounds)

        self.t += 1
        done = self.t >= self.episode_steps
        rew = self._reward()
        rew = self._regularize(rew, a)
        return self._obs(), rew, done

    def _regularize(self, rew, a):
        """Energy + CAPS temporal-smoothness penalties, then reward clip."""
        if self.energy_coef > 0:
            rew = rew - self.energy_coef * (a ** 2).mean(-1)
        if self.smooth_coef > 0:
            rew = rew - self.smooth_coef * ((a - self.prev_ctrl) ** 2).mean(-1)
        self.prev_ctrl = a
        return rew.clamp(self.rew_clip[0], self.rew_clip[1])

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

    def fitness(self):
        """Unshaped follow fitness, exp(-c * ||player - target||), per world.

        The gate metric, and the one number --vel-shaping cannot inflate: the shaping
        term pays for velocity TOWARD the target, so a policy can farm it while never
        actually arriving. Fitness only rewards being there.
        """
        pos, _ = self._root_frames()
        dist = torch.linalg.norm(self.target_xy - pos[:, :2], dim=-1)
        return torch.exp(-self.reward_coef * dist)

    def _obs(self):
        n = self.n
        pos, rot = self._root_frames()
        # bodies_pos: each creature body rel root, in root frame
        bp = self.xpos[:, self.body_ids, :] - pos.unsqueeze(1)          # [n,3,3]
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


        tgt_now = self._to_ego(self.target_xy)
        future = (self.target_xy + self.target_vel * self.lookahead).clamp(-self.bounds, self.bounds)
        tgt_fut = self._to_ego(future)

        # world z-axis in the body frame == MuJoCo xmat[6:9] (third row), which is
        # exactly what dm_control's WalkerObservables.world_zaxis returns.
        world_zaxis = rot.reshape(n, 9)[:, 6:9]

        return torch.cat([
            bodies_ego,                        # creature/bodies_pos       (9)
            pos[:, 2:3],                       # creature/body_height      (1)
            self.qpos[:, self.jq],             # creature/joints_pos       (n_joints_pos)
            self.qvel[:, self.jv],             # creature/joints_vel       (n_joints_vel)
            sa, sg, sv,                        # accel, gyro, velocimeter  (9)
            touch,                             # creature/touch_sensors    (3)
            world_zaxis,                       # creature/world_zaxis      (3)
            tgt_now, tgt_fut,                  # target_ego, _future       (4)
        ], -1)
