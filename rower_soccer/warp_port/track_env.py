"""Batched GPU motion-tracking env on MuJoCo Warp -- the NPMP stage.

This is the mocap-tracking step of Liu et al. 2022 (and Merel et al. 2019's
NPMP), with the evolved Karl-Sims gait standing in for motion capture. The
policy learns to reproduce the reference gait under OUR physics and OUR
observation contract; the decoder it leaves behind is the reusable low-level
controller that carries evolution's motor style into follow/dribble/kick.

The information split is the whole point, and it is enforced by the obs layout
rather than by the network:

    proprio           -> [0, P)   decoder sees ONLY this (+ z)
    reference frames  -> [P, P+T) expert/encoder sees this too

The expert may look at where the gait is going; the decoder may not. That is
what makes the decoder reusable by a task policy that has no reference at all.
ActorCritic (warp_port/ppo.py) reads `proprio_indices`/`task_indices` off this
env and wires LatentExtractor accordingly, so the split costs nothing here.

WE TRACK KINEMATICS, NOT THE RECORDED TORQUES. The npz's torques came from
Unity-style velocity servos at 2x blueprint scale and are unit-incompatible
with this rower's torque actuators. Joint angles are dimensionless, so they
transfer across the scale change untouched -- which is also how NPMP does it
(mocap gives kinematics; the tracking policy learns its own actions).

Proprio layout matches follow_env exactly (dm_control sorted-key order), so
weights and intuitions carry across drills. Sizes are creature-generic:

  bodies_pos (3*nbody), body_height (1), joints_pos (nu), joints_vel (nu),
  accelerometer (3), gyro (3), velocimeter (3), touch (nbody),
  world_zaxis (3)                                              -> proprio
  reference joint angles at +1/+5/+10 ctrl steps (3*nu),
  sin/cos of gait phase (2)                                    -> task

Episodes are world-synchronized (global reset every `episode_steps`), as in
follow_env: it keeps resets out of the CUDA graph.
"""

import os

import mujoco
import numpy as np
import torch
import warp as wp

import mujoco_warp as mjw

from rower_soccer.warp_port.scene import build_creature_scene, touch_slices

CONTROL_DT = 0.025          # 40 Hz, matching rower_ref.py's CTRL_HZ
SUBSTEPS = 10               # physics dt 0.0025
LOOKAHEAD_STEPS = (1, 5, 10)   # +25 ms, +125 ms, +250 ms
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_REF = os.path.join(REPO, "runs_v2", "rower_ref_gait.npz")
DEFAULT_XML = os.path.join(REPO, "creature_configs", "two_arm_rower_scaled.xml")
SCENE_PREFIX = "c-"          # build_creature_scene namespaces creature elements


class WarpTrackEnv:
    def __init__(self, num_worlds=1024, creature_xml=DEFAULT_XML, ref_path=DEFAULT_REF,
                 episode_seconds=10.0, device="cuda", seed=0, use_graph=True,
                 track_coef=2.0, upright_coef=1.0, energy_coef=0.0, smooth_coef=0.0,
                 rew_clip=(-10.0, 10.0), nconmax=256, njmax=1024, rsi=True):
        self.n = num_worlds
        self.device = device
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.episode_steps = int(round(episode_seconds / CONTROL_DT))
        self.track_coef = track_coef
        self.upright_coef = upright_coef
        self.energy_coef = energy_coef
        self.smooth_coef = smooth_coef
        self.rew_clip = rew_clip
        self.rsi = rsi
        self.n_diverged = 0

        # ---- reference -------------------------------------------------
        z = np.load(ref_path, allow_pickle=True)
        ref = np.asarray(z["ref_qpos"], dtype=np.float64)        # [K, nu] radians
        ref_names = [str(s) for s in z["joint_names"]]
        self.ref_hz = float(z["ctrl_hz"])
        self.freq_tgt = float(z["freq_tgt"])
        if abs(self.ref_hz - 1.0 / CONTROL_DT) > 1e-6:
            raise ValueError(
                f"reference is at {self.ref_hz} Hz but the env runs at "
                f"{1/CONTROL_DT} Hz; rebuild with "
                f"`rower_ref.py build --ctrl-hz {1/CONTROL_DT:g}`")

        self.model, self.meta = build_creature_scene(creature_xml, prefix=SCENE_PREFIX)
        m = self.meta

        # Reference column i must land on the joint of the SAME NAME. The npz and
        # the XML happen to agree today; asserting it means a renamed or reordered
        # joint fails loudly here instead of training a policy against a scrambled
        # reference that still looks like a plausible gait.
        # build_creature_scene namespaces the creature's joints with `prefix`.
        hinge = []
        for j in range(self.model.njnt):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)
            if n and self.model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
                hinge.append(n[len(SCENE_PREFIX):] if n.startswith(SCENE_PREFIX) else n)
        if hinge != ref_names:
            raise ValueError(f"reference joints {ref_names} != scene hinges {hinge}")

        self.K = ref.shape[0]
        self.ref = torch.as_tensor(ref, dtype=torch.float32, device=device)
        # periodic finite difference -> reference joint velocities, for RSI
        dref = (np.roll(ref, -1, 0) - np.roll(ref, 1, 0)) / (2.0 / self.ref_hz)
        self.ref_vel = torch.as_tensor(dref, dtype=torch.float32, device=device)
        # per-joint tracking weights (phase-average coherence; see rower_ref.py)
        w = np.asarray(z["track_weight"], dtype=np.float32) if "track_weight" in z \
            else np.ones(ref.shape[1], np.float32)
        self.track_w = torch.as_tensor(w, device=device).clamp(min=1e-3)
        self.track_w = self.track_w / self.track_w.sum()

        # Spawn height per reference phase. The default spawn_z is computed for the
        # rest pose; a reference pose can hang a limb lower, and starting a world
        # interpenetrating the floor is exactly how a contact blows up on step 1.
        self.spawn_z = torch.as_tensor(
            self._spawn_heights(ref), dtype=torch.float32, device=device)

        # ---- warp model/data -------------------------------------------
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        self.wm = mjw.put_model(self.model)
        self.wd = mjw.put_data(self.model, data, nworld=num_worlds,
                               nconmax=nconmax, njmax=njmax)

        self.qpos = wp.to_torch(self.wd.qpos)
        self.qvel = wp.to_torch(self.wd.qvel)
        self.ctrl = wp.to_torch(self.wd.ctrl)
        self.xpos = wp.to_torch(self.wd.xpos)
        self.xmat = wp.to_torch(self.wd.xmat).reshape(self.n, -1, 3, 3)
        self.sensordata = wp.to_torch(self.wd.sensordata)

        self.jq = torch.as_tensor(m.joint_qpos, device=device)
        self.jv = torch.as_tensor(m.joint_qvel, device=device)
        self.body_ids = torch.as_tensor(m.body_ids, device=device)
        ss = m.sensor_slices
        self.sl_touch = touch_slices(m)
        self.sl_vel, self.sl_gyro, self.sl_accel = (
            ss["torso_vel"], ss["torso_gyro"], ss["torso_accel"])

        nb, nu = len(m.body_ids), m.nu
        self.act_dim = nu
        p_dim = 3 * nb + 1 + nu + nu + 9 + len(self.sl_touch) + 3
        t_dim = len(LOOKAHEAD_STEPS) * nu + 2
        self.obs_dim = p_dim + t_dim
        self.proprio_indices = np.arange(0, p_dim)
        self.task_indices = np.arange(p_dim, p_dim + t_dim)

        self.phase = torch.zeros(self.n, dtype=torch.long, device=device)
        self.prev_ctrl = torch.zeros(self.n, nu, device=device)
        self.t = 0
        # Running joint-error accumulator. Sampling joint_err() at an arbitrary
        # moment is misleading: RSI makes the error exactly 0 at every episode
        # start, so an instantaneous read taken just after a reset logs a
        # perfect score the policy did not earn.
        self._jerr_sum = 0.0
        self._jerr_n = 0

        self._graph = None
        if use_graph:
            with wp.ScopedCapture() as cap:
                for _ in range(SUBSTEPS):
                    mjw.step(self.wm, self.wd)
            self._graph = cap.graph

    # ------------------------------------------------------------------
    def _spawn_heights(self, ref):
        """Root z per reference phase that puts the lowest geom just above ground."""
        model = self.model
        d = mujoco.MjData(model)
        qr = self.meta.qpos_root
        out = np.zeros(len(ref))
        for k in range(len(ref)):
            d.qpos[:] = 0.0
            d.qpos[qr + 2] = 0.0
            d.qpos[qr + 3] = 1.0
            d.qpos[self.meta.joint_qpos] = ref[k]
            mujoco.mj_forward(model, d)
            # lowest point of any creature geom, allowing for its radius
            zs = []
            for g in range(model.ngeom):
                if model.geom_bodyid[g] in self.meta.body_ids:
                    zs.append(d.geom_xpos[g, 2] - float(model.geom_rbound[g]))
            out[k] = (-min(zs) if zs else self.meta.spawn_z) + 0.02
        return out

    def _physics_step(self):
        if self._graph is not None:
            wp.capture_launch(self._graph)
        else:
            for _ in range(SUBSTEPS):
                mjw.step(self.wm, self.wd)
        wp.synchronize_device()

    def _forward(self):
        mjw.forward(self.wm, self.wd)
        wp.synchronize_device()

    def reset(self):
        m = self.meta
        self.qvel.zero_()
        self.qpos.zero_()
        qr = m.qpos_root
        # Reference state initialisation: start each world at a random point in
        # the gait, already in that pose. Without it every episode begins from
        # rest at phase 0 and the policy only ever sees the stroke's opening,
        # which is the classic tracking failure mode.
        if self.rsi:
            ph = torch.randint(0, self.K, (self.n,), generator=self.gen, device=self.device)
        else:
            ph = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.phase = ph

        yaw = torch.rand(self.n, generator=self.gen, device=self.device) * (2 * np.pi)
        self.qpos[:, qr + 0] = 0.0
        self.qpos[:, qr + 1] = 0.0
        self.qpos[:, qr + 2] = self.spawn_z[ph]
        self.qpos[:, qr + 3] = torch.cos(yaw / 2)
        self.qpos[:, qr + 6] = torch.sin(yaw / 2)
        self.qpos[:, self.jq] = self.ref[ph]
        if self.rsi:
            self.qvel[:, self.jv] = self.ref_vel[ph]

        self.t = 0
        self.prev_ctrl = torch.zeros(self.n, self.act_dim, device=self.device)
        self._forward()
        return self._obs()

    def _sanitize(self):
        """Reset any world whose physics diverged, in place, BEFORE obs/reward.

        Same guarantee as follow_env._sanitize: mujoco_warp occasionally blows a
        contact up and the world's qvel races to inf. Catching it at the source
        keeps obs and reward clean by construction. Diverged worlds restart at
        their current phase so the reference stays aligned.
        """
        bad = ((~torch.isfinite(self.qvel).all(-1))
               | (~torch.isfinite(self.qpos).all(-1))
               | (self.qvel.abs().amax(-1) > 500.0))
        if not bool(bad.any()):
            return
        self.n_diverged += int(bad.sum().item())
        idx = bad.nonzero(as_tuple=True)[0]
        qr = self.meta.qpos_root
        ph = self.phase[idx]
        self.qvel[idx] = 0.0
        self.qpos[idx] = 0.0
        self.qpos[idx, qr + 2] = self.spawn_z[ph]
        self.qpos[idx, qr + 3] = 1.0
        self.qpos[idx.unsqueeze(1), self.jq.unsqueeze(0)] = self.ref[ph]
        self._forward()

    def step(self, actions):
        a = actions.clamp(-1.0, 1.0)
        self.ctrl.copy_(a)
        self._physics_step()
        self._sanitize()
        self.phase = (self.phase + 1) % self.K
        self.t += 1
        self._jerr_sum += float(self.joint_err().mean())
        self._jerr_n += 1
        done = self.t >= self.episode_steps
        rew = self._reward()
        if self.energy_coef > 0:
            rew = rew - self.energy_coef * (a ** 2).mean(-1)
        if self.smooth_coef > 0:
            rew = rew - self.smooth_coef * ((a - self.prev_ctrl) ** 2).mean(-1)
        self.prev_ctrl = a
        return self._obs(), rew.clamp(*self.rew_clip), done

    # ------------------------------------------------------------------
    def _upright(self):
        """+1 when the body z-axis points up, 0 when horizontal or inverted."""
        return self.xmat[:, self.meta.root_body][:, 2, 2].clamp(min=0.0)

    def joint_err(self):
        """Weighted RMS joint-tracking error in radians (the gate metric)."""
        err = self.qpos[:, self.jq] - self.ref[self.phase]
        return torch.sqrt((self.track_w * err ** 2).sum(-1))

    def _reward(self):
        """exp(-c * weighted joint error) * uprightness.

        A product, not a sum, exactly as the paper's tracking rewards compose:
        a policy that lies on its side and waves the arms correctly should score
        near zero, not half marks. Joint error is weighted by each joint's
        phase-average coherence, so joints whose reference is not reproducible
        cannot dominate the gradient.
        """
        r_track = torch.exp(-self.track_coef * self.joint_err())
        return r_track * self._upright() ** self.upright_coef

    def mean_joint_err(self):
        """Mean joint-tracking error since the last call, in radians.

        Averaged over every step taken, then reset -- the honest number to log.
        An instantaneous joint_err() is 0 immediately after an RSI reset.
        """
        if self._jerr_n == 0:
            return float("nan")
        out = self._jerr_sum / self._jerr_n
        self._jerr_sum, self._jerr_n = 0.0, 0
        return out

    def fitness(self):
        """Unshaped tracking fitness per world -- the gate metric."""
        return torch.exp(-self.track_coef * self.joint_err()) * self._upright()

    def _obs(self):
        n = self.n
        rb = self.meta.root_body
        pos, rot = self.xpos[:, rb, :], self.xmat[:, rb]
        bp = self.xpos[:, self.body_ids, :] - pos.unsqueeze(1)
        bodies_ego = torch.einsum("nij,nbj->nbi", rot.transpose(1, 2), bp).reshape(n, -1)

        touch = torch.cat([self.sensordata[:, s:s + d] for s, d in self.sl_touch],
                          -1) / 10000.0
        sv, sg, sa = (self.sensordata[:, s:s + d] for s, d in
                      (self.sl_vel, self.sl_gyro, self.sl_accel))
        # Accelerometer is the only unbounded input; same /100 + clamp as
        # follow_env. This is part of the obs contract, not a tuning knob.
        sa = (sa / 100.0).clamp(-50.0, 50.0)
        world_zaxis = rot.reshape(n, 9)[:, 6:9]

        # task: where the gait is going, plus where it is in the cycle
        future = [self.ref[(self.phase + k) % self.K] for k in LOOKAHEAD_STEPS]
        ang = 2 * np.pi * self.phase.float() / self.K
        phase_feat = torch.stack([torch.sin(ang), torch.cos(ang)], -1)

        return torch.cat([
            bodies_ego, pos[:, 2:3],
            self.qpos[:, self.jq], self.qvel[:, self.jv],
            sa, sg, sv, touch, world_zaxis,
            *future, phase_feat,
        ], -1)
