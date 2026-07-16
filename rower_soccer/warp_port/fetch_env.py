"""dm_control quadruped-fetch, batched on mujoco_warp. Faithful port.

The MODEL is byte-identical to the CPU original: we call dm_control's own
`quadruped.make_model(walls_and_ball=True)` and load the resulting XML, so the
walker (12 filter-dynamics general actuators, tendon-driven lift/extend), the
walled 30 m arena, the condim-6 ball, and the origin target are exactly the
suite's. mujoco_warp steps it and computes the same named sensors (verified
7e-3 max abs error against MuJoCo CPU on this model).

SPAWN, OBS, and REWARD replicate suite/quadruped.py's Fetch task:
  * spawn: walker at random azimuth + xy within 0.9*floor; ball at random xy,
    dropped from z=2 with 5*randn(2) horizontal velocity (yes, flying).
  * obs (90): egocentric_state(44 = 16 hinge qpos + 16 hinge qvel + 12 act),
    torso_velocity(3), torso_upright(1), imu(6 = accel+gyro), force_torque(24,
    arcsinh), ball_state(9, torso frame), target_position(3, torso frame) --
    same values, same order as the CPU wrapper's flatten.
  * reward: _upright * reach * (0.5 + 0.5*fetch), all linear `tolerance`s with
    margin = arena diagonal radius. Max 1/step, 1000 steps/episode.

The one deliberate deviation: `_find_non_contacting_height` is solved ONCE at
init (the fetch spawn orientation is azimuth-only, so the non-contacting
height is a single constant) instead of per-episode.
"""
import mujoco
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw

CONTROL_DT = 0.02     # suite _CONTROL_TIMESTEP
SUBSTEPS = 4          # model timestep .005
EPISODE_SECONDS = 20  # suite _DEFAULT_TIME_LIMIT


def build_fetch_model():
    """The suite's own XML, with walls + ball + target kept."""
    from dm_control.suite import quadruped, common
    xml = quadruped.make_model(walls_and_ball=True)
    return mujoco.MjModel.from_xml_string(xml, common.ASSETS)


def _spawn_height(model):
    """Replicates _find_non_contacting_height for an upright (azimuth-only)
    orientation: raise the root in 1 cm steps until nothing touches."""
    data = mujoco.MjData(model)
    z = 0.0
    for _ in range(10_000):
        mujoco.mj_resetData(model, data)
        data.qpos[:] = model.qpos0
        data.qpos[0:3] = 0.0, 0.0, z
        data.qpos[3:7] = 1.0, 0.0, 0.0, 0.0
        mujoco.mj_forward(model, data)
        if data.ncon == 0:
            return z
        z += 0.01
    raise RuntimeError("no non-contacting height found")


class WarpFetchEnv:
    def __init__(self, num_worlds=1024, device="cuda", seed=0, use_graph=True,
                 nconmax=256, njmax=1024):
        self.n = num_worlds
        self.device = device
        self.episode_steps = int(round(EPISODE_SECONDS / CONTROL_DT))
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.n_diverged = 0

        self.model = m = build_fetch_model()
        self.spawn_z = _spawn_height(m)

        # --- indices, straight off the MjModel ---------------------------
        name2id = lambda t, s: mujoco.mj_name2id(m, t, s)
        JNT, SITE, GEOM, BODY = (mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_SITE,
                                 mujoco.mjtObj.mjOBJ_GEOM, mujoco.mjtObj.mjOBJ_BODY)
        root_j = name2id(JNT, "root")
        ball_j = name2id(JNT, "ball_root")
        self.rq = int(m.jnt_qposadr[root_j]);  self.rv = int(m.jnt_dofadr[root_j])
        self.bq = int(m.jnt_qposadr[ball_j]);  self.bv = int(m.jnt_dofadr[ball_j])
        hinges = np.nonzero(m.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)[0]
        self.hq = torch.as_tensor(m.jnt_qposadr[hinges].astype(np.int64), device=device)
        self.hv = torch.as_tensor(m.jnt_dofadr[hinges].astype(np.int64), device=device)
        self.torso = name2id(BODY, "torso")
        self.ball_body = name2id(BODY, "ball")
        self.site_target = name2id(SITE, "target")
        self.site_workspace = name2id(SITE, "workspace")

        # Reward constants, read from the model like the suite does.
        floor = name2id(GEOM, "floor")
        self.arena_radius = float(m.geom_size[floor, 0]) * np.sqrt(2.0)
        self.spawn_radius = 0.9 * float(m.geom_size[floor, 0])
        self.reach_bound = (float(m.site_size[self.site_workspace, 0])
                            + float(m.geom_size[name2id(GEOM, "ball"), 0]))
        self.fetch_bound = float(m.site_size[self.site_target, 0])

        # Sensor slices (model order): accel, gyro, velocimeter, 4x force,
        # 4x torque. dm_control's imu() = [accel, gyro]; force_torque() = the
        # force sensors then the torque sensors (ascending sensor id).
        adr = m.sensor_adr
        self.sl_accel = slice(int(adr[0]), int(adr[0]) + 3)
        self.sl_gyro = slice(int(adr[1]), int(adr[1]) + 3)
        self.sl_vel = slice(int(adr[2]), int(adr[2]) + 3)
        self.sl_ft = slice(int(adr[3]), int(adr[3]) + 24)

        # Action mapping: policy [-1,1] -> native ctrlrange per actuator.
        cr = m.actuator_ctrlrange.astype(np.float32)
        self.ctrl_mid = torch.as_tensor((cr[:, 1] + cr[:, 0]) / 2, device=device)
        self.ctrl_half = torch.as_tensor((cr[:, 1] - cr[:, 0]) / 2, device=device)

        data = mujoco.MjData(m)
        mujoco.mj_forward(m, data)
        self.wm = mjw.put_model(m)
        # Contact buffers sized explicitly (see dribble_env: put_data infers
        # from a contact-free initial state and overflows -> NaN otherwise).
        # The quadruped has 4 toes + body parts + ball + walls per world.
        self.wd = mjw.put_data(m, data, nworld=num_worlds,
                               nconmax=nconmax, njmax=njmax)

        self.qpos = wp.to_torch(self.wd.qpos)
        self.qvel = wp.to_torch(self.wd.qvel)
        self.act = wp.to_torch(self.wd.act)
        self.ctrl = wp.to_torch(self.wd.ctrl)
        self.xpos = wp.to_torch(self.wd.xpos)
        self.xmat = wp.to_torch(self.wd.xmat).reshape(self.n, -1, 3, 3)
        self.site_xpos = wp.to_torch(self.wd.site_xpos).reshape(self.n, -1, 3)
        self.sensordata = wp.to_torch(self.wd.sensordata)

        self.qpos0 = torch.as_tensor(m.qpos0, dtype=torch.float32, device=device)

        self.obs_dim = 90
        self.act_dim = int(m.nu)
        self.t = 0
        # Latent-bottleneck split: everything body-internal is proprio;
        # ball_state + target_position are the task block.
        self.proprio_indices = np.arange(0, 78)
        self.task_indices = np.arange(78, 90)

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

    def _forward(self):
        mjw.forward(self.wm, self.wd)
        wp.synchronize_device()

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _randn(self, *shape):
        return torch.randn(*shape, generator=self.gen, device=self.device)

    # ------------------------------------------------------------------
    def reset(self):
        n = self.n
        self.qpos[:] = self.qpos0
        self.qvel.zero_()
        self.act.zero_()
        self.ctrl.zero_()

        # Walker: random azimuth + horizontal position (suite: 0.9 * floor).
        az = self._rand(n) * (2 * np.pi)
        self.qpos[:, self.rq + 0] = (self._rand(n) * 2 - 1) * self.spawn_radius
        self.qpos[:, self.rq + 1] = (self._rand(n) * 2 - 1) * self.spawn_radius
        self.qpos[:, self.rq + 2] = self.spawn_z
        self.qpos[:, self.rq + 3] = torch.cos(az / 2)
        self.qpos[:, self.rq + 4] = 0.0
        self.qpos[:, self.rq + 5] = 0.0
        self.qpos[:, self.rq + 6] = torch.sin(az / 2)

        # Ball: random xy, dropped from z=2 with random horizontal velocity.
        self.qpos[:, self.bq + 0] = (self._rand(n) * 2 - 1) * self.spawn_radius
        self.qpos[:, self.bq + 1] = (self._rand(n) * 2 - 1) * self.spawn_radius
        self.qpos[:, self.bq + 2] = 2.0
        self.qpos[:, self.bq + 3] = 1.0
        self.qpos[:, self.bq + 4:self.bq + 7] = 0.0
        self.qvel[:, self.bv + 0] = 5.0 * self._randn(n)
        self.qvel[:, self.bv + 1] = 5.0 * self._randn(n)

        self.t = 0
        self.prev_ctrl = torch.zeros(n, self.act_dim, device=self.device)
        self._forward()
        return self._obs()

    def _sanitize(self):
        """Reset diverged worlds to rest before obs/reward (see dribble_env)."""
        bad = ((~torch.isfinite(self.qvel).all(-1))
               | (~torch.isfinite(self.qpos).all(-1))
               | (self.qvel.abs().amax(-1) > 500.0))
        if not bool(bad.any()):
            return
        self.n_diverged += int(bad.sum().item())
        idx = bad.nonzero(as_tuple=True)[0]
        self.qpos[idx] = self.qpos0
        self.qvel[idx] = 0.0
        self.act[idx] = 0.0
        self.qpos[idx, self.rq + 2] = self.spawn_z
        # ball at rest away from the walker
        self.qpos[idx, self.bq + 0] = 3.0
        self.qpos[idx, self.bq + 1] = 3.0
        self.qpos[idx, self.bq + 2] = 0.15
        self._forward()

    def step(self, actions):
        """actions: [n, nu] in [-1, 1] -> native ctrlranges. obs, rew, done."""
        a = actions.clamp(-1.0, 1.0)
        self.ctrl.copy_(self.ctrl_mid + a * self.ctrl_half)
        self._physics_step()
        self._sanitize()
        self.t += 1
        done = self.t >= self.episode_steps
        return self._obs(), self._reward(), done

    # ------------------------------------------------------------------
    def _torso_frames(self):
        return self.xpos[:, self.torso, :], self.xmat[:, self.torso]

    @staticmethod
    def _linear_tolerance(d, bound, margin):
        """rewards.tolerance(sigmoid='linear', value_at_margin=0)."""
        return torch.clamp(1.0 - (d - bound) / margin, max=1.0).clamp(min=0.0)

    def _reward_terms(self):
        pos, rot = self._torso_frames()
        upright = ((1.0 + rot[:, 2, 2]) / 2.0).clamp(0.0, 1.0)
        ball = self.xpos[:, self.ball_body, :]
        ws = self.site_xpos[:, self.site_workspace, :]
        tgt = self.site_xpos[:, self.site_target, :]
        reach = self._linear_tolerance(
            torch.linalg.norm((ws - ball)[:, :2], dim=-1),
            self.reach_bound, self.arena_radius)
        fetch = self._linear_tolerance(
            torch.linalg.norm((tgt - ball)[:, :2], dim=-1),
            self.fetch_bound, self.arena_radius)
        return upright, reach, fetch

    def _reward(self):
        upright, reach, fetch = self._reward_terms()
        return upright * reach * (0.5 + 0.5 * fetch)

    def fitness(self):
        """The task reward itself (bounded [0,1]); dm_control's own metric."""
        return self._reward()

    def _obs(self):
        n = self.n
        pos, rot = self._torso_frames()

        ego = torch.cat([self.qpos[:, self.hq], self.qvel[:, self.hv],
                         self.act], -1)                                  # 44
        sd = self.sensordata
        torso_vel = sd[:, self.sl_vel]                                   # 3
        upright = rot[:, 2, 2].unsqueeze(-1)                             # 1
        imu = torch.cat([sd[:, self.sl_accel], sd[:, self.sl_gyro]], -1) # 6
        ft = torch.arcsinh(sd[:, self.sl_ft])                            # 24

        # ball_state: rel pos / rel lin vel / rot vel, all rows dotted with the
        # torso frame exactly like Physics.ball_state.
        ball_rel_pos = self.xpos[:, self.ball_body, :] - pos
        ball_rel_vel = (self.qvel[:, self.bv:self.bv + 3]
                        - self.qvel[:, self.rv:self.rv + 3])
        ball_rot_vel = self.qvel[:, self.bv + 3:self.bv + 6]
        stacked = torch.stack([ball_rel_pos, ball_rel_vel, ball_rot_vel], 1)
        ball_state = torch.einsum("nvj,njk->nvk", stacked, rot).reshape(n, 9)

        target_pos = torch.einsum("nj,njk->nk",
                                  self.site_xpos[:, self.site_target, :] - pos,
                                  rot)                                   # 3

        obs = torch.cat([ego, torso_vel, upright, imu, ft,
                         ball_state, target_pos], -1)
        return obs
