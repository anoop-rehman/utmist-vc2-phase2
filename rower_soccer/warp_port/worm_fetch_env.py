"""Fetch (dm_control quadruped-fetch, adapted) for OUR worm, on mujoco_warp.

Two scene modes:
  * arena (default): a worm-scaled version of the fetch arena -- square floor
    of half-size --floor-half (5 m -> 10x10 arena vs the quadruped's 30x30),
    the same inward-tilted walls, ball dropped from z=1 with 1.5*randn kick.
    Scale factor ~ the bodies' speed ratio (worm 0.76 m/s vs quadruped ~5).
  * pitch: the full-size dm_soccer pitch (the 2v2 game world). Spawns stay in
    a +/-spawn_radius region around the centre so the task remains completable;
    the pitch's own walls are the boundary.

REWARD is the Fetch formula with one substitution: the quadruped's
torso_upright (torso z . world z) becomes (labeled_up_local . world z), where
up_local comes from the browser labeling tool (label_up.py) -- the GA-evolved
worm has no canonical belly, so a human picks it.

    reward = upright * reach * (0.5 + 0.5 * fetch)      all linear tolerances

OBS (41) = worm proprio (29, byte-identical layout to follow/dribble -- the
decoder contract, so --init-from follow/dribble checkpoints transfers the
low-level controller) + fetch task obs (ball_state 9 + target_position 3,
computed exactly like the quadruped port).
"""
import json

import mujoco
import numpy as np
import torch
import warp as wp
import mujoco_warp as mjw

from rower_soccer.warp_port.scene import build_creature_scene, BallSpec

CONTROL_DT = 0.025    # the worm stack's control rate (follow/dribble/decoder)
SUBSTEPS = 10         # worm scene timestep 0.0025
EPISODE_SECONDS = 20  # fetch's episode length


def _arena_xml(floor_half):
    """The fetch arena, worm-scaled: plane + 4 inward-tilted walls."""
    fh = floor_half
    wall = fh + 0.7
    return f"""
<mujoco model="worm_fetch_arena">
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


class WarpWormFetchEnv:
    def __init__(self, num_worlds=1024,
                 creature_xml="creature_configs/three_seg_worm.xml",
                 up_axis_json="creature_configs/three_seg_worm_up_axis.json",
                 scene="arena", floor_half=5.0, spawn_frac=0.9,
                 ball_drop_z=1.0, ball_kick_std=1.5,
                 device="cuda", seed=0, use_graph=True,
                 nconmax=64, njmax=512):
        self.n = num_worlds
        self.device = device
        self.episode_steps = int(round(EPISODE_SECONDS / CONTROL_DT))
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.n_diverged = 0
        self.ball_drop_z = ball_drop_z
        self.ball_kick_std = ball_kick_std

        if scene == "arena":
            base = _arena_xml(floor_half)
            self.spawn_radius = spawn_frac * floor_half
            self.arena_radius = floor_half * np.sqrt(2.0)
        elif scene == "pitch":
            base = None  # scene.py's soccer pitch
            self.spawn_radius = spawn_frac * floor_half  # spawn REGION, not pitch
            self.arena_radius = floor_half * np.sqrt(2.0)
        else:
            raise ValueError(scene)
        self.scene = scene

        self.model, self.meta = build_creature_scene(
            creature_xml, ball=fetch_ball(), base_xml=base)
        m = self.meta
        # Soften ALL contacts to the ball's 0.010 timeconst. The upright spawn
        # rests the worm on capsule EDGES, and edge contacts at the follow/
        # dribble 0.005 stiffness NaN a few percent of worlds within steps --
        # the same failure the ball had at 0.005. This env shares no
        # checkpoints with follow/dribble (fetch runs are from scratch), so
        # contact-parity with them buys nothing.
        self.model.geom_solref[:, 0] = 0.010

        with open(up_axis_json) as f:
            lbl = json.load(f)
        self.up_local = torch.tensor(lbl["up_local"], dtype=torch.float32,
                                     device=device)
        # Labeled REST POSE: the labeler saves every hinge angle (radians);
        # the creature spawns in exactly that pose. Unlabeled joints stay 0.
        self._label_joints = {}
        for name, rad in (lbl.get("joints") or {}).items():
            try:
                self._label_joints[int(self.model.joint(name).qposadr[0])] = float(rad)
            except KeyError:
                pass
        # Spawn RIGHT-SIDE-UP, like quadruped fetch spawns its walker upright:
        # the labeled quat IS the upright orientation (random yaw composes on
        # top in reset). Without this the worm spawns in its model-default
        # pose -- 90 deg off the label -- where upright plateaus at ~0.48 and a
        # 2-DOF worm cannot roll itself over to fix it.
        self.spawn_quat = torch.tensor(
            lbl.get("quat_wxyz", [1.0, 0.0, 0.0, 0.0]),
            dtype=torch.float32, device=device)
        self.spawn_z_up = self._noncontact_height(lbl.get("quat_wxyz"))

        # The worm's root (seg0) sits at one END of the body, so a root placed
        # at spawn_radius can bury the OTHER end inside a wall -- the contact
        # solver then fires it into the sky (measured: root z 8.7 m, zero
        # action). The worm's spawn region shrinks by its body reach so the
        # whole body always clears the walls; the ball (r=0.15) keeps the full
        # region. Pitch has no walls near the spawn region, so no shrink.
        data0 = mujoco.MjData(self.model)
        for adr, rad in self._label_joints.items():
            data0.qpos[adr] = rad
        mujoco.mj_forward(self.model, data0)
        root_xy = data0.xpos[self.meta.root_body][:2]
        body_set = set(int(b) for b in self.meta.body_ids)
        # geom_rbound = each geom's bounding-sphere radius, so this is a hard
        # upper bound on how far ANY part of the worm sticks out from the root
        # (body xpos alone sits at joint pivots and undercounts the last tip).
        reach = max(float(np.linalg.norm(data0.geom_xpos[g][:2] - root_xy))
                    + float(self.model.geom_rbound[g])
                    for g in range(self.model.ngeom)
                    if int(self.model.geom_bodyid[g]) in body_set)
        if scene == "arena":
            self.worm_spawn_radius = max(0.5, floor_half - reach - 0.2)
        else:
            self.worm_spawn_radius = self.spawn_radius

        # Fetch reward geometry: reach bound = "at the ball" for the worm's
        # footprint (root within half a body length + ball radius); fetch
        # bound = the target site's radius, same as the quadruped's 0.4.
        self.reach_bound = 0.5 + 0.15
        self.fetch_bound = 0.4

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
        touch_keys = sorted(k for k in ss if k.endswith("_touch"))
        self.sl_touch = [ss[k] for k in touch_keys]
        self.sl_vel, self.sl_gyro, self.sl_accel = (ss["torso_vel"],
                                                    ss["torso_gyro"],
                                                    ss["torso_accel"])
        self.bq, self.bv = m.ball_qpos, m.ball_qvel
        self.rv = m.qvel_root

        # proprio size follows the creature: bodies_ego + height + joints
        # pos/vel + accel/gyro/vel + touch + world_zaxis. Worm: 29. Rower: 65.
        n_proprio = (3 * len(m.body_ids) + 1 + 2 * len(m.joint_qpos)
                     + 9 + len(touch_keys) + 3)
        self.obs_dim = n_proprio + 12
        self.act_dim = m.nu
        self.t = 0
        self.proprio_indices = np.arange(0, n_proprio)
        self.task_indices = np.arange(n_proprio, self.obs_dim)
        # fetch target is FIXED at the arena/pitch centre, like dm_control's.
        self.target_xy = torch.zeros(self.n, 2, device=device)

        self._graph = None
        if use_graph:
            with wp.ScopedCapture() as cap:
                for _ in range(SUBSTEPS):
                    mjw.step(self.wm, self.wd)
            self._graph = cap.graph

    # -- physics plumbing, identical to the other worm envs ---------------
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

    def _noncontact_height(self, quat_wxyz):
        """CPU, once at init: lowest z where the LABELED orientation touches
        nothing (dm_control's _find_non_contacting_height)."""
        m = self.model
        data = mujoco.MjData(m)
        qr, q = self.meta.qpos_root, quat_wxyz or [1.0, 0.0, 0.0, 0.0]
        z = 0.0
        for _ in range(10_000):
            mujoco.mj_resetData(m, data)
            # Park the ball high in the air during the search: a sideways
            # offset lands INSIDE the pitch wall on the soccer pitch, and a
            # ball in permanent wall contact means ncon never reaches 0.
            data.qpos[self.meta.ball_qpos:self.meta.ball_qpos + 3] = 0, 0, 50
            data.qpos[qr + 0:qr + 3] = 0.0, 0.0, z
            data.qpos[qr + 3:qr + 7] = q
            for adr, rad in self._label_joints.items():
                data.qpos[adr] = rad
            mujoco.mj_forward(m, data)
            if data.ncon == 0:
                return z
            z += 0.01
        raise RuntimeError("no non-contacting height for the labeled pose")

    def _spawn_quats(self, yaw):
        """Random world-yaw composed onto the labeled upright quat."""
        cy, sy = torch.cos(yaw / 2), torch.sin(yaw / 2)
        lw, lx, ly, lz = self.spawn_quat
        return (cy * lw - sy * lz, cy * lx - sy * ly,
                cy * ly + sy * lx, cy * lz + sy * lw)

    def reset(self):
        n, m = self.n, self.meta
        self.qpos.zero_()
        self.qvel.zero_()
        qr = m.qpos_root
        yaw = self._rand(n) * (2 * np.pi)
        qw, qx, qy, qz = self._spawn_quats(yaw)
        self.qpos[:, qr + 0] = (self._rand(n) * 2 - 1) * self.worm_spawn_radius
        self.qpos[:, qr + 1] = (self._rand(n) * 2 - 1) * self.worm_spawn_radius
        self.qpos[:, qr + 2] = self.spawn_z_up
        for adr, rad in self._label_joints.items():
            self.qpos[:, adr] = rad
        self.qpos[:, qr + 3] = qw
        self.qpos[:, qr + 4] = qx
        self.qpos[:, qr + 5] = qy
        self.qpos[:, qr + 6] = qz

        self.qpos[:, self.bq + 0] = (self._rand(n) * 2 - 1) * self.spawn_radius
        self.qpos[:, self.bq + 1] = (self._rand(n) * 2 - 1) * self.spawn_radius
        self.qpos[:, self.bq + 2] = self.ball_drop_z
        self.qpos[:, self.bq + 3] = 1.0
        self.qvel[:, self.bv + 0] = self.ball_kick_std * self._randn(n)
        self.qvel[:, self.bv + 1] = self.ball_kick_std * self._randn(n)

        self.t = 0
        self.prev_ctrl = torch.zeros(n, self.act_dim, device=self.device)
        self._forward()
        return self._obs()

    def _sanitize(self):
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
        self.qpos[idx, qr + 2] = self.spawn_z_up
        self.qpos[idx, qr + 3:qr + 7] = self.spawn_quat
        for adr, rad in self._label_joints.items():
            self.qpos[idx, adr] = rad
        self.qpos[idx, self.bq + 0] = 1.5
        self.qpos[idx, self.bq + 1] = 1.5
        self.qpos[idx, self.bq + 2] = 0.15
        self.qpos[idx, self.bq + 3] = 1.0
        self._forward()

    def step(self, actions):
        a = actions.clamp(-1.0, 1.0)
        self.ctrl.copy_(a)
        self._physics_step()
        self._sanitize()
        self.t += 1
        done = self.t >= self.episode_steps
        return self._obs(), self._reward(), done

    # -- reward ------------------------------------------------------------
    def _root_frames(self):
        rb = self.meta.root_body
        return self.xpos[:, rb, :], self.xmat[:, rb]

    @staticmethod
    def _linear_tolerance(d, bound, margin):
        return torch.clamp(1.0 - (d - bound) / margin, max=1.0).clamp(min=0.0)

    def _reward_terms(self):
        pos, rot = self._root_frames()
        up_world_z = torch.einsum("nij,j->ni", rot, self.up_local)[:, 2]
        upright = ((1.0 + up_world_z) / 2.0).clamp(0.0, 1.0)
        ball = self.qpos[:, self.bq:self.bq + 2]
        reach = self._linear_tolerance(
            torch.linalg.norm(ball - pos[:, :2], dim=-1),
            self.reach_bound, self.arena_radius)
        fetch = self._linear_tolerance(
            torch.linalg.norm(ball - self.target_xy, dim=-1),
            self.fetch_bound, self.arena_radius)
        return upright, reach, fetch

    def _reward(self):
        upright, reach, fetch = self._reward_terms()
        return upright * reach * (0.5 + 0.5 * fetch)

    def fitness(self):
        return self._reward()

    # -- obs: worm proprio contract + fetch task block ----------------------
    def _obs(self):
        n = self.n
        pos, rot = self._root_frames()
        bp = self.xpos[:, self.body_ids, :] - pos.unsqueeze(1)
        bodies_ego = torch.einsum("nij,nbj->nbi", rot.transpose(1, 2), bp).reshape(n, -1)
        touch = torch.cat([self.sensordata[:, s:s + d]
                           for s, d in self.sl_touch], -1) / 10000.0
        sv, sg, sa = (self.sensordata[:, s:s + d] for s, d in
                      (self.sl_vel, self.sl_gyro, self.sl_accel))
        sa = (sa / 100.0).clamp(-50.0, 50.0)   # the obs contract's accel scaling
        world_zaxis = rot.reshape(n, 9)[:, 6:9]

        # fetch task obs, exactly like the quadruped port's ball_state /
        # target_position (torso -> root frame).
        ball_rel_pos = self.xpos[:, self.meta.ball_body, :] - pos
        ball_rel_vel = (self.qvel[:, self.bv:self.bv + 3]
                        - self.qvel[:, self.rv:self.rv + 3])
        ball_rot_vel = self.qvel[:, self.bv + 3:self.bv + 6]
        stacked = torch.stack([ball_rel_pos, ball_rel_vel, ball_rot_vel], 1)
        ball_state = torch.einsum("nvj,njk->nvk", stacked, rot).reshape(n, 9)
        tgt3 = torch.cat([self.target_xy,
                          torch.zeros(n, 1, device=self.device)], -1)
        target_pos = torch.einsum("nj,njk->nk", tgt3 - pos, rot)

        return torch.cat([
            bodies_ego,                 # (9)   proprio, dribble-identical order
            pos[:, 2:3],                # (1)
            self.qpos[:, self.jq],      # (2)
            self.qvel[:, self.jv],      # (2)
            sa, sg, sv,                 # (9)
            touch,                      # (3)
            world_zaxis,                # (3)   -- proprio ends at 29
            ball_state,                 # (9)   task
            target_pos,                 # (3)   task
        ], -1)
