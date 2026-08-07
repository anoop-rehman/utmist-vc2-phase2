"""Fetch (dm_control quadruped-fetch, adapted) for OUR worm, on the
backend-agnostic WormEnv base.

Scene is the worm-scaled arena (square floor of half-size --floor-half, inward-
tilted walls); the pitch option is gone. Ball dropped from z=ball_drop_z with a
ball_kick_std kick. REWARD (FetchReward) = upright * reach * (0.5 + 0.5*fetch),
where upright uses a human-labeled up-axis (label_up.py) since the GA worm has no
canonical belly.

Obs (proprio-first): proprio(29) | ball_state(9) | target_position(3) = 41. The
proprio block is byte-identical to follow/dribble (the decoder contract), so
--init-from follow/dribble checkpoints transfers the low-level controller.
"""
import json

import mujoco
import numpy as np
import torch

# fetch_ball / _arena_xml re-exported so existing imports
# (train_worm_fetch_warp.py) keep working.
from rower_soccer.warp_port.worm_env_base import (WormEnv, FetchReward,  # noqa: F401
                                                  fetch_ball, _arena_xml)

EPISODE_SECONDS = 20  # fetch's episode length


class WarpWormFetchEnv(WormEnv):
    def __init__(self, num_worlds=1024,
                 creature_xml="creature_configs/three_seg_worm.xml",
                 up_axis_json="creature_configs/three_seg_worm_up_axis.json",
                 floor_half=5.0, spawn_frac=0.9, ball_drop_z=1.0,
                 ball_kick_std=1.5, device=None, seed=0, use_graph=True,
                 nconmax=64, njmax=512, episode_seconds=EPISODE_SECONDS,
                 reward=None, use_gpu=True, backend_cls=None):
        self._up_axis_json = up_axis_json
        self._spawn_frac = spawn_frac
        self.ball_drop_z = ball_drop_z
        self.ball_kick_std = ball_kick_std
        self.spawn_radius = spawn_frac * floor_half
        self.arena_radius = floor_half * np.sqrt(2.0)
        # Fetch reward geometry: reach = "at the ball" (half body length + ball
        # radius); fetch bound = target site radius, same as the quadruped's 0.4.
        self.reach_bound = 0.5 + 0.15
        self.fetch_bound = 0.4
        reward = reward or FetchReward(reach_bound=self.reach_bound,
                                       fetch_bound=self.fetch_bound)
        super().__init__(num_worlds=num_worlds, creature_xml=creature_xml,
                         episode_seconds=episode_seconds, use_gpu=use_gpu,
                         device=device, seed=seed, use_graph=use_graph,
                         nconmax=nconmax, njmax=njmax, reward=reward,
                         floor_half=floor_half, backend_cls=backend_cls)

    # -- scene / model hooks ------------------------------------------------
    def _ball_spec(self):
        return fetch_ball()

    def _post_build_model(self, model):
        # Soften ALL contacts to the ball's 0.010 timeconst: the upright spawn
        # rests the worm on capsule EDGES, which NaN at the follow/dribble 0.005.
        model.geom_solref[:, 0] = 0.010

    # -- task hooks ---------------------------------------------------------
    def _task_dim(self):
        return 12  # ball_state(9) + target_position(3)

    def _task_init(self):
        with open(self._up_axis_json) as f:
            lbl = json.load(f)
        self.up_local = torch.tensor(lbl["up_local"], dtype=torch.float32,
                                     device=self.device)
        # Labeled rest pose: spawn in exactly the labeled hinge angles.
        self._label_joints = {}
        for name, rad in (lbl.get("joints") or {}).items():
            try:
                self._label_joints[int(self.model.joint(name).qposadr[0])] = float(rad)
            except KeyError:
                pass
        # The labeled quat IS the upright orientation (random yaw composes on top).
        self.spawn_quat = torch.tensor(lbl.get("quat_wxyz", [1.0, 0.0, 0.0, 0.0]),
                                       dtype=torch.float32, device=self.device)
        self.spawn_z_up = self._noncontact_height(lbl.get("quat_wxyz"))
        self._spawn_z = self.spawn_z_up  # override base default

        # Shrink the worm's spawn region so its whole body clears the walls (its
        # root sits at one END, so a root at spawn_radius can bury the far end).
        data0 = mujoco.MjData(self.model)
        for adr, rad in self._label_joints.items():
            data0.qpos[adr] = rad
        mujoco.mj_forward(self.model, data0)
        root_xy = data0.xpos[self.meta.root_body][:2]
        body_set = set(int(b) for b in self.meta.body_ids)
        reach = max(float(np.linalg.norm(data0.geom_xpos[g][:2] - root_xy))
                    + float(self.model.geom_rbound[g])
                    for g in range(self.model.ngeom)
                    if int(self.model.geom_bodyid[g]) in body_set)
        self.worm_spawn_radius = max(0.5, self._floor_half - reach - 0.2)

        # Fetch target is FIXED at the arena centre, like dm_control's.
        self.target_xy = torch.zeros(self.n, 2, device=self.device)

    def _noncontact_height(self, quat_wxyz):
        """CPU, once at init: lowest z where the LABELED orientation touches
        nothing (dm_control's _find_non_contacting_height)."""
        m = self.model
        data = mujoco.MjData(m)
        qr, q = self.meta.qpos_root, quat_wxyz or [1.0, 0.0, 0.0, 0.0]
        z = 0.0
        for _ in range(10_000):
            mujoco.mj_resetData(m, data)
            # Park the ball high during the search so a wall-contacting ball never
            # keeps ncon > 0.
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

    def _task_obs(self):
        n = self.n
        pos, rot = self._root_frames()
        # ball_state / target_position, exactly the quadruped port (torso frame).
        ball_rel_pos = self.xpos[:, self.ball_body, :] - pos
        ball_rel_vel = (self.qvel[:, self.bv:self.bv + 3]
                        - self.qvel[:, self.rv:self.rv + 3])
        ball_rot_vel = self.qvel[:, self.bv + 3:self.bv + 6]
        stacked = torch.stack([ball_rel_pos, ball_rel_vel, ball_rot_vel], 1)
        ball_state = torch.einsum("nvj,njk->nvk", stacked, rot).reshape(n, 9)
        tgt3 = torch.cat([self.target_xy,
                          torch.zeros(n, 1, device=self.device)], -1)
        target_pos = torch.einsum("nj,njk->nk", tgt3 - pos, rot)
        return torch.cat([ball_state, target_pos], -1)

    def _reset_state(self):
        n = self.n
        yaw = self._rand(n) * (2 * np.pi)
        qw, qx, qy, qz = self._spawn_quats(yaw)
        xy = torch.stack([(self._rand(n) * 2 - 1) * self.worm_spawn_radius,
                          (self._rand(n) * 2 - 1) * self.worm_spawn_radius], -1)
        self._spawn_root(xy=xy, quat=(qw, qx, qy, qz))
        for adr, rad in self._label_joints.items():
            self.qpos[:, adr] = rad
        # Ball: random xy within spawn_radius, dropped from ball_drop_z with kick.
        self.qpos[:, self.bq + 0] = (self._rand(n) * 2 - 1) * self.spawn_radius
        self.qpos[:, self.bq + 1] = (self._rand(n) * 2 - 1) * self.spawn_radius
        self.qpos[:, self.bq + 2] = self.ball_drop_z
        self.qpos[:, self.bq + 3] = 1.0
        self.qvel[:, self.bv + 0] = self.ball_kick_std * self._randn(n)
        self.qvel[:, self.bv + 1] = self.ball_kick_std * self._randn(n)

    def _sanitize_task(self, idx):
        qr = self.meta.qpos_root
        self.qpos[idx, qr + 3:qr + 7] = self.spawn_quat
        for adr, rad in self._label_joints.items():
            self.qpos[idx, adr] = rad
        self.qpos[idx, self.bq + 0] = 1.5
        self.qpos[idx, self.bq + 1] = 1.5
        self.qpos[idx, self.bq + 2] = 0.15
        self.qpos[idx, self.bq + 3] = 1.0
