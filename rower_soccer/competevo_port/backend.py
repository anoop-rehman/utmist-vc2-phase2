"""Backends for the CompetEvo port: warp_port's, plus the fields their env reads.

`rower_soccer.warp_port.backend` exposes qpos/qvel/ctrl/xpos/xmat/sensordata --
everything the worm drills need. CompetEvo's reward and goal test are built on
`data.subtree_com[torso]` (their `Agent.get_body_com`, agent.py:188-191), which is
the whole ant's centre of mass, not the torso body frame; and their contact cost
reads `data.cfrc_ext`. Both are additional Data fields, so this module subclasses
the two backends to alias them rather than editing warp_port (owned by other work
in flight).

On `cfrc_ext`: MuJoCo only fills it from `mj_rnePostConstraint`, which runs only
for acceleration-stage sensors. Their scene declares none, so cfrc_ext is
IDENTICALLY ZERO through their whole training run -- measured on both mujoco 2.3.5
(their venv) and 3.11 (ours), with contacts active. Their ant's contact cost is
therefore a constant 0. We alias the field anyway so the env can assert that
rather than assume it.
"""

import copy

import mujoco
import torch

from rower_soccer.warp_port.backend import CpuBackend, WarpBackend


class CompeteWarpBackend(WarpBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subtree_com = self._wp.to_torch(self.wd.subtree_com)
        self.cfrc_ext = self._wp.to_torch(self.wd.cfrc_ext)


class CompeteWarpDevBackend(CompeteWarpBackend):
    """Warp backend whose MODEL is batched too, so each world can be a different
    ant (stage 2's per-world morphology).

    mujoco_warp's `Model` arrays default to a leading dimension of 1, shared
    across worlds; every kernel reads them as `field[worldid % field.shape[0]]`.
    `put_model(mjm, batch_sizes={...})` allocates the listed fields with leading
    dimension `nworld` instead, tiling the compiled value into every row. That is
    the only supported way to get a per-world axis: the shape is baked into the
    kernels at build time (`collision_driver.py:307` specializes on
    `wp.static(ngeom_rbound > 1)`), so it must be chosen BEFORE the model is put
    on device and cannot be reshaped afterwards.

    Writes into those arrays are ordinary in-place tensor writes and do NOT
    invalidate a captured CUDA graph -- the graph replays kernels that read the
    same device pointers. Only a shape change would, and shapes are fixed here.
    Verified in `tests/test_design_parity.py::graph survives a design write`.
    """

    def __init__(self, model, num_worlds, substeps, *, batched_fields=(),
                 **kwargs):
        self._batched_fields = tuple(batched_fields)
        self._num_worlds = num_worlds
        super().__init__(model, num_worlds, substeps, **kwargs)
        self.model_arrays = {f: self._wp.to_torch(getattr(self.wm, f))
                             for f in self._batched_fields}

    def _put_model(self, model):
        return self._mjw.put_model(
            model, batch_sizes={f: self._num_worlds
                                for f in self._batched_fields})


class CompeteCpuBackend(CpuBackend):
    """CPU MuJoCo, one MjData per world. Used by the parity gate (which needs
    mujoco, not mujoco_warp, as the reference for our own math) and by tests on
    machines with no GPU."""

    def __init__(self, model, num_worlds, substeps, dtype=torch.float64, **kwargs):
        super().__init__(model, num_worlds, substeps, **kwargs)
        # float64 by default, unlike the Warp path. The parity gate compares
        # against their float64 env at 1e-6, and the forward-progress reward
        # divides a COM difference by dt=0.015, multiplying any representation
        # error by 67x -- fp32 state would put the reward comparison at ~1e-5
        # and the gate would be measuring our own rounding.
        self.dtype = dtype
        for name in ("qpos", "qvel", "ctrl", "xpos", "xmat", "sensordata"):
            setattr(self, name, getattr(self, name).to(dtype))
        nb = model.nbody
        self.subtree_com = torch.zeros(num_worlds, nb, 3, device=self.device,
                                       dtype=dtype)
        self.cfrc_ext = torch.zeros(num_worlds, nb, 6, device=self.device,
                                    dtype=dtype)
        self._pull()
        self._pull_extra()

    def _pull_extra(self):
        for w, d in enumerate(self.datas):
            self.subtree_com[w] = torch.from_numpy(d.subtree_com.copy())
            self.cfrc_ext[w] = torch.from_numpy(d.cfrc_ext.copy())

    def step(self):
        super().step()
        self._pull_extra()

    def forward(self):
        super().forward()
        self._pull_extra()


class CompeteCpuDevBackend(CompeteCpuBackend):
    """The CPU mirror of `CompeteWarpDevBackend`: one MjModel PER WORLD, so the
    design writer has the same `model_arrays` interface on both stacks.

    This exists for the parity gate. It is deliberately the same code path as
    the GPU one -- the same `design.py` tensors are written into the same field
    names -- so a gate that passes here is testing the writer, not a CPU-only
    reimplementation of it. Small world counts only; it is a Python loop.
    """

    def __init__(self, model, num_worlds, substeps, *, batched_fields=(),
                 **kwargs):
        super().__init__(model, num_worlds, substeps, **kwargs)
        self.models = [copy.deepcopy(model) for _ in range(num_worlds)]
        self.datas = [mujoco.MjData(m) for m in self.models]
        self.model_arrays = {}
        # CPU MuJoCo has fields mujoco_warp does not (its compile-time body BVH),
        # and its broadphase reads them, so the mirror has to keep them current
        # or it quietly drops contacts on a design with longer legs.
        from rower_soccer.competevo_port.design import CPU_EXTRA_FIELDS
        batched_fields = tuple(batched_fields) + tuple(
            f for f in CPU_EXTRA_FIELDS if f not in batched_fields)
        for f in batched_fields:
            a = torch.as_tensor(
                getattr(model, f).reshape(_field_shape(model, f)).copy(),
                dtype=self.dtype)
            self.model_arrays[f] = a.unsqueeze(0).repeat(
                num_worlds, *([1] * a.dim())).clone()
        self._model_dirty = True
        for m, d in zip(self.models, self.datas):
            mujoco.mj_forward(m, d)
        self._pull()
        self._pull_extra()

    def _push_model(self):
        if not self._model_dirty:
            return
        for w, m in enumerate(self.models):
            for f, t in self.model_arrays.items():
                getattr(m, f)[:] = t[w].cpu().numpy().reshape(
                    getattr(m, f).shape)
        self._model_dirty = False

    def mark_model_dirty(self):
        self._model_dirty = True

    def _push(self):
        self._push_model()
        for w, d in enumerate(self.datas):
            d.qpos[:] = self.qpos[w].cpu().numpy()
            d.qvel[:] = self.qvel[w].cpu().numpy()
            d.ctrl[:] = self.ctrl[w].cpu().numpy()

    def step(self):
        self._push()
        for m, d in zip(self.models, self.datas):
            for _ in range(self.substeps):
                mujoco.mj_step(m, d)
        self._pull()
        self._pull_extra()

    def forward(self):
        self._push()
        for m, d in zip(self.models, self.datas):
            mujoco.mj_forward(m, d)
        self._pull()
        self._pull_extra()


def _field_shape(model, field):
    """`geom_aabb` is flat `(ngeom, 6)` on MjModel and `(ngeom, 2, 3)` in
    mujoco_warp; everything else agrees. Keep the warp convention so one design
    tensor writes into either stack."""
    return ((model.ngeom, 2, 3) if field == "geom_aabb"
            else getattr(model, field).shape)
