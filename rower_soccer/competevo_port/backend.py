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

import torch

from rower_soccer.warp_port.backend import CpuBackend, WarpBackend


class CompeteWarpBackend(WarpBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.subtree_com = self._wp.to_torch(self.wd.subtree_com)
        self.cfrc_ext = self._wp.to_torch(self.wd.cfrc_ext)


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
