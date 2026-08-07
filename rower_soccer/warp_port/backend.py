"""Batched physics backend behind which ALL mujoco_warp specifics live.

The env classes (worm_env_base.WormEnv + task subclasses) are written against
`PhysicsBackend` and contain zero `mjw.`/`wp.` calls, so a CPU dm_control/MuJoCo
backend can be swapped in later without touching env/obs/reward code.

Mutation contract
-----------------
In-place writes to the state tensors (`qpos`, `qvel`, `ctrl`) take effect at the
NEXT `forward()` or `step()`. The Warp backend aliases GPU buffers, so writes are
immediate; a future CPU backend may stage writes and flush them on the call.
Either way env code must call `forward()` after writing qpos/qvel before reading
`xpos`/`xmat`/`sensordata` -- which `reset()`/`_sanitize()` already do.
"""
import abc

import mujoco


class PhysicsBackend(abc.ABC):
    """Uploads a compiled MjModel, allocates `num_worlds` of batched data, and
    exposes the simulator state as torch tensors plus `step()`/`forward()`.

    Required attributes (set by subclasses):
        device      : "cuda" / "cpu"; the env builds its torch tensors here.
        qpos, qvel, ctrl        : [n, n{q,v,u}] torch views.
        xpos                    : [n, nbody, 3] torch view.
        xmat                    : [n, nbody, 3, 3] torch view.
        sensordata              : [n, nsensordata] torch view.
    """

    device: str

    @abc.abstractmethod
    def step(self):
        """Advance the physics by SUBSTEPS substeps for every world."""

    @abc.abstractmethod
    def forward(self):
        """Recompute kinematics + sensors from current qpos/qvel WITHOUT
        integrating (mj_forward), flushing any staged state writes first."""


class WarpBackend(PhysicsBackend):
    """mujoco_warp backend: GPU-resident, CUDA-graph-accelerated, N worlds in
    parallel. This is the only implementation built today; it holds every
    `mjw.`/`wp.` call that used to be smeared across the env classes."""

    def __init__(self, model, num_worlds, substeps, *, use_graph=True,
                 nconmax=64, njmax=512, device="cuda"):
        # Import lazily so the module (and the env classes that import it) load
        # on machines without warp/CUDA -- only *instantiating* needs a GPU.
        import warp as wp
        import mujoco_warp as mjw

        self._wp, self._mjw = wp, mjw
        self.device = device
        self.n = num_worlds
        self.substeps = substeps

        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        self.wm = mjw.put_model(model)
        # Size the contact/constraint buffers EXPLICITLY: put_data otherwise
        # infers them from the initial (contact-free) MjData, and any runtime
        # overflow silently drops constraints and NaNs the sim.
        self.wd = mjw.put_data(model, data, nworld=num_worlds,
                               nconmax=nconmax, njmax=njmax)

        # Zero-copy torch views aliasing the Warp GPU buffers.
        self.qpos = wp.to_torch(self.wd.qpos)
        self.qvel = wp.to_torch(self.wd.qvel)
        self.ctrl = wp.to_torch(self.wd.ctrl)
        self.xpos = wp.to_torch(self.wd.xpos)
        self.xmat = wp.to_torch(self.wd.xmat).reshape(num_worlds, -1, 3, 3)
        self.sensordata = wp.to_torch(self.wd.sensordata)

        self._graph = None
        if use_graph:
            with wp.ScopedCapture() as cap:
                for _ in range(substeps):
                    mjw.step(self.wm, self.wd)
            self._graph = cap.graph

    def step(self):
        if self._graph is not None:
            self._wp.capture_launch(self._graph)
        else:
            for _ in range(self.substeps):
                self._mjw.step(self.wm, self.wd)
        self._wp.synchronize_device()

    def forward(self):
        self._mjw.forward(self.wm, self.wd)
        self._wp.synchronize_device()
