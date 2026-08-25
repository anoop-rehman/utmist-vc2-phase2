"""D3 unit 3d step 4: Transform2Act's EXECUTION stage, batched, one topology.

The port map's step 4 is deliberately the narrowest thing that can be gated:
**one body plan, N worlds, execution only.** The design stages involve no
physics at all (`hopper.py:114-142` return `reward = 0.0` and never touch the
sim), so they are not what a GPU port is for; the execution stage is the 1,000
steps per episode that actually cost something, and it is where a port can be
silently wrong.

What this is NOT: the topology strategy (step 5). Every world here shares a
skeleton. Worlds may still differ in ATTRIBUTES -- capsule radii, segment
lengths, gears -- which is what `batched_fields` is for and what section 11's
grouping decision assumes.

--------------------------------------------------------------------------
The three things a port of this env has to get exactly right
--------------------------------------------------------------------------
All three are gated in `gate_batched_exec.py` against observations pulled out
of their live env, at IDENTICAL qpos/qvel -- so the check is of this pipeline
alone and the physics divergence measured in `PORT_MAP.md` section 13 cannot
hide a bug in it.

1. **The root node's sim observation is FLIPPED and the others are PADDED.**
   `hopper.py:210` is `[np.flip(qpos[1:3]), np.flip(qvel[:3])]`, i.e.
   `[ang, height, ang_vel, z_vel, x_vel]` -- reversed, and it is the *second and
   third* qpos entries, not the first two. Every other node is
   `[qpos[js], 0, qvel[js], 0, 0]` (`hopper.py:214`): its single joint angle,
   its single joint velocity, and three structural zeros in between. A port that
   packs those five slots in the obvious order produces an env that trains and
   is meaningless.

2. **`clip_qvel` applies to the OBSERVATION only.** `hopper.py:207` clips a
   *copy* to +/-10 before reading it; the physics keeps the unclipped velocity.

3. **Actuators are resolved BY NAME, not by body order.**
   `action_to_control` looks up `model.actuator_names.index(body.get_actuator_
   name())` per body. On `hopper_gpu_s2` the two orders happen to coincide,
   which is exactly the kind of coincidence that makes an index-order port pass
   its first test and fail on the next morphology.

--------------------------------------------------------------------------
Time limits
--------------------------------------------------------------------------
`hopper.py:179` folds `control_nsteps < max_nsteps` into `done`, so the 1,000
step time limit sets `done = True` and their GAE bootstraps ZERO at it. That is
the opposite of the CompetEvo convention (`competevo_port/ppo.py:187`, mask = 1
on truncation). `last_end` distinguishes the two causes so a trainer can tell a
fall from a time limit, but `done` deliberately does not -- matching them is the
point. See `D3_HANDOFF.md`, "M2 acceptance criterion, settled".
"""

import numpy as np
import torch

# hopper.py's defaults, read at `cfg.done_condition.get(...)`. `max_ang` is the
# only one hopper.yml overrides (20 degrees, against the 3600 default that
# disables the check).
DONE_DEFAULTS = {"min_height": 0.7, "max_height": 2.0,
                 "max_ang": 3600.0, "max_nsteps": 1000}
QVEL_CLIP = 10.0

# `last_end` codes. running=0 so a freshly reset world reads as "no ending yet".
END_RUNNING, END_FELL, END_NONFINITE, END_TIMEOUT = 0, 1, 2, 3


def topology_spec(model, bodies, depths, max_body_depth=4, index_base=4):
    """Everything about a morphology the env needs, resolved once by NAME.

    `bodies` and `depths` come from their `robot.bodies` -- this port does not
    re-derive the tree, because their `Body.depth` is the authority and a
    re-derivation that disagreed would be a silent observation change.
    """
    import mujoco

    n = len(bodies)
    name2act = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i): i
                for i in range(model.nu)}
    # bodies[0] is the root and has no actuator (their `get_actuator_name()`
    # returns None for it); its control slot stays zero forever.
    act_of_node = np.full(n, -1, dtype=np.int64)
    for i, b in enumerate(bodies):
        if i == 0:
            continue
        aid = name2act.get(f"{b}_joint")
        assert aid is not None, f"no actuator named {b}_joint"
        act_of_node[i] = aid

    # Each non-root body owns exactly one hinge, named `<body>_joint`; their
    # `get_single_body_qposaddr` asserts the width is 1 and so do we.
    name2jnt = {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): j
                for j in range(model.njnt)}
    qadr_of_node = np.full(n, -1, dtype=np.int64)
    vadr_of_node = np.full(n, -1, dtype=np.int64)
    for i, b in enumerate(bodies):
        if i == 0:
            continue
        j = name2jnt[f"{b}_joint"]
        assert model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
        qadr_of_node[i] = model.jnt_qposadr[j]
        vadr_of_node[i] = model.jnt_dofadr[j]

    attr_fixed = np.zeros((n, max_body_depth), dtype=np.float64)
    for i, d in enumerate(depths):
        attr_fixed[i, d] = 1.0

    body_index = np.array([int(b, base=index_base) for b in bodies],
                          dtype=np.int64)
    return {"bodies": list(bodies), "n_nodes": n,
            "act_of_node": act_of_node, "qadr_of_node": qadr_of_node,
            "vadr_of_node": vadr_of_node, "attr_fixed": attr_fixed,
            "body_index": body_index}


class T2ABatchedExecEnv:
    """N worlds of one topology, execution stage only.

    `design_params` is `[n_nodes, D]` shared, or `[N, n_nodes, D]` per world --
    it is an observation, not a physics input, so it is carried rather than
    written into the model. Whatever writes the corresponding *model* fields
    (`xml_to_fields`) is the caller's job; this class asserts nothing about
    their consistency because at step 4 every world is the same body.
    """

    def __init__(self, model, spec, design_params, num_worlds=64,
                 frame_skip=4, done_condition=None, alive_bonus=1.0,
                 exec_reward_scale=1.0, abs_displacement=False,
                 clip_qvel=True, backend="warp", device=None,
                 batched_fields=(), init_noise=0.005, seed=0, **backend_kw):
        from rower_soccer.competevo_port.backend import (CompeteCpuDevBackend,
                                                         CompeteWarpDevBackend)

        self.model = model
        self.spec = spec
        self.n = num_worlds
        self.frame_skip = frame_skip
        self.dt = float(model.opt.timestep) * frame_skip
        self.dc = dict(DONE_DEFAULTS, **(done_condition or {}))
        self.alive_bonus = alive_bonus
        self.exec_reward_scale = exec_reward_scale
        self.abs_displacement = abs_displacement
        self.clip_qvel = clip_qvel
        self.init_noise = init_noise

        cls = CompeteWarpDevBackend if backend == "warp" else CompeteCpuDevBackend
        self.backend = cls(model, num_worlds, frame_skip,
                           batched_fields=tuple(batched_fields), **backend_kw)
        self.device = self.backend.qpos.device
        self.dtype = self.backend.qpos.dtype
        dev, dt_ = self.device, self.dtype

        self.qadr = torch.as_tensor(spec["qadr_of_node"], device=dev)
        self.vadr = torch.as_tensor(spec["vadr_of_node"], device=dev)
        self.act_of = torch.as_tensor(spec["act_of_node"], device=dev)
        self.attr_fixed = torch.as_tensor(spec["attr_fixed"], device=dev,
                                          dtype=dt_)
        self.body_index = torch.as_tensor(spec["body_index"], device=dev)
        d = torch.as_tensor(np.asarray(design_params), device=dev, dtype=dt_)
        self.design = d if d.dim() == 3 else d.unsqueeze(0).expand(self.n, -1, -1)

        # Their reset is `init_qpos +/- U(0.005)`, and init_qpos/init_qvel come
        # from the compiled model's keyframe-free defaults.
        import mujoco
        md = mujoco.MjData(model)
        mujoco.mj_forward(model, md)
        self.init_qpos = torch.as_tensor(md.qpos.copy(), device=dev, dtype=dt_)
        self.init_qvel = torch.as_tensor(md.qvel.copy(), device=dev, dtype=dt_)

        self.g = torch.Generator(device="cpu").manual_seed(seed)
        self.control_nsteps = torch.zeros(self.n, dtype=torch.long, device=dev)
        self.last_end = torch.zeros(self.n, dtype=torch.long, device=dev)
        self.reset()

    # ---- state -----------------------------------------------------------
    def _write_initial(self, idx, add_noise=True):
        k = idx.numel()
        if k == 0:
            return
        q = self.init_qpos.unsqueeze(0).repeat(k, 1)
        v = self.init_qvel.unsqueeze(0).repeat(k, 1)
        if add_noise and self.init_noise:
            lo, hi = -self.init_noise, self.init_noise
            q += torch.rand(q.shape, generator=self.g, dtype=torch.float64
                            ).to(q.device, q.dtype) * (hi - lo) + lo
            v += torch.rand(v.shape, generator=self.g, dtype=torch.float64
                            ).to(v.device, v.dtype) * (hi - lo) + lo
        self.backend.qpos[idx] = q
        self.backend.qvel[idx] = v
        self.control_nsteps[idx] = 0

    def reset(self, add_noise=True):
        self._write_initial(torch.arange(self.n, device=self.device), add_noise)
        self.last_end[:] = END_RUNNING
        self.backend.forward()
        return self.obs()

    def set_state(self, qpos, qvel, control_nsteps=None):
        """Force every world onto a given state. The gate's whole method: it
        removes the physics divergence from the comparison."""
        q = torch.as_tensor(np.asarray(qpos), device=self.device, dtype=self.dtype)
        v = torch.as_tensor(np.asarray(qvel), device=self.device, dtype=self.dtype)
        self.backend.qpos[:] = q if q.dim() == 2 else q.unsqueeze(0)
        self.backend.qvel[:] = v if v.dim() == 2 else v.unsqueeze(0)
        if control_nsteps is not None:
            self.control_nsteps[:] = int(control_nsteps)
        self.backend.forward()

    # ---- observation -----------------------------------------------------
    def sim_obs(self):
        """`[N, n_nodes, 5]`, matching `hopper.get_sim_obs` slot for slot."""
        qpos, qvel = self.backend.qpos, self.backend.qvel
        if self.clip_qvel:
            qvel = qvel.clamp(-QVEL_CLIP, QVEL_CLIP)
        out = torch.zeros(self.n, self.spec["n_nodes"], 5,
                          device=self.device, dtype=self.dtype)
        # Root: np.flip(qpos[1:3]) then np.flip(qvel[:3]) -- REVERSED, and note
        # qpos[1:3] is (height, ang) so flipped it is (ang, height).
        out[:, 0, 0] = qpos[:, 2]
        out[:, 0, 1] = qpos[:, 1]
        out[:, 0, 2] = qvel[:, 2]
        out[:, 0, 3] = qvel[:, 1]
        out[:, 0, 4] = qvel[:, 0]
        # Others: [q, 0, v, 0, 0].
        out[:, 1:, 0] = qpos.index_select(1, self.qadr[1:])
        out[:, 1:, 2] = qvel.index_select(1, self.vadr[1:])
        return out

    def obs(self):
        """`[N, n_nodes, 4 + 5 + D]` -- attr_fixed | sim | design, their order."""
        sim = self.sim_obs()
        af = self.attr_fixed.unsqueeze(0).expand(self.n, -1, -1)
        return torch.cat([af, sim, self.design], dim=-1)

    # ---- dynamics --------------------------------------------------------
    def terms(self, posbefore, qpos, qvel, control_nsteps):
        """`hopper.py:158-179`'s reward and done, as a pure function.

        Split out so the gate can drive it with THEIR recorded states rather
        than ours -- physics divergence would otherwise make an exact reward
        comparison impossible. `step()` calls this same function, which is the
        part that matters: the CompetEvo port once billed a control cost on the
        clipped action and every parity gate missed it, because the gates drove
        `terms()` while the bug lived in `step()`. Nothing here may be
        recomputed inline at the call site.
        """
        height, ang = qpos[:, 1], qpos[:, 2]
        dx = qpos[:, 0] - posbefore
        if self.abs_displacement:
            dx = dx.abs()
        reward = (dx / self.dt + self.alive_bonus) * self.exec_reward_scale

        # `mujoco_env_gym.state_vector()` is the FULL qpos concatenated with
        # qvel -- gym's own Hopper drops the first two entries, theirs does not,
        # so the root x is inside the finiteness test. Checked, not assumed.
        finite = (torch.isfinite(qpos).all(1) & torch.isfinite(qvel).all(1))
        upright = ((height > self.dc["min_height"])
                   & (height < self.dc["max_height"])
                   & (ang.abs() < np.deg2rad(self.dc["max_ang"])))
        timeout = control_nsteps >= self.dc["max_nsteps"]
        done = ~(finite & upright & ~timeout)

        # Ordered so a genuine failure is never reported as a time limit.
        self.last_end = torch.where(
            done,
            torch.where(~finite, torch.full_like(self.last_end, END_NONFINITE),
                        torch.where(~upright,
                                    torch.full_like(self.last_end, END_FELL),
                                    torch.full_like(self.last_end, END_TIMEOUT))),
            torch.full_like(self.last_end, END_RUNNING))
        return reward, done

    def step(self, action, auto_reset=True):
        """`action` is `[N, n_nodes, ...]` in their node order; column 0 of each
        node is its control and the root's is discarded, exactly as
        `action_to_control` does."""
        a = torch.as_tensor(action, device=self.device, dtype=self.dtype)
        if a.dim() == 3:
            a = a[..., 0]
        ctrl = torch.zeros(self.n, self.model.nu, device=self.device,
                           dtype=self.dtype)
        ctrl.scatter_(1, self.act_of[1:].unsqueeze(0).expand(self.n, -1),
                      a[:, 1:])
        self.backend.ctrl[:] = ctrl

        posbefore = self.backend.qpos[:, 0].clone()
        self.backend.step()
        self.control_nsteps += 1
        reward, done = self.terms(posbefore, self.backend.qpos,
                                  self.backend.qvel, self.control_nsteps)
        obs = self.obs()
        if auto_reset and bool(done.any()):
            self._write_initial(done.nonzero(as_tuple=True)[0])
            self.backend.forward()
        return obs, reward, done, {"last_end": self.last_end}
