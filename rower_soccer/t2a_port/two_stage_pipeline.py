"""D3 unit 3d step 5: the two-stage pipeline -- design on the CPU, execution
batched on the GPU, worlds grouped by topology.

    design (CPU, no MuJoCo)          execution (batched, one model per group)
    -----------------------          ----------------------------------------
    N x DesignWorld                  group by ordered body-name tuple
    5 skeleton steps + 1 attribute   compile ONE model per group
    -> N morphologies                write the OTHER members' fields per world
                                     -> one T2ABatchedExecEnv per group

`topology_census.py` settled the strategy (PORT_MAP section 11): group, do not
compile a superset and mask. This module is that decision made real.

--------------------------------------------------------------------------
Where the per-world numbers come from, and why that changed
--------------------------------------------------------------------------
`xml_to_fields.py` computes the per-world model fields ARITHMETICALLY, on the
premise (its docstring) that "compiling 50,000 of them per epoch is the thing
that would sink the whole approach". That premise is wrong, and the arithmetic
is what nearly sank it:

* **50,000 is agent-steps, not designs.** A batch of ~57,000 agent-steps at
  ~1,000 steps per episode contains ~57 DESIGNS. One compile each.
* **Measured here: 4.0 ms to compile and 0.5 ms to convert, per world** (40
  real designs off `hopper_gpu_s2` epoch 1000). 57 worlds is 0.26 s of CPU per
  PPO iteration, against an iteration that samples 57,344 steps. It is not the
  bottleneck and it never was.
* **The arithmetic is a much bigger surface than it looks.** Asking MuJoCo
  which fields actually differ between two designs of the SAME topology
  returns 21 of them, including `body_iquat`, `dof_M0`, `dof_length`,
  `geom_sameframe`, `bvh_aabb`, `cam_poscom0` and `light_poscom0`. Closed forms
  for all of those would be a large body of untested code whose failure mode is
  a physically-wrong-but-numerically-fine env -- the bug class this project has
  shipped twice.

So the pipeline COMPILES each world and reads the fields off the compile.
`xml_to_fields.py` keeps its job: it is a gate on those closed forms, and it is
the fallback if a future task really does need thousands of designs per batch.

--------------------------------------------------------------------------
The coverage check is the part that makes this safe
--------------------------------------------------------------------------
`differing_fields()` compares every array on the K compiled models of a group
and returns the ones that are not identical. Every one of them must either be
written per world, or be on `WARP_INERT` with a reason. A field that differs,
is not written, and is not known-inert raises. That converts "did I remember
every field?" from a judgement call into an assertion -- and it is what caught
`dof_M0`, `geom_sameframe` and `body_iquat`, none of which were on anyone's
list.
"""

import collections
import time

import mujoco
import numpy as np
import torch

from rower_soccer.t2a_port.batched_exec_env import T2ABatchedExecEnv, topology_spec
from rower_soccer.t2a_port.xml_global_to_local import convert

# Fields that differ between designs but cannot change mujoco_warp's physics.
# Each entry is a REASON, checked against the installed mujoco_warp by
# `_check_warp_inert()` rather than trusted.
WARP_INERT = {
    "bvh_aabb": "not a mujoco_warp Model field; warp broadphases from "
                "geom_aabb/geom_rbound (CompeteCpuDevBackend writes it because "
                "CPU MuJoCo's broadphase descends it)",
    "dof_M0": "not a mujoco_warp Model field",
    "geom_sameframe": "not a mujoco_warp Model field (a CPU mj_kinematics fast "
                      "path flag; the CPU mirror writes it)",
    "body_sameframe": "not a mujoco_warp Model field (same)",
    "dof_length": "read only by mujoco_warp/_src/sleep.py, which runs only "
                  "under mjENBL_SLEEP -- asserted off below",
}


def compile_design(xml_str):
    """Their global-coordinate design XML -> a modern MjModel.

    `legacy_inertial=True` is NOT optional. PORT_MAP section 13: without it the
    port simulates a robot 1.7% off in mass and 5.2% off in inertia, because
    MuJoCo 2.1 computed capsule inertials differently and that is what
    Transform2Act trained against.
    """
    return mujoco.MjModel.from_xml_string(convert(xml_str, legacy_inertial=True))


def _arrays(model):
    for name in dir(model):
        if name.startswith("_"):
            continue
        try:
            v = getattr(model, name)
        except Exception:
            continue
        if isinstance(v, np.ndarray) and v.dtype.kind in "fiub":
            yield name, v


def differing_fields(models):
    """Every array field that is not identical across `models`.

    Structural mismatches (a shape difference) mean the group key is wrong and
    raise here rather than corrupting a write.
    """
    m0 = models[0]
    ref = dict(_arrays(m0))
    out = []
    for name, v0 in ref.items():
        for m in models[1:]:
            v = getattr(m, name)
            if v.shape != v0.shape:
                raise ValueError(
                    f"field {name} has shape {v.shape} vs {v0.shape} inside one "
                    f"topology group -- the grouping key is wrong")
            if not np.array_equal(v0, v):
                out.append(name)
                break
    return sorted(out)


def _warp_model_fields():
    import dataclasses

    from mujoco_warp._src import types
    batchable, present = set(), set()
    for f in dataclasses.fields(types.Model):
        present.add(f.name)
        shape = getattr(f.type, "shape", ())
        if shape and shape[0] == "*":
            batchable.add(f.name)
    return batchable, present


def _check_warp_inert(model, inert):
    """A differing field may be skipped ONLY if mujoco_warp cannot read it.

    The rule is deliberately not "does it look like it matters" -- if warp can
    batch a field, it gets written, whatever it is for. This check exists for
    the handful warp has no array for at all, and it is verified against the
    INSTALLED mujoco_warp rather than against this list. It has already earned
    its keep: `cam_poscom0` and `light_poscom0` were on the skip list as
    "rendering only" and the check refused them, because warp does carry them.
    """
    _, present = _warp_model_fields()
    for f in inert:
        if f not in present:
            continue
        if f == "dof_length":
            assert not (model.opt.enableflags
                        & mujoco.mjtEnableBit.mjENBL_SLEEP), (
                "mjENBL_SLEEP is on, so dof_length is live and must be batched")
            continue
        raise AssertionError(
            f"{f} differs between designs and IS a mujoco_warp Model field; "
            f"write it instead of skipping it")


def _field_shape(model, field):
    """`geom_aabb` is flat `(ngeom, 6)` on MjModel and `(ngeom, 2, 3)` in
    mujoco_warp. Same convention as `competevo_port.backend`."""
    return ((model.ngeom, 2, 3) if field == "geom_aabb"
            else getattr(model, field).shape)


class TopologyGroup:
    """K worlds that share a skeleton, on ONE compiled model plus per-world
    field writes.

    `models[0]` is the model actually uploaded; every other member exists only
    as a source of numbers. `write_fields()` is the whole port surface, and
    `gate_two_stage.py` checks it by rolling the group against the same K
    worlds compiled and run one at a time.
    """

    def __init__(self, key, worlds, models, spec_cfg, *, backend="warp",
                 device=None, done_condition=None, reward_specs=None,
                 clip_qvel=True, frame_skip=4, init_noise=0.005, seed=0,
                 nconmax_per_world=48, njmax_per_world=96, write=True,
                 field_perm=None, drop_fields=(), **backend_kw):
        assert len(worlds) == len(models) and worlds
        self.key = key
        self.worlds = worlds
        self.models = models
        self.n = len(worlds)
        w0 = worlds[0]
        self.bodies = [b.name for b in w0.robot.bodies]
        self.depths = [int(b.depth) for b in w0.robot.bodies]
        self.edges = np.asarray(w0.edges())
        self.body_index = w0.body_index()

        # Everything the key promises, asserted rather than assumed.
        for w in worlds[1:]:
            assert [b.name for b in w.robot.bodies] == self.bodies
            assert np.array_equal(np.asarray(w.edges()), self.edges)
            assert np.array_equal(w.body_index(), self.body_index)

        # Coverage. Anything warp CAN batch is written, no judgement calls;
        # anything it cannot must be on WARP_INERT with a reason, and
        # `_check_warp_inert` re-derives that reason from the installed warp.
        self.diff = differing_fields(models)
        batchable, _present = _warp_model_fields()
        self.written = [f for f in self.diff if f in batchable]
        self.inert = [f for f in self.diff if f not in batchable]
        unknown = [f for f in self.inert if f not in WARP_INERT]
        if unknown:
            raise NotImplementedError(
                f"fields differ between designs, mujoco_warp cannot batch them "
                f"and no reason is recorded: {unknown}")
        if backend == "warp":
            _check_warp_inert(models[0], self.inert)
        else:
            # CPU MuJoCo holds a whole MjModel per world, so nothing is inert:
            # write every differing field.
            self.written = list(self.diff)

        # `qpos0` decides the reset pose and `T2ABatchedExecEnv` reads it from
        # the representative model only.
        assert "qpos0" not in self.diff, "qpos0 differs inside a group"

        self.written = [f for f in self.written if f not in set(drop_fields)]

        self.spec = topology_spec(models[0], self.bodies, self.depths,
                                  max_body_depth=spec_cfg.max_body_depth,
                                  index_base=spec_cfg.index_base)
        design = np.stack([w.design_cur_params for w in worlds])
        rs = reward_specs or {}
        self.env = T2ABatchedExecEnv(
            models[0], self.spec, design, num_worlds=self.n,
            frame_skip=frame_skip, done_condition=done_condition,
            alive_bonus=rs.get("alive_bonus", 1.0),
            exec_reward_scale=rs.get("exec_reward_scale", 1.0),
            abs_displacement=rs.get("abs_displacement", False),
            clip_qvel=clip_qvel, backend=backend,
            device=device if device is not None
            else ("cpu" if backend == "cpu" else None),
            batched_fields=tuple(self.written), init_noise=init_noise,
            seed=seed,
            nconmax=max(64, nconmax_per_world * self.n),
            njmax=max(512, njmax_per_world * self.n), **backend_kw)
        if write:
            self.write_fields(perm=field_perm)
        self.env.reset()

    def write_fields(self, perm=None):
        """Push each member's own compiled values into the batched model.

        `perm` exists for the gate's negative control: rolling it by one gives
        every world its NEIGHBOUR's body, which must move the trajectory.
        """
        order = list(range(self.n)) if perm is None else list(perm)
        arrays = self.env.backend.model_arrays
        for f in self.written:
            dst = arrays[f]
            src = np.stack([np.asarray(getattr(self.models[j], f)).reshape(
                _field_shape(self.models[j], f)) for j in order])
            dst.copy_(torch.as_tensor(src, dtype=dst.dtype, device=dst.device))
        if hasattr(self.env.backend, "mark_model_dirty"):
            self.env.backend.mark_model_dirty()

    # ---- what the policy needs -------------------------------------------
    def adj(self):
        dev, dt_ = self.env.device, self.env.dtype
        n = len(self.bodies)
        a = torch.zeros(1, n, n, device=dev, dtype=dt_)
        a[0, self.edges[0], self.edges[1]] = 1.0
        return a.expand(self.n, -1, -1).contiguous()

    def ind(self):
        return torch.as_tensor(self.body_index, device=self.env.device
                               ).unsqueeze(0).expand(self.n, -1).contiguous()


def group_designs(worlds):
    """`OrderedDict` topology-key -> list of world indices, biggest first.

    Ordering the groups by size means the big GPU batches run first, which
    matters only for wall-clock, and makes the census readable.
    """
    g = collections.defaultdict(list)
    for i, w in enumerate(worlds):
        g[w.topo_key()].append(i)
    return collections.OrderedDict(
        sorted(g.items(), key=lambda kv: -len(kv[1])))


def run_design_stages(spec, init_xml, n_worlds, policy, *, device, dtype,
                      mean_action=False, generator=None):
    """The CPU half. All `n_worlds` designs are advanced in lockstep so the
    policy sees one batched forward per design step per topology group.

    Returns `(worlds, records)`; `records` is the per-step trajectory data the
    PPO update needs -- observation, action and log-prob per world per design
    step, which is exactly what their buffer holds for a design step.
    """
    from rower_soccer.t2a_port.design_stage import DesignWorld

    worlds = [DesignWorld(spec, init_xml) for _ in range(n_worlds)]
    records = []
    for t in range(spec.skel_transform_nsteps + 1):
        stage = "skel_trans" if t < spec.skel_transform_nsteps else "attr_trans"
        # Even before the first edit the worlds share one topology, so this
        # grouping is a no-op at t=0 and becomes real from t=1.
        groups = group_designs(worlds)
        step = {"stage": stage, "obs": [None] * n_worlds,
                "action": [None] * n_worlds, "log_prob": [None] * n_worlds,
                "adj": {}, "ind": {}, "members": {}}
        for key, idx in groups.items():
            obs = torch.as_tensor(
                np.stack([worlds[i].obs() for i in idx]), device=device,
                dtype=dtype)
            e = np.asarray(worlds[idx[0]].edges())
            n = len(worlds[idx[0]].robot.bodies)
            a = torch.zeros(1, n, n, device=device, dtype=dtype)
            a[0, e[0], e[1]] = 1.0
            adj = a.expand(len(idx), -1, -1).contiguous()
            ind = torch.as_tensor(worlds[idx[0]].body_index(), device=device
                                  ).unsqueeze(0).expand(len(idx), -1).contiguous()
            act, lp = policy.act(stage, obs, adj, ind, mean_action=mean_action,
                                 generator=generator)
            for r, i in enumerate(idx):
                step["obs"][i] = obs[r]
                step["action"][i] = act[r]
                step["log_prob"][i] = lp[r]
            step["adj"][key] = adj[:1]
            step["ind"][key] = ind[:1]
            step["members"][key] = idx
            act_np = act.detach().cpu().numpy()
            for r, i in enumerate(idx):
                if stage == "skel_trans":
                    worlds[i].skel_step(act_np[r][:, -1])
                else:
                    worlds[i].attr_step(
                        act_np[r][:, policy.control_action_dim:-1])
        records.append(step)
    return worlds, records


def build_groups(worlds, spec_cfg, **kw):
    """Compile and group. Returns `(groups, index_map, timings)` where
    `index_map[i]` is `(group, row)` for world `i`."""
    t0 = time.time()
    models = [compile_design(w.cur_xml_str) for w in worlds]
    t_compile = time.time() - t0
    t0 = time.time()
    out, index_map = [], [None] * len(worlds)
    for key, idx in group_designs(worlds).items():
        g = TopologyGroup(key, [worlds[i] for i in idx],
                          [models[i] for i in idx], spec_cfg, **kw)
        for row, i in enumerate(idx):
            index_map[i] = (len(out), row)
        out.append(g)
    return out, index_map, {"compile_s": t_compile,
                            "build_s": time.time() - t0}
