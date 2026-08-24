"""Stage-2 gate: the per-world design writer vs the model THEIR code compiles.

Plain-python (no pytest in this venv):

    PYTHONPATH=. .venv/bin/python -m rower_soccer.competevo_port.tests.test_design_parity
    ... --designs 10 --seed 0
    ... --gpu                      # + the mujoco_warp checks (needs a GPU)

The gate, cheapest first:

  1. BASE SCENE -- our generated dev MJCF compiles to their merged
     `world_body.dev_ant_body.dev_ant_body.xml`, field for field, at their
     physics options.
  2. DESIGN -> MODEL FIELDS -- for random genomes, our per-world writer equals
     the model their `set_design_params` + `load_tmp_mujoco_env` path produces.
     This is the check PORT_STATUS asked for: `test_model_matches_theirs`
     parameterized by a design vector, INCLUDING body_mass / body_inertia /
     body_ipos, not just geom sizes.

     Two references are reported, because they answer different questions:
       (a) their emitted MJCF compiled by OUR mujoco -- "is the writer right?";
       (b) their MjModel as compiled in THEIR venv -- "is anything else
           different?". (b) minus (a) is the mujoco 2.3.5 -> 3.11 delta, which
           is not zero (`geom_aabb` is padded by `geom_margin` in 2.3.5 and not
           in 3.11) and is not a port bug.
  3. mj_setConst CONSTANTS -- `body_invweight0` / `dof_invweight0` /
     `actuator_acc0` have no closed form in the design, so the writer calls
     `mj_setConst` on a host scratch model. Checked against a freshly compiled
     model, and the cost of NOT doing it is measured in metres of trajectory.
  4. OBS + REWARD PARITY at hand-set states, with a design applied first: the
     52-dim dev observation in their order, and their dense/sparse rewards and
     termination (which for the dev ant has an UPPER standing bound).
  5. RESET / EPISODE SHAPE -- the design step returns reward 0 and leaves the
     world at qpos0 with zero velocity, as their MjData rebuild does.
  6. GPU (--gpu) -- mujoco_warp really does batch these fields per world, a
     design write changes the dynamics, and the CUDA graph captured before the
     write is still valid afterwards.

Parts 2-5 need CompetEvo's venv and are skipped (loudly) without it.
"""

import argparse
import json
import os
import subprocess
import sys
import time

import mujoco
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from rower_soccer.competevo_port import parity
from rower_soccer.competevo_port.backend import CompeteCpuDevBackend
from rower_soccer.competevo_port.design import (BATCHED_FIELDS,
                                                CONST_FIELDS,
                                                CPU_EXTRA_FIELDS,
                                                WRITTEN_FIELDS, DesignWriter,
                                                HostConstants,
                                                build_design_spec,
                                                design_fields)
from rower_soccer.competevo_port.dev_env import RunToGoalDevEnv
from rower_soccer.competevo_port.scene import (DESIGN_DIM, build_dev_scene,
                                               dev_run_to_goal_xml,
                                               their_scene_path)

DRIVER = os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "their_dev_driver.py")
TOL = 1e-6
N_DESIGNS = 10

# Compiler-derived fields the writer owns, plus the ones it deliberately leaves
# alone; both are reported so the second group cannot hide.
DERIVED_UNWRITTEN = CONST_FIELDS

_results = []


def check(name, fn):
    t0 = time.perf_counter()
    try:
        detail = fn() or ""
        ok = True
    except Exception as exc:                             # noqa: BLE001
        detail, ok = f"{type(exc).__name__}: {exc}", False
    _results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} "
          f"({time.perf_counter() - t0:.1f}s) {detail}")
    return ok


def query_their_dev(payload, timeout=1200):
    proc = subprocess.run([parity.THEIR_PYTHON, DRIVER],
                          input=json.dumps(payload), capture_output=True,
                          text=True, timeout=timeout, cwd="/workspace/competevo")
    return parity._decode_driver_reply(proc, "their_dev_driver")


def random_designs(n, seed=0, n_agents=2):
    """Their reset draw: `scale_vector ~ U(-1, 1)^20` per agent
    (dev_ant.py:365). The policy's sampled design is clamped to the same box."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, (n, n_agents, DESIGN_DIM))


def _field(model, name):
    a = np.asarray(getattr(model, name), dtype=np.float64)
    return a.reshape(model.ngeom, 2, 3) if name == "geom_aabb" else a


# ---------------------------------------------------------------------------
# 1. base scene
# ---------------------------------------------------------------------------
BASE_FIELDS = ("body_mass", "body_inertia", "body_ipos", "body_iquat",
               "body_pos", "body_quat", "body_parentid", "body_subtreemass",
               "dof_damping", "dof_armature", "dof_bodyid", "dof_invweight0",
               "body_invweight0", "jnt_type", "jnt_range", "jnt_axis",
               "jnt_pos", "geom_type", "geom_size", "geom_pos", "geom_quat",
               "geom_friction", "geom_margin", "geom_condim", "geom_contype",
               "geom_conaffinity", "geom_solref", "geom_solimp", "geom_rbound",
               "geom_aabb", "actuator_gear", "actuator_ctrlrange",
               "actuator_trnid", "actuator_acc0", "qpos0")


def t_base_scene():
    ours, meta = build_dev_scene(solver="PGS", iterations=1000)
    theirs = mujoco.MjModel.from_xml_path(their_scene_path(dev=True))
    assert (ours.nq, ours.nv, ours.nu) == (theirs.nq, theirs.nv, theirs.nu)
    bad = [f for f in BASE_FIELDS
           if np.abs(_field(ours, f) - _field(theirs, f)).max() > 1e-12]
    assert not bad, f"dev scene differs from theirs: {bad}"
    order = lambda m: [m.jnt(m.actuator_trnid[i, 0]).name for i in range(m.nu)]
    assert order(ours) == order(theirs), "actuator order differs"
    # The evo merger's two-agent contact bitmask really is live here (unlike the
    # gym_compete merger), so neither dev ant self-collides.
    g0 = [i for i in range(ours.ngeom) if ours.geom(i).name.startswith("agent0/")]
    g1 = [i for i in range(ours.ngeom) if ours.geom(i).name.startswith("agent1/")]
    hit = lambda i, j: (ours.geom_contype[i] & ours.geom_conaffinity[j]) or (
        ours.geom_contype[j] & ours.geom_conaffinity[i])
    assert not hit(g0[0], g0[1]) and not hit(g1[0], g1[1]), "self-collision on"
    assert hit(g0[0], g1[0]), "agents do not collide with each other"
    assert (meta.obs_dim, meta.act_dim) == (52, 28), (meta.obs_dim, meta.act_dim)
    assert [a.goal_x for a in meta.agents] == [4.0, -4.0]
    return (f"{len(BASE_FIELDS)} fields bit-equal, nq/nv/nu "
            f"{ours.nq}/{ours.nv}/{ours.nu}, obs 52 / act 28, no self-collision")


# ---------------------------------------------------------------------------
# 2-3. design -> model fields
# ---------------------------------------------------------------------------
def run_design_parity(n_designs, seed):
    S = random_designs(n_designs, seed)
    theirs = query_their_dev(
        {"cases": [{"design": [S[k, a].tolist() for a in range(S.shape[1])]}
                   for k in range(n_designs)]})
    model, meta = build_dev_scene(solver="PGS", iterations=1000)
    spec = build_design_spec(model, meta)
    ours = design_fields(spec, torch.as_tensor(S))

    rep, ref_ours, ref_theirs = {}, {}, {}
    for k, case in enumerate(theirs["cases"]):
        # (a) their emitted MJCF, through OUR compiler.
        m = mujoco.MjModel.from_xml_string(case["xml"])
        for f in WRITTEN_FIELDS + DERIVED_UNWRITTEN:
            ref_ours.setdefault(f, []).append(_field(m, f))
            ref_theirs.setdefault(f, []).append(
                np.asarray(case["model"][f], dtype=np.float64).reshape(
                    _field(m, f).shape))

    base = {f: _field(model, f) for f in DERIVED_UNWRITTEN}
    for f in WRITTEN_FIELDS:
        a = ours[f].numpy().reshape(np.array(ref_ours[f]).shape)
        rep[f"writer/{f}"] = float(np.abs(a - np.array(ref_ours[f])).max())
        rep[f"compiler_version/{f}"] = float(
            np.abs(np.array(ref_ours[f]) - np.array(ref_theirs[f])).max())
    # The mj_setConst block: what the host scratch model produces vs a fresh
    # compile (must be zero), and how far the BASE values it replaces would have
    # been (which is why the call is there at all).
    # `compute` is handed the GENOME, not `ours`: it owns a CPU twin of the spec
    # and recomputes the fields host-side (that is the whole point of the host
    # round-trip fix), so this drives the exact production path.
    consts = HostConstants(model, spec).compute(torch.as_tensor(S))
    for f in DERIVED_UNWRITTEN:
        t = np.array(ref_ours[f])
        scale = np.maximum(np.abs(t).max(), 1e-12)
        rep[f"writer/{f}"] = float(
            np.abs(consts[f].reshape(t.shape) - t).max() / scale)
        rep[f"stale_if_skipped/{f}_rel"] = float(
            np.abs(t - base[f][None]).max() / scale)
    rep["_designs"] = n_designs
    # Sanity that the designs actually moved the robot around.
    mm = np.array(ref_ours["body_mass"])
    rep["_leg_mass_ratio_max"] = float(
        (mm[:, 1:].max(0) / np.maximum(mm[:, 1:].min(0), 1e-12)).max())
    return rep


def t_design_parity(n_designs, seed):
    rep = run_design_parity(n_designs, seed)
    print(f"  --- design parity: {rep['_designs']} random genomes, seed {seed}"
          f" (max mass spread across designs "
          f"{rep['_leg_mass_ratio_max']:.2f}x) ---")
    for k in sorted(rep):
        if not k.startswith("_"):
            print(f"    {k:36s} {rep[k]:.3e}")
    worst = [f"{k}={v:.2e}" for k, v in rep.items()
             if k.startswith("writer/") and v > 1e-12]
    assert not worst, "writer disagrees with the compiler: " + ", ".join(worst)
    return (f"{len(WRITTEN_FIELDS)} fields, worst |ours - their compiler| "
            f"{max(v for k, v in rep.items() if k.startswith('writer/')):.1e}")


# ---------------------------------------------------------------------------
# 4. obs + reward parity with a design applied
# ---------------------------------------------------------------------------
def _cpu_dev_env(n, **kw):
    return RunToGoalDevEnv(num_worlds=n, use_gpu=False,
                           backend_cls=CompeteCpuDevBackend, auto_reset=False,
                           scene_kwargs={"solver": "PGS", "iterations": 1000},
                           **kw)


DEV_OBS_GROUPS = (("stage_flag", slice(0, 1)),
                  ("scale_vector", slice(1, 21)),
                  ("sim/own_root_pos", slice(21, 24)),
                  ("sim/own_root_quat", slice(24, 28)),
                  ("sim/own_joint_pos", slice(28, 36)),
                  ("sim/own_root_linvel", slice(36, 39)),
                  ("sim/own_root_angvel", slice(39, 42)),
                  ("sim/own_joint_vel", slice(42, 50)),
                  ("sim/opponent_root_xy", slice(50, 52)))


def run_state_parity(n_cases, seed):
    states = parity.random_states(n_cases, seed=seed)
    S = random_designs(n_cases, seed + 1000)
    cases = [dict(states[k], design=[S[k, a].tolist() for a in range(2)])
             for k in range(n_cases)]
    theirs = query_their_dev({"cases": cases})["cases"]

    env = _cpu_dev_env(n_cases)
    env.reset()
    idx = torch.arange(n_cases)
    env.set_design(idx, torch.as_tensor(S, dtype=env.dtype))
    to = lambda x: torch.as_tensor(np.asarray(x), dtype=env.dtype)
    env.qpos.copy_(to([c["qpos_prev"] for c in cases]))
    env.qvel.copy_(to([c["qvel_prev"] for c in cases]))
    env.backend.forward()
    env._com_before = env._agent_com_x().clone()
    env.qpos.copy_(to([c["qpos"] for c in cases]))
    env.qvel.copy_(to([c["qvel"] for c in cases]))
    env.backend.forward()
    t = env.terms(to([c["action"] for c in cases]))
    obs = env.obs()

    npy = lambda x: x.detach().double().cpu().numpy()
    rep = {}
    their_obs = np.array([c["obs"] for c in theirs])
    our_obs = npy(obs)
    for name, sl in DEV_OBS_GROUPS:
        rep[f"obs/{name}"] = float(
            np.abs(our_obs[..., sl] - their_obs[..., sl]).max())
    rep["obs/ALL"] = float(np.abs(our_obs - their_obs).max())
    for key, ours_key in (("subtree_com_x", "com_x"),
                          ("reward_forward", "forward"),
                          ("reward_ctrl", "ctrl_cost"),
                          ("reward_contact", "contact_cost"),
                          ("reward_dense", "dense"),
                          ("reward_parse", "parse"),
                          ("reward_total", "reward")):
        ref = np.array([c[key] for c in theirs])
        rep[f"reward/{key}"] = float(np.abs(npy(t[ours_key]) - ref).max())
    rep["flag/agent_fell"] = int((npy(t["fell"]).astype(bool)
                                  != np.array([c["agent_done"] for c in theirs])
                                  ).sum())
    rep["flag/reached_goal"] = int(
        (npy(t["reached"]).astype(bool)
         != np.array([c["reached_goal"] for c in theirs])).sum())
    rep["flag/winner"] = int((npy(t["winner"]).astype(bool)
                              != np.array([c["winner"] for c in theirs])).sum())
    rep["flag/terminated"] = int(
        (npy(t["terminated"]).astype(bool)
         != np.array([c["terminated"][0] for c in theirs])).sum())
    rep["_cases"] = n_cases
    rep["_cases_with_a_fall"] = int(sum(any(c["agent_done"]) for c in theirs))
    rep["_cases_with_goal_crossed"] = int(
        sum(any(c["reached_goal"]) for c in theirs))
    rep["_cases_in_contact"] = int(sum(c["ncon"] > 0 for c in theirs))
    rep["_their_cfrc_ext_absmax"] = float(
        max(max(c["cfrc_ext_absmax"]) for c in theirs))
    return rep


def t_state_parity(n_cases, seed):
    rep = run_state_parity(n_cases, seed)
    print(f"  --- dev obs + reward parity: {rep['_cases']} hand-set states "
          f"with random designs ---")
    for k in sorted(rep):
        v = rep[k]
        print(f"    {k:32s} {v:.3e}" if isinstance(v, float)
              and not k.startswith("_") else f"    {k:32s} {v}")
    worst = [f"{k}={v}" for k, v in rep.items()
             if not k.startswith("_") and v > (TOL if k.startswith(
                 ("obs/", "reward/")) else 0)]
    assert not worst, "over tolerance: " + ", ".join(worst)
    return f"worst obs {max(v for k, v in rep.items() if k.startswith('obs/')):.1e}"


# ---------------------------------------------------------------------------
# 5. episode shape
# ---------------------------------------------------------------------------
def t_episode_shape():
    """Their design step: reward 0, no termination, and a state that is exactly
    qpos0 with zero velocity -- the reset noise dies with the old MjData."""
    n = 4
    S = random_designs(n, seed=5)
    theirs = query_their_dev(
        {"cases": [{"design": [S[k, a].tolist() for a in range(2)]}
                   for k in range(n)]})["cases"]
    q_theirs = np.array([c["qpos_after_design"] for c in theirs])
    v_theirs = np.array([c["qvel_after_design"] for c in theirs])

    env = _cpu_dev_env(n)
    obs0 = env.reset()
    assert obs0.shape == (n, 2, 52), obs0.shape
    assert bool(env.stage.all()) and float(obs0[..., 0].abs().max()) == 0.0, \
        "stage flag should be 0 before the design action"
    act = torch.zeros(n, 2, 28, dtype=env.dtype)
    act[..., :DESIGN_DIM] = torch.as_tensor(S, dtype=env.dtype)
    obs1, rew, done, info = env.step(act)
    assert float(rew.abs().max()) == 0.0, "design step must pay 0"
    assert not bool(done.any()), "design step must not terminate"
    assert float((obs1[..., 0] - 1.0).abs().max()) == 0.0, "flag should be 1"
    assert float((obs1[..., 1:21] - torch.as_tensor(S, dtype=env.dtype)
                  ).abs().max()) < 1e-12, "obs scale block != applied design"
    dq = np.abs(env.qpos.numpy() - q_theirs).max()
    dv = np.abs(env.qvel.numpy() - v_theirs).max()
    assert dq < 1e-12 and dv == 0.0, f"post-design state differs: {dq}, {dv}"
    assert int(env.ep_step.max()) == 0, "the design step is not an elapsed step"
    # ... and the next step is a normal one.
    _, rew2, _, info2 = env.step(torch.zeros(n, 2, 28, dtype=env.dtype))
    assert not bool(info2["was_design"].any())
    assert float(rew2.min()) > 0.0, "survive bonus missing after the design step"
    return (f"reward 0, qpos == their fresh MjData ({dq:.1e}), qvel 0, "
            f"flag 0 -> 1, first live reward {float(rew2.mean()):.3f}")


def t_setconst_trajectory(n_designs, seed, steps=40):
    """Field equality is necessary, not sufficient: this is the whole model,
    stepped, against a fresh compile of their XML.

    Three models per design, identical state and identical open-loop torques for
    40 control steps (0.6 s):
      (a) their emitted MJCF compiled normally -- the reference;
      (b) the base model + our writer + the `mj_setConst` call;
      (c) the base model + our writer, constants left stale.
    (b) must be the reference; (c) is what skipping the call costs, which is the
    number that justifies paying for it.

    This runs on CPU MuJoCo, so it also exercises `CPU_EXTRA_FIELDS` -- MuJoCo's
    compile-time body BVH, which is the whole of the residual if it is left
    stale (0.048 m) and which mujoco_warp does not have.
    """
    S = random_designs(n_designs, seed)
    theirs = query_their_dev(
        {"cases": [{"design": [S[k, a].tolist() for a in range(2)]}
                   for k in range(n_designs)], "fields": ["qpos0"]})["cases"]
    model, meta = build_dev_scene()
    spec = build_design_spec(model, meta)
    fields = design_fields(spec, torch.as_tensor(S))
    rng = np.random.default_rng(seed + 77)
    ctrl = rng.uniform(-0.5, 0.5, (steps, model.nu))

    def rollout(m):
        d = mujoco.MjData(m)
        d.qpos[:] = m.qpos0
        mujoco.mj_forward(m, d)
        xs = []
        for t in range(steps):
            d.ctrl[:] = ctrl[t]
            mujoco.mj_step(m, d, nstep=5)
            xs.append(d.qpos.copy())
        return np.array(xs)

    exact_drift, stale_drift, stale_rel = [], [], []
    for k, case in enumerate(theirs):
        ref = mujoco.MjModel.from_xml_string(case["xml"])
        ref.opt.solver, ref.opt.iterations = model.opt.solver, model.opt.iterations
        built = []
        for do_const in (True, False):
            m = mujoco.MjModel.from_xml_string(dev_run_to_goal_xml())
            # `CPU_EXTRA_FIELDS` (the body BVH) is written on the CPU path only;
            # mujoco_warp does not have the field. Include it here because this
            # check IS the CPU path.
            for f in WRITTEN_FIELDS + CPU_EXTRA_FIELDS:
                getattr(m, f)[:] = fields[f][k].numpy().reshape(
                    getattr(m, f).shape)
            if do_const:
                mujoco.mj_setConst(m, mujoco.MjData(m))
            built.append(m)
        q_ref = rollout(ref)
        exact_drift.append(np.abs(rollout(built[0]) - q_ref).max())
        stale_drift.append(np.abs(rollout(built[1]) - q_ref).max())
        stale_rel.append(
            np.abs(ref.dof_invweight0 - built[1].dof_invweight0).max()
            / max(np.abs(ref.dof_invweight0).max(), 1e-12))
    print(f"  --- whole-model rollout, {steps} control steps (0.6 s), "
          f"{n_designs} designs ---")
    print(f"    writer + mj_setConst |dqpos|   max {max(exact_drift):.3e}")
    print(f"    writer, stale constants        max {max(stale_drift):.3e}   "
          f"(dof_invweight0 up to {max(stale_rel) * 100:.0f}% off)")
    assert max(exact_drift) < 1e-9, (
        f"writer+mj_setConst diverges from a fresh compile: {max(exact_drift)}")
    return (f"identical to a fresh compile ({max(exact_drift):.1e}); skipping "
            f"mj_setConst would cost {max(stale_drift):.2e} over 0.6 s")


def t_designs_actually_bite():
    """A different genome has to give a different trajectory -- otherwise every
    check above could pass with a writer that quietly does nothing."""
    n = 2
    env = _cpu_dev_env(n, max_episode_steps=50)
    env.reset()
    S = np.zeros((n, 2, DESIGN_DIM))
    S[1] = 1.0                                   # world 1: every part maximal
    act = torch.zeros(n, 2, 28, dtype=env.dtype)
    act[..., :DESIGN_DIM] = torch.as_tensor(S, dtype=env.dtype)
    env.step(act)
    mass = env.backend.model_arrays["body_mass"]
    ratio = float(mass[1].sum() / mass[0].sum())
    a = torch.zeros(n, 2, 28, dtype=env.dtype)
    a[..., DESIGN_DIM:] = 0.5
    for _ in range(20):
        env.step(a)
    drift = float((env.qpos[0] - env.qpos[1]).abs().max())
    assert ratio > 1.3, f"max design is only {ratio:.3f}x the base mass"
    assert drift > 1e-3, "two very different ants moved identically"
    return f"max-design ant is {ratio:.2f}x heavier; qpos drift {drift:.3f} m"


# ---------------------------------------------------------------------------
# 6. GPU: batching + graph
# ---------------------------------------------------------------------------
def t_gpu_batched_model():
    from rower_soccer.competevo_port.backend import CompeteWarpDevBackend
    n = 8
    model, meta = build_dev_scene()
    be = CompeteWarpDevBackend(model, n, 5, batched_fields=BATCHED_FIELDS,
                               use_graph=True)
    for f in BATCHED_FIELDS:
        assert be.model_arrays[f].shape[0] == n, (f, be.model_arrays[f].shape)
    spec = build_design_spec(model, meta, device="cuda", dtype=torch.float32)
    writer = DesignWriter(spec, be.model_arrays, model=model)
    # Same start state everywhere; only the morphology differs.
    q0 = torch.as_tensor(np.asarray(model.qpos0), device="cuda",
                         dtype=be.qpos.dtype)
    be.qpos.copy_(q0.unsqueeze(0).expand_as(be.qpos))
    be.qvel.zero_()
    S = torch.zeros(n, meta.n_agents, DESIGN_DIM, device="cuda")
    S[n // 2:] = 1.0
    writer.write(torch.arange(n, device="cuda"), S)
    mass = be.model_arrays["body_mass"]
    assert float(mass[0].sum()) < float(mass[-1].sum()), "no per-world mass"
    be.forward()
    be.ctrl.copy_(torch.full_like(be.ctrl, 0.5))
    for _ in range(20):
        be.step()                     # replays the graph captured BEFORE the write
    q = be.qpos.detach().cpu().numpy()
    spread_same = np.abs(q[0] - q[1]).max()
    spread_diff = np.abs(q[0] - q[-1]).max()
    assert np.isfinite(q).all(), "NaN after a per-world design write"
    assert spread_same < 1e-6, "identical designs diverged"
    assert spread_diff > 1e-3, "different designs did not"
    return (f"{len(BATCHED_FIELDS)} fields batched [{n}, ...]; graph replay "
            f"after the write: identical designs agree to {spread_same:.1e}, "
            f"different ones separate by {spread_diff:.3f} m")


def t_gpu_env_smoke():
    env = RunToGoalDevEnv(num_worlds=64, use_gpu=True, max_episode_steps=30)
    obs = env.reset()
    assert obs.shape == (64, 2, 52)
    for _ in range(40):
        a = torch.rand(64, 2, 28, device=env.device) * 2 - 1
        obs, rew, done, info = env.step(a)
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
    return (f"{env.games} episodes closed, {env.n_diverged} diverged worlds, "
            f"reward |max| {float(rew.abs().max()):.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--designs", type=int, default=N_DESIGNS)
    p.add_argument("--cases", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()

    check("dev scene compiles to their exact MjModel", t_base_scene)
    if not os.path.exists(parity.THEIR_PYTHON):
        print(f"SKIP: CompetEvo venv missing at {parity.THEIR_PYTHON} -- "
              "the cross-stack design gate did NOT run")
    else:
        check("design -> model fields vs their compiler",
              lambda: t_design_parity(args.designs, args.seed))
        check("dev obs + reward parity under a design",
              lambda: t_state_parity(args.cases, args.seed))
        check("design step: reward 0, fresh state, flag flip", t_episode_shape)
        check("whole model, stepped: writer + mj_setConst == fresh compile",
              lambda: t_setconst_trajectory(min(args.designs, 6), args.seed))
    check("different designs -> different physics", t_designs_actually_bite)
    if args.gpu:
        check("mujoco_warp batches the model per world; graph survives",
              t_gpu_batched_model)
        check("batched dev env steps on GPU without NaNs", t_gpu_env_smoke)

    n_fail = sum(1 for _, ok in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
