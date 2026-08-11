"""Stage-0 parity gate: our batched run-to-goal env vs CompetEvo's CPU env.

Plain-python (no pytest in this venv):

    PYTHONPATH=. .venv/bin/python -m rower_soccer.competevo_port.tests.test_parity
    ... --cases 96 --seed 3        # more states
    ... --diverge                  # + the solver-divergence diagnostic

The gate has three parts, cheapest first:

  1. MODEL EQUIVALENCE -- our generated MJCF compiles to the same MjModel as
     their merged scene: every mass, inertia, joint range, damping, armature,
     friction, margin, gear, geom size, and the actuator->joint ORDER (which IS
     the action layout). Catches "the scene is subtly a different robot" before
     any state math runs.
  2. OBS + REWARD PARITY on hand-set states -- the real gate. See parity.py for
     why hand-set states and not a shared rollout.
  3. SOLVER DIVERGENCE -- printed, never asserted: how fast the two stacks
     separate under identical open-loop actions. That is the price of the
     PGS -> Newton deviation and no amount of porting removes it.

Parts 2 and 3 need CompetEvo's venv and are skipped (loudly) without it.
"""

import argparse
import os
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
from rower_soccer.competevo_port.backend import CompeteCpuBackend
from rower_soccer.competevo_port.run_to_goal_env import RunToGoalEnv
from rower_soccer.competevo_port.scene import (build_run_to_goal_scene,
                                               their_scene_path)

# Kinematic fields are the same algebra over the same doubles on both sides, so
# the bar is representation error, not tolerance shopping.
TOL = 1e-6
N_CASES = 48

MODEL_FIELDS = ("body_mass", "body_inertia", "body_ipos", "body_pos",
                "body_quat", "body_parentid", "dof_damping", "dof_armature",
                "dof_bodyid", "jnt_type", "jnt_range", "jnt_axis", "jnt_pos",
                "geom_type", "geom_size", "geom_pos", "geom_quat",
                "geom_friction", "geom_margin", "geom_condim", "geom_contype",
                "geom_conaffinity", "geom_solref", "geom_solimp",
                "actuator_gear", "actuator_ctrlrange", "actuator_trnid",
                "qpos0")

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


# ---------------------------------------------------------------------------
# 1. model equivalence
# ---------------------------------------------------------------------------
def t_model_matches_theirs():
    """Our builder vs their merged XML at THEIR physics options, so the option
    deviation is isolated from the geometry."""
    ours, meta = build_run_to_goal_scene(solver="PGS", iterations=1000)
    theirs = mujoco.MjModel.from_xml_path(their_scene_path())
    assert (ours.nq, ours.nv, ours.nu) == (theirs.nq, theirs.nv, theirs.nu)
    bad = []
    for f in MODEL_FIELDS:
        a = np.asarray(getattr(ours, f), dtype=np.float64)
        b = np.asarray(getattr(theirs, f), dtype=np.float64)
        if a.shape != b.shape or (a.size and np.abs(a - b).max() > 1e-12):
            bad.append(f)
    assert not bad, f"model fields differ from theirs: {bad}"
    order = lambda m: [m.jnt(m.actuator_trnid[i, 0]).name for i in range(m.nu)]
    assert order(ours) == order(theirs), "actuator order differs"
    assert [a.qpos for a in meta.agents] == [(0, 15), (15, 30)]
    assert [a.qvel for a in meta.agents] == [(0, 14), (14, 28)]
    assert [a.goal_x for a in meta.agents] == [4.0, -4.0]
    assert [a.move_left for a in meta.agents] == [False, True]
    return f"{len(MODEL_FIELDS)} model fields bit-equal, nq/nv/nu {ours.nq}/{ours.nv}/{ours.nu}"


# ---------------------------------------------------------------------------
# 2. obs + reward parity
# ---------------------------------------------------------------------------
def _cpu_env(n, **kw):
    return RunToGoalEnv(num_worlds=n, use_gpu=False,
                        backend_cls=CompeteCpuBackend, auto_reset=False, **kw)


def run_parity(n_cases, seed, contact=False, label=""):
    cases = parity.random_states(n_cases, seed=seed, contact=contact)
    theirs = parity.query_their_env({"cases": cases})
    env = _cpu_env(n_cases)
    rep = parity.compare(cases, theirs, parity.evaluate_ours(cases, env))
    print(f"  --- parity{label}: {n_cases} states, seed {seed}, "
          f"max contacts {rep['_their_ncon_max']} ---")
    for k in sorted(rep):
        v = rep[k]
        print(f"    {k:32s} {v:.3e}" if isinstance(v, float) and not k.startswith("_")
              else f"    {k:32s} {v}")
    return rep


def _assert_report(rep):
    worst = []
    for k, v in rep.items():
        if k.startswith("_"):
            continue
        limit = TOL if k.startswith(("obs/", "reward/")) else 0
        if v > limit:
            worst.append(f"{k}={v}")
    assert not worst, "over tolerance: " + ", ".join(worst)


def t_obs_and_reward_parity(n_cases, seed):
    rep = run_parity(n_cases, seed)
    _assert_report(rep)
    # Their contact cost is structurally zero: the scene declares no
    # acceleration-stage sensor, so mj_rnePostConstraint never runs and cfrc_ext
    # stays 0 -- measured, not assumed. If this ever fires,
    # RunToGoalEnv(contact_cost_from_cfrc=True) becomes the honest port.
    assert rep["_their_cfrc_ext_absmax"] == 0.0, "their cfrc_ext is NOT zero"
    return f"worst obs {max(v for k, v in rep.items() if k.startswith('obs/')):.2e}"


def t_contact_parity():
    """Same gate on states forced into floor contact -- the regime where the two
    solvers disagree most. State-derived quantities must still match, because
    constraint forces do not enter the observation."""
    rep = run_parity(24, 7, contact=True, label=" (in contact)")
    _assert_report(rep)
    return f"max contacts {rep['_their_ncon_max']}"


# ---------------------------------------------------------------------------
# 3. plumbing + diagnostics
# ---------------------------------------------------------------------------
def t_batched_step():
    env = RunToGoalEnv(num_worlds=4, use_gpu=False,
                       backend_cls=CompeteCpuBackend, max_episode_steps=5)
    obs = env.reset()
    assert obs.shape == (4, 2, 31), obs.shape
    for _ in range(6):
        obs, rew, done, info = env.step(torch.zeros(4, 2, 8, dtype=env.dtype))
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
    assert env.games >= 4, env.games
    return f"{env.games} episodes closed, ep_len {int(env.last_len[0])}"


def t_solver_divergence():
    env = _cpu_env(1)
    d = parity.rollout_divergence(env, steps=40, seed=1)
    marks = ", ".join(f"t={i + 1}: {d[i]:.2e}" for i in (0, 4, 19, len(d) - 1)
                      if i < len(d))
    print(f"    root-position |ours - theirs| over an open-loop rollout: {marks}")
    return "diagnostic only (PGS -> Newton)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cases", type=int, default=N_CASES)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--diverge", action="store_true",
                   help="also run the solver-divergence diagnostic (slow)")
    args = p.parse_args()

    check("scene compiles to their exact MjModel", t_model_matches_theirs)
    check("batched env resets, steps, auto-resets", t_batched_step)
    if not os.path.exists(parity.THEIR_PYTHON):
        print(f"SKIP: CompetEvo venv missing at {parity.THEIR_PYTHON} -- "
              "the cross-stack parity gate did NOT run")
    else:
        check("obs + reward parity vs their env",
              lambda: t_obs_and_reward_parity(args.cases, args.seed))
        check("obs + reward parity in contact", t_contact_parity)
        if args.diverge:
            check("solver divergence diagnostic", t_solver_divergence)

    n_fail = sum(1 for _, ok in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
