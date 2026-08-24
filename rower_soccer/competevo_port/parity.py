"""Cross-stack parity harness: our batched env vs CompetEvo's CPU env.

Their stack (python 3.8 / mujoco 2.3.5 / gymnasium 0.28) and ours (3.11 / mujoco
3.11 / warp) cannot share a process, so the comparison is JSON over a subprocess:
we generate hand-set states here, `their_env_driver.py` evaluates them inside
`/workspace/competevo/.venv`, and we diff field by field.

WHY HAND-SET STATES rather than a shared rollout: mujoco_warp does not implement
PGS, so the ported scene solves contacts with Newton (scene.py's docstring). Two
different solvers cannot produce equal trajectories, and a trajectory comparison
would grade the solver swap instead of the port. Every quantity the policy and
the learner actually see -- the 31-dim observation, its ordering, the dense and
sparse rewards, the termination and win predicates -- is a function of state, so
pinning the state isolates the port. `rollout_divergence()` measures the solver
delta separately, as a diagnostic.
"""

import json
import os
import subprocess
import sys

import numpy as np
import torch

THEIR_PYTHON = "/workspace/competevo/.venv/bin/python"
DRIVER = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                      "their_env_driver.py")

# Obs field groups, in their order (Ant._get_obs): the parity report is per
# group because they have different provenance -- root pose comes straight out
# of qpos, joint velocities out of the integrator, opponent xy out of the other
# agent's slice, and a mis-slice shows up as one group failing, not all of them.
OBS_GROUPS = (("own_root_pos", slice(0, 3)),
              ("own_root_quat", slice(3, 7)),
              ("own_joint_pos", slice(7, 15)),
              ("own_root_linvel", slice(15, 18)),
              ("own_root_angvel", slice(18, 21)),
              ("own_joint_vel", slice(21, 29)),
              ("opponent_root_xy", slice(29, 31)))


def random_states(n, seed=0, nq=30, nv=28, n_agents=2, contact=False):
    """`n` hand-set (prev, current) state pairs plus actions.

    Spread across the regimes the reward branches on: airborne, standing,
    fallen (root z below 0.28), and past a goal line -- a state set that never
    crosses x=+/-4 would leave the +/-1000 sparse reward untested.
    """
    rng = np.random.default_rng(seed)
    cases = []
    for k in range(n):
        qpos = np.zeros(nq)
        qvel = rng.uniform(-2.0, 2.0, nv)
        mode = k % 4
        for a in range(n_agents):
            o = a * (nq // n_agents)
            x0 = -1.0 if a == 0 else 1.0
            # mode 3 puts agent 0 across its goal line at x=+4 so the +/-1000
            # sparse reward and the winner flag are actually exercised.
            x = (4.2 if a == 0 else 1.0) if mode == 3 else x0 + rng.uniform(-1.5, 1.5)
            z = {0: rng.uniform(0.5, 1.0),     # airborne / tall
                 1: rng.uniform(0.28, 0.6),    # standing
                 2: rng.uniform(0.10, 0.30),   # near / past the fall threshold
                 3: rng.uniform(0.3, 0.7)}[mode]
            if contact:                        # force geoms into the floor
                z = min(z, 0.30)
            qpos[o:o + 3] = [x, rng.uniform(-2.0, 2.0), z]
            q = rng.normal(size=4)
            qpos[o + 3:o + 7] = q / np.linalg.norm(q)
            qpos[o + 7:o + 15] = rng.uniform(-0.6, 0.6, 8)
        # The "before" state differs by a plausible one-control-step delta, so
        # the forward-progress term is exercised with a realistic magnitude.
        qpos_prev = qpos + rng.uniform(-0.02, 0.02, nq)
        cases.append({
            "qpos_prev": qpos_prev.tolist(),
            "qvel_prev": (qvel + rng.uniform(-0.1, 0.1, nv)).tolist(),
            "qpos": qpos.tolist(), "qvel": qvel.tolist(),
            "action": [rng.uniform(-1.0, 1.0, 8).tolist()
                       for _ in range(n_agents)],
        })
    return cases



def _decode_driver_reply(proc, name):
    """Their driver's JSON, tolerating a crash that happens AFTER it is written.

    On the 2026-08-24 pod rebuild their venv started aborting at interpreter
    teardown -- `free(): invalid pointer`, SIGABRT, returncode -6 -- with the
    complete reply already on stdout. That is a C-level double-free somewhere
    in the mujoco/gymnasium/torch teardown on this box, not a parity failure,
    and treating it as one silently disables four cross-checks against their
    code, which is the last thing this file should do.

    So the contract is "a COMPLETE, PARSEABLE reply", not "exit 0":

      * no marker, or JSON that does not parse -> failure, as before. That is
        what a crash MID-computation looks like, and it is not tolerated.
      * a complete reply plus death by signal -> accepted, and WARNED about on
        stderr every single time, because a silently tolerated crash is how a
        real failure hides.
      * a complete reply plus a positive exit code -> failure. Their driver
        exits 0 on success, so a nonzero one means it decided something went
        wrong, and it is entitled to be believed.
    """
    tail = (f"stdout tail:\n{proc.stdout[-2000:]}\n"
            f"stderr tail:\n{proc.stderr[-2000:]}")
    if "@@JSON@@" not in proc.stdout:
        raise RuntimeError(f"{name} failed (no reply):\n{tail}")
    try:
        reply = json.loads(proc.stdout.split("@@JSON@@", 1)[1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{name} reply is truncated ({exc}):\n{tail}")
    if proc.returncode > 0:
        raise RuntimeError(f"{name} exited {proc.returncode}:\n{tail}")
    if proc.returncode < 0:
        print(f"  WARNING: {name} produced a complete reply and then died with "
              f"signal {-proc.returncode} at teardown. Reply used; the crash "
              f"is an environment artefact, not a parity result.",
              file=sys.stderr)
    return reply


def query_their_env(payload, timeout=600):
    """Run the driver in their venv and return its JSON reply."""
    proc = subprocess.run([THEIR_PYTHON, DRIVER], input=json.dumps(payload),
                          capture_output=True, text=True, timeout=timeout,
                          cwd="/workspace/competevo")
    return _decode_driver_reply(proc, "their_env_driver")


def evaluate_ours(cases, env):
    """Drive OUR env through the same door: latch the COM at the prev state,
    hand-set the current state, then call the production `terms()`."""
    n = len(cases)
    assert env.n == n, "one world per case"
    dev, dt = env.device, env.dtype
    to = lambda x: torch.as_tensor(np.asarray(x), device=dev, dtype=dt)

    env.qpos.copy_(to([c["qpos_prev"] for c in cases]))
    env.qvel.copy_(to([c["qvel_prev"] for c in cases]))
    env.backend.forward()
    env._com_before = env._agent_com_x().clone()

    env.qpos.copy_(to([c["qpos"] for c in cases]))
    env.qvel.copy_(to([c["qvel"] for c in cases]))
    env.backend.forward()

    a = to([c["action"] for c in cases])
    t = env.terms(a)
    t["obs"] = env.obs()
    t["subtree_com_x"] = env._agent_com_x()
    t["com_before"] = env._com_before
    return t


def compare(cases, theirs, ours):
    """Field-by-field max |ours - theirs|, grouped for a readable report."""
    npy = lambda x: x.detach().double().cpu().numpy()
    T = theirs["cases"]
    rep = {}

    their_obs = np.array([c["obs"] for c in T])          # [n, A, 31]
    our_obs = npy(ours["obs"])
    for name, sl in OBS_GROUPS:
        rep[f"obs/{name}"] = float(np.abs(our_obs[..., sl]
                                          - their_obs[..., sl]).max())
    rep["obs/ALL"] = float(np.abs(our_obs - their_obs).max())

    for key, ours_key in (("subtree_com_x", "subtree_com_x"),
                          ("reward_forward", "forward"),
                          ("reward_ctrl", "ctrl_cost"),
                          ("reward_contact", "contact_cost"),
                          ("reward_dense", "dense"),
                          ("reward_parse", "parse"),
                          ("reward_total", "reward")):
        t = np.array([c[key] for c in T])
        rep[f"reward/{key}"] = float(np.abs(npy(ours[ours_key]) - t).max())

    for key, ours_key in (("reached_goal", "reached"), ("winner", "winner")):
        t = np.array([c[key] for c in T])
        rep[f"flag/{key}"] = int((npy(ours[ours_key]).astype(bool) != t).sum())
    rep["flag/agent_fell"] = int(
        (npy(ours["fell"]).astype(bool)
         != np.array([c["agent_done"] for c in T])).sum())
    rep["flag/terminated"] = int(
        (npy(ours["terminated"]).astype(bool)
         != np.array([c["terminated"][0] for c in T])).sum())
    rep["_their_cfrc_ext_absmax"] = float(
        max(max(c["cfrc_ext_absmax"]) for c in T))
    rep["_their_ncon_max"] = int(max(c["ncon"] for c in T))
    # Coverage: a parity report over states that never cross a goal line or
    # never fall would agree perfectly on branches neither side took. Report how
    # many cases actually exercised each branch, so "0 mismatches" means
    # something.
    rep["_cases"] = len(T)
    rep["_cases_with_goal_crossed"] = int(
        sum(any(c["reached_goal"]) for c in T))
    rep["_cases_with_a_fall"] = int(sum(any(c["agent_done"]) for c in T))
    rep["_cases_terminated"] = int(sum(c["terminated"][0] for c in T))
    rep["_cases_in_contact"] = int(sum(c["ncon"] > 0 for c in T))
    return rep


def rollout_divergence(env, steps=50, seed=0, actions=None):
    """DIAGNOSTIC (not a gate): how far the two stacks drift under identical
    open-loop actions from an identical start state. Measures the PGS->Newton
    solver swap, which no amount of porting can remove."""
    rng = np.random.default_rng(seed)
    qpos = np.array(env.model.qpos0, dtype=np.float64)
    qpos[[0, 1, 15, 16]] += rng.uniform(-0.05, 0.05, 4)
    qvel = np.zeros(env.meta.nv)
    if actions is None:
        actions = rng.uniform(-0.5, 0.5, (steps, 2, 8))
    theirs = query_their_env({"rollout": {"qpos": qpos.tolist(),
                                          "qvel": qvel.tolist(),
                                          "actions": actions.tolist()}})
    from rower_soccer.competevo_port.run_to_goal_env import set_state
    set_state(env, qpos, qvel)
    env.ep_step.zero_()
    ours = []
    for t in range(len(theirs["rollout"])):
        a = torch.as_tensor(actions[t], device=env.device,
                            dtype=env.dtype).unsqueeze(0).expand(env.n, -1, -1)
        env.step(a)
        ours.append(env.qpos[0].detach().double().cpu().numpy().copy())
    their_q = np.array([s["qpos"] for s in theirs["rollout"]])
    ours = np.array(ours)
    # Root-position drift of agent 0, per step.
    return np.abs(ours[:, :3] - their_q[:, :3]).max(-1)
