"""JSON-over-stdio driver for CompetEvo's CPU env. RUNS IN THEIR VENV, NOT OURS.

The two stacks cannot share a process: theirs is Python 3.8 + mujoco 2.3.5 +
gymnasium 0.28, ours is Python 3.11 + mujoco 3.11 + warp. So the parity gate
spawns `/workspace/competevo/.venv/bin/python -m ... their_env_driver`, hands it
a batch of hand-set states on stdin, and reads their env's own obs/reward numbers
back on stdout. This file is the only code in the port that imports gym_compete;
it is executed, never imported, by our side.

Protocol (one JSON object in, one JSON object out):

  in : {"cases": [{"qpos_prev": [30], "qvel_prev": [28],
                   "qpos": [30], "qvel": [28], "action": [[8], [8]]}, ...]}
  out: {"ok": true, "cases": [{...per-agent fields...}], "meta": {...}}

Why hand-set STATES and not a rollout: the port changes the solver (PGS is not
implemented in mujoco_warp), so stepped trajectories are guaranteed to diverge
and would measure the integrator, not the port. Instead each case pins the state
the reward is computed FROM (`qpos_prev`, via `before_step()`, which latches the
torso subtree-COM) and the state it is computed AT, then calls their real
`after_step`/`goal_rewards` with no simulate() in between. Every term of their
reward except the contact cost is then a pure function of the two hand-set states
and the action, and can be matched exactly.
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/workspace/competevo")

from types import SimpleNamespace

from gym_compete.new_envs import MultiAgentEnv


def build_env():
    """Their `run-to-goal-ants-v0`, constructed with the exact registration
    kwargs from `gym_compete/__init__.py:96-108` (bypassing gym.make, which only
    adds gymnasium's OrderEnforcing/TimeLimit wrappers around the same object)
    and the exact cfg flags from `config/run-to-goal-ants-v0.yaml`."""
    cfg = SimpleNamespace(use_parse_reward=True)
    # `scene_xml_path` is an OUTPUT path: MultiAgentEnv.__init__:114-119 always
    # regenerates the merged scene and writes it there (their registration aims
    # it at their own assets dir, so simply constructing the env dirties their
    # checkout with freshly randomized `anon<int>` body names). Point it at a
    # temp file: /workspace/competevo is a read-only reference for this port, and
    # a their-code sanity run is live in that tree.
    return MultiAgentEnv(
        cfg=cfg,
        agent_names=["ant", "ant"],
        scene_xml_path=os.path.join(
            os.environ.get("COMPETEVO_TMPDIR", "/tmp"),
            "parity_world_body.ant_body.ant_body.xml"),
        rgb=[(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
        init_pos=[(-1, 0, 0.75), (1, 0, 0.75)],
        ini_euler=[(0, 0, 0), (0, 0, 180)],
    )


def run_case(env, case):
    qpos_prev = np.asarray(case["qpos_prev"], dtype=np.float64)
    qvel_prev = np.asarray(case["qvel_prev"], dtype=np.float64)
    qpos = np.asarray(case["qpos"], dtype=np.float64)
    qvel = np.asarray(case["qvel"], dtype=np.float64)
    actions = [np.asarray(a, dtype=np.float64) for a in case["action"]]

    # 1. the "before" state: this is what latches each agent's _xposbefore.
    env.set_state(qpos_prev, qvel_prev)
    for i in range(env.n_agents):
        env.agents[i].before_step()
    xposbefore = [float(env.agents[i]._xposbefore) for i in range(env.n_agents)]

    # 2. the "after" state, in place of env_scene.simulate(actions). set_state
    #    runs mj_forward, so xpos/subtree_com/cfrc_ext are all consistent with it.
    env.set_state(qpos, qvel)

    move_rews, dones, infos = [], [], []
    for i in range(env.n_agents):
        r, done, rinfo = env.agents[i].after_step(actions[i])
        move_rews.append(float(r))
        dones.append(bool(done))
        infos.append({k: float(v) for k, v in rinfo.items()})
    goal_rews, game_done = env.goal_rewards(infos=infos, agent_dones=dones)
    terminateds = env._get_done(dones, game_done)
    obs = env._get_obs()

    out = {
        "obs": [np.asarray(o, dtype=np.float64).tolist() for o in obs],
        "xposbefore": xposbefore,
        "subtree_com_x": [float(env.agents[i].get_body_com("torso")[0])
                          for i in range(env.n_agents)],
        "reward_dense": move_rews,
        "reward_parse": [float(g) for g in goal_rews],
        "reward_total": [float(goal_rews[i] + env.move_reward_weight * move_rews[i])
                         for i in range(env.n_agents)],
        "reward_forward": [infos[i]["reward_forward"] for i in range(env.n_agents)],
        "reward_ctrl": [infos[i]["reward_ctrl"] for i in range(env.n_agents)],
        "reward_contact": [infos[i]["reward_contact"] for i in range(env.n_agents)],
        "reward_survive": [infos[i]["reward_survive"] for i in range(env.n_agents)],
        "agent_done": dones,
        "reached_goal": [bool(env.agents[i].reached_goal())
                         for i in range(env.n_agents)],
        "winner": [bool("winner" in infos[i]) for i in range(env.n_agents)],
        "game_done": bool(game_done),
        "terminated": [bool(t) for t in terminateds],
        "cfrc_ext_absmax": [float(np.abs(env.agents[i].get_cfrc_ext()).max())
                            for i in range(env.n_agents)],
        "ncon": int(env.env_scene.data.ncon),
    }
    return out


def run_rollout(env, spec):
    """Open-loop reference trajectory: set a state, then step their env with a
    fixed action sequence. Used only as a DIAGNOSTIC of solver divergence
    (RK4/PGS vs RK4/Newton), never as a gate."""
    qpos = np.asarray(spec["qpos"], dtype=np.float64)
    qvel = np.asarray(spec["qvel"], dtype=np.float64)
    env.set_state(qpos, qvel)
    for i in range(env.n_agents):
        env.agents[i].reset_agent()
    traj = []
    for a in spec["actions"]:
        acts = [np.asarray(x, dtype=np.float64) for x in a]
        obs, rews, term, trunc, infos = env._step(acts)
        traj.append({"qpos": env.env_scene.data.qpos.tolist(),
                     "reward": [float(r) for r in rews],
                     "terminated": [bool(t) for t in term]})
        if any(term):
            break
    return traj


def run_resets(env, n):
    """Their post-reset state, `n` draws. Their `_reset` randomizes qpos twice
    (once in `env_scene.reset()`, again inside `reset_model()`) and then
    `set_xyz` zeroes qvel, so only the last draw survives -- this samples the
    distribution our `reset_idx` has to reproduce."""
    out = []
    for _ in range(n):
        env.reset()
        out.append({"qpos": env.env_scene.data.qpos.tolist(),
                    "qvel": env.env_scene.data.qvel.tolist()})
    return out


def main():
    req = json.load(sys.stdin)
    env = build_env()
    out = {"ok": True,
           "meta": {"nq": int(env.env_scene.model.nq),
                    "nv": int(env.env_scene.model.nv),
                    "nu": int(env.env_scene.model.nu),
                    "dt": float(env.dt),
                    "goal_x": [float(env.agents[i].GOAL)
                               for i in range(env.n_agents)],
                    "move_left": [bool(env.agents[i].move_left)
                                  for i in range(env.n_agents)],
                    "qpos0": env.env_scene.init_qpos.tolist()}}
    if "cases" in req:
        out["cases"] = [run_case(env, c) for c in req["cases"]]
    if "rollout" in req:
        out["rollout"] = run_rollout(env, req["rollout"])
    if "resets" in req:
        out["resets"] = run_resets(env, int(req["resets"]))
    # Their env prints during construction; keep stdout clean by writing the
    # payload with a sentinel the caller splits on.
    sys.stdout.write("\n@@JSON@@" + json.dumps(out))


if __name__ == "__main__":
    main()
