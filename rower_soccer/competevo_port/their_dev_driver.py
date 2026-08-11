"""JSON-over-stdio driver for CompetEvo's DEV env. RUNS IN THEIR VENV.

Sibling of `their_env_driver.py` (see that file for why the two stacks cannot
share a process). This one drives `run-to-goal-devants-v0`, whose morphology is
an action: each case carries a 20-dim scale vector per agent, and the driver
walks their real episode entry points --

    env.reset()                      # fresh agent objects, fresh base XML trees
    env.step([design0, design1])     # stage 'attribute_transform':
                                     #   set_design_params -> lxml mutation
                                     #   load_tmp_mujoco_env -> from_xml_string
                                     #   transit_execution

-- and then reports (a) the model their compiler produced, field by field, and
(b) the 52-dim observation and the reward/termination for an optional hand-set
state. (a) is what the GPU port's per-world field writer has to equal; it is the
only definition of "correct design" that does not involve trusting our own
reading of their string surgery.

Protocol:

  in : {"cases": [{"design": [[20], [20]],
                   "qpos_prev": [30], "qvel_prev": [28],      # optional
                   "qpos": [30], "qvel": [28],
                   "action": [[8], [8]]}, ...],
        "fields": ["body_mass", ...]}                          # optional
  out: {"ok": true, "cases": [{"model": {...}, "obs": [...], ...}], "meta": {...}}
"""

import json
import os
import sys

import numpy as np

sys.path.insert(0, "/workspace/competevo")

from types import SimpleNamespace

from competevo.evo_envs import MultiDevAgentEnv

# Everything the per-world writer touches, plus the derived constants it does
# NOT write (so the gate can measure what that costs).
DEFAULT_FIELDS = ("geom_size", "geom_pos", "geom_quat", "geom_rbound",
                  "geom_aabb", "body_pos", "body_quat", "body_mass",
                  "body_inertia", "body_ipos", "body_iquat",
                  "body_subtreemass", "actuator_gear", "qpos0",
                  "body_invweight0", "dof_invweight0", "actuator_acc0")


def build_env():
    """Their `run-to-goal-devants-v0` with the registration kwargs from
    `competevo/__init__.py:84-95`. `rundir` is an OUTPUT directory (the merged
    scene is regenerated and written there on construction), so it is pointed at
    a temp dir -- /workspace/competevo is a read-only reference for this port."""
    cfg = SimpleNamespace(use_parse_reward=True)
    return MultiDevAgentEnv(
        cfg=cfg,
        agent_names=["dev_ant", "dev_ant"],
        rundir=os.environ.get("COMPETEVO_TMPDIR", "/tmp"),
        rgb=[(0.98, 0.87, 0.67), (0.98, 0.87, 0.67)],
        init_pos=[(-1, 0, 0.75), (1, 0, 0.75)],
        ini_euler=[(0, 0, 0), (0, 0, 180)],
    )


def dump_model(model, fields):
    out = {}
    for f in fields:
        out[f] = np.asarray(getattr(model, f), dtype=np.float64).tolist()
    return out


def run_case(env, case, fields):
    design = [np.asarray(d, dtype=np.float64) for d in case["design"]]
    env.reset()
    # Their step-0 action is [design(20) | motor(8)]; `set_design_params` slices
    # the head and `step` (execution) slices the tail, so the padding is real.
    actions = [np.concatenate([d, np.zeros(8)]) for d in design]
    obs0, rew0, term0, trunc0, infos0 = env.step(actions)
    model = env.env_scene.model
    out = {
        "model": dump_model(model, fields),
        # The merged MJCF their string surgery produced for this genome.
        # Compiling THIS with our mujoco is the primary gate: it compares our
        # writer against the compiler, with the mujoco 2.3.5 -> 3.11 version
        # delta (which shows up in `geom_aabb`) factored out.
        "xml": env._env_xml_str,
        # The obs their policy sees AFTER the design step: stage flag now 1
        # ('execution'), scale block now the design that was just applied.
        "obs_after_design": [np.concatenate([np.asarray(x).ravel()
                                             for x in o]).tolist()
                             for o in obs0],
        "reward_after_design": [float(r) for r in rew0],
        "terminated_after_design": [bool(t) for t in term0],
        # Their fresh MjData after the rebuild: no reset noise survives it.
        "qpos_after_design": env.env_scene.data.qpos.tolist(),
        "qvel_after_design": env.env_scene.data.qvel.tolist(),
    }

    if "qpos" not in case:
        return out

    # Same door as the fixed-morph gate: latch the COM at the prev state, set the
    # current state in place of simulate(), then call their real after_step.
    qpos_prev = np.asarray(case["qpos_prev"], dtype=np.float64)
    qvel_prev = np.asarray(case["qvel_prev"], dtype=np.float64)
    qpos = np.asarray(case["qpos"], dtype=np.float64)
    qvel = np.asarray(case["qvel"], dtype=np.float64)
    acts = [np.asarray(a, dtype=np.float64) for a in case["action"]]

    env.set_state(qpos_prev, qvel_prev)
    for i in range(env.n_agents):
        env.agents[i].before_step()
    env.set_state(qpos, qvel)

    move_rews, dones, infos = [], [], []
    for i in range(env.n_agents):
        r, done, rinfo = env.agents[i].after_step(acts[i])
        move_rews.append(float(r))
        dones.append(bool(done))
        infos.append({k: float(v) for k, v in rinfo.items()})
    goal_rews, game_done = env.goal_rewards(infos=infos, agent_dones=dones)
    terminateds = env._get_done(dones, game_done)
    obs = env._get_obs()

    out.update({
        "obs": [np.concatenate([np.asarray(x).ravel() for x in o]).tolist()
                for o in obs],
        "subtree_com_x": [float(env.agents[i].get_body_com("0")[0])
                          for i in range(env.n_agents)],
        "reward_dense": move_rews,
        "reward_parse": [float(g) for g in goal_rews],
        "reward_total": [float(goal_rews[i] + env.move_reward_weight
                               * move_rews[i]) for i in range(env.n_agents)],
        "reward_forward": [infos[i]["reward_forward"] for i in range(2)],
        "reward_ctrl": [infos[i]["reward_ctrl"] for i in range(2)],
        "reward_contact": [infos[i]["reward_contact"] for i in range(2)],
        "agent_done": dones,
        "reached_goal": [bool(env.agents[i].reached_goal())
                         for i in range(env.n_agents)],
        "winner": [bool("winner" in infos[i]) for i in range(env.n_agents)],
        "terminated": [bool(t) for t in terminateds],
        "cfrc_ext_absmax": [float(np.abs(env.agents[i].get_cfrc_ext()).max())
                            for i in range(env.n_agents)],
        "ncon": int(env.env_scene.data.ncon),
    })
    return out


def run_resets(env, n):
    """Their post-reset state and obs BEFORE the design step, `n` draws. The
    interesting part is what happens to it: `load_tmp_mujoco_env` builds a fresh
    MjData, so the reset noise never reaches the simulator."""
    out = []
    for _ in range(n):
        obs, _ = env.reset()
        out.append({
            "qpos": env.env_scene.data.qpos.tolist(),
            "qvel": env.env_scene.data.qvel.tolist(),
            "obs": [np.concatenate([np.asarray(x).ravel() for x in o]).tolist()
                    for o in obs],
        })
    return out


def main():
    req = json.load(sys.stdin)
    fields = tuple(req.get("fields", DEFAULT_FIELDS))
    env = build_env()
    out = {"ok": True,
           "meta": {"nq": int(env.env_scene.model.nq),
                    "nv": int(env.env_scene.model.nv),
                    "nu": int(env.env_scene.model.nu),
                    "dt": float(env.dt),
                    "state_dim": int(env.agents[0].state_dim),
                    "action_dim": int(env.agents[0].action_dim),
                    "scale_max": float(
                        __import__("competevo.evo_envs.agents.dev_ant",
                                   fromlist=["SCALE_MAX"]).SCALE_MAX),
                    "goal_x": [float(env.agents[i].GOAL) for i in range(2)],
                    "move_left": [bool(env.agents[i].move_left)
                                  for i in range(2)],
                    "body_names": [env.env_scene.model.body(i).name
                                   for i in range(env.env_scene.model.nbody)],
                    "geom_names": [env.env_scene.model.geom(i).name
                                   for i in range(env.env_scene.model.ngeom)]}}
    if "resets" in req:
        out["resets"] = run_resets(env, int(req["resets"]))
    if "cases" in req:
        out["cases"] = [run_case(env, c, fields) for c in req["cases"]]
    sys.stdout.write("\n@@JSON@@" + json.dumps(out))


if __name__ == "__main__":
    main()
