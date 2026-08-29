"""D3 M3 E0 gate: is Transform2Act's OWN ant task runnable, unmodified?

E0 starts the search from a competent quadruped instead of a 3-segment line.
Two implementations could host it and they are not interchangeable:

  * **our GPU port** (`train_t2a.py`) -- validated on hopper only. It opens
    `assets/mujoco_envs/hopper.xml` at `train_t2a.py:295`, sets
    `sim_obs_dim = 5` at `train_t2a.py:297`, and `batched_exec_env.sim_obs`
    builds exactly five columns from the PLANAR root layout
    `(qpos[1], qpos[2]) = (height, ang)`. Their `ant.py:197-220` builds
    THIRTEEN (`qpos[2:7], qvel[:6], zeros(2)` for a free root), and
    `ant.py:167-176` takes the tilt from a quaternion rather than a hinge.
    `DesignSpec.index_base` is `max_nchild + 1 = 3` on `ant.yml`; their
    `ant.py:32` hardcodes 5. The port has no ant path -- not a flag, a port.
  * **their CPU reference**, which ships `ant.yml` + `ant.py` + `ant.xml`.

This gate is the evidence for choosing the second: it drives their ant end to
end and checks the body it starts from is the ant in `ant.xml`, that the
observation dimensions are the ant's and not the hopper's, and that an
untrained policy can push a design through both design stages into execution
without the XML round-trip failing.

    cd /workspace/Transform2Act && source env-gpu.sh
    .venv-gpu/bin/python \
        /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/gate_their_ant.py

CPU only: it opens no CUDA context, so it is safe to run beside live MPS
clients.
"""

import os
import sys

sys.path.append("/workspace/Transform2Act")
os.chdir("/workspace/Transform2Act")

import numpy as np  # noqa: E402
import torch  # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{'  ' + detail if detail else ''}")
    if not ok:
        FAILURES.append(name)


def main():
    torch.set_default_dtype(torch.float64)
    from design_opt.utils.config import Config
    from design_opt.envs import env_dict
    from design_opt.models.transform2act_policy import Transform2ActPolicy

    cfg = Config("ant", tmp=True)
    np.random.seed(0)
    torch.manual_seed(0)
    env = env_dict[cfg.env_name](cfg, None)

    print("\n1. the starting body is the ant in assets/mujoco_envs/ant.xml")
    check("5 robot bodies (torso + 4 single-segment limbs)",
          [b.name for b in env.robot.bodies] == ["0", "1", "2", "3", "4"],
          str([b.name for b in env.robot.bodies]))
    check("4 motors", env.model.nu == 4, f"nu={env.model.nu}")
    # njnt is 5, not the 6 PLAN_D3_M3 section 0c records: a free root plus four
    # hinges. The free root carries 7 qpos / 6 qvel, hence nq 11 / nv 10.
    check("free root + 4 hinges", env.model.njnt == 5
          and env.model.nq == 11 and env.model.nv == 10,
          f"njnt={env.model.njnt} nq={env.model.nq} nv={env.model.nv}")
    check("limbs radiate in the xy plane (a quadruped, not a planar walker)",
          cfg.robot_cfg["body_params"]["offset"]["type"] == "xy")

    print("\n2. observation dimensions are the ANT's, not the hopper's")
    check("sim_obs_dim == 13 (hopper's is 5)", env.sim_obs_dim == 13,
          f"got {env.sim_obs_dim}")
    check("attr_fixed_dim == max_body_depth == 4", env.attr_fixed_dim == 4,
          f"got {env.attr_fixed_dim}")
    check("attr_design_dim == 5 (offset_x, offset_y, gear, size, ext_start)",
          env.attr_design_dim == 5, f"got {env.attr_design_dim}")
    check("skel_num_action == 3 (enable_remove defaults true)",
          env.skel_num_action == 3)

    print("\n3. an untrained policy drives design -> design -> execution")
    policy = Transform2ActPolicy(cfg.policy_specs, env)
    policy.eval()

    # `transform2act_agent.tensorfy`: a batch of states is a list of lists.
    def tensorfy(np_list):
        if isinstance(np_list[0], list):
            return [[torch.tensor(x) for x in y] for y in np_list]
        return [torch.tensor(y) for y in np_list]

    reached_exec = 0
    n_bodies_seen = set()
    exec_rewards = []
    ends = []
    n_eps = 20
    for ep in range(n_eps):
        state = env.reset()
        stages = []
        for t in range(cfg.skel_transform_nsteps + 1 + 50):
            with torch.no_grad():
                action = policy.select_action(
                    tensorfy([state]), False).numpy().astype(np.float64)
            state, reward, done, info = env.step(action)
            stages.append(info["stage"])
            if info["stage"] == "execution":
                exec_rewards.append(reward)
            if done:
                ends.append(t)
                break
        if "execution" in stages:
            reached_exec += 1
        n_bodies_seen.add(len(env.robot.bodies))

    check(f"all {n_eps} sampled designs reached execution",
          reached_exec == n_eps, f"{reached_exec}/{n_eps}")
    check("the skeleton stage actually edits the body (untrained policy)",
          len(n_bodies_seen) > 1, f"body counts seen: {sorted(n_bodies_seen)}")
    check("execution rewards are finite",
          len(exec_rewards) > 0 and np.isfinite(exec_rewards).all(),
          f"{len(exec_rewards)} steps, mean {np.mean(exec_rewards):.3f}")

    print("\n4. the execution reward is ant.py's, recomputed independently")
    # ant.py:159-165: (x_after - x_before)/dt - 1e-4 * mean(ctrl^2) + 0.0
    env.reset()
    for _ in range(cfg.skel_transform_nsteps + 1):
        with torch.no_grad():
            a = policy.select_action(tensorfy([env._get_obs()]),
                                     True).numpy().astype(np.float64)
        env.step(a)
    with torch.no_grad():
        a = policy.select_action(tensorfy([env._get_obs()]),
                                 True).numpy().astype(np.float64)
    xb = env.get_body_com("0")[0]
    ctrl = env.action_to_control(a[:, :env.control_action_dim])
    _, reward, _, info = env.step(a)
    xa = env.get_body_com("0")[0]
    expect = (xa - xb) / env.dt - 1e-4 * np.square(ctrl).mean()
    check("reward == forward + ctrl cost, no alive bonus",
          info["stage"] == "execution" and abs(reward - expect) < 1e-9,
          f"reward {reward:.6f} vs recomputed {expect:.6f}")

    print("\n" + ("GATE PASSED" if not FAILURES
                  else f"GATE FAILED: {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
