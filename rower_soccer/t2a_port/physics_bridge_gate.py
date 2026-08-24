"""Do THEIR physics and OUR physics agree on the same morphology?

D3 3d step 3 is the batched execution env. Everything in it rests on an
assumption nobody has checked: Transform2Act simulates with **mujoco-py
2.1.2.14 against the mujoco210 binary**, and the batched port simulates with the
**modern `mujoco` bindings** under `mujoco_warp`. Two different engines, two
different Pythons, two different venvs.

If they disagree, every downstream number in the port is measuring a different
robot from the one their published curves describe -- and it would show up as
"the port trains to a lower reward", which is exactly the shape of the M1 gap we
are already chasing. So this runs first, before any batched env exists.

Two phases, because the two stacks cannot coexist in one interpreter:

    # 1. in THEIR venv: export the morphology and a reference trajectory
    cd /workspace/Transform2Act && source env-gpu.sh
    .venv-gpu/bin/python .../t2a_port/physics_bridge_gate.py --emit \
        --cfg hopper_gpu --checkpoint latest --steps 300

    # 2. in OURS: replay the SAME actions from the SAME initial state
    PYTHONPATH=. .venv/bin/python .../t2a_port/physics_bridge_gate.py --check

The actions are recorded, not re-derived, so the comparison is of the
integrator and the model compiler alone -- no policy, no observation pipeline,
nothing that could hide a physics difference behind a matching action.

What "agree" means here is deliberately not "bit-identical": these are different
MuJoCo builds and divergence is expected to grow. The gate reports the step at
which the trajectories separate by more than a threshold, because for a
1,000-step episode what matters is whether they stay together for the episode,
not whether they agree to machine precision at step 1.
"""

import argparse
import json
import os
import sys

import numpy as np

OUT = "/tmp/claude-0/-root/453bc0de-a27f-4894-ad03-7d048158ee36/scratchpad/t2a_bridge.json"


def emit(args):
    sys.path.append("/workspace/Transform2Act")
    os.chdir("/workspace/Transform2Act")
    import glob
    import re

    import torch
    from design_opt.agents.transform2act_agent import (Transform2ActAgent,
                                                       tensorfy)
    from design_opt.utils.config import Config

    torch.set_default_dtype(torch.float64)
    cfg = Config(args.cfg, tmp=False)
    ckpt = args.checkpoint
    if ckpt == "latest":
        eps = sorted(int(m.group(1)) for m in
                     (re.search(r"epoch_(\d+)\.p$", f)
                      for f in glob.glob(os.path.join(cfg.model_dir, "epoch_*.p")))
                     if m)
        ckpt = eps[-1] if eps else "best"
    elif ckpt != "best":
        ckpt = int(ckpt)

    agent = Transform2ActAgent(cfg=cfg, dtype=torch.float64,
                               device=torch.device("cpu"), seed=cfg.seed,
                               num_threads=1, training=False, checkpoint=ckpt)
    env, policy = agent.env, agent.policy_net
    policy.eval()

    # Run the design stages so the morphology is the trained one, then stop at
    # the first execution step -- that is the state the batched env would take
    # over from.
    state = env.reset()
    with torch.no_grad():
        while True:
            action = policy.select_action(tensorfy([state]), True).numpy().astype(np.float64)
            info = env.step(action)[3]
            if info.get("stage") == "execution":
                break
            state = env.step.__self__.cur_state if False else state
            state = env._get_obs()

    xml = env.cur_xml_str
    qpos0 = env.sim.data.qpos.copy()
    qvel0 = env.sim.data.qvel.copy()
    nu = env.sim.model.nu

    # Recorded actions, not a policy: the check is of physics alone. A fixed
    # pseudo-random sequence excites more of the dynamics than zeros would.
    rng = np.random.default_rng(0)
    ctrls = rng.uniform(-0.3, 0.3, size=(args.steps, nu))

    qpos_traj = []
    for t in range(args.steps):
        env.sim.data.ctrl[:] = ctrls[t]
        for _ in range(env.frame_skip):
            env.sim.step()
        qpos_traj.append(env.sim.data.qpos.copy().tolist())

    blob = {
        "cfg": args.cfg, "checkpoint": str(ckpt),
        "xml": xml, "qpos0": qpos0.tolist(), "qvel0": qvel0.tolist(),
        "frame_skip": int(env.frame_skip), "nu": int(nu),
        "ctrls": ctrls.tolist(), "qpos_traj": qpos_traj,
        "nq": int(env.sim.model.nq), "nv": int(env.sim.model.nv),
        "timestep": float(env.sim.model.opt.timestep),
        # The COMPILED model, so the conversion can be checked against what
        # their compiler produced rather than only against a trajectory.
        "body_pos": env.sim.model.body_pos.tolist(),
        "geom_pos": env.sim.model.geom_pos.tolist(),
        "geom_size": env.sim.model.geom_size.tolist(),
        "body_mass": env.sim.model.body_mass.tolist(),
        "dof_damping": env.sim.model.dof_damping.tolist(),
        "body_inertia": env.sim.model.body_inertia.tolist(),
        "body_ipos": env.sim.model.body_ipos.tolist(),
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(blob, f)
    print(f"emitted {args.steps} steps, nq={blob['nq']} nv={blob['nv']} "
          f"nu={nu} frame_skip={blob['frame_skip']} -> {OUT}")


def check(args):
    import mujoco

    from rower_soccer.t2a_port.xml_global_to_local import convert

    with open(OUT) as f:
        blob = json.load(f)
    # Their generator emits coordinate="global", which MuJoCo removed in
    # 2.3.3. Convert rather than give up: see xml_global_to_local.
    local_xml = convert(blob["xml"], legacy_capsule_mass=args.legacy_mass,
                        legacy_inertial=args.legacy_inertial)
    model = mujoco.MjModel.from_xml_string(local_xml)
    data = mujoco.MjData(model)
    print(f"legacy: mass={'ON' if args.legacy_mass else 'off'} "
          f"inertial={'ON' if args.legacy_inertial else 'off'}")

    print(f"their model: nq={blob['nq']} nv={blob['nv']} nu={blob['nu']} "
          f"timestep={blob['timestep']}")
    print(f"our model:   nq={model.nq} nv={model.nv} nu={model.nu} "
          f"timestep={model.opt.timestep}")
    ok_shape = (model.nq == blob["nq"] and model.nv == blob["nv"]
                and model.nu == blob["nu"])
    print(f"[{'PASS' if ok_shape else 'FAIL'}] the same XML compiles to the "
          f"same model dimensions")
    if not ok_shape:
        raise SystemExit(1)

    if "body_pos" in blob:
        theirs = np.array(blob["body_pos"])
        ours = np.array(model.body_pos)
        d = np.abs(theirs - ours).max()
        print(f"[{'PASS' if d < 1e-9 else 'FAIL'}] body_pos identical after "
              f"conversion  max |d| = {d:.3e}")
        for key in ("geom_pos", "geom_size", "body_mass", "dof_damping",
                    "body_inertia", "body_ipos"):
            if key in blob:
                t_ = np.array(blob[key]); o_ = np.array(getattr(model, key))
                d = np.abs(t_ - o_).max()
                rel = d / max(np.abs(t_).max(), 1e-12)
                print(f"[{'PASS' if d < 1e-9 else 'FAIL'}] {key:13s} "
                      f"max |d| = {d:.3e}  ({100 * rel:.3f}% of range)")

    data.qpos[:] = np.array(blob["qpos0"])
    data.qvel[:] = np.array(blob["qvel0"])
    mujoco.mj_forward(model, data)

    ref = np.array(blob["qpos_traj"])
    ctrls = np.array(blob["ctrls"])
    got = np.zeros_like(ref)
    for t in range(len(ctrls)):
        data.ctrl[:] = ctrls[t]
        for _ in range(blob["frame_skip"]):
            mujoco.mj_step(model, data)
        got[t] = data.qpos

    err = np.abs(got - ref).max(axis=1)
    print(f"\nper-step max |qpos difference| over {len(err)} steps:")
    for t in (0, 9, 49, 99, 199, len(err) - 1):
        if t < len(err):
            print(f"    step {t + 1:4d}   {err[t]:.3e}")
    # Where does it stop being the same trajectory? 1 cm is the scale at which
    # a hopper's contact sequence can differ, which is the thing that would
    # change the reward.
    for thr in (1e-6, 1e-3, 1e-2):
        idx = np.argmax(err > thr) if (err > thr).any() else None
        where = f"step {idx + 1}" if idx is not None else "never"
        print(f"    first exceeds {thr:.0e}: {where}")
    print(f"\n    max over the whole trajectory: {err.max():.3e}")
    print("\nNOTE: different MuJoCo builds; exact agreement is not expected. "
          "What matters is whether they stay together for an episode.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emit", action="store_true")
    p.add_argument("--check", action="store_true")
    p.add_argument("--cfg", default="hopper_gpu")
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--legacy-inertial", action="store_true",
                   help="emit explicit <inertial> from the recovered MuJoCo "
                        "2.1 closed form -- exact, not corrective")
    p.add_argument("--legacy-mass", action="store_true",
                   help="reproduce MuJoCo 2.1's capsule mass, which is what "
                        "Transform2Act actually trained against")
    args = p.parse_args()
    if args.emit:
        emit(args)
    elif args.check:
        check(args)
    else:
        raise SystemExit("pass --emit (their venv) or --check (ours)")


if __name__ == "__main__":
    main()
