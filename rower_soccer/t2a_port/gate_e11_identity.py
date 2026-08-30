"""D3 M3 E1.1 gate: is the "identity" design stage genuinely an identity?

The E1.1 comparison is only a controller comparison if both arms run the same
body. An identity transform that quietly nudged a bone length would turn it
into a comparison of two different bodies while looking exactly like a
controller comparison -- which is the failure mode the user named. So this
gates the claim rather than assuming it, and it gates it against a NON-IDENTITY
action: `env_specs.force_identity_design` is supposed to discard whatever the
policy asked for, so feeding it zeros would prove nothing.

Per episode, with a deliberately destructive random design action:

  1. every `mjModel` array is snapshotted right after `reset()`;
  2. after EACH of the 5 skeleton steps and the 1 attribute step, every array
     is compared against the snapshot;
  3. the episode is run to termination and the arrays, the exported XML string
     and `Robot.get_params()` are compared again -- first step to last;
  4. the stage sequence is asserted to be exactly 5 skeleton + 1 attribute +
     N execution, so the episode STRUCTURE the MLP arm sees is E1's.

And the negative control, without which none of the above means anything:
the identical random actions on the SAME cfg with `force_identity_design`
absent must CHANGE the model.

    cd /workspace/Transform2Act && source env-gpu.sh
    .venv-gpu/bin/python .../t2a_port/gate_e11_identity.py \
        --cfg ant_e11_mlp_s1 --control-cfg ant_e1_s1 --episodes 20

CPU only -- no CUDA context.
"""

import argparse
import os
import sys

sys.path.append("/workspace/Transform2Act")
os.chdir("/workspace/Transform2Act")

import numpy as np  # noqa: E402
import torch  # noqa: E402

FAILED = []
TOL = 0.0          # EXACT. An identity that is only close is not an identity.


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


def arrays(m):
    for nm in dir(m):
        if nm.startswith("_"):
            continue
        try:
            v = getattr(m, nm)
        except Exception:
            continue
        if isinstance(v, np.ndarray) and v.dtype.kind in "fiub":
            yield nm, np.array(v)


def snapshot(m):
    return dict(arrays(m))


def diff(ref, m):
    out = []
    for nm, v in arrays(m):
        if nm not in ref or ref[nm].shape != v.shape:
            out.append(nm)
            continue
        if not np.array_equal(ref[nm], v):
            out.append(nm)
    return out


def rand_design_action(env, rng):
    """A destructive action: every body asked to add or remove, and a large
    attribute delta. `robot_param_scale` is 1, so this is a full-range kick."""
    n = len(env.robot.bodies)
    w = env.control_action_dim + env.attr_design_dim + 1
    a = np.zeros((n, w))
    a[:, :env.control_action_dim] = rng.normal(size=(n, env.control_action_dim))
    a[:, env.control_action_dim:-1] = rng.normal(size=(n, env.attr_design_dim))
    a[:, -1] = rng.integers(0, 3, size=n)      # 0 keep, 1 add, 2 remove
    return a


def run_episode(env, rng, max_exec, snap_check):
    """Returns (stages, n_arrays_diff_seen, final model, final xml, params)."""
    env.reset()
    ref = snapshot(env.model)
    n_ref = len(ref)
    stages = []
    seen = set()
    while env.if_use_transform_action() != 2:
        a = rand_design_action(env, rng)
        _, _, done, info = env.step(a)
        stages.append(info["stage"])
        if snap_check:
            seen |= set(diff(ref, env.model))
        if done:
            break
    n_exec = 0
    while n_exec < max_exec:
        a = np.zeros((len(env.robot.bodies),
                      env.control_action_dim + env.attr_design_dim + 1))
        a[:, 0] = rng.normal(size=len(env.robot.bodies))
        _, _, done, info = env.step(a)
        stages.append(info["stage"])
        n_exec += 1
        if done:
            break
    if snap_check:
        seen |= set(diff(ref, env.model))
    return stages, seen, n_ref


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="ant_e11_mlp_s1")
    p.add_argument("--control-cfg", default="ant_e1_s1",
                   help="the same body WITHOUT force_identity_design")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--their-cfg", default="ant_e0_s1",
                   help="THEIR 5-body ant, to show the 1e-8 root-offset guard "
                        "is their code and not the conversion's")
    p.add_argument("--max-exec", type=int, default=1000)
    args = p.parse_args()

    torch.set_default_dtype(torch.float64)
    from design_opt.envs.ant import AntEnv
    from design_opt.utils.config import Config

    cfg = Config(args.cfg, tmp=False)
    check(f"cfg {args.cfg} sets force_identity_design",
          cfg.env_specs.get("force_identity_design", False) is True)
    env = AntEnv(cfg, agent=None)
    nb0, nu0 = len(env.robot.bodies), env.model.nu
    xml0 = env.robot.export_xml_string()
    par0 = np.asarray(env.robot.get_params())

    print(f"\nIDENTITY ARM -- {args.cfg}, {args.episodes} episodes, "
          f"destructive random design actions")
    all_seen, all_stages, n_ref = set(), [], 0
    ep_lens = []
    for i in range(args.episodes):
        rng = np.random.default_rng(1000 + i)
        stages, seen, n_ref = run_episode(env, rng, args.max_exec, True)
        all_seen |= seen
        all_stages.append(stages)
        ep_lens.append(sum(1 for s in stages if s == "execution"))

    check("E1.1: every mjModel array is unchanged from the first design step "
          "to the last step of the episode",
          not all_seen,
          f"{n_ref} arrays x {args.episodes} episodes"
          + (f"; CHANGED: {sorted(all_seen)}" if all_seen else
             "; zero differences"))
    check("E1.1: body count and actuator count constant",
          len(env.robot.bodies) == nb0 and env.model.nu == nu0,
          f"{nb0} bodies / {nu0} motors throughout")
    check("E1.1: the exported XML string is byte-identical after "
          f"{args.episodes} episodes",
          env.robot.export_xml_string() == xml0)
    # `Robot.get_params()` is NOT bit-identical, and the reason is worth
    # stating exactly rather than tolerating. `Body.set_params`
    # (`khrylib/robot/xml_robot.py:444-446`) guards against a zero-length bone:
    #
    #     if np.all(offset == 0.0):
    #         offset[0] += 1e-8
    #
    # Our ant's ROOT is a sphere torso whose `bone_offset` is exactly
    # [0, 0, 0], so the first attribute transform of every episode bumps its
    # x by 1e-8 m. It is inert: `robot.no_root_offset: true` in the cfg makes
    # `Body.rebuild` (line 368) set `bone_end = bone_start` for the root, so
    # the root's bone_offset places no geom and no joint. That is why the 134
    # model arrays and the XML string above are EXACTLY unchanged. It fires on
    # THEIR ant identically (checked below), so it is their epsilon, not
    # anything the conversion or E1.1 introduced, and it is idempotent -- once
    # the offset is non-zero the guard never fires again.
    par1 = np.asarray(env.robot.get_params())
    names = list(env.robot.get_params(get_name=True))
    moved = [i for i in range(len(par0)) if par0[i] != par1[i]]
    root_offx = names.index("offset_x")      # the root is bodies[0], emitted first
    check("E1.1: Robot.get_params() moves in EXACTLY ONE entry, the root's "
          "offset_x", moved == [root_offx],
          f"{len(moved)} of {len(par0)} entries moved"
          + (f": {[names[i] for i in moved]}" if moved else ""))
    bo = np.asarray(env.robot.bodies[0].bone_offset)
    check("E1.1: and the physical quantity behind it moves by EXACTLY the "
          "1e-8 m zero-bone guard of xml_robot.py:445, in x only",
          bo[0] == 1e-8 and bo[1] == 0.0 and bo[2] == 0.0,
          f"root bone_offset {bo.tolist()}")
    check("E1.1: the root's offset places nothing (no_root_offset), which is "
          "why the model and the XML are exactly unchanged",
          cfg.robot_cfg.get("no_root_offset", False) is True)
    # idempotence: the guard must not fire a second time and creep.
    rng = np.random.default_rng(7)
    run_episode(env, rng, 50, False)
    check("E1.1: idempotent -- another episode does not move it again",
          np.array_equal(np.asarray(env.robot.get_params()), par1),
          f"root bone_offset {np.asarray(env.robot.bodies[0].bone_offset).tolist()}")

    nskel = cfg.skel_transform_nsteps
    ok_struct = all(
        s[:nskel] == ["skeleton_transform"] * nskel
        and s[nskel] == "attribute_transform"
        and set(s[nskel + 1:]) == {"execution"}
        for s in all_stages)
    check(f"E1.1: episode structure is {nskel} skeleton + 1 attribute + N "
          "execution on every episode", ok_struct,
          f"execution lengths {min(ep_lens)}-{max(ep_lens)}, "
          f"mean {np.mean(ep_lens):.0f}")

    # ---- negative control -------------------------------------------------
    print(f"\nNEGATIVE CONTROL -- {args.control_cfg}, same random actions, "
          "force_identity_design ABSENT")
    cfg2 = Config(args.control_cfg, tmp=False)
    check(f"control cfg {args.control_cfg} does NOT set force_identity_design",
          not cfg2.env_specs.get("force_identity_design", False))
    env2 = AntEnv(cfg2, agent=None)
    seen2, nb_seen = set(), set()
    for i in range(min(args.episodes, 5)):
        rng = np.random.default_rng(1000 + i)
        _, seen, _ = run_episode(env2, rng, 50, True)
        seen2 |= seen
        nb_seen.add(len(env2.robot.bodies))
    check("control: the SAME actions DO change the model (so the gate above "
          "is not vacuous)", len(seen2) > 0,
          f"{len(seen2)} arrays changed, e.g. "
          f"{sorted(seen2)[:6]}; body counts seen {sorted(nb_seen)}")

    # ---- is the 1e-8 root bump ours or theirs? ---------------------------
    print("\nPROVENANCE of the 1e-8 root offset -- THEIR ant, their cfg")
    cfg3 = Config(args.their_cfg, tmp=False)
    env3 = AntEnv(cfg3, agent=None)
    b_before = np.array(env3.robot.bodies[0].bone_offset)
    env3.reset()
    z = np.zeros((len(env3.robot.bodies),
                  env3.control_action_dim + env3.attr_design_dim + 1))
    while env3.if_use_transform_action() != 2:
        env3.step(z)
    b_after = np.array(env3.robot.bodies[0].bone_offset)
    check(f"provenance: {args.their_cfg} (THEIR 5-body ant) shows the same "
          "1e-8 root bump, so it is their code and not the conversion",
          b_before.tolist() == [0.0, 0.0, 0.0]
          and b_after.tolist() == [1e-8, 0.0, 0.0],
          f"{b_before.tolist()} -> {b_after.tolist()}")

    print("\n" + ("GATE PASSED" if not FAILED
                  else f"GATE FAILED: {len(FAILED)} -> {FAILED}"))
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
