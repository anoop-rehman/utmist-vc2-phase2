"""D3 M3 E1: what body plans can the skeleton stage actually reach?

`ant_competevo.yml` inherits `add_body_condition.max_nchild: 2`,
`max_body_depth: 4` and (by omission) `min_body_depth: 1` from `ant.yml`. Our
ant's torso already has four children, so the obvious worry is that it can
never grow a fifth leg. This measures the reachable space rather than reasoning
about it, on BOTH ants, using the env's own predicates
(`design_opt/envs/ant.py:47-59`) and the env's own mutators.

Reported per starting body:
  * depth, number of children, whether the body has a hinge (hence whether a
    child cloned from it would be ACTUATED -- `Robot.add_child_to_body` clones
    the body itself, joints included, so a child of a jointless body is a
    passive link that no motor drives);
  * `allow_add_body` and `allow_remove_body`.

Then the space is saturated: add a child to every body that allows one, repeat
until no body does, and report the largest body plan reachable. And it is
eroded: remove every removable leaf, repeat, and report the smallest.

    cd /workspace/Transform2Act && source env-gpu.sh
    .venv-gpu/bin/python .../t2a_port/e1_designspace_probe.py --cfg ant_e1_s1
    .venv-gpu/bin/python .../t2a_port/e1_designspace_probe.py --cfg ant_e0_s1

CPU only -- no CUDA context, no physics beyond one compile.
"""

import argparse
import os
import sys

sys.path.append("/workspace/Transform2Act")
os.chdir("/workspace/Transform2Act")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def describe(env, tag):
    print(f"\n--- {tag}: {len(env.robot.bodies)} bodies, {env.model.nu} motors")
    print("    name   depth  nchild  hinge  allow_add  allow_remove")
    for b in env.robot.bodies:
        print(f"    {b.name:<6} {b.depth:>5}  {len(b.child):>6}  "
              f"{'yes' if b.joints else ' no':>5}  "
              f"{str(env.allow_add_body(b)):>9}  {str(env.allow_remove_body(b)):>12}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    args = p.parse_args()
    torch.set_default_dtype(torch.float64)
    from design_opt.envs.ant import AntEnv
    from design_opt.utils.config import Config

    cfg = Config(args.cfg, tmp=False)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    env = AntEnv(cfg, agent=None)
    print(f"cfg {args.cfg}  xml {env.model_xml_file}")
    print(f"  min_body_depth {cfg.min_body_depth}  max_body_depth "
          f"{cfg.max_body_depth}  max_nchild "
          f"{cfg.add_body_condition.get('max_nchild', 3)}  "
          f"enable_remove {cfg.enable_remove}")
    describe(env, "INITIAL")

    # --- saturate: grow until no body allows a child -------------------------
    rounds = 0
    while True:
        targets = [b for b in env.robot.bodies if env.allow_add_body(b)]
        if not targets:
            break
        for b in targets:
            env.robot.add_child_to_body(b)
        rounds += 1
        if rounds > 12:
            raise SystemExit("saturation did not terminate -- check the rules")
    env.reload_sim_model(env.robot.export_xml_string().decode('utf-8'))
    n_passive = sum(1 for b in env.robot.bodies if not b.joints)
    print(f"\n  SATURATED after {rounds} growth rounds: "
          f"{len(env.robot.bodies)} bodies, {env.model.nu} motors, "
          f"{n_passive} jointless (passive, unactuated) bodies")
    depths = {}
    for b in env.robot.bodies:
        depths[b.depth] = depths.get(b.depth, 0) + 1
    print(f"  bodies per depth: " +
          ", ".join(f"d{k}={v}" for k, v in sorted(depths.items())))

    # --- erode: strip removable leaves until none remain ---------------------
    env2 = AntEnv(cfg, agent=None)
    rounds = 0
    while True:
        targets = [b for b in env2.robot.bodies if env2.allow_remove_body(b)]
        if not targets:
            break
        for b in targets:
            env2.robot.remove_body(b)
        rounds += 1
        if rounds > 12:
            raise SystemExit("erosion did not terminate")
    env2.reload_sim_model(env2.robot.export_xml_string().decode('utf-8'))
    n_passive = sum(1 for b in env2.robot.bodies if not b.joints)
    print(f"  ERODED after {rounds} removal rounds: "
          f"{len(env2.robot.bodies)} bodies, {env2.model.nu} motors, "
          f"{n_passive} jointless; names "
          f"[{','.join(sorted(b.name for b in env2.robot.bodies))}]")


if __name__ == "__main__":
    main()
