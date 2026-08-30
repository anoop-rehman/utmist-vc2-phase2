"""D3 M3 E1: dump the mean-action design of a checkpoint as an XML file.

Two reasons this is a separate step from rendering:

  * their stack (mujoco-py 2.1) has no free-camera offscreen path that is
    pleasant to drive, while the repo's `mujoco` 3.12 + EGL renderer already
    exists and is proven (`render_e1_ant.py`);
  * doing it this way EXERCISES the converter's claim that "every design
    descended from this ant compiles under modern MuJoCo directly, with no
    conversion step" (`D3_M3_E1_ANT_CONVERTER.md` section 1). If an evolved
    design failed to load in mujoco 3.12, the renderer would say so.

    cd /workspace/Transform2Act && source env-gpu.sh
    .venv-gpu/bin/python .../t2a_port/e1_dump_design.py \
        --cfg ant_e1_s1 --epochs 0,10,...,100 --outdir .../designs

CPU only -- no CUDA context.
"""

import argparse
import json
import os
import sys

sys.path.append("/workspace/Transform2Act")
os.chdir("/workspace/Transform2Act")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def tensorfy(np_list):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x) for x in y] for y in np_list]
    return [torch.tensor(y) for y in np_list]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True)
    p.add_argument("--epochs", default="100")
    p.add_argument("--outdir", required=True)
    p.add_argument("--initial-body", action="store_true")
    args = p.parse_args()

    torch.set_default_dtype(torch.float64)
    from design_opt.agents.transform2act_agent import Transform2ActAgent
    from design_opt.utils.config import Config
    from khrylib.utils.torch import to_test

    os.makedirs(args.outdir, exist_ok=True)
    for ep in [int(e) for e in args.epochs.split(",")]:
        cfg = Config(args.cfg, tmp=False)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        agent = Transform2ActAgent(cfg=cfg, dtype=torch.float64,
                                   device=torch.device("cpu"), seed=cfg.seed,
                                   num_threads=1, training=False, checkpoint=ep)
        env, policy = agent.env, agent.policy_net
        to_test(policy)
        state = env.reset()
        if agent.running_state is not None:
            state = agent.running_state(state)
        with torch.no_grad():
            while env.if_use_transform_action() != 2:
                if args.initial_body:
                    a = np.zeros((len(env.robot.bodies),
                                  env.attr_design_dim + 2))
                else:
                    a = policy.select_action(
                        tensorfy([state]), True).numpy().astype(np.float64)
                state, _, done, info = env.step(a)
                if agent.running_state is not None:
                    state = agent.running_state(state)
                if done:
                    raise SystemExit(f"{args.cfg} e{ep}: done during design")
        tag = f"{args.cfg}_e{ep:04d}" + ("_initial" if args.initial_body else "")
        xml = env.robot.export_xml_string().decode()
        open(os.path.join(args.outdir, tag + ".xml"), "w").write(xml)
        names = sorted(b.name for b in env.robot.bodies)
        meta = {"cfg": args.cfg, "epoch": ep, "n_bodies": len(names),
                "n_motors": int(env.model.nu), "names": names,
                "mass_theirs": float(env.model.body_mass.sum())}
        json.dump(meta, open(os.path.join(args.outdir, tag + ".json"), "w"),
                  indent=1)
        print(f"  {tag}: {len(names)} bodies, {env.model.nu} motors -> "
              f"{tag}.xml")


if __name__ == "__main__":
    main()
