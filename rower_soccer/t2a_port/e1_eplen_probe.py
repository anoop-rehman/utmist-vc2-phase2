"""D3 M3 E1: is `done_condition.max_ang: 60` terminating our quadruped early?

`ant_competevo.yml` inherits `done_condition` from `ant.yml` unchanged, and
`D3_M3_E1_ANT_CONVERTER.md` section 6 flagged `max_ang: 60` as untested on a
creature whose torso is a free-jointed sphere that can roll. `ant.py:169-178`
ends an execution episode when ANY of

    height <= min_height (0.0) | height >= max_height (2.0)
    |ang| >= max_ang (60 deg, torso z-axis vs world z) | control_nsteps >= 1000

so this rolls N episodes from a checkpoint (or an untrained policy), records
the execution-episode length, and re-evaluates all four predicates at the step
the episode ended on to say WHICH one fired. A one-line histogram plus a
cause breakdown.

    cd /workspace/Transform2Act && source env-gpu.sh
    .venv-gpu/bin/python .../t2a_port/e1_eplen_probe.py --cfg ant_e1_s1 \
        --untrained --episodes 60

CPU only -- no CUDA context, safe beside live MPS clients.
"""

import argparse
import collections
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
    p.add_argument("--epoch", default="0")
    p.add_argument("--episodes", type=int, default=60)
    p.add_argument("--untrained", action="store_true")
    p.add_argument("--initial-body", action="store_true",
                   help="zero design action: the task's starting XML exactly.")
    p.add_argument("--mean-action", action="store_true")
    p.add_argument("--seed", type=int, default=12345)
    args = p.parse_args()

    torch.set_default_dtype(torch.float64)
    from design_opt.agents.transform2act_agent import Transform2ActAgent
    from design_opt.utils.config import Config
    from khrylib.utils.torch import to_test
    from khrylib.utils.transformation import quaternion_matrix

    cfg = Config(args.cfg, tmp=False)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    epoch = 0 if args.untrained else (int(args.epoch) if args.epoch.isnumeric()
                                      else args.epoch)
    agent = Transform2ActAgent(cfg=cfg, dtype=torch.float64,
                               device=torch.device("cpu"), seed=cfg.seed,
                               num_threads=1, training=False, checkpoint=epoch)
    env, policy = agent.env, agent.policy_net
    to_test(policy)

    dc = cfg.done_condition
    min_h = dc.get('min_height', 0.0)
    max_h = dc.get('max_height', 2.0)
    max_ang = dc.get('max_ang', 3600)
    max_n = dc.get('max_nsteps', 1000)
    print(f"cfg {args.cfg}  checkpoint "
          f"{'UNTRAINED' if args.untrained else epoch}  "
          f"done_condition: min_height {min_h} max_height {max_h} "
          f"max_ang {max_ang} deg  max_nsteps {max_n}")

    lens, causes, angs = [], collections.Counter(), []
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    for _ in range(args.episodes):
        state = env.reset()
        if agent.running_state is not None:
            state = agent.running_state(state)
        n = 0
        with torch.no_grad():
            while True:
                if args.initial_body and env.if_use_transform_action() != 2:
                    a = np.zeros((len(env.robot.bodies),
                                  env.attr_design_dim + 2))
                else:
                    a = policy.select_action(
                        tensorfy([state]), args.mean_action
                    ).numpy().astype(np.float64)
                state, _, done, info = env.step(a)
                if agent.running_state is not None:
                    state = agent.running_state(state)
                if info.get("stage") == "execution":
                    n += 1
                    s = env.state_vector()
                    ang = np.rad2deg(np.arccos(
                        np.clip(quaternion_matrix(s[3:7])[2, 2], -1, 1)))
                    angs.append(ang)
                if done:
                    break
        if n == 0:
            causes["design stage failed (never reached execution)"] += 1
            continue
        s = env.state_vector()
        h = s[2]
        ang = np.rad2deg(np.arccos(
            np.clip(quaternion_matrix(s[3:7])[2, 2], -1, 1)))
        why = []
        if not np.isfinite(s).all():
            why.append("non-finite state")
        if h <= min_h:
            why.append(f"height<={min_h}")
        if h >= max_h:
            why.append(f"height>={max_h}")
        if ang >= max_ang:
            why.append(f"tilt>={max_ang}deg")
        if n >= max_n:
            why.append(f"reached max_nsteps {max_n}")
        causes[" + ".join(why) if why else "unknown"] += 1
        lens.append(n)

    a = np.asarray(lens)
    print(f"\n  {len(a)} episodes reached execution of {args.episodes}")
    print(f"  execution episode length: mean {a.mean():.1f}  median "
          f"{np.median(a):.0f}  min {a.min()}  max {a.max()}  "
          f"p10 {np.percentile(a, 10):.0f}  p90 {np.percentile(a, 90):.0f}")
    edges = [0, 10, 25, 50, 100, 200, 400, 700, 1000, 1001]
    hist, _ = np.histogram(a, bins=edges)
    print("  histogram  " + "  ".join(
        f"[{edges[i]}-{edges[i+1]-1}]:{hist[i]}" for i in range(len(hist))))
    print(f"  torso tilt over all {len(angs)} execution steps: mean "
          f"{np.mean(angs):.1f} deg  p90 {np.percentile(angs, 90):.1f}  "
          f"max {np.max(angs):.1f}")
    print("  termination cause:")
    for k, v in causes.most_common():
        print(f"    {v:4d} ({100*v/args.episodes:5.1f}%)  {k}")


if __name__ == "__main__":
    main()
