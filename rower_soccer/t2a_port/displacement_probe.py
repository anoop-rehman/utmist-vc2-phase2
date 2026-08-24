"""Does the agent LOCOMOTE, or does it just accumulate |Δx|?

The reward-form discrepancy (docs/repro/TRANSFORM2ACT_M1_REPRO_NOTES.md): the
paper's equation 17 pays `|p^x_{t+1} - p^x_t| / δt + 1` and the released code
pays the signed version. `|Δx|` learns 2.5-6.5x faster by reward, but reward is
not the quantity "2D Locomotion" names. An agent vibrating in place racks up
`|Δx|` without going anywhere, and would look excellent on the y-axis of
Figure 3 while failing the task the figure is titled after.

So measure the thing the reward is a proxy for:

  net       = |x_final - x_start|          how far it actually got
  path      = sum |x_{t+1} - x_t|          how much it moved at all
  net/path  = 1.0 pure locomotion, 0.0 pure oscillation

`net/path` is comparable across both reward forms, which reward is not. It is
the number that decides whether the paper's ~9,000 describes a runner.

    cd /workspace/Transform2Act && source env-gpu.sh
    .venv-gpu/bin/python .../t2a_port/displacement_probe.py \
        --cfg hopper_gpu_abs --checkpoint latest --episodes 20
    ... --cfg hopper_gpu --checkpoint 50 --episodes 20
"""

import argparse
import glob
import os
import re
import sys

sys.path.append("/workspace/Transform2Act")
sys.path.append("/workspace/utmist-vc2-phase2")
os.chdir("/workspace/Transform2Act")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from design_opt.agents.transform2act_agent import (Transform2ActAgent,  # noqa: E402
                                                    tensorfy)
from design_opt.utils.config import Config  # noqa: E402


def resolve(model_dir, spec):
    if spec not in ("latest", "best"):
        return int(spec)
    if spec == "best":
        return "best"
    eps = sorted(int(m.group(1)) for m in
                 (re.search(r"epoch_(\d+)\.p$", f)
                  for f in glob.glob(os.path.join(model_dir, "epoch_*.p"))) if m)
    if not eps:
        raise SystemExit(f"no epoch_*.p under {model_dir}")
    return eps[-1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", default="hopper_gpu_abs")
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--episodes", type=int, default=20)
    args = p.parse_args()

    torch.set_default_dtype(torch.float64)
    cfg = Config(args.cfg, tmp=False)
    ckpt = resolve(cfg.model_dir, args.checkpoint)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    agent = Transform2ActAgent(cfg=cfg, dtype=torch.float64,
                               device=torch.device("cpu"), seed=cfg.seed,
                               num_threads=1, training=False, checkpoint=ckpt)
    env, policy = agent.env, agent.policy_net
    policy.eval()

    nets, paths, rewards, steps = [], [], [], []
    with torch.no_grad():
        for _ in range(args.episodes):
            state = env.reset()
            x0 = env.sim.data.qpos[0]
            prev, path, R, n = x0, 0.0, 0.0, 0
            for _ in range(10000):
                # Exactly their eval call (transform2act_agent.py:60-62):
                # no trans_policy wrapper, no [0], mean_action=True.
                action = policy.select_action(
                    tensorfy([state]), True).numpy().astype(np.float64)
                state, reward, done, info = env.step(action)
                if info.get("stage") == "execution":
                    x = env.sim.data.qpos[0]
                    path += abs(x - prev)
                    prev = x
                    R += reward
                    n += 1
                if done:
                    break
            nets.append(prev - x0)
            paths.append(path)
            rewards.append(R)
            steps.append(n)

    nets, paths = np.array(nets), np.array(paths)
    # Guard the ratio: an episode that never moved has path 0 and no defined
    # straightness, and averaging 0/0 in would be silently wrong.
    moved = paths > 1e-9
    ratio = np.abs(nets[moved]) / paths[moved]
    print(f"\ncfg {args.cfg}  checkpoint {ckpt}  {args.episodes} episodes, "
          f"mean actions")
    print(f"  exec_R_eps        {np.mean(rewards):9.1f}   "
          f"(the y-axis of Figure 3)")
    print(f"  episode length    {np.mean(steps):9.1f} steps")
    print(f"  NET displacement  {np.mean(nets):9.2f} m   "
          f"(what '2D Locomotion' means)")
    print(f"  PATH length       {np.mean(paths):9.2f} m   "
          f"(what |dx| pays for)")
    if moved.any():
        print(f"  net / path        {np.mean(ratio):9.3f}     "
              f"1.0 = pure locomotion, 0.0 = pure oscillation")
    print(f"  mean speed        {np.mean(nets) / max(np.mean(steps), 1) / 0.008:9.2f} m/s "
          f"(net, at dt = 0.008)")
    if moved.sum() < len(paths):
        print(f"  ({len(paths) - moved.sum()} episodes never moved at all)")


if __name__ == "__main__":
    main()
