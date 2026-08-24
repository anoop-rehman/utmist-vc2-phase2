"""Score CompetEvo's OWN run the way `score_policies.py` scores ours.

M2E §9's reference figures — 96.9% goal / 1.0% falls at their epoch_0200 — came
from a script that lived inside `/workspace/competevo`. That checkout was
re-cloned when the pod was replaced and the script went with it, along with the
checkpoints it read. This is the replacement, and it lives HERE, in a repo that
gets committed.

    /workspace/competevo/.venv/bin/python \
      rower_soccer/competevo_port/reference_endings.py \
      --run /workspace/competevo/tmp/run-to-goal-devants-v0/<stamp> \
      --epoch 200 --episodes 288

Runs in THEIR venv against THEIR env and THEIR policies, so nothing about our
port is in the loop. The output is deliberately the same table
`score_policies.py` prints, because the point is to compare them.

--------------------------------------------------------------------------
Two things this has to get right
--------------------------------------------------------------------------
**Headless.** `BaseRunner.setup_env` (base_runner.py:72-78) branches on
`self.training`, and the eval branch passes `render_mode="human"` — which needs
a display and dies on GLFW under a pod. `HeadlessRunner` overrides the method
rather than flipping `training`, because `training` also controls checkpoint
loading and logger wiring.

**Their ending semantics, not ours.** Taken from their own sampling loop
(`multi_evo_agent_runner.py:284-300`):

* `truncated`               -> a draw, i.e. ran out of time;
* `terminateds[0]` with a `"winner"` key in some agent's info -> that agent scored;
* `terminateds[0]` with no winner -> somebody fell over.

Deriving it any other way would be measuring our definition against their run.
"""

import argparse
import collections
import glob
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/competevo")


def build(run_dir, epoch, seed):
    import logging

    from config.config import Config
    from logger.logger import Logger
    from runner.multi_evo_agent_runner import MultiEvoAgentRunner

    cfg_file = os.path.join(run_dir, "config.yml")
    cfg = Config(cfg_file if os.path.exists(cfg_file)
                 else "config/run-to-goal-devants-v0.yaml")
    logger = Logger(name="endings", cfg=cfg)
    logger.propagate = False
    logger.setLevel(logging.CRITICAL)
    logger.set_output_handler()

    class HeadlessRunner(MultiEvoAgentRunner):
        def setup_env(self, env_name):
            import gymnasium as gym
            # Their eval branch forces render_mode="human"; this is the same
            # call without it. Nothing else about eval mode changes.
            self.env = gym.make(env_name, cfg=self.cfg)

    torch.set_default_dtype(torch.float64)
    np.random.seed(seed)
    torch.manual_seed(seed)
    runner = HeadlessRunner(cfg, logger, torch.float64, torch.device("cpu"),
                            num_threads=1, training=False,
                            ckpt_dir=os.path.join(run_dir, "models"),
                            # PER-AGENT. base_runner.py:33 indexes ckpt[0], so
                            # an int raises before anything else happens.
                            ckpt=[epoch] * 2)
    runner.epoch = epoch
    return cfg, runner


def load_samplers(runner, cfg, model_dir, epoch):
    # Same imports their runner uses (multi_evo_agent_runner.py:3-7).
    from custom.learners.dev_sampler import DevSampler
    from custom.learners.evo_sampler import EvoSampler
    from custom.learners.sampler import Sampler

    samplers = {}
    for i in range(runner.agent_num):
        flag = getattr(runner.env.agents[i], "flag", None)
        cls = {"evo": EvoSampler, "dev": DevSampler}.get(flag) or Sampler
        samplers[i] = cls(cfg, torch.float64, "cpu", runner.env.agents[i])
        path = f"{model_dir}/agent_{i}/epoch_{epoch:04d}.p"
        with open(path, "rb") as f:
            samplers[i].load_ckpt(pickle.load(f))
    return samplers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="their tmp/<env>/<stamp> dir")
    p.add_argument("--epoch", type=int, required=True)
    p.add_argument("--episodes", type=int, default=288)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()

    model_dir = os.path.join(args.run, "models")
    have = sorted(int(os.path.basename(f)[6:10]) for f in
                  glob.glob(os.path.join(model_dir, "agent_0", "epoch_*.p")))
    if args.epoch not in have:
        raise SystemExit(f"epoch {args.epoch} not saved. Present: "
                         f"{have[:3]}..{have[-3:] if have else []} "
                         f"({len(have)} total)")

    cfg, runner = build(args.run, args.epoch, args.seed)
    samplers = load_samplers(runner, cfg, model_dir, args.epoch)

    from runner.multi_evo_agent_runner import mix_tensorfy
    endings = collections.Counter()
    lens, wins = [], [0] * runner.agent_num

    # Their sample_worker runs the whole rollout under no_grad; without it
    # select_action returns a tensor that still requires grad and .numpy()
    # raises.
    torch.set_grad_enabled(False)
    for _ in range(args.episodes):
        states, _ = runner.env.reset()
        for i, s in samplers.items():
            if s.running_state is not None:
                states[i] = s.running_state(states[i])
        for t in range(10000):
            sv = mix_tensorfy(states)
            actions = []
            for i, s in samplers.items():
                flag = getattr(runner.env.agents[i], "flag", None)
                arg = [sv[i]] if flag in ("evo", "dev") else sv[i]
                # mean_action=True: their non-sampling branch, which is what
                # makes the number a property of the policy.
                actions.append(s.policy_net.select_action(arg, True)
                               .squeeze().numpy().astype(np.float64))
            states, _, terminateds, truncated, infos = runner.env.step(actions)
            for i, s in samplers.items():
                if s.running_state is not None:
                    states[i] = s.running_state(states[i])
            if truncated:
                endings["timeout"] += 1
                lens.append(t + 1)
                break
            if terminateds[0]:
                won = [i for i in range(runner.agent_num)
                       if "winner" in infos[i]]
                for i in won:
                    wins[i] += 1
                endings["goal" if won else "fell"] += 1
                lens.append(t + 1)
                break

    total = max(sum(endings.values()), 1)
    print(f"\nTHEIRS  {args.run}  epoch {args.epoch}")
    print(f"  {total} games, mean actions")
    for k in ("goal", "fell", "timeout"):
        print(f"    {k:8s} {endings[k]:6d}   {100.0 * endings[k] / total:5.1f}%")
    print(f"    mean episode length {np.mean(lens) if lens else 0:.1f}")
    print(f"    win rate per agent  {[round(w / total, 3) for w in wins]}")
    print(f"    win rate summed     {sum(wins) / total:.3f}")


if __name__ == "__main__":
    main()
