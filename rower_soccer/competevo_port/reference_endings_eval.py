"""How do THEIR run-to-goal games end? goal / fall / timeout, counted.

Our port's answer at the end of the 107-epoch 2e run is 6.6% goal, 31.8% fell,
61.7% timeout -- i.e. the dominant failure is not reaching the line, not falling
over. That number is worthless on its own: it only means something against the
same count from the reference. Their runner logs win rate and episode length and
nothing else, so this script drives their own `MultiEvoAgentRunner` (their env,
their policy, their `mean_action=True` eval branch) and classifies each episode.

Nothing in `competevo/` is modified -- this is a reader, so the comparison
cannot be an artefact of a change made to make it agree.

    .venv/bin/python endings_eval.py \
        --run-dir tmp/run-to-goal-devants-v0/20260810_211247 \
        --ckpt 107 --episodes 100 --seed 0

`--ckpt N` loads `models/agent_i/epoch_%04dN.p`, which their `save_checkpoint`
writes as `epoch + 1`, so `--ckpt 107` is the policy AFTER training epoch 106.
Ours is the pair saved at the end of a 107-epoch run, so the two line up.
"""

import argparse
import collections
import json
import logging
import os
import sys

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config.config import Config  # noqa: E402
from logger.logger import Logger  # noqa: E402
from runner.multi_evo_agent_runner import MultiEvoAgentRunner, mix_tensorfy  # noqa: E402


class HeadlessRunner(MultiEvoAgentRunner):
    """Their runner with one line changed, in a subclass rather than in place.

    `BaseRunner.setup_env` hard-codes `render_mode="human"` whenever
    `training=False`, which opens a GLFW window and aborts on a pod with no
    display. Overriding it to the branch TRAINING takes is not just a headless
    workaround -- it is the stricter choice, because the episodes being counted
    are then produced by exactly the env construction that produced the
    reference run's own numbers.
    """

    def setup_env(self, env_name):
        import gymnasium as gym
        self.env = gym.make(env_name, cfg=self.cfg)


def build_runner(run_dir, ckpt):
    """display.py's construction, minus the rendering."""
    cfg_file = os.path.join(run_dir, "config.yml")
    cfg = Config(cfg_file)

    class _Args:
        pass

    args = _Args()
    args.cfg = cfg_file
    args.run_dir = run_dir + "/"
    logger = Logger(name="current", args=args, cfg=cfg)
    logger.propagate = False
    logger.setLevel(logging.WARNING)
    logger.set_output_handler()
    logger.run_dir = args.run_dir
    logger.model_dir = "%smodels" % logger.run_dir
    logger.log_dir = "%slog" % logger.run_dir
    logger.tb_dir = "%stb" % logger.run_dir

    torch.set_default_dtype(torch.float64)
    ckpt_dir = os.path.join(run_dir, "models")
    runner = HeadlessRunner(cfg, logger, torch.float64, torch.device("cpu"),
                                 training=False, ckpt_dir=ckpt_dir,
                                 ckpt=[ckpt, ckpt])
    return runner, cfg


def classify(runner, episodes, seed):
    """Their `display` loop, with the ending recorded instead of discarded."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    env = runner.env
    # gymnasium's wrapper refuses `__getattr__` on names starting with '_', and
    # the step counter is one of those, so reach past it once here.
    base = env.unwrapped

    endings = collections.Counter()
    lens, wins = [], [0, 0]
    both_reached = 0
    goal_and_fall = 0

    for _ in range(episodes):
        states, _ = env.reset()
        for i, learner in runner.learners.items():
            if learner.running_state is not None:
                states[i] = learner.running_state(states[i])

        for t in range(10000):
            state_var = mix_tensorfy(states)
            with torch.no_grad():
                actions = []
                for i, learner in runner.learners.items():
                    a = learner.policy_net.select_action([state_var[i]], True) \
                        if hasattr(learner, "flag") else \
                        learner.policy_net.select_action(state_var[i], True)
                    actions.append(a.squeeze().numpy().astype(np.float64))
            states, _, terminateds, truncated, infos = env.step(actions)
            for i, learner in runner.learners.items():
                if learner.running_state is not None:
                    states[i] = learner.running_state(states[i])

            if not (terminateds[0] or truncated):
                continue

            # `goal_rewards` only stamps 'winner' when EXACTLY one agent is over
            # the line; ask the agents directly so a two-agent crossing (which
            # ends the game with no winner) is not silently filed as a fall.
            reached = [bool(base.agents[i].reached_goal()) for i in base.agents]
            fell = any(bool(infos[i].get("agent_done", False)) for i in base.agents)
            won = [bool(infos[i].get("winner", False)) for i in base.agents]

            if sum(reached) > 1:
                both_reached += 1
            if any(won) and fell:
                goal_and_fall += 1

            if any(won):
                endings["goal"] += 1
                for i in range(2):
                    wins[i] += int(won[i])
            elif sum(reached) > 0:
                endings["goal_no_winner"] += 1
            elif fell:
                endings["fell"] += 1
            elif truncated:
                endings["timeout"] += 1
            else:
                # `_get_done` also fires on a non-finite state vector.
                endings["nonfinite"] += 1
            lens.append(base._elapsed_steps)
            break

    return endings, lens, wins, both_reached, goal_and_fall


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", default="tmp/run-to-goal-devants-v0/20260810_211247")
    p.add_argument("--ckpt", type=int, default=107)
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    runner, cfg = build_runner(args.run_dir, args.ckpt)
    endings, lens, wins, both, gf = classify(runner, args.episodes, args.seed)

    total = sum(endings.values()) or 1
    print(f"reference: {args.run_dir}  ckpt epoch_{args.ckpt:04d}  seed {args.seed}")
    print(f"episodes {total}   mean length {np.mean(lens):.1f} of "
          f"{env_cap(runner)}")
    for k in ("goal", "goal_no_winner", "fell", "timeout", "nonfinite"):
        print(f"  {k:15s} {endings[k]:5d}  {100.0 * endings[k] / total:5.1f}%")
    print(f"  (both agents crossed: {both}; goal and fall same step: {gf})")
    print(f"win rate per agent: [{wins[0] / total:.4f}, {wins[1] / total:.4f}]"
          f"  summed {sum(wins) / total:.4f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"run_dir": args.run_dir, "ckpt": args.ckpt,
                       "seed": args.seed, "episodes": total,
                       "endings": dict(endings), "mean_len": float(np.mean(lens)),
                       "wins": wins, "both_reached": both,
                       "goal_and_fall": gf}, f, indent=2)
        print(f"wrote {args.out}")


def env_cap(runner):
    return getattr(runner.env.unwrapped, "_max_episode_steps", "?")


if __name__ == "__main__":
    main()
