"""Milestone 2e: how much of their iter-0 eval number is the seed?

    PYTHONPATH=. python -m rower_soccer.competevo_port.baseline_spread \
        --seeds 8 --eval-worlds 64

Their `run-to-goal-devants-v0` sanity run prints `428.04 / 428.52` at epoch 0 and
the port map treats "~428-440" as a gate. But at epoch 0 the eval is run with
MEAN actions, so the design action is deterministic and every eval episode plays
the same body plan -- a body plan that is a function of the random init of the
scale head, whose output weights their `init_fc_weights` scales by 1.0 and not
the 0.1 the control head gets (`custom/models/dev_actor.py:29-30` vs `50-51`).
So epoch-0 eval reward is a draw from a seed-dependent distribution, and a
single-number comparison against it is meaningless without the spread. This
measures the spread on our side: same untrained architecture, N seeds, one eval
env built once.
"""

import argparse

import numpy as np
import torch

from rower_soccer.competevo_port.dev_env import RunToGoalDevEnv
from rower_soccer.competevo_port.dev_ppo import DevActorCritic
from rower_soccer.competevo_port.selfplay import evaluate_pair


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--eval-worlds", type=int, default=64)
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    env = RunToGoalDevEnv(num_worlds=args.eval_worlds, use_gpu=(dev == "cuda"),
                          seed=1000)
    rows = []
    for s in range(args.seeds):
        torch.manual_seed(s)
        acs = [DevActorCritic(design_dim=env.design_dim,
                              sim_obs_dim=env.sim_obs_dim,
                              n_motor=env.n_motor).to(dev) for _ in range(2)]
        # alpha = 1 at epoch 0, so the curriculum return IS the dense return,
        # which is what their epoch-0 line reports.
        ev = evaluate_pair(env, acs, alpha=1.0)
        rows.append((s, ev["ret_curriculum"][0], ev["ret_curriculum"][1],
                     ev["ep_len"], ev["win_rate"][0], ev["win_rate"][1],
                     ev["games"]))
        print(f"seed {s}: curriculum ret {rows[-1][1]:8.1f} / {rows[-1][2]:8.1f}"
              f"  len {ev['ep_len']:5.1f}  win {ev['win_rate'].tolist()}"
              f"  games {ev['games']}")
    r = np.array([[a, b] for _, a, b, *_ in rows])
    print(f"\n{args.seeds} seeds, {args.eval_worlds} eval worlds")
    print(f"  per-agent epoch-0 curriculum return: mean {r.mean():.1f}, "
          f"sd {r.std():.1f}, min {r.min():.1f}, max {r.max():.1f}")
    print(f"  theirs (measured, their CPU run, seed 42): 428.04 / 428.52")
    z = (428.28 - r.mean()) / (r.std() + 1e-9)
    print(f"  their value sits {z:+.2f} sd from our seed mean")


if __name__ == "__main__":
    main()
