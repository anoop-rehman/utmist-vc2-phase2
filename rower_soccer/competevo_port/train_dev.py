"""Stage-2 PPO smoke for `run-to-goal-devants-v0`: two ants that choose bodies.

    PYTHONPATH=. .venv/bin/python -m rower_soccer.competevo_port.train_dev \
        --minutes 12 --worlds 1024 --out runs/competevo_port/dev_smoke

The point of this run is NOT a learning curve -- 12 minutes against their 1000
epochs is nothing. It is that the loop closes with morphology in it: a per-world
genome is emitted by the policy every episode, written into the batched model,
simulated, and credited through GAE, for millions of transitions, without NaNs
and without the throughput collapsing.

What is logged that stage 1 did not have:
  * `design_std` -- the spread of the emitted genomes across worlds. If the
    design head is doing anything at all this moves; if it collapses to a
    constant, every world is the same ant and the whole stage is decorative.
  * `mass` -- the mean total ant mass the writer produced, i.e. proof the model
    arrays are actually being rewritten each episode.
  * the usual stage-1 columns (forward progress per step, curriculum alpha).
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from rower_soccer.competevo_port.dev_env import RunToGoalDevEnv
from rower_soccer.competevo_port.dev_ppo import DevActorCritic, DevSelfPlayPPO
from rower_soccer.competevo_port.ppo import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, default=1024)
    p.add_argument("--minutes", type=float, default=12.0)
    p.add_argument("--iters", type=int, default=10_000)
    p.add_argument("--rollout", type=int, default=64)
    p.add_argument("--policy-lr", type=float, default=5e-5)
    p.add_argument("--value-lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=8192)
    p.add_argument("--curriculum-steps", type=int, default=None,
                   help="agent-steps over which alpha anneals 1->0; default is "
                        "their dev config, 1000 epochs x 50k steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-worlds", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--out", default="runs/competevo_port/dev_smoke")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    env = RunToGoalDevEnv(num_worlds=args.worlds, use_gpu=(dev == "cuda"),
                          seed=args.seed)
    eval_env = RunToGoalDevEnv(num_worlds=args.eval_worlds,
                               use_gpu=(dev == "cuda"), seed=args.seed + 1000)
    ac = DevActorCritic(design_dim=env.design_dim,
                        sim_obs_dim=env.sim_obs_dim,
                        n_motor=env.n_motor).to(dev)
    kw = ({} if args.curriculum_steps is None
          else {"curriculum_steps": args.curriculum_steps})
    trainer = DevSelfPlayPPO(env, ac, rollout_len=args.rollout,
                             epochs=args.epochs, minibatch_size=args.minibatch,
                             policy_lr=args.policy_lr, value_lr=args.value_lr,
                             device=dev, **kw)

    log = {"args": vars(args), "iters": []}
    base = evaluate(eval_env, ac)
    print(f"untrained net, mean actions: ret={base['ret'].round(1).tolist()} "
          f"len={base['ep_len']:.0f} win={base['win_rate'].tolist()} "
          f"games={base['games']}")
    log["baseline_untrained_eval"] = {k: np.asarray(v).tolist()
                                      for k, v in base.items()}

    print("\n=== training ===")
    t0 = time.time()
    deadline = t0 + args.minutes * 60
    for it in range(args.iters):
        stats = trainer.train_iter()
        elapsed = time.time() - t0
        mass = float(env.backend.model_arrays["body_mass"].sum(-1).mean())
        row = {"iter": it, "steps": trainer.total_steps, "sec": elapsed,
               "train_ret": env.last_return.float().mean(0).cpu().numpy().tolist(),
               "train_len": float(env.last_len.float().mean()),
               "fwd_per_step": trainer.ep_fwd, "alpha": trainer.alpha(),
               "design_mean": float(env.scale.float().mean()),
               "design_std": float(env.scale.float().std()),
               "scale_log_std": float(ac.scale_log_std.mean()),
               "control_log_std": float(ac.control_log_std.mean()),
               "mass": mass, "diverged": env.n_diverged, **stats}
        if it % args.eval_every == 0 or time.time() >= deadline:
            ev = evaluate(eval_env, ac)
            row["eval_ret"] = ev["ret"].tolist()
            row["eval_win"] = ev["win_rate"].tolist()
            row["eval_len"] = ev["ep_len"]
            print(f"it {it:4d} {trainer.total_steps / 1e6:6.2f}M "
                  f"{trainer.total_steps / elapsed:,.0f} sps | train_ret "
                  f"{np.round(row['train_ret'], 1).tolist()} len "
                  f"{row['train_len']:.0f} | EVAL ret "
                  f"{np.round(ev['ret'], 1).tolist()} win "
                  f"{ev['win_rate'].tolist()} len {ev['ep_len']:.0f} | "
                  f"fwd/step {row['fwd_per_step']:+.3f} design_std "
                  f"{row['design_std']:.3f} mass {mass:.3f} "
                  f"nan_worlds {env.n_diverged}")
        elif it % 5 == 0:
            print(f"it {it:4d} {trainer.total_steps / 1e6:6.2f}M "
                  f"{trainer.total_steps / elapsed:,.0f} sps | train_ret "
                  f"{np.round(row['train_ret'], 1).tolist()} len "
                  f"{row['train_len']:.0f} fwd/step {row['fwd_per_step']:+.3f} "
                  f"design_std {row['design_std']:.3f} mass {mass:.3f} "
                  f"kl {stats['kl']:+.2e} nan_worlds {env.n_diverged}")
        log["iters"].append(row)
        with open(os.path.join(args.out, "log.json"), "w") as f:
            json.dump(log, f, indent=1)
        if time.time() >= deadline:
            print(f"time budget reached after {it + 1} iters")
            break

    torch.save({"ac": ac.state_dict(), "args": vars(args)},
               os.path.join(args.out, "policy.pt"))
    print(f"saved {os.path.join(args.out, 'policy.pt')}")


if __name__ == "__main__":
    main()
