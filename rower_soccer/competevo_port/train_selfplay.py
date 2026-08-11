"""Stage-3 smoke: two learners co-evolving against an opponent checkpoint ring.

    PYTHONPATH=. python -m rower_soccer.competevo_port.train_selfplay \
        --minutes 12 --worlds 1024 --out runs/competevo_port/selfplay_smoke

What this run is for, and what it is not: at a few million agent-steps against
their 50M it cannot say anything about who wins. It is here to show that the
stage-3 loop closes -- two independent learners, each sampling its opponent from
a bounded ring by their `delta` rule, over a batched env with per-world
morphology -- without diverging and without the ring quietly doing nothing.

Columns stage 2 did not have:
  * `opp_lag`  -- mean (current epoch - sampled opponent epoch) over the live
                  slots. 0 means every opponent is the current policy, i.e. the
                  ring is decorative. delta=0.5 predicts it climbs to ~epoch/4.
  * `ring`     -- entries held per side, and the clamp counter. A non-zero clamp
                  count means eviction has started biasing the draw.
  * per-learner `pi_loss_{0,1}` / `kl_{0,1}`, because there are now two of them
    and they can diverge from each other.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from rower_soccer.competevo_port.dev_env import RunToGoalDevEnv
from rower_soccer.competevo_port.dev_ppo import DevActorCritic
from rower_soccer.competevo_port.selfplay import (CoEvoPPO, DEV_DELTA,
                                                  OPPONENT_BLOCKS,
                                                  RING_CAPACITY,
                                                  evaluate_pair)


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
    p.add_argument("--delta", type=float, default=DEV_DELTA,
                   help="their opponent WINDOW parameter: the checkpoint is "
                        "uniform on [max(1, floor(delta*epoch)), epoch-1]. "
                        "0.5 = dev configs, 0 = their fixed-morph ants")
    p.add_argument("--ring-capacity", type=int, default=RING_CAPACITY)
    p.add_argument("--checkpoint-every", type=int, default=1,
                   help="their save_model_interval; 1 is theirs")
    p.add_argument("--blocks", type=int, default=OPPONENT_BLOCKS,
                   help="distinct sampled opponents live per side per iteration")
    p.add_argument("--no-opponent-sample", action="store_true",
                   help="their use_opponent_sample: false -- always the "
                        "opponent's CURRENT weights (the stage-2 behaviour, "
                        "but still two learners)")
    p.add_argument("--curriculum-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-worlds", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--out", default="runs/competevo_port/selfplay_smoke")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    env = RunToGoalDevEnv(num_worlds=args.worlds, use_gpu=(dev == "cuda"),
                          seed=args.seed)
    eval_env = RunToGoalDevEnv(num_worlds=args.eval_worlds,
                               use_gpu=(dev == "cuda"), seed=args.seed + 1000)
    acs = [DevActorCritic(design_dim=env.design_dim,
                          sim_obs_dim=env.sim_obs_dim,
                          n_motor=env.n_motor).to(dev) for _ in range(2)]
    kw = ({} if args.curriculum_steps is None
          else {"curriculum_steps": args.curriculum_steps})
    trainer = CoEvoPPO(env, acs, delta=args.delta,
                       ring_capacity=args.ring_capacity,
                       checkpoint_every=args.checkpoint_every,
                       blocks=args.blocks,
                       use_opponent_sample=not args.no_opponent_sample,
                       rollout_len=args.rollout, epochs=args.epochs,
                       minibatch_size=args.minibatch, policy_lr=args.policy_lr,
                       value_lr=args.value_lr, seed=args.seed, device=dev, **kw)

    log = {"args": vars(args), "iters": []}
    base = evaluate_pair(eval_env, acs)
    print(f"untrained pair, mean actions: ret={base['ret'].round(1).tolist()} "
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
        rings = [len(r) for r in trainer.rings]
        row = {"iter": it, "epoch": trainer.epoch,
               "steps": trainer.total_steps, "sec": elapsed,
               "train_ret": env.last_return.float().mean(0).cpu().numpy().tolist(),
               "train_len": float(env.last_len.float().mean()),
               "fwd_per_step": trainer.ep_fwd,
               "alpha": trainer.learners[0].alpha(),
               "opp_lag": trainer.opponent_lag(),
               "opp_epochs": [list(s) for s in trainer.opp_epoch],
               "ring": rings,
               "ring_clamped": [r.n_clamped for r in trainer.rings],
               "ring_mb": [r.nbytes() / 1e6 for r in trainer.rings],
               "design_std": float(env.scale.float().std()),
               "mass": float(env.backend.model_arrays["body_mass"].sum(-1).mean()),
               "diverged": env.n_diverged, **stats}
        if it % args.eval_every == 0 or time.time() >= deadline:
            ev = evaluate_pair(eval_env, acs)
            row["eval_ret"] = ev["ret"].tolist()
            row["eval_win"] = ev["win_rate"].tolist()
            row["eval_len"] = ev["ep_len"]
            print(f"it {it:4d} {trainer.total_steps / 1e6:6.2f}M "
                  f"{trainer.total_steps / elapsed:,.0f} sps | EVAL ret "
                  f"{np.round(ev['ret'], 1).tolist()} win "
                  f"{ev['win_rate'].tolist()} len {ev['ep_len']:.0f} | "
                  f"fwd/step {row['fwd_per_step'][0]:+.3f}/"
                  f"{row['fwd_per_step'][1]:+.3f} | opp_lag {row['opp_lag']:.1f} "
                  f"ring {rings} clamp {row['ring_clamped']} "
                  f"({row['ring_mb'][0]:.1f} MB) | nan {env.n_diverged}")
        elif it % 5 == 0:
            print(f"it {it:4d} {trainer.total_steps / 1e6:6.2f}M "
                  f"{trainer.total_steps / elapsed:,.0f} sps | train_ret "
                  f"{np.round(row['train_ret'], 1).tolist()} len "
                  f"{row['train_len']:.0f} | fwd/step "
                  f"{row['fwd_per_step'][0]:+.3f}/{row['fwd_per_step'][1]:+.3f} "
                  f"kl {stats['kl_0']:+.1e}/{stats['kl_1']:+.1e} | opp_lag "
                  f"{row['opp_lag']:.1f} ring {rings} nan {env.n_diverged}")
        log["iters"].append(row)
        with open(os.path.join(args.out, "log.json"), "w") as f:
            json.dump(log, f, indent=1)
        if time.time() >= deadline:
            print(f"time budget reached after {it + 1} iters")
            break

    torch.save({"ac_0": acs[0].state_dict(), "ac_1": acs[1].state_dict(),
                "args": vars(args)}, os.path.join(args.out, "policies.pt"))
    print(f"saved {os.path.join(args.out, 'policies.pt')}")


if __name__ == "__main__":
    main()
