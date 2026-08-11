"""Stage-1 PPO smoke for the ported two-ant run-to-goal env.

    MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python \
        -m rower_soccer.competevo_port.train_run_to_goal \
        --minutes 18 --worlds 1024 --out runs/competevo_port/smoke

One shared policy plays BOTH ants (see ppo.py for why that is the honest stage-1
target and where faithful opponent sampling goes). What this run is for:

  * a baseline number for the random policy and for the UNTRAINED net evaluated
    with mean actions -- the latter is what the port map records from their code
    as `iter-0 eval ~= 490-510 per agent, win rate 0.00`, and reproducing that
    magnitude is the cheapest evidence that reward, episode length and
    termination are all wired up the way theirs are;
  * evidence that reward improves over that baseline once PPO runs;
  * a video, because metrics lie and videos do not.
"""

import argparse
import json
import os
import time

import numpy as np
import torch

from rower_soccer.competevo_port.ppo import ActorCritic, SelfPlayPPO, evaluate
from rower_soccer.competevo_port.run_to_goal_env import RunToGoalEnv


def random_policy_baseline(env, steps, stochastic=True):
    """Their `use_opponent_sample` eval at epoch 0 loads no weights, so the
    baseline that matters is an untrained net; this is the cruder floor -- pure
    uniform noise -- which separates "the policy learned something" from "the
    survive bonus is 1.0 per step"."""
    env.reset()
    env.reset_win_stats()
    rets, lens = [], []
    for _ in range(steps):
        a = (torch.rand(env.n, env.n_agents, env.act_dim, device=env.device) * 2 - 1
             if stochastic else torch.zeros(env.n, env.n_agents, env.act_dim,
                                            device=env.device))
        _, _, done, _ = env.step(a)
        if bool(done.any()):
            idx = done.nonzero(as_tuple=True)[0]
            rets.append(env.last_return[idx].float().cpu().numpy())
            lens.append(env.last_len[idx].float().cpu().numpy())
    rets = np.concatenate(rets) if rets else np.zeros((1, env.n_agents))
    return {"ret": rets.mean(0), "ep_len": float(np.concatenate(lens).mean()),
            "win_rate": env.win_rate(), "games": env.games}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, default=1024)
    p.add_argument("--minutes", type=float, default=18.0,
                   help="wall-clock budget; the loop stops between iterations")
    p.add_argument("--iters", type=int, default=10_000)
    p.add_argument("--rollout", type=int, default=64)
    p.add_argument("--policy-lr", type=float, default=5e-5)
    p.add_argument("--value-lr", type=float, default=3e-4)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--minibatch", type=int, default=2048)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-worlds", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=20)
    p.add_argument("--out", default="runs/competevo_port/smoke")
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    env = RunToGoalEnv(num_worlds=args.worlds, use_gpu=(dev == "cuda"),
                       seed=args.seed)
    # A separate, smaller fleet for evaluation: eval runs FULL episodes with mean
    # actions (their eval pass), which cannot share a segmented training rollout.
    eval_env = RunToGoalEnv(num_worlds=args.eval_worlds,
                            use_gpu=(dev == "cuda"), seed=args.seed + 1000)
    ac = ActorCritic(env.obs_dim, env.act_dim).to(dev)
    trainer = SelfPlayPPO(env, ac, rollout_len=args.rollout,
                          epochs=args.epochs, minibatch_size=args.minibatch,
                          policy_lr=args.policy_lr, value_lr=args.value_lr,
                          device=dev)

    log = {"args": vars(args), "iters": []}
    print("=== baselines (before any training) ===")
    base_rand = random_policy_baseline(eval_env, eval_env.max_episode_steps + 50)
    print(f"uniform-random actions : ret={base_rand['ret'].round(1).tolist()} "
          f"len={base_rand['ep_len']:.0f} win={base_rand['win_rate'].tolist()} "
          f"games={base_rand['games']}")
    base_eval = evaluate(eval_env, ac)
    print(f"untrained net, mean act: ret={base_eval['ret'].round(1).tolist()} "
          f"len={base_eval['ep_len']:.0f} win={base_eval['win_rate'].tolist()} "
          f"games={base_eval['games']}   "
          f"(their iter-0 eval reference: 490-510 per agent, win rate 0.00)")
    log["baseline_random"] = {k: np.asarray(v).tolist()
                              for k, v in base_rand.items()}
    log["baseline_untrained_eval"] = {k: np.asarray(v).tolist()
                                      for k, v in base_eval.items()}

    print("\n=== training ===")
    t0 = time.time()
    deadline = t0 + args.minutes * 60
    for it in range(args.iters):
        stats = trainer.train_iter()
        elapsed = time.time() - t0
        sps = trainer.total_steps / elapsed
        row = {"iter": it, "steps": trainer.total_steps, "sec": elapsed,
               "train_ret": env.last_return.float().mean(0).cpu().numpy().tolist(),
               "train_len": float(env.last_len.float().mean()),
               "log_std": float(ac.log_std.mean()), **stats}
        if it % args.eval_every == 0 or time.time() >= deadline:
            ev = evaluate(eval_env, ac)
            row["eval_ret"] = ev["ret"].tolist()
            row["eval_win"] = ev["win_rate"].tolist()
            row["eval_len"] = ev["ep_len"]
            print(f"it {it:4d} {trainer.total_steps/1e6:6.2f}M steps "
                  f"{sps:,.0f} sps | train_ret "
                  f"{np.round(row['train_ret'], 1).tolist()} len "
                  f"{row['train_len']:.0f} | EVAL ret "
                  f"{np.round(ev['ret'], 1).tolist()} win "
                  f"{ev['win_rate'].tolist()} len {ev['ep_len']:.0f}")
        elif it % 5 == 0:
            print(f"it {it:4d} {trainer.total_steps/1e6:6.2f}M steps "
                  f"{sps:,.0f} sps | train_ret "
                  f"{np.round(row['train_ret'], 1).tolist()} len "
                  f"{row['train_len']:.0f} kl {stats['kl']:+.2e}")
        log["iters"].append(row)
        with open(os.path.join(args.out, "log.json"), "w") as f:
            json.dump(log, f, indent=1)
        if time.time() >= deadline:
            print(f"time budget reached after {it + 1} iters")
            break

    torch.save({"ac": ac.state_dict(), "args": vars(args)},
               os.path.join(args.out, "policy.pt"))

    if not args.no_video:
        from rower_soccer.competevo_port.render import (RunToGoalRenderer,
                                                        eval_video)
        vid_env = RunToGoalEnv(num_worlds=1, use_gpu=(dev == "cuda"),
                               seed=args.seed + 7)
        path = os.path.join(args.out, "eval.mp4")
        ret, n, winner = eval_video(vid_env, ac, path,
                                    RunToGoalRenderer(), fps=int(1 / 0.015))
        print(f"video: {path}  ({n} frames, return {np.round(ret, 1).tolist()}, "
              f"winner {winner})")
        log["video"] = {"path": path, "frames": n, "return": ret.tolist(),
                        "winner": winner}
        with open(os.path.join(args.out, "log.json"), "w") as f:
            json.dump(log, f, indent=1)


if __name__ == "__main__":
    main()
