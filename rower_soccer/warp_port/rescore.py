"""Re-score a finished run's checkpoints with the batched deterministic scorer.

`best.pt` for the drills that finished before `score.py` existed was selected by
the old method: a ONE-WORLD stochastic eval, whose noise is comparable to the
differences it was choosing between. The scorer that replaced it runs N worlds
deterministically against a fixed seed, so every candidate faces the same task
draws and the comparison is paired.

This re-runs that comparison after the fact, so a run's pinned checkpoint is
justified by the good measurement rather than the one that happened to be
available when it finished.

    python -m rower_soccer.warp_port.rescore --run runs_v2/dribble_ant_v3 --drill dribble

Prints one line per checkpoint with a standard error, because the point of the
exercise is that differences smaller than the error bar are not differences.
"""

import argparse
import json
import os

import torch

from rower_soccer.warp_port.ppo import _flatten_checkpoint
from rower_soccer.warp_port.score import score_policy

DRILLS = ("follow", "dribble", "kick", "shoot")


def load_trainer_module(drill):
    import importlib
    return importlib.import_module(f"rower_soccer.warp_port.train_{drill}_warp")


def build_env(mod, args):
    """The trainers grew different helpers at different times -- dribble and
    follow have `make_score_env`, shoot only has `make_env`. Try them in order
    of specificity rather than picking one and refactoring four live trainers."""
    if hasattr(mod, "make_score_env"):
        return mod.make_score_env(args)
    if hasattr(mod, "make_eval_env"):
        return mod.make_eval_env(args, num_worlds=args.score_worlds,
                                 seed=args.score_seed)
    return mod.make_env(args, num_worlds=args.score_worlds,
                        seed=args.score_seed)


def build_ac(args, env, state_dict):
    """The same rule every trainer applies, plus a check that it was the right
    rule: if the architecture implied by `config.json` does not have exactly the
    checkpoint's parameters, stop. A scorer that silently grades a mis-built
    network is worse than no scorer."""
    from rower_soccer.warp_port.ppo import ActorCritic, SimpleActorCritic
    if args.plain:
        ac = SimpleActorCritic(env.obs_dim, env.act_dim)
    else:
        ac = ActorCritic(env.obs_dim, env.act_dim,
                         proprio_indices=env.proprio_indices.tolist(),
                         task_indices=env.task_indices.tolist(),
                         z_dim=args.z_dim,
                         state_dependent_std=args.state_dependent_std)
    want, have = set(ac.state_dict()), set(state_dict)
    if want != have:
        raise SystemExit(
            f"architecture from config.json does not match the checkpoint: "
            f"missing {sorted(want - have)[:4]}, unexpected {sorted(have - want)[:4]}")
    ac.load_state_dict(state_dict)
    return ac


class _Args:
    """`make_eval_env` reads its settings off an argparse namespace; a finished
    run's `config.json` is that namespace, saved."""

    def __init__(self, cfg, overrides):
        for k, v in cfg.items():
            setattr(self, k, v)
        for k, v in overrides.items():
            setattr(self, k, v)

    def __getattr__(self, name):
        raise AttributeError(
            f"config.json has no '{name}' -- this run predates the flag, so "
            f"pass it explicitly rather than letting a default stand in")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True, help="run directory")
    p.add_argument("--drill", required=True, choices=DRILLS)
    p.add_argument("--worlds", type=int, default=256,
                   help="scoring worlds; 64 is the trainer's in-loop default, "
                        "more is cheap when nothing else is waiting on it")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--ckpts", nargs="*", default=None,
                   help="checkpoint filenames (default: every .pt in the run)")
    args = p.parse_args()

    cfg = json.load(open(os.path.join(args.run, "config.json")))
    mod = load_trainer_module(args.drill)
    env_args = _Args(cfg, {"score_worlds": args.worlds, "score_seed": args.seed})
    env = build_env(mod, env_args)

    names = args.ckpts or sorted(
        f for f in os.listdir(args.run) if f.endswith(".pt"))
    print(f"{args.run}  drill={args.drill}  {args.worlds} worlds  seed {args.seed}")
    rows = []
    for name in names:
        path = os.path.join(args.run, name)
        # `_flatten_checkpoint` handles both layouts the trainers write: a
        # resume checkpoint keyed under "ac", and the per-submodule export
        # ({mlp_extractor, action_net, value_net, log_std}) that best.pt uses.
        blob = torch.load(path, map_location="cpu")
        sd = _flatten_checkpoint(blob)
        ac = build_ac(env_args, env, sd)
        ac.eval().to(env.device)
        r = score_policy(env, ac, seed=args.seed, deterministic=True)
        rows.append((name, r))
        step = blob.get("total_steps", blob.get("step"))
        print(f"  {name:24s} fitness {r.fitness:.4f} +/- {r.fitness_sem:.4f}"
              f"   ep_rew {r.ep_rew:8.1f}"
              f"   {'step ' + format(step, ',') if step else ''}")

    best = max(rows, key=lambda kv: kv[1].fitness)
    within = [n for n, r in rows
              if best[1].fitness - r.fitness <= 2 * (r.fitness_sem
                                                     + best[1].fitness_sem)]
    print(f"\nhighest: {best[0]} at {best[1].fitness:.4f}")
    if len(within) > 1:
        print("statistically indistinguishable from it (within 2 combined SEM): "
              + ", ".join(n for n in within if n != best[0]))


if __name__ == "__main__":
    main()
