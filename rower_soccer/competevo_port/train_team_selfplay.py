"""2f step 5/6: two-learner team co-evolution over the 2v2 env.

`CoEvoPPO` was generalised in place from two AGENTS to two SIDES rather than
forked, so 1v1 is the L = 1 case of the same code and
`tests/test_selfplay.py` (15/15) is the regression that says so. This file is
the trainer around it: env, warm start, logging, eval.

    PYTHONPATH=. MUJOCO_GL=osmesa .venv/bin/python \
        -m rower_soccer.competevo_port.train_team_selfplay \
        --worlds 512 --iters 200 --out runs/competevo_port/t2v2

World layout, inherited and now team-shaped:

    worlds [0, N/2)   ego = team 0 (agents 0, 2), opponents from ring[1]
    worlds [N/2, N)   ego = team 1 (agents 1, 3), opponents from ring[0]

One ring entry per TEAM and one slot per WORLD, so a world plays a whole past
team rather than two independently sampled halves of one -- design doc section
6. That is the property `gate_team_selfplay` checks directly, because a
per-agent slot would still train, still look plausible, and would quietly be a
different algorithm.

Defaults are the config M2E validated at 1v1 (policy_lr 5e-5, value_lr 3e-4,
10 PPO epochs, minibatch 2048, delta 0.5, blocks 4) so that the 1v1 run is a
usable control on every shared quantity. `--worlds 512` rather than 128: the
2v2 step is launch-bound and 128 wastes the batch (design doc, "What 2v2
costs").
"""

import argparse
import json
import os
import time

import numpy as np
import torch


def team_eval(env, acs, team_lanes, max_steps=None):
    """Mean-action rollout with each team on its own current weights.

    `evaluate_pair` assumes one policy per AGENT; here two policies drive four
    ants, so the lane mapping has to be explicit.
    """
    steps = max_steps or env.max_episode_steps
    env.reset_win_stats()
    obs = env.reset()
    lens, endings = [], {"goal": 0, "wipeout": 0, "fell": 0,
                         "timeout": 0, "other": 0}
    with torch.no_grad():
        for _ in range(steps + 2):
            o = obs.float()
            act = torch.zeros(env.n, env.n_agents, env.act_dim,
                              device=env.device, dtype=o.dtype)
            for e, lanes in enumerate(team_lanes):
                act[:, lanes] = acs[e].mean_action(o[:, lanes])
            obs, _, done, info = env.step(act.to(env.dtype))
            if bool(done.any()):
                idx = done.nonzero(as_tuple=True)[0]
                # The env's OWN classification. Re-deriving it from
                # winner/fell/truncated merges two different endings: under
                # down_rule="team_down" one agent leaving the standing band
                # does not end anything, so an episode that timed out with
                # somebody on the floor is a TIMEOUT, not a fall, and only a
                # whole team going down is a wipeout.
                for e in env.last_end[idx].tolist():
                    endings[{0: "other", 1: "goal", 2: "wipeout",
                             3: "fell", 4: "timeout"}[e]] += 1
                lens.extend(env.last_len[idx].float().cpu().tolist())
    total = max(sum(endings.values()), 1)
    per_agent = np.atleast_1d(env.win_rate())
    # NOT the sum. Under win_rule="team_first" the env sets
    # `winner = mine & one_team`, marking BOTH members of the scoring team, so
    # summing a team's agents double-counts and a team that wins every game
    # would report 2.0. `team_win_rate` divides by team size, which is the
    # per-team probability of winning and the number the 1v1 control compares
    # against. Per-agent is kept alongside it because the split within a team
    # is the division of labour 2f is looking for -- but at team_first that
    # split is about who CROSSED, which `reached` tracks, not `winner`.
    per_team = [float(x) for x in np.atleast_1d(env.team_win_rate())]
    return {"win_rate": [float(x) for x in per_agent],
            "win_rate_team": per_team,
            "len": float(np.mean(lens)) if lens else 0.0,
            "episodes": total,
            **{f"end_{k}": v / total for k, v in endings.items()}}



def _flat(row, prefix=""):
    """Flatten one log row into wandb scalars.

    The row carries per-agent lists (`train_ret`, `fwd_per_step`, `ring`) and a
    nested `eval` dict; wandb wants scalars, and a list logged as a list is not
    plottable. Everything becomes `group/name` or `group/name_i`.
    """
    out = {}
    for k, v in row.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{prefix}{k}/"))
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                if isinstance(x, (int, float)):
                    out[f"{prefix}{k}_{i}"] = float(x)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f"{prefix}{k}"] = float(v)
    return out


def _wandb_init(args, kind):
    """Start a wandb run, or return None if disabled/unavailable.

    Never fatal: a training run must not die because a metrics service is
    unreachable. `--no-wandb` and a missing WANDB_API_KEY both just turn it off.
    """
    if getattr(args, "no_wandb", False):
        return None
    if not os.environ.get("WANDB_API_KEY"):
        print("[wandb] no WANDB_API_KEY -- logging to disk only", flush=True)
        return None
    try:
        import wandb
        name = os.path.basename(os.path.normpath(args.out))
        run = wandb.init(project=args.wandb_project, name=name, id=name,
                         config=vars(args), tags=["D2", kind],
                         resume="allow")
        print(f"[wandb] {run.url}", flush=True)
        return wandb
    except Exception as exc:                       # noqa: BLE001
        print(f"[wandb] disabled ({exc})", flush=True)
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, default=512)
    p.add_argument("--rollout", type=int, default=100)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--minutes", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--minibatch", type=int, default=2048)
    p.add_argument("--policy-lr", type=float, default=5e-5)
    p.add_argument("--value-lr", type=float, default=3e-4)
    p.add_argument("--delta", type=float, default=0.5)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--ring-capacity", type=int, default=512)
    p.add_argument("--checkpoint-every", type=int, default=1)
    p.add_argument("--save-policies-every", type=int, default=20)
    p.add_argument("--eval-worlds", type=int, default=64)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--down-rule", default="team_down")
    p.add_argument("--win-rule", default="team_first")
    p.add_argument("--goal-credit", default="team")
    p.add_argument("--creatures", default=None,
                   help="2h: comma-separated creature per agent, in the port's "
                        "agent order (A1, B1, A2, B2) -- e.g. "
                        "'ant,ant,spider,spider' for an ant/spider team on both "
                        "sides. Default: ant everywhere, i.e. the 2f/2g scene. "
                        "A MIXED composition requires --per-slot.")
    p.add_argument("--per-slot", action="store_true",
                   help="2h Option A: an independent actor-critic per (side, "
                        "slot) instead of one per side with a role one-hot. "
                        "Mutually exclusive with --role-in-design")
    p.add_argument("--role-in-design", action="store_true",
                   help="let the DESIGN head see the role one-hot. Off, both "
                        "teammates run the same function of the same random "
                        "draw and converge on the same body (measured: 0.052 "
                        "SMD). This is the smallest change that makes "
                        "morphological specialisation expressible.")
    p.add_argument("--warm-start", default=None,
                   help="a 1v1 policies.pt; ac_0/ac_1 widen into the two team "
                        "nets. Legal because team_policy orders opp_near first")
    p.add_argument("--out", default="runs/competevo_port/team_selfplay")
    p.add_argument("--wandb-project", default="creature-soccer")
    p.add_argument("--no-wandb", action="store_true")
    args = p.parse_args()

    from rower_soccer.competevo_port.dev_ppo import DevActorCritic
    from rower_soccer.competevo_port.selfplay import CoEvoPPO
    from rower_soccer.competevo_port.team_env import TeamRunToGoalDevEnv
    from rower_soccer.competevo_port.team_policy import (TeamActorCritic,
                                                         widen_from_1v1)

    os.makedirs(args.out, exist_ok=True)
    _wb = _wandb_init(args, "2v2")
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env_kw = dict(down_rule=args.down_rule, win_rule=args.win_rule,
                  goal_credit=args.goal_credit)
    if args.creatures:
        # Left as `scene_kwargs` rather than a first-class env argument: the
        # composition is a property of the SCENE, and `build_dev_team_scene`
        # is the one place that validates it.
        env_kw["scene_kwargs"] = {
            "creatures": [c.strip() for c in args.creatures.split(",")]}
    env = TeamRunToGoalDevEnv(num_worlds=args.worlds,
                              use_gpu=(device == "cuda"), seed=args.seed,
                              **env_kw)
    from rower_soccer.competevo_port.slot_policy import env_is_mixed, wrap_env
    mixed = env_is_mixed(env)
    if mixed:
        # A shared net per side cannot drive two different creatures: one
        # `design_dim`, one motor count, one own-state width. Option A is the
        # architecture 2h is built on, so this is a hard requirement, not a
        # default.
        assert args.per_slot, (
            f"composition {env.meta.creatures} is heterogeneous; it needs "
            "--per-slot (one actor-critic per (side, slot))")
        assert not args.warm_start, (
            "--warm-start widens a 1v1 ANT policy; there is no warm start for "
            "a mixed composition")
        print(f"heterogeneous composition {env.meta.creatures}: "
              f"design {env.meta.design_dims}, motors {env.meta.n_motors}, "
              f"obs {env.meta.obs_dims} -> padded {env.obs_dim}")

    if args.warm_start:
        blob = torch.load(args.warm_start, map_location="cpu")
        acs = []
        for key in ("ac_0", "ac_1"):
            src = DevActorCritic()
            src.load_state_dict(blob[key])
            acs.append(widen_from_1v1(src, n_agents=env.n_agents,
                                      role_in_design=args.role_in_design))
        print(f"warm start from {args.warm_start}")
    elif args.per_slot:
        # 2h Option A: one net per (side, SLOT) instead of one per side with a
        # role one-hot. Role becomes the identity of the network rather than an
        # input, so `--role-in-design` is meaningless here and refused rather
        # than silently ignored.
        from rower_soccer.competevo_port.slot_policy import from_env
        assert not args.role_in_design, (
            "--per-slot and --role-in-design are mutually exclusive: under "
            "per-slot the role IS the net")
        team = env.meta.team
        sides = [[i for i in range(env.n_agents) if team[i] == t]
                 for t in (0, 1)]
        acs = [from_env(env, s) for s in sides]
        print(f"per-slot policies (Option A): lanes {sides}, "
              f"{sum(p.numel() for p in acs[0].parameters()):,} params/side")
    else:
        acs = [TeamActorCritic(n_agents=env.n_agents,
                               role_in_design=args.role_in_design)
               for _ in range(2)]

    # The wrapper presents the POLICY's 58-dim observation; every consumer --
    # the trainer, the opponent stack, the evaluator -- must see the same one,
    # so it wraps the env once here rather than at each call site. On a mixed
    # composition `wrap_env` returns `MixedPolicyObsEnv` instead, which expands
    # the PADDED observation and leaves the per-slot gather to the policy.
    wrapped = wrap_env(env, acs[0])
    trainer = CoEvoPPO(wrapped, acs=acs, delta=args.delta,
                       ring_capacity=args.ring_capacity,
                       checkpoint_every=args.checkpoint_every,
                       blocks=args.blocks, rollout_len=args.rollout,
                       seed=args.seed, device=device, epochs=args.epochs,
                       minibatch_size=args.minibatch,
                       policy_lr=args.policy_lr, value_lr=args.value_lr)
    lanes = trainer.team_lanes
    # The policy width is per SLOT on a mixed team, so report what each net
    # actually consumes rather than slot 0's, which was the only one that ever
    # differed from the buffer width before 2h.
    widths = ([n.obs_dim for n in acs[0].nets] if hasattr(acs[0], "n_slots")
              else acs[0].obs_dim)
    print(f"worlds {env.n} x {env.n_agents} agents, teams "
          f"{[l.tolist() for l in lanes]}, rollout {args.rollout}, "
          f"obs {env.obs_dim} -> buffer {wrapped.obs_dim} -> nets {widths}")

    eval_env = TeamRunToGoalDevEnv(num_worlds=args.eval_worlds,
                                   use_gpu=(device == "cuda"),
                                   seed=args.seed + 7, **env_kw)
    eval_wrapped = wrap_env(eval_env, acs[0])

    rows, t0 = [], time.time()
    for it in range(args.iters):
        stats = trainer.train_iter()
        row = {"iter": it, "epoch": trainer.epoch,
               "sec": time.time() - t0, "steps": trainer.total_steps,
               "fwd_per_step": list(trainer.ep_fwd),
               "opp_lag": trainer.opponent_lag(),
               "ring": [len(r) for r in trainer.rings],
               "diverged": env.n_diverged,
               "design_std": float(acs[0].scale_log_std.exp().mean()),
               **{k: float(v) for k, v in stats.items()}}
        if args.eval_every and (it + 1) % args.eval_every == 0:
            [ac.eval() for ac in acs]
            row["eval"] = team_eval(eval_wrapped, acs, lanes)
        rows.append(row)
        if _wb is not None:
            _wb.log(_flat(row), step=int(row["steps"]))
        with open(os.path.join(args.out, "log.json"), "w") as f:
            json.dump({"args": vars(args), "iters": rows}, f, indent=1)
        ev = row.get("eval")
        print(f"  it {it:3d}  fwd {row['fwd_per_step'][0]:+.3f}/"
              f"{row['fwd_per_step'][1]:+.3f}  lag {row['opp_lag']:5.1f}  "
              f"ring {row['ring']}  nan {row['diverged']}"
              + (f"  | team win {[round(x, 3) for x in ev['win_rate_team']]} "
                 f"goal {ev['end_goal']:.2f} wipe {ev['end_wipeout']:.2f} "
                 f"len {ev['len']:.0f}" if ev else ""), flush=True)

        if args.save_policies_every and (it + 1) % args.save_policies_every == 0:
            torch.save({"ac_0": acs[0].state_dict(),
                        "ac_1": acs[1].state_dict(), "args": vars(args)},
                       os.path.join(args.out, f"policies_ep{it + 1:04d}.pt"))
        if args.minutes and (time.time() - t0) / 60.0 > args.minutes:
            print(f"stopping at the {args.minutes} minute budget")
            break

    torch.save({"ac_0": acs[0].state_dict(), "ac_1": acs[1].state_dict(),
                "args": vars(args)}, os.path.join(args.out, "policies.pt"))
    print(f"done: {len(rows)} iters, {(time.time() - t0) / 60:.1f} min, "
          f"{args.out}")


if __name__ == "__main__":
    main()
