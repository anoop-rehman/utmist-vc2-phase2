"""Can a morphology play this game AT ALL? -- the prerequisite the 2h sweep skipped.

The 27-run creature sweep ranked teams by win rate, and the ranking was
meaningless: every cell containing a bug or a spider times out 82-99% of the
time, so the "winner" is whoever loses less slowly. Ranking teams presupposes
that the creatures in them can reach the goal, and nothing established that.

This measures the prerequisite directly, and separates the two ways a team can
fail to score:

  CAN IT MOVE?   mean forward speed toward its own goal, and the fraction of
                 steps it spends upright. A creature that never learned to
                 locomote reads 0 here regardless of what the opponent does.
  CAN IT SCORE?  goal rate against an IDLE opponent -- one that receives zero
                 torque. With nobody to interfere, reaching the line is pure
                 locomotion plus navigation. A creature that moves well and
                 still cannot score against a statue has a navigation problem,
                 not a gait problem.

Both are read from policies the sweep already trained, so this costs minutes
and answers "did they learn to score?". It does NOT answer "could they learn,
given a fair task" -- these policies were trained against a live opponent and
may simply never have been paid for anything. That question needs its own
training run, and this script is what tells you whether it is worth launching.

    PYTHONPATH=. .venv/bin/python -m rower_soccer.competevo_port.competence_eval \\
        --runs runs/competevo_port/t2h_*_s42 --opponent idle

`--action {mean,sample}` exists because the training eval samples and the
renderers do not. It is NOT what explained the spider-vs-spider discrepancy
(83.4% wipeout in the training log against 100% timeout in the clip): sampling
reproduces the clip, and the real cause was averaging a REGIME CHANGE. That run
is 100% wipeout on 36-72 step episodes through iter 184 and 100% timeout on
500-step episodes by iter 194, so a mean over the last 8 evals reported a state
the policy was never in. Aggregate before comparing rates -- but check first
that the window you are aggregating holds one regime.
"""

import argparse
import collections
import glob
import json
import os

import numpy as np
import torch

from rower_soccer.competevo_port.scene import CONTROL_DT

END = {0: "running", 1: "goal", 2: "wipeout", 3: "fall", 4: "timeout"}


def evaluate(run, worlds, seed, opponent, action, steps):
    from rower_soccer.competevo_port.render_sweep import (build_env,
                                                          build_policies,
                                                          load_args)
    from rower_soccer.competevo_port.slot_policy import wrap_env

    a = load_args(run)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    env, _ = build_env(a, worlds, seed)
    acs, sides = build_policies(env, a, run, device)
    driver = wrap_env(env, acs[0])
    lanes = [torch.tensor(s, device=env.device) for s in sides]

    endings = collections.Counter()
    # Progress is accumulated the way `terms()` computes forward reward --
    # move_sign * d(com_x) -- so "forward" means each agent's OWN attacking
    # direction and the two sides are directly comparable.
    prog = torch.zeros(env.n, env.n_agents, device=env.device,
                       dtype=env.dtype)
    upright = torch.zeros(env.n, env.n_agents, device=env.device,
                          dtype=env.dtype)
    # Counted PER WORLD, because the reset mask below rejects different steps
    # in different worlds. A single global step count would divide every world
    # by the same denominator and quietly understate the ones that reset most.
    pcount = torch.zeros(env.n, 1, device=env.device, dtype=env.dtype)
    nstep = 0
    obs = driver.reset()
    env.reset_win_stats()
    with torch.no_grad():
        for _ in range(steps):
            o = obs.float()
            act = torch.zeros(env.n, env.n_agents, env.act_dim,
                              device=env.device, dtype=o.dtype)
            for e, ln in enumerate(lanes):
                # Side 1 is the opponent. Under `idle` it is left at exactly
                # zero torque -- not a frozen body, an unactuated one, which is
                # what makes the goal rate a pure locomotion-plus-navigation
                # measurement.
                if e == 1 and opponent == "idle":
                    continue
                if action == "sample":
                    act[:, ln] = acs[e].act(o[:, ln])[0]
                else:
                    act[:, ln] = acs[e].mean_action(o[:, ln])
            before = env._agent_com_x().clone()
            obs, _, done, info = driver.step(act.to(env.dtype))
            if not bool(info["was_design"][0]):
                d = env.move_sign * (env._agent_com_x() - before)
                # DROP THE RESET STEPS. On reset the body teleports back to
                # spawn, so `com_x - before` is a whole field-length of
                # NEGATIVE "progress" -- and it lands most often on the teams
                # that score most, which biases speed downward exactly where
                # it should be highest. `gate_drill_priors.roll` drops these
                # the same way. 0.5 m inside one 0.015 s control step would
                # be 33 m/s, so the threshold separates physics from teleports
                # with room to spare rather than by tuning.
                ok = (d.abs() < 0.5).all(-1, keepdim=True)
                prog += d * ok
                pcount += ok.to(env.dtype)
                upright += (env._root_z() >= 0.28).to(env.dtype)
                nstep += 1
            if bool(done.any()):
                idx = done.nonzero(as_tuple=True)[0]
                for e in env.last_end[idx].tolist():
                    endings[END[e]] += 1

    tot = max(1, sum(endings.values()))
    m = max(1, nstep)
    # Per-side means: agents 0,2 are side A and 1,3 are side B.
    side = [[i for i in range(env.n_agents) if env.meta.team[i] == t]
            for t in (0, 1)]
    return {
        "run": os.path.basename(run),
        "creatures": env.meta.creatures,
        "episodes": tot,
        "goal": endings["goal"] / tot,
        "wipeout": endings["wipeout"] / tot,
        "timeout": endings["timeout"] / tot,
        # -> m/s using THIS env's control step, and divided by each world's
        # own accepted-step count rather than a global one. Not the drills'
        # 0.025: the competevo scene is frame_skip 5 x timestep 0.003 =
        # 0.015 s, which is what `terms()` divides by to build forward_r.
        # Hardcoding the drill value understated every speed here by 1.67x.
        "speed_A": float((prog[:, side[0]] / pcount.clamp(min=1)).mean()) / CONTROL_DT,
        "speed_B": float((prog[:, side[1]] / pcount.clamp(min=1)).mean()) / CONTROL_DT,
        "upright_A": float(upright[:, side[0]].mean()) / m,
        "upright_B": float(upright[:, side[1]].mean()) / m,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--opponent", default="idle", choices=("idle", "policy"))
    p.add_argument("--action", default="mean", choices=("mean", "sample"))
    p.add_argument("--worlds", type=int, default=64)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    runs = sorted({r for pat in args.runs for r in glob.glob(pat)})
    rows = []
    for r in runs:
        if not os.path.exists(os.path.join(r, "policies.pt")):
            continue
        rows.append(evaluate(r, args.worlds, args.seed, args.opponent,
                             args.action, args.steps))
        d = rows[-1]
        print(f"  {d['run']:22s} {'/'.join(c[:2] for c in d['creatures'])}  "
              f"goal {100 * d['goal']:5.1f}%  "
              f"speedA {d['speed_A']:+6.3f} m/s  upA {100 * d['upright_A']:5.1f}%",
              flush=True)

    print(f"\nopponent={args.opponent}  action={args.action}  "
          f"worlds={args.worlds}  steps={args.steps}")
    print(f"{'run':22s} {'A':7s} {'B':7s} | {'goal%':>6s} {'wipe%':>6s} "
          f"{'t/o%':>6s} | {'A m/s':>7s} {'B m/s':>7s} | {'A up%':>6s} {'B up%':>6s}")
    print("-" * 92)
    for d in rows:
        print(f"{d['run']:22s} {d['creatures'][0][:7]:7s} "
              f"{d['creatures'][1][:7]:7s} | {100 * d['goal']:6.1f} "
              f"{100 * d['wipeout']:6.1f} {100 * d['timeout']:6.1f} | "
              f"{d['speed_A']:+7.3f} {d['speed_B']:+7.3f} | "
              f"{100 * d['upright_A']:6.1f} {100 * d['upright_B']:6.1f}")

    # Per-morphology roll-up over side A only: side B is idle under the default
    # and its speed is a measure of nothing.
    by = collections.defaultdict(list)
    for d in rows:
        by[d["creatures"][0]].append(d)
    print("\nside A by morphology (the side that is actually driven):")
    for k in sorted(by):
        v = by[k]
        print(f"  {k:8s} goal {100 * np.mean([x['goal'] for x in v]):5.1f}%   "
              f"speed {np.mean([x['speed_A'] for x in v]):+6.3f} m/s   "
              f"upright {100 * np.mean([x['upright_A'] for x in v]):5.1f}%   "
              f"n={len(v)}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"args": vars(args), "rows": rows}, fh, indent=1)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
