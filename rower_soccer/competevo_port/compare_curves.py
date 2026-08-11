"""Milestone 2e: align our per-iteration log against their per-epoch log.

    PYTHONPATH=. python -m rower_soccer.competevo_port.compare_curves \
        --ours runs/competevo_port/m2e_validation/log.json \
        --theirs /workspace/competevo/sanity_run.log

Their run of `config/run-to-goal-devants-v0.yaml` prints, once per epoch,

    Agent_i gets eval reward: R.
    Agent_i gets win rate: W.

`R` is the CURRICULUM reward at that epoch's alpha, not the env reward -- their
eval sampler goes through `custom_reward` exactly as training does. The column
of ours that is comparable is therefore `eval_ret_curriculum`, which is why
`evaluate_pair` had to learn to compute it. `eval_ret` (the raw env return) is
printed alongside so the size of the difference is visible rather than assumed.

The x axes need no rescaling: with `--worlds 1000 --rollout 100` one of our
iterations trains on 50,000 ego transitions per learner, which is their
`min_batch_size`, so iteration == epoch. `--assert-epoch-mapping` fails loudly
if the log says otherwise.
"""

import argparse
import json
import re

import numpy as np

ANSI = re.compile(r"\x1b\[[0-9;]*m")
IT = re.compile(r"Iteration (\d+)")
REW = re.compile(r"Agent_(\d) gets eval reward: (-?[\d.]+?)\.\s*$")
WIN = re.compile(r"Agent_(\d) gets win rate: (-?[\d.]+?)\.\s*$")
SAMPLE = re.compile(r"Sampling (\d+) steps by (\d+) slaves, spending ([\d.]+) s")
UPDATE = re.compile(r"Policy update, spending: ([\d.]+) s")


def parse_theirs(path):
    """{epoch: {r0, r1, w0, w1, sec}} from their runner's stdout log."""
    rows, ep = {}, None
    with open(path, errors="replace") as f:
        for raw in f:
            line = ANSI.sub("", raw).strip()
            m = IT.search(line)
            if m:
                ep = int(m.group(1))
                rows.setdefault(ep, {})
                continue
            if ep is None:
                continue
            m = REW.search(line)
            if m:
                rows[ep][f"r{m.group(1)}"] = float(m.group(2))
                continue
            m = WIN.search(line)
            if m:
                rows[ep][f"w{m.group(1)}"] = float(m.group(2))
                continue
            m = SAMPLE.search(line)
            if m:
                rows[ep]["sample_s"] = float(m.group(3))
                continue
            m = UPDATE.search(line)
            if m:
                rows[ep]["update_s"] = float(m.group(1))
    return {k: v for k, v in rows.items() if "r0" in v and "w0" in v}


def parse_ours(path):
    """{iter: row} for the iterations that carry an eval."""
    log = json.load(open(path))
    return log, {r["iter"]: r for r in log["iters"] if "eval_win" in r}


def first_nonzero_win(rows, key0, key1):
    for k in sorted(rows):
        if rows[k].get(key0, 0) > 0 or rows[k].get(key1, 0) > 0:
            return k
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ours", default="runs/competevo_port/m2e_validation/log.json")
    p.add_argument("--theirs", default="/workspace/competevo/sanity_run.log")
    p.add_argument("--every", type=int, default=10)
    p.add_argument("--csv", default=None)
    args = p.parse_args()

    theirs = parse_theirs(args.theirs)
    log, ours = parse_ours(args.ours)
    a = log["args"]
    per_iter = (a["worlds"] // 2) * a["rollout"]
    print(f"ours:   {a['worlds']} worlds x rollout {a['rollout']} "
          f"=> {per_iter:,} ego transitions/learner/iteration "
          f"({'== their min_batch_size 50,000' if per_iter == 50_000 else 'MISMATCH vs their 50,000 -- the epoch axes do NOT line up'})")
    print(f"        {len(log['iters'])} iterations, {len(ours)} evals, "
          f"{log['iters'][-1]['sec'] / 60:.1f} min, "
          f"{log['iters'][-1]['steps'] / log['iters'][-1]['sec']:,.0f} "
          f"ego-transitions/s")
    print(f"theirs: {len(theirs)} epochs with an eval, max epoch {max(theirs)}")

    fn_t = first_nonzero_win(theirs, "w0", "w1")
    fn_o = first_nonzero_win({k: {"w0": v["eval_win"][0], "w1": v["eval_win"][1]}
                              for k, v in ours.items()}, "w0", "w1")
    print(f"\nfirst epoch with a nonzero eval win rate: "
          f"theirs {fn_t}, ours {fn_o}")

    hdr = (f"{'epoch':>6} | {'alpha':>6} | {'THEIRS eval R':>15} "
           f"{'win':>11} | {'OURS eval R (curric)':>21} {'win':>11} "
           f"| {'ours env R':>15} {'len':>5} {'games':>5}")
    print("\n" + hdr)
    print("-" * len(hdr))
    keys = sorted(set(theirs) | set(ours))
    for k in keys:
        if k % args.every and k not in (0, max(keys)):
            continue
        t = theirs.get(k)
        o = ours.get(k)
        ts = (f"{t['r0']:7.1f}/{t['r1']:7.1f} {t['w0']:5.2f}/{t['w1']:5.2f}"
              if t else f"{'--':>15} {'--':>11}")
        if o:
            c = o.get("eval_ret_curriculum", o["eval_ret"])
            os_ = (f"{c[0]:10.1f}/{c[1]:10.1f} "
                   f"{o['eval_win'][0]:5.2f}/{o['eval_win'][1]:5.2f}")
            ex = (f"{o['eval_ret'][0]:7.1f}/{o['eval_ret'][1]:7.1f} "
                  f"{o['eval_len']:5.0f} {o.get('eval_games', 0):5d}")
            al = f"{o['eval_alpha']:6.3f}" if o.get("eval_alpha") is not None else "    --"
        else:
            os_, ex, al = f"{'--':>21} {'--':>11}", f"{'--':>15} {'--':>5} {'--':>5}", "    --"
        print(f"{k:6d} | {al} | {ts} | {os_} | {ex}")

    # Where both sides have a number, how far apart are they?
    both = [k for k in keys if k in theirs and k in ours]
    if both:
        dt = np.array([[theirs[k]["r0"], theirs[k]["r1"]] for k in both])
        do = np.array([ours[k].get("eval_ret_curriculum", ours[k]["eval_ret"])
                       for k in both])
        wt = np.array([[theirs[k]["w0"], theirs[k]["w1"]] for k in both])
        wo = np.array([ours[k]["eval_win"] for k in both])
        print(f"\noverlap: {len(both)} epochs "
              f"[{min(both)}, {max(both)}]")
        print(f"  mean eval reward   theirs {dt.mean():8.1f}   ours {do.mean():8.1f}")
        print(f"  mean win rate      theirs {wt.mean():8.3f}   ours {wo.mean():8.3f}")
        print(f"  corr(reward)  agent0 {np.corrcoef(dt[:, 0], do[:, 0])[0, 1]:+.3f}"
              f"  agent1 {np.corrcoef(dt[:, 1], do[:, 1])[0, 1]:+.3f}"
              if len(both) > 2 else "")

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("epoch,alpha,their_r0,their_r1,their_w0,their_w1,"
                    "our_cur_r0,our_cur_r1,our_env_r0,our_env_r1,"
                    "our_w0,our_w1,our_len,our_games\n")
            for k in keys:
                t, o = theirs.get(k, {}), ours.get(k)
                c = o.get("eval_ret_curriculum", o["eval_ret"]) if o else [None, None]
                f.write(",".join("" if v is None else str(v) for v in [
                    k, o.get("eval_alpha") if o else None,
                    t.get("r0"), t.get("r1"), t.get("w0"), t.get("w1"),
                    c[0], c[1],
                    o["eval_ret"][0] if o else None,
                    o["eval_ret"][1] if o else None,
                    o["eval_win"][0] if o else None,
                    o["eval_win"][1] if o else None,
                    o["eval_len"] if o else None,
                    o.get("eval_games") if o else None]) + "\n")
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
