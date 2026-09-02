"""D3 M3 E2.1: the curriculum ablation's table, and the correlations.

    .venv/bin/python .../t2a_port/e21_analyse.py runs/d3_e21_curriculum/posthoc/*.json

E2's central finding is that RETURN DOES NOT MEASURE COMPETENCE on this task:
across its seven arms r(fall rate, return) = +0.989 and
r(forward progress, return) = +0.019. So the headline column here is FORWARD
PROGRESS, and both correlations are recomputed per condition.

Two levels, because the two answer different questions and E2 only had one:

  * EPISODE level, pooled within a condition (2 seeds x 20 episodes = 40).
    `fell` is 0/1 per episode. This is the well-powered one and it is what
    "does return track progress in this condition" means.
  * ARM level, over every arm's mean, which is E2's own statistic and is
    reported only so the two are comparable. With 2 arms per condition it
    cannot be computed within a condition and is given over the pooled set.
"""
import json
import sys

import numpy as np


def pear(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def main(paths):
    rows = []
    for p in paths:
        d = json.load(open(p))
        m, s = d["results"]["mean_action"], d["results"]["stochastic"]
        rows.append(dict(name=p.split("/")[-1][:-5], arm=d["arm"],
                         cfg=d["cfg"], tag=d.get("tag"), epoch=d["epoch"],
                         frozen=d["body_frozen"], std=d["action_std"],
                         m=m, s=s, eps=m["episodes"]))
    rows.sort(key=lambda r: r["name"])

    print("\n=== mean-action headline (stochastic in the last column) ===")
    hdr = ("arm", "R", "sd", "goal", "lost", "fell", "len", "fwd m",
           "%goal", "best m", "net dx", "a.std", "stoch R")
    print(f"{hdr[0]:<14}" + "".join(f"{h:>9}" for h in hdr[1:]))
    for r in rows:
        m = r["m"]
        print(f"{r['name']:<14}{m['R_mean']:9.1f}{m['R_sd']:9.1f}"
              f"{m['goal_rate']:9.2f}{m['loss_rate']:9.2f}"
              f"{m['fall_rate']:9.2f}{m['ep_len']:9.1f}"
              f"{m['max_fwd']:9.2f}{100 * m['frac_of_goal']:9.1f}"
              f"{m['max_fwd_best']:9.2f}{m['net_dx']:9.2f}"
              f"{r['std']:9.3f}{r['s']['R_mean']:9.1f}")

    print("\n=== body frozen under each arm's OWN trained policy ===")
    for r in rows:
        print(f"  {r['name']:<14} {'YES' if r['frozen'] else 'NO'}")

    # -------------------------------------------------- episode level -----
    def cond(pred):
        e = [ep for r in rows if pred(r) for ep in r["eps"]]
        return e

    groups = {}
    for r in rows:
        key = ("idle" if r["arm"] == "idle"
               else (r["tag"] or "untagged"))
        groups.setdefault(key, []).extend(r["eps"])

    print("\n=== EPISODE-LEVEL correlations, within each condition ===")
    print(f"{'condition':<12}{'n':>5}{'r(fell,R)':>12}{'r(fwd,R)':>12}"
          f"{'r(len,R)':>12}{'mean fwd':>10}{'fall rate':>11}{'goal':>7}")
    for k, e in sorted(groups.items()):
        if not e:
            continue
        R = [x["R"] for x in e]
        fell = [float(x["fell"]) for x in e]
        fwd = [x["max_fwd"] for x in e]
        n = [x["n"] for x in e]
        print(f"{k:<12}{len(e):5d}{pear(fell, R):12.3f}{pear(fwd, R):12.3f}"
              f"{pear(n, R):12.3f}{np.mean(fwd):10.2f}"
              f"{np.mean(fell):11.2f}"
              f"{np.mean([float(x['reached']) for x in e]):7.2f}")

    # -------------------------------------------------- arm level ---------
    print("\n=== ARM-LEVEL correlations (E2's own statistic) ===")
    R = [r["m"]["R_mean"] for r in rows]
    fell = [r["m"]["fall_rate"] for r in rows]
    fwd = [r["m"]["max_fwd"] for r in rows]
    print(f"  over {len(rows)} arms:  r(fall rate, R) = {pear(fell, R):+.3f}"
          f"   r(forward progress, R) = {pear(fwd, R):+.3f}")

    # -------------------------------------------------- by ending ---------
    print("\n=== decomposition by how the episode ended (mean-action) ===")
    print(f"{'arm':<14}{'fell n':>8}{'fell R':>10}{'fell len':>10}"
          f"{'lost n':>8}{'lost R':>10}{'lost len':>10}{'goal n':>8}")
    for r in rows:
        e = r["eps"]
        f = [x for x in e if x["fell"]]
        l = [x for x in e if x["opp_reached"]]
        g = [x for x in e if x["reached"]]
        fmt = lambda v, k: (f"{np.mean([x[k] for x in v]):10.1f}" if v
                            else f"{'-':>10}")
        print(f"{r['name']:<14}{len(f):8d}{fmt(f, 'R')}{fmt(f, 'n')}"
              f"{len(l):8d}{fmt(l, 'R')}{fmt(l, 'n')}{len(g):8d}")


if __name__ == "__main__":
    main(sys.argv[1:])
