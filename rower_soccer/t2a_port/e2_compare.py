"""D3 M3 E2: assemble the arms' post-hoc JSONs into the comparison table.

Reads what `e2_posthoc.py --out` wrote (nothing is recomputed here, and
nothing comes from a training log), prints the mean-action table with the
stochastic column beside it, and runs the episode-level Welch t-test between
the GNN and each MLP configuration -- the same statistic E1.1 reported.

    .venv/bin/python .../t2a_port/e2_compare.py runs/d3_e2_rtg/posthoc/*.json
"""
import argparse
import json
import math

import numpy as np


def welch(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return float("nan"), float("nan")
    t = (a.mean() - b.mean()) / se
    df = (va / na + vb / nb) ** 2 / ((va / na) ** 2 / (na - 1)
                                     + (vb / nb) ** 2 / (nb - 1))
    return t, df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    a = p.parse_args()
    runs = [json.load(open(f)) for f in a.files]

    def key(r):
        return (r["arm"], r.get("tag") or "matched", r["cfg"])
    runs.sort(key=key)

    print(f"\n{'arm':<22} {'seed cfg':<14} {'mean-action R':>14} {'sd':>8} "
          f"{'goal':>6} {'lost':>6} {'fell':>6} {'len':>7} {'dx m':>7} "
          f"{'m/s':>7} {'std':>7} | {'stochastic R':>13} {'goal':>6}")
    print("-" * 132)
    for r in runs:
        m, s = r["results"]["mean_action"], r["results"]["stochastic"]
        arm = r["arm"].upper() + (f" ({r['tag']})" if r.get("tag") else
                                  (" (matched)" if r["arm"] == "mlp" else ""))
        print(f"{arm:<22} {r['cfg']:<14} {m['R_mean']:14.1f} {m['R_sd']:8.1f} "
              f"{m['goal_rate']:6.2f} {m['loss_rate']:6.2f} {m['fall_rate']:6.2f} "
              f"{m['ep_len']:7.1f} {m['net_dx']:7.2f} {m['speed']:7.3f} "
              f"{r['action_std']:7.4f} | {s['R_mean']:13.1f} {s['goal_rate']:6.2f}")

    groups = {}
    for r in runs:
        g = r["arm"] + ("_" + r["tag"] if r.get("tag") else
                        ("_matched" if r["arm"] == "mlp" else ""))
        groups.setdefault(g, []).append(r)
    print("\nseed means (mean-action protocol)")
    means = {}
    for g, rs in sorted(groups.items()):
        v = [x["results"]["mean_action"]["R_mean"] for x in rs]
        gr = [x["results"]["mean_action"]["goal_rate"] for x in rs]
        means[g] = float(np.mean(v))
        print(f"  {g:<16} R {np.mean(v):9.1f}  seeds {[round(x, 1) for x in v]}"
              f"   goal rate {np.mean(gr):.3f}  seeds {[round(x, 3) for x in gr]}")

    if "gnn" in groups:
        ge = [e["R"] for r in groups["gnn"]
              for e in r["results"]["mean_action"]["episodes"]]
        for g, rs in sorted(groups.items()):
            if g == "gnn":
                continue
            me = [e["R"] for r in rs
                  for e in r["results"]["mean_action"]["episodes"]]
            t, df = welch(me, ge)
            ratio = means[g] / means["gnn"] if means["gnn"] else float("nan")
            print(f"\n  {g} vs gnn: seed-mean ratio {ratio:+.3f}x, "
                  f"episode-level diff {np.mean(me) - np.mean(ge):+.1f}, "
                  f"Welch t = {t:.2f} (df {df:.1f}, n {len(me)} vs {len(ge)})")
            print("    NOTE: a ratio computed from returns that are dominated "
                  "by the +/-1000 goal term is a ratio of MIXTURES, not of "
                  "speeds -- read the goal rate beside it.")


if __name__ == "__main__":
    main()
