"""D3 M3 E4: the pre-registered divergence statistic.

    Delta(e) = D_self(e) - D_null(e)

  D_self  distance between the two lineages WITHIN a run -- the pair that
          co-evolved against each other (3 pairs, one per seed).
  D_null  distance between lineages in DIFFERENT runs, role-matched a-to-a and
          b-to-b (12 pairs).

Both at the same epoch, in the same experiment, under identical conditions;
the only difference is whether the pair co-evolved. The null is INTERNAL to
E4 on purpose -- comparing against E3.1's cross-seed numbers would confound
co-evolution with opponent type (scripted vs learned).

Distances are the two already implemented in `e0_analyse.compare`, computed
from each epoch's `mean_action_design` and standardised by the pooled
`sampled_genome_std` population at that same epoch:

  SMD      mean |delta genome| over shared bodies / sampled-population std.
           PRIMARY.
  Jaccard  on body-name sets. Secondary -- it barely moves.

VERDICT, fixed before running (D3_E4_PREREQ.md section D), on epochs 200-400,
aggregating per-pair distances into the window mean BEFORE comparing:

  DIVERGENCE   mean Delta >= +0.15 and Delta(e) > 0 in >= 80% of epochs
  CONVERGENCE  mean Delta <= -0.15 and Delta(e) < 0 in >= 80% of epochs
  NULL         anything else

0.15 is 3.1 standard errors of the measured pair spread. The criterion is on
the TRAJECTORY, not the endpoint, because the cross-seed null rises
monotonically through training (SMD 0.17 -> 0.93 on E3.1) -- a fixed threshold
against a final-epoch number would be measuring elapsed training.

Two guards report a silenced channel as UNTESTABLE rather than null:
  * draw rate > 50% over the window -- mirror-symmetric lineages arriving on
    the same physics step score parse = 0, so the coupled term is off;
  * both lineages above goal 0.95 for > 100 consecutive epochs -- returns stop
    differentiating. Such a run's Delta is reported separately, not pooled.
"""
import argparse, itertools, json, os
import numpy as np

R = "/workspace/Transform2Act/results"
LO, HI = 200, 400
THRESH, SIGN_FRAC = 0.15, 0.80
DRAW_MAX, CEIL_GOAL, CEIL_RUN = 0.50, 0.95, 100


def load(cfg):
    p = f"{R}/{cfg}/e4_epochs.jsonl"
    if not os.path.exists(p):
        return None
    out = {}
    for l in open(p):
        try:
            r = json.loads(l)
        except Exception:
            continue
        out[r["epoch"]] = r
    return out


def dist(ra, rb, sd):
    ga = ra["mean_action_design"]["genome"]
    gb = rb["mean_action_design"]["genome"]
    na = set(ra["mean_action_design"]["names"])
    nb = set(rb["mean_action_design"]["names"])
    jac = len(na & nb) / len(na | nb)
    shared = [k for k in ga if k in gb and k != "0"]
    if not shared:
        return jac, None
    raw = np.abs(np.asarray([ga[k] for k in shared])
                 - np.asarray([gb[k] for k in shared]))
    return jac, float((raw / sd).mean())


def std_at(rows):
    var = np.mean([np.square(r["census"]["sampled_genome_std"]) for r in rows],
                  axis=0)
    return np.maximum(np.sqrt(var), 1e-3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1,2,3")
    ap.add_argument("--out", default="/workspace/utmist-vc2-phase2/rower_soccer/"
                                     "docs/t2a/e4_null/e4_divergence.json")
    a = ap.parse_args()
    seeds = [s.strip() for s in a.seeds.split(",") if s.strip()]

    runs = {}
    for s in seeds:
        for L in "ab":
            c = f"rtg_e4_s{s}{L}"
            r = load(c)
            if r:
                runs[(s, L)] = r
    if not runs:
        print("no E4 runs found yet")
        return
    for k, v in sorted(runs.items()):
        print(f"  rtg_e4_s{k[0]}{k[1]}: {len(v)} epochs")

    epochs = sorted(set.intersection(*[set(v) for v in runs.values()]))
    if not epochs:
        print("no common epochs yet")
        return

    rows = []
    for e in epochs:
        present = {k: v[e] for k, v in runs.items() if e in v}
        for (ka, kb) in itertools.combinations(sorted(present), 2):
            sa, La = ka
            sb, Lb = kb
            within = (sa == sb)
            if not within and La != Lb:
                kind = "cross-role"          # reported as a check only
            else:
                kind = "self" if within else "null"
            sd = std_at([present[ka], present[kb]])
            jac, smd = dist(present[ka], present[kb], sd)
            rows.append(dict(epoch=e, a=f"{sa}{La}", b=f"{sb}{Lb}",
                             kind=kind, jaccard=jac, smd=smd))

    def wmean(kind, e=None):
        v = [r["smd"] for r in rows if r["kind"] == kind
             and r["smd"] is not None and (e is None or r["epoch"] == e)
             and (e is not None or LO <= r["epoch"] <= HI)]
        return float(np.mean(v)) if v else None

    per_epoch = []
    for e in epochs:
        ds, dn = wmean("self", e), wmean("null", e)
        if ds is not None and dn is not None:
            per_epoch.append((e, ds - dn, ds, dn))

    win = [(e, d) for e, d, _, _ in per_epoch if LO <= e <= HI]
    out = dict(window=[LO, HI], threshold=THRESH, sign_frac=SIGN_FRAC,
               n_epochs_window=len(win), rows=rows,
               per_epoch=[dict(epoch=e, delta=d, d_self=s, d_null=n)
                          for e, d, s, n in per_epoch])

    # ---- degeneracy guards ------------------------------------------------
    draws, ceil = {}, {}
    for k, v in runs.items():
        dr = [x["race"]["draw_rate"] for x in v.values()
              if "race" in x and "draw_rate" in x["race"] and LO <= x["epoch"] <= HI]
        draws[f"{k[0]}{k[1]}"] = float(np.mean(dr)) if dr else None
        g = [(x["epoch"], x["eval"]["goal_rate"]) for x in v.values() if "eval" in x]
        g.sort()
        run_len, best = 0, 0
        for _, gr in g:
            run_len = run_len + 1 if gr > CEIL_GOAL else 0
            best = max(best, run_len)
        ceil[f"{k[0]}{k[1]}"] = best
    out["draw_rate_window"] = draws
    out["max_consecutive_evals_above_goal_%.2f" % CEIL_GOAL] = ceil
    hi_draw = [k for k, v in draws.items() if v is not None and v > DRAW_MAX]

    # ---- verdict ----------------------------------------------------------
    if not win:
        verdict, detail = "PENDING", f"no epochs in [{LO},{HI}] yet"
    elif hi_draw:
        verdict = "UNTESTABLE"
        detail = (f"draw rate above {DRAW_MAX:.0%} for {hi_draw} -- the coupled "
                  f"channel is off, so this is not a null result")
    else:
        m = float(np.mean([d for _, d in win]))
        pos = float(np.mean([d > 0 for _, d in win]))
        neg = float(np.mean([d < 0 for _, d in win]))
        if m >= THRESH and pos >= SIGN_FRAC:
            verdict = "DIVERGENCE"
        elif m <= -THRESH and neg >= SIGN_FRAC:
            verdict = "CONVERGENCE"
        else:
            verdict = "NULL"
        detail = (f"mean Delta {m:+.3f} over {len(win)} epochs; "
                  f"Delta>0 in {pos:.0%}, Delta<0 in {neg:.0%}")
        out.update(mean_delta=m, frac_positive=pos, frac_negative=neg)
    out["verdict"] = verdict
    out["verdict_detail"] = detail

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\n  D_self (window) {wmean('self')}   D_null (window) {wmean('null')}")
    print(f"  draw rate: {draws}")
    print(f"  VERDICT: {verdict} -- {detail}")
    print(f"  -> {a.out}")


if __name__ == "__main__":
    main()
