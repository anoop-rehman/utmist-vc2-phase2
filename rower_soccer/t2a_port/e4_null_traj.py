"""D3 M3 E4: regenerate the cross-seed morphology-distance NULL from E3.1.

Establishes two things E4's verdict depends on:

  (a) the null RISES MONOTONICALLY with epoch (SMD 0.17 -> 0.93), so a fixed
      threshold applied to a final-epoch number would be measuring elapsed
      training rather than divergence -- which is why E4's criterion is on the
      trajectory;
  (b) the between-pair spread, which calibrates E4's +-0.15 threshold at
      3.1 standard errors.

Writes `docs/t2a/e4_null/e31_crossseed_null.json`. The file carries a
`calibration_status` of PROVISIONAL until every seed has reached epoch 400,
and MUST NOT be cited for an E4 verdict while it says so -- with a seed still
running, the last window rests on a truncated epoch range for every pair that
seed belongs to.

    python3 e4_null_traj.py            # regenerate and report
"""
import itertools, json, os
import numpy as np

R = "/workspace/Transform2Act/results"
SEEDS = ["rtg_e31_s1", "rtg_e31_s2", "rtg_e31_s3"]
OUT = ("/workspace/utmist-vc2-phase2/rower_soccer/docs/t2a/e4_null/"
       "e31_crossseed_null.json")
FINAL_EPOCH = 399
WINDOWS = [(0, 49), (50, 99), (100, 199), (200, 299), (300, 399)]
LO = 200


def load(cfg):
    p = f"{R}/{cfg}/e3_epochs.jsonl"
    if not os.path.exists(p):
        return None
    out = {}
    for l in open(p):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if "mean_action_design" in r and "census" in r:
            out[r["epoch"]] = r
    return out


def main():
    runs = {s: load(s) for s in SEEDS}
    runs = {s: v for s, v in runs.items() if v}
    for s, v in runs.items():
        print(f"  {s}: {len(v)} epochs, last {max(v)}")
    epochs = sorted(set.intersection(*[set(v) for v in runs.values()]))
    rows = []
    for e in epochs:
        sd = np.maximum(np.sqrt(np.mean(
            [np.square(runs[s][e]["census"]["sampled_genome_std"])
             for s in runs], axis=0)), 1e-3)
        for a, b in itertools.combinations(sorted(runs), 2):
            ga = runs[a][e]["mean_action_design"]["genome"]
            gb = runs[b][e]["mean_action_design"]["genome"]
            na = set(runs[a][e]["mean_action_design"]["names"])
            nb = set(runs[b][e]["mean_action_design"]["names"])
            shared = [k for k in ga if k in gb and k != "0"]
            smd = None
            if shared:
                raw = np.abs(np.asarray([ga[k] for k in shared])
                             - np.asarray([gb[k] for k in shared]))
                smd = float((raw / sd).mean())
            rows.append(dict(epoch=e, pair=f"{a[-2:]}-{b[-2:]}",
                             jaccard=len(na & nb) / len(na | nb), smd=smd,
                             n_shared=len(shared),
                             same_topo=(runs[a][e]["mean_action_design"]["topo"]
                                        == runs[b][e]["mean_action_design"]["topo"])))

    late = [r for r in rows if r["epoch"] >= LO and r["smd"] is not None]
    by_pair = {}
    for p in sorted({r["pair"] for r in late}):
        v = [r["smd"] for r in late if r["pair"] == p]
        by_pair[p] = dict(mean=float(np.mean(v)), sd=float(np.std(v)), n=len(v))
    pm = [v["mean"] for v in by_pair.values()]
    pair_sd = float(np.std(pm, ddof=1)) if len(pm) > 1 else float("nan")
    se = float(np.sqrt(pair_sd ** 2 / 3 + pair_sd ** 2 / 12))

    last = max(epochs)
    final = all(max(v) >= FINAL_EPOCH for v in runs.values())
    status = ("FINAL. Every seed reached epoch %d; this file may be cited as "
              "the calibration for an E4 verdict." % FINAL_EPOCH) if final else (
              "PROVISIONAL. epochs_common ends at %d because at least one seed "
              "is still running, so the last window rests on a truncated range "
              "for every pair that seed belongs to. Regenerate with "
              "e4_null_traj.py once all seeds reach %d. MUST NOT be cited for "
              "an E4 verdict while it says PROVISIONAL." % (last, FINAL_EPOCH))

    doc = dict(
        what="Cross-seed morphology-distance NULL from D3 E3.1's three seeds. "
             "Establishes (a) that the null RISES monotonically with epoch, so "
             "an endpoint threshold measures elapsed training, and (b) the "
             "between-pair spread that calibrates E4's +-0.15 threshold.",
        source=f"{R}/rtg_e31_s{{1,2,3}}/e3_epochs.jsonl "
               "(per-epoch mean_action_design + census.sampled_genome_std)",
        generated_by="rower_soccer/t2a_port/e4_null_traj.py",
        calibration_status=status,
        all_seeds_final=final,
        last_epoch_per_seed={s: max(v) for s, v in runs.items()},
        epochs_common=[int(epochs[0]), int(epochs[-1])], n_rows=len(rows),
        window_means_smd={f"{lo}-{hi}": float(np.mean(
            [r["smd"] for r in rows if lo <= r["epoch"] <= hi
             and r["smd"] is not None]) or 0.0) for lo, hi in WINDOWS},
        late_half_pooled=dict(
            smd_mean=float(np.mean([r["smd"] for r in late])),
            smd_sd=float(np.std([r["smd"] for r in late])),
            jaccard_mean=float(np.mean([r["jaccard"] for r in late]))),
        late_half_by_pair=by_pair,
        threshold_derived=dict(
            pair_sd=pair_sd, se_delta=se, threshold=0.15,
            threshold_in_se=(0.15 / se) if se else None,
            formula="SE(delta) = pair_sd * sqrt(1/3 + 1/12), for 3 within-run "
                    "and 12 between-run role-matched pairs"),
        rows=rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(doc, open(OUT, "w"), indent=1)

    print("\n  window means (SMD):",
          {k: round(v, 3) for k, v in doc["window_means_smd"].items()})
    print("  late-half by pair :",
          {k: round(v["mean"], 3) for k, v in by_pair.items()})
    print("  pair sd %.4f -> SE(delta) %.4f -> 0.15 is %.1f SE"
          % (pair_sd, se, 0.15 / se if se else float("nan")))
    print("  status:", "FINAL" if final else "PROVISIONAL")
    print("  ->", OUT)


if __name__ == "__main__":
    main()
