"""D3 M3 E4 pre-registration: the comparison set for the convergence
discriminator, and the measured thresholds that decide "matches an E3.1 body".

Committed BEFORE E4 produces anything, so that "does E4's converged body match
one of E3.1's three" is a threshold test rather than a judgement call made
after seeing the answer.

Thresholds are measured, not chosen:

  MATCH    SMD <= 0.44   the pooled p95 of WITHIN-lineage drift over a 40-epoch
                         lag in E3.1's late window -- i.e. the largest distance
                         a single lineage moves while still being "the same
                         body". (lag 10 -> 0.281, lag 20 -> 0.367, lag 40 ->
                         0.437; the most permissive is used.)
  DISTINCT SMD >= 0.75   the p05 of the BETWEEN-seed null in the same window --
                         i.e. as far apart as two independent searches get.
  between  AMBIGUOUS     reported as such, never rounded to either verdict.

The two scales do not overlap (0.44 vs 0.75), which is what makes the test
meaningful: "same body" and "independently different body" are separated by a
factor of 1.7 with a gap between them.

Re-run with --refresh once rtg_e31_s1 reaches epoch 400 to finalise its entry.
"""
import json, os, sys, numpy as np

R = "/workspace/Transform2Act/results"
SEEDS = ["rtg_e31_s1", "rtg_e31_s2", "rtg_e31_s3"]
OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "t2a", "e4_null")
MATCH, DISTINCT = 0.44, 0.75


def load(s):
    return {json.loads(l)["epoch"]: json.loads(l)
            for l in open(f"{R}/{s}/e3_epochs.jsonl")}


def smd(ga, gb, sd):
    sh = [k for k in ga if k in gb and k != "0"]
    if not sh:
        return None, 0
    raw = np.abs(np.asarray([ga[k] for k in sh]) - np.asarray([gb[k] for k in sh]))
    return float((raw / sd).mean()), len(sh)


def build():
    runs = {s: load(s) for s in SEEDS}
    entries = {}
    for s, r in runs.items():
        # the latest epoch carrying BOTH a design and an eval, so the body and
        # the score it earned are the same epoch's (eval runs every Nth epoch)
        e = max(k for k, v in r.items()
                if "eval" in v and "mean_action_design" in v)
        ma, ev = r[e]["mean_action_design"], r[e]["eval"]
        entries[s] = dict(
            epoch=e, final=(e >= 395), topo=ma["topo"], names=ma["names"],
            n_bodies=ma["n_bodies"], n_motors=ma["model_nu_ours"],
            mass=ma["model_mass_ours"], limb_len_sum=ma["limb_length"]["sum"],
            genome=ma["genome"],
            sampled_genome_std=r[e]["census"]["sampled_genome_std"],
            own_eval=dict(goal=ev["goal_rate"], speed=ev["speed"],
                          ep_len=ev["ep_len"], max_fwd=ev["max_fwd"]))
    return entries


def main():
    entries = build()
    # pairwise distances within the comparison set itself, for context
    pw = {}
    for a in SEEDS:
        for b in SEEDS:
            if a >= b:
                continue
            sd = np.maximum(np.sqrt(np.mean(
                [np.square(entries[x]["sampled_genome_std"]) for x in (a, b)],
                axis=0)), 1e-3)
            d, n = smd(entries[a]["genome"], entries[b]["genome"], sd)
            na, nb = set(entries[a]["names"]), set(entries[b]["names"])
            pw[f"{a[-2:]}-{b[-2:]}"] = dict(smd=d, n_shared=n,
                                            jaccard=len(na & nb) / len(na | nb))
    doc = dict(
        what="Pre-registered comparison set and match thresholds for D3 M3 E4's "
             "convergence discriminator. If E4's lineages converge, the converged "
             "body is compared against these three E3.1 bodies: a MATCH means the "
             "race merely selected a pre-existing optimum (task-limited); a "
             "DISTINCT body none of E3.1's seeds found means selection genuinely "
             "sharpened (architecture-limited).",
        committed="2026-09-05, before any E4 run existed",
        generated_by="rower_soccer/t2a_port/e4_compset.py",
        thresholds=dict(
            match_smd_max=MATCH, distinct_smd_min=DISTINCT,
            match_basis="pooled p95 of within-lineage SMD drift over a 40-epoch "
                        "lag, E3.1 late window (lag10 0.281 / lag20 0.367 / "
                        "lag40 0.437)",
            distinct_basis="p05 of the between-seed null SMD, same window",
            rule="SMD to the NEAREST comparison-set body decides. <=0.44 MATCH; "
                 ">=0.75 DISTINCT; between is AMBIGUOUS and reported as such.",
            standardiser="pooled sampled-population genome std of the two "
                         "designs being compared, floored at 1e-3, exactly as "
                         "e0_analyse.genome_std"),
        caveat_s3="s3's own final controller scored goal 0.00; its BODY is in the "
                  "set because the frozen-body diagnostic (rtg_e31d_s3body, which "
                  "freezes this same epoch-400 design) showed the body reaches the "
                  "goal with a fresh controller. The set is of BODIES, not of "
                  "controllers.",
        caveat_s1=("s1 was still running at epoch %d when this was committed; its "
                   "entry is PROVISIONAL. Re-run with --refresh at epoch 400. This "
                   "does not affect the thresholds, which come from pooled "
                   "statistics." % entries["rtg_e31_s1"]["epoch"]),
        all_final=all(e["final"] for e in entries.values()),
        within_set_distances=pw,
        bodies=entries)
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "e31_comparison_set.json")
    json.dump(doc, open(p, "w"), indent=1)
    print("wrote", os.path.normpath(p))
    print("all_final =", doc["all_final"])
    for s, e in entries.items():
        print("  %-12s e%-4d %s  %2db/%dm  goal %.2f speed %6.3f  %s"
              % (s, e["epoch"], e["topo"], e["n_bodies"], e["n_motors"],
                 e["own_eval"]["goal"], e["own_eval"]["speed"],
                 "FINAL" if e["final"] else "PROVISIONAL"))
    print("  within-set distances:",
          {k: round(v["smd"], 3) for k, v in pw.items()})


if __name__ == "__main__":
    main()
