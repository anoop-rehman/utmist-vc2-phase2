import json, numpy as np, itertools, os
SEEDS = ["rtg_e31_s1", "rtg_e31_s2", "rtg_e31_s3"]
R = "/workspace/Transform2Act/results"

def load(s):
    p = f"{R}/{s}/e3_epochs.jsonl"
    if not os.path.exists(p): return None
    out = {}
    for l in open(p):
        r = json.loads(l)
        out[r["epoch"]] = r
    return out

runs = {s: load(s) for s in SEEDS}
for s, v in runs.items():
    print(f"{s}: {len(v) if v else 0} epochs")
runs = {s: v for s, v in runs.items() if v}

def dists(ra, rb, sd):
    ga, gb = ra["mean_action_design"]["genome"], rb["mean_action_design"]["genome"]
    na, nb = set(ra["mean_action_design"]["names"]), set(rb["mean_action_design"]["names"])
    J = len(na & nb) / len(na | nb)
    shared = [k for k in ga if k in gb and k != "0"]
    if not shared: return J, None, 0
    raw = np.abs(np.asarray([ga[k] for k in shared]) - np.asarray([gb[k] for k in shared]))
    return J, float((raw / sd).mean()), len(shared)

epochs = sorted(set.intersection(*[set(v) for v in runs.values()]))
print(f"common epochs: {len(epochs)} ({epochs[0]}..{epochs[-1]})")
rows = []
for e in epochs:
    sd = np.maximum(np.sqrt(np.mean(
        [np.square(runs[s][e]["census"]["sampled_genome_std"]) for s in runs], axis=0)), 1e-3)
    for a, b in itertools.combinations(sorted(runs), 2):
        J, smd, ns = dists(runs[a][e], runs[b][e], sd)
        rows.append(dict(epoch=e, pair=f"{a[-2:]}-{b[-2:]}", J=J, smd=smd, nshared=ns,
                         same=runs[a][e]["mean_action_design"]["topo"] == runs[b][e]["mean_action_design"]["topo"]))
json.dump(rows, open("/tmp/claude-0/-root/453bc0de-a27f-4894-ad03-7d048158ee36/scratchpad/null_traj.json","w"))

import collections
print("\n=== cross-seed distance by epoch window (the NULL: independent runs, same task) ===")
print(f"{'window':>10} {'pair':>7} {'Jaccard':>18} {'SMD':>18} {'same topo':>10}")
wins = [(0,49),(50,99),(100,199),(200,299),(300,399)]
for lo,hi in wins:
    for p in sorted(set(r["pair"] for r in rows)):
        sel=[r for r in rows if lo<=r["epoch"]<=hi and r["pair"]==p]
        if not sel: continue
        Js=[r["J"] for r in sel]; Ss=[r["smd"] for r in sel if r["smd"] is not None]
        print(f"{lo:>4}-{hi:<5} {p:>7} {np.mean(Js):>8.3f}+-{np.std(Js):<8.3f} "
              f"{np.mean(Ss):>8.3f}+-{np.std(Ss):<8.3f} {100*np.mean([r['same'] for r in sel]):>8.0f}%")
    print()
allS=[r["smd"] for r in rows if r["smd"] is not None]; allJ=[r["J"] for r in rows]
print(f"POOLED over all pairs/epochs: Jaccard {np.mean(allJ):.3f}+-{np.std(allJ):.3f}   SMD {np.mean(allS):.3f}+-{np.std(allS):.3f}")
late=[r for r in rows if r["epoch"]>=200]
lS=[r["smd"] for r in late if r["smd"] is not None]; lJ=[r["J"] for r in late]
print(f"LATE HALF (>=200)          : Jaccard {np.mean(lJ):.3f}+-{np.std(lJ):.3f}   SMD {np.mean(lS):.3f}+-{np.std(lS):.3f}")
print(f"  SMD percentiles (late): p05 {np.percentile(lS,5):.3f}  p50 {np.percentile(lS,50):.3f}  p95 {np.percentile(lS,95):.3f}")
print(f"  Jac percentiles (late): p05 {np.percentile(lJ,5):.3f}  p50 {np.percentile(lJ,50):.3f}  p95 {np.percentile(lJ,95):.3f}")
