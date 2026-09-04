#!/bin/bash
# D3 M3 E3: distil each arm's per-epoch JSONL into a compact CSV inside the
# REPO, so every number a decision rests on is readable from the filesystem
# without wandb and without the Transform2Act results tree.
#
# The trainer already writes `results/<cfg>/e3_epochs.jsonl` -- full genome,
# census, evaluation -- but that lives outside the repo and is not tracked.
# This mirrors the decision-relevant columns, motor count first among them,
# into runs/d3_e3_adversarial/census/<cfg>_morph.csv every 5 minutes.
#
#   setsid nohup ./census_sidecar.sh > logs/census_sidecar.log 2>&1 &
set -uo pipefail
D=/workspace/utmist-vc2-phase2/runs/d3_e3_adversarial
P=/workspace/utmist-vc2-phase2/.venv/bin/python
mkdir -p "$D/census"
while true; do
  for c in rtg_e3_s1 rtg_e3_s2 rtg_e3_s3 rtg_e3c_s1 rtg_e3c_s2; do
    f="/workspace/Transform2Act/results/$c/e3_epochs.jsonl"
    [ -f "$f" ] || continue
    "$P" - "$f" "$D/census/${c}_morph.csv" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
cols = ["epoch","alpha","n_bodies","n_motors","mass","limb_len_mean",
        "limb_len_sum","gear_mean","max_depth","topo",
        "distinct_topologies","top_topology_share","sampled_bodies_mean",
        "design_failed","eval_fall_rate","eval_goal_rate","eval_max_fwd",
        "eval_R_mean","eval_design_fail_rate","r_fall_return","r_fwd_return"]
rows = []
for line in open(src):
    try: d = json.loads(line)
    except Exception: continue
    m = d.get("mean_action_design") or {}
    c = d.get("census") or {}
    e = d.get("eval") or {}
    dg = d.get("dodge_pooled") or {}
    rows.append([
        d.get("epoch"), d.get("alpha"),
        m.get("n_bodies"), m.get("model_nu_ours"), m.get("model_mass_ours"),
        (m.get("limb_length") or {}).get("mean"),
        (m.get("limb_length") or {}).get("sum"),
        (m.get("gear") or {}).get("mean"),
        max((int(k) for k in (m.get("depth_hist") or {})), default=None),
        m.get("topo"),
        c.get("distinct_topologies"), c.get("top_topology_share"),
        c.get("bodies_mean"), c.get("design_failed"),
        e.get("fall_rate"), e.get("goal_rate"), e.get("max_fwd"),
        e.get("R_mean"), e.get("design_fail_rate"),
        dg.get("r_fall_return"), dg.get("r_fwd_return")])
with open(dst, "w") as fh:
    fh.write(",".join(cols) + "\n")
    for r in rows:
        fh.write(",".join("" if v is None else str(v) for v in r) + "\n")
PY
  done
  sleep 300
done
