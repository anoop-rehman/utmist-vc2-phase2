#!/bin/bash
# D3 M3 E3.1 instrumentation, pointed at THIS rung's arms.
#
# WHY THIS EXISTS AS A NEW FILE. E3's four watchers all carried hardcoded cfg
# lists (rtg_e3_s*, rtg_e3c_s*) and kept polling faithfully after every one of
# those runs ended -- so the instrumentation LOOKED healthy while collecting
# nothing for the new arms. Both instrumentation failures in this experiment
# happened at a TRANSITION (new arms, new directory), which is exactly when a
# watcher silently keeps pointing at the old target. This one takes its cfg
# list from CFGS so repointing is a variable, not an edit.
#
# What it does, every 120 s:
#   1. distils each arm's e3_epochs.jsonl into census/<cfg>_morph.csv
#      -- p_act4 and control_log_std are ALREADY written per epoch by the
#         trainer itself (e3_morph.census gained the motor columns before
#         these arms launched), so this is a distiller, not a probe;
#   2. evaluates BOTH pre-registered falsifiers and shouts if either fires.
set -uo pipefail
D=/workspace/utmist-vc2-phase2/runs/d3_e31_fix
CFGS="${CFGS:-rtg_e31_s1 rtg_e31_s2 rtg_e31_s3}"
P=/workspace/utmist-vc2-phase2/.venv/bin/python
mkdir -p "$D/census"
while true; do
  for c in $CFGS; do
    f="/workspace/Transform2Act/results/$c/e3_epochs.jsonl"
    [ -f "$f" ] || continue
    "$P" - "$f" "$D/census/${c}_morph.csv" "$c" <<'PY'
import json, sys
src, dst, cfg = sys.argv[1], sys.argv[2], sys.argv[3]
cols = ["epoch","alpha","control_log_std","attr_log_std","n_bodies","n_motors",
        "mass","limb_len_mean","gear_mean","topo","distinct_topologies",
        "top_topology_share","sampled_bodies_mean","pop_motors_mean",
        "pop_motors_max","p_act1","p_act4","design_failed",
        "eval_fall_rate","eval_goal_rate","eval_max_fwd","eval_R_mean",
        "r_fall_return","r_fwd_return"]
rows=[]
for line in open(src):
    try: d=json.loads(line)
    except Exception: continue
    m=d.get("mean_action_design") or {}; c=d.get("census") or {}
    e=d.get("eval") or {}; g=d.get("dodge_pooled") or {}
    rows.append([d.get("epoch"), d.get("alpha"), d.get("control_log_std"),
        d.get("attr_log_std"), m.get("n_bodies"), m.get("model_nu_ours"),
        m.get("model_mass_ours"), (m.get("limb_length") or {}).get("mean"),
        (m.get("gear") or {}).get("mean"), m.get("topo"),
        c.get("distinct_topologies"), c.get("top_topology_share"),
        c.get("bodies_mean"), c.get("motors_mean"), c.get("motors_max"),
        c.get("p_act1"), c.get("p_act4"), c.get("design_failed"),
        e.get("fall_rate"), e.get("goal_rate"), e.get("max_fwd"),
        e.get("R_mean"), g.get("r_fall_return"), g.get("r_fwd_return")])
with open(dst,"w") as fh:
    fh.write(",".join(cols)+"\n")
    for r in rows: fh.write(",".join("" if v is None else str(v) for v in r)+"\n")
PY
  done
  # ---- the two pre-registered falsifiers, D3_E31_FIX.md -----------------
  "$P" - "$D/census" "$CFGS" <<'PY'
import csv, glob, os, sys
cen, cfgs = sys.argv[1], sys.argv[2].split()
for cfg in cfgs:
    f=os.path.join(cen, f"{cfg}_morph.csv")
    if not os.path.exists(f): continue
    rows=[r for r in csv.DictReader(open(f)) if r.get("epoch")]
    early=[r for r in rows if int(r["epoch"])<=20]
    for r in early:
        if r.get("control_log_std") and float(r["control_log_std"]) > -0.9645:
            print(f"FALSIFIER 1 FIRED  {cfg} epoch {r['epoch']}: control_log_std "
                  f"{r['control_log_std']} > -0.9645 inside the first 20 epochs")
        if r.get("p_act4") and float(r["p_act4"]) == 0.0:
            print(f"FALSIFIER 2 FIRED  {cfg} epoch {r['epoch']}: p_act4 collapsed to 0")
PY
  sleep 120
done
