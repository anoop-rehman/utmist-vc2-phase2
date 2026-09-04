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
    "$P" - "$f" "$D/census/${c}_morph.csv" "$D/census" "$c" <<'PY'
import glob, json, os, sys
src, dst, cendir, cfg = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]

# THE POPULATION COLUMNS COME FROM CHECKPOINT PROBES, NOT FROM THE LIVE LOG.
# `e3_morph.census` gained its motor columns AFTER these arms launched, so the
# running trainers never write them and the live JSONL never will. Joining the
# probe JSONs in by epoch is the only way these columns can carry a value for
# this run, and an epoch with no probe is genuinely blank rather than missing.
pop = {}
for pf in glob.glob(os.path.join(cendir, f"pop_{cfg}_e*.json")):
    try:
        d = json.load(open(pf))
    except Exception:
        continue
    e = d.get("ckpt_epoch")
    if e is None:
        continue
    c = d.get("census", {})
    pop[int(e)] = (c.get("motors_mean"), c.get("motors_max"),
                   c.get("p_act1"), c.get("p_act4"),
                   d.get("step_share_act4"))

cols = ["epoch","alpha","n_bodies","n_motors","mass","limb_len_mean",
        "limb_len_sum","gear_mean","max_depth","topo",
        "distinct_topologies","top_topology_share","sampled_bodies_mean",
        "pop_motors_mean","pop_motors_max","p_act1","p_act4","step_share_act4",
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
    ep = d.get("epoch")
    # prefer the live census if a future run ever has it, else the probe
    pm, px, p1, p4, ss = pop.get(ep, (None, None, None, None, None))
    rows.append([
        ep, d.get("alpha"),
        m.get("n_bodies"), m.get("model_nu_ours"), m.get("model_mass_ours"),
        (m.get("limb_length") or {}).get("mean"),
        (m.get("limb_length") or {}).get("sum"),
        (m.get("gear") or {}).get("mean"),
        max((int(k) for k in (m.get("depth_hist") or {})), default=None),
        m.get("topo"),
        c.get("distinct_topologies"), c.get("top_topology_share"),
        c.get("bodies_mean"),
        c.get("motors_mean", pm), c.get("motors_max", px),
        c.get("p_act1", p1), c.get("p_act4", p4), ss,
        c.get("design_failed"),
        e.get("fall_rate"), e.get("goal_rate"), e.get("max_fwd"),
        e.get("R_mean"), e.get("design_fail_rate"),
        dg.get("r_fall_return"), dg.get("r_fwd_return")])
with open(dst, "w") as fh:
    fh.write(",".join(cols) + "\n")
    for r in rows:
        fh.write(",".join("" if v is None else str(v) for v in r) + "\n")
PY
  done

  # One row per population probe, across every arm and checkpoint. THIS is the
  # artefact the epoch-100 decision reads: `p_act4` and `step_share_act4` are
  # what section 3c's first row is judged on, and they exist only here.
  "$P" - "$D/census" <<'PY'
import glob, json, os, sys
cendir = sys.argv[1]
cols = ["cfg","ckpt","ckpt_epoch","designs","readout_n_bodies",
        "readout_n_motors","readout_gear_mean","pop_motors_mean",
        "pop_motors_max","p_act1","p_act4","step_share_act1","step_share_act4",
        "distinct_topologies","top_topology_share","bodies_mean","motors_hist"]
rows = []
for pf in sorted(glob.glob(os.path.join(cendir, "pop_*.json"))):
    try: d = json.load(open(pf))
    except Exception: continue
    c = d.get("census", {})
    rows.append([d.get("cfg"), d.get("ckpt"), d.get("ckpt_epoch"),
                 d.get("designs"), d.get("mean_action_n_bodies"),
                 d.get("mean_action_n_motors"), d.get("mean_action_gear_mean"),
                 c.get("motors_mean"), c.get("motors_max"), c.get("p_act1"),
                 c.get("p_act4"), d.get("step_share_act1"),
                 d.get("step_share_act4"), c.get("distinct_topologies"),
                 c.get("top_topology_share"), c.get("bodies_mean"),
                 json.dumps(c.get("motors_hist", {})).replace(",", ";")])
rows.sort(key=lambda r: (str(r[0]), -1 if r[2] is None else r[2]))
with open(os.path.join(cendir, "population.csv"), "w") as fh:
    fh.write(",".join(cols) + "\n")
    for r in rows:
        fh.write(",".join("" if v is None else str(v) for v in r) + "\n")
PY
  sleep 300
done
