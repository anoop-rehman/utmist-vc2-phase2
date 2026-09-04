#!/bin/bash
# D3 M3 E3.1: the next two milestones, benchmarked against the frozen-body
# control's own timeline (locomotion at epochs 79-84, goal>=0.5 at 144-149).
set -uo pipefail
D=/workspace/utmist-vc2-phase2/runs/d3_e31_fix
seen_loco=0
while true; do
  /workspace/utmist-vc2-phase2/.venv/bin/python - "$seen_loco" <<'PY' > /tmp/e31_ms.txt 2>/dev/null
import csv, sys
seen=int(sys.argv[1]); out=[]
for s in (2,3):
    try: rows=[r for r in csv.DictReader(open(f"/workspace/utmist-vc2-phase2/runs/d3_e31_fix/census/rtg_e31_s{s}_morph.csv")) if r.get("eval_max_fwd")]
    except Exception: continue
    if not rows: continue
    last=rows[-1]
    if float(last["eval_goal_rate"])>0:
        out.append(f"GOAL SCORED  rtg_e31_s{s} epoch {last['epoch']}: goal {last['eval_goal_rate']}, fwd {last['eval_max_fwd']} m -- the design+control loop has scored on an adversarial task")
    elif not seen and float(last["eval_max_fwd"])>1.0:
        out.append(f"LOCOMOTION  rtg_e31_s{s} epoch {last['epoch']}: max_fwd {last['eval_max_fwd']} m (>1.0) -- controls reached this at epochs 79-84")
print("\n".join(out))
PY
  if [ -s /tmp/e31_ms.txt ]; then cat /tmp/e31_ms.txt; grep -q LOCOMOTION /tmp/e31_ms.txt && seen_loco=1; grep -q "GOAL SCORED" /tmp/e31_ms.txt && exit 0; fi
  sleep 300
done
