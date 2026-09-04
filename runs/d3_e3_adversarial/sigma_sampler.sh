#!/bin/bash
# D3 M3 E3: sample control_log_std every 6 epochs instead of every 20.
#
# THE REAL RESOLUTION PROBLEM. Archival checkpoints land every 20 epochs, so
# sigma is only observable at 19/39/59/79/... But the milestones the
# pre-registered prediction is tested against fall BETWEEN them: affordability
# at ~69-70 and the boundary crossing at ~96-98. At 20-epoch granularity the
# crossing can only be bracketed to +/-10 epochs, against a predicted window of
# 89-114 -- the measurement would be nearly as wide as the prediction.
#
# train_e3_gnn.py writes models/_video_tmp.p every --video-every (6) epochs,
# hands it to the renderer and deletes it in a `finally`. Polling at 2 s and
# reading log_std out of it while it exists gives 6-epoch resolution for free.
# It is a read; it cannot perturb the trainer.
set -uo pipefail
D=/workspace/utmist-vc2-phase2/runs/d3_e3_adversarial
CSV="$D/census/sigma_fine.csv"
[ -f "$CSV" ] || echo "cfg,epoch,control_log_std,sigma,cost_per_step,below_survive,below_boundary" > "$CSV"
while true; do
  for c in rtg_e3c_s1 rtg_e3c_s2; do
    f="/workspace/Transform2Act/results/$c/models/_video_tmp.p"
    [ -f "$f" ] || continue
    python3 - "$f" "$c" "$CSV" <<'PY' 2>/dev/null
import math, pickle, sys, os
f, cfg, csv = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    d = pickle.load(open(f, "rb"))
except Exception:
    sys.exit(0)
ep = d.get("epoch")
ls = float(d["policy_dict"]["control_action_log_std"].mean().item())
seen = set()
if os.path.exists(csv):
    for line in open(csv):
        p = line.split(",")
        if len(p) > 1 and p[0] == cfg:
            seen.add(p[1])
if str(ep) in seen:
    sys.exit(0)
sig = math.exp(ls); cost = 0.5 * 8 * sig * sig
with open(csv, "a") as fh:
    fh.write(f"{cfg},{ep},{ls:.6f},{sig:.6f},{cost:.6f},"
             f"{int(cost < 1.0)},{int(ls < -0.9645)}\n")
print(f"{cfg} epoch {ep}: log_std {ls:+.4f} sigma {sig:.4f} cost {cost:.4f}"
      f"{'  <-- COST BELOW SURVIVE BONUS' if cost < 1.0 else ''}"
      f"{'  <-- CROSSED THE BOUNDARY -0.9645' if ls < -0.9645 else ''}")
PY
  done
  sleep 2
done
