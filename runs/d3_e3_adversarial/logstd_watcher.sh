#!/bin/bash
# D3 M3 E3: track control_log_std on the two frozen-body control arms.
#
# Section 3g predicts they cross the empirical boundary -0.9645 at epoch
# 151-205 and locomote 18-27 epochs later. That crossing is the prediction's
# key event, so it has to be observable as it happens.
#
# These two arms were launched BEFORE train_e3_gnn.py gained per-epoch log_std
# logging, so their sigma is only recoverable from checkpoints. This reads it
# out of whatever checkpoints exist, every 10 minutes, into a tracked CSV. It
# is a pickle load and no simulation, so it costs no measurable CPU against the
# 10.2-core quota.
set -uo pipefail
D=/workspace/utmist-vc2-phase2/runs/d3_e3_adversarial
while true; do
  python3 "$D/../../rower_soccer/t2a_port/e3_logstd_trace.py" \
      --cfgs rtg_e3c_s1,rtg_e3c_s2 \
      --out "$D/census/logstd_ctl.json" > "$D/census/logstd_ctl.txt" 2>&1
  sleep 600
done
