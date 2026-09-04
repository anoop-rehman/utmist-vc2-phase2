#!/bin/bash
# D3 M3 E3: fire on the two milestones the pre-registered prediction is tested
# against, read from the 6-epoch sigma series (census/sigma_fine.csv).
#
#   affordability   ctrl cost/step drops below the 1.0 survive bonus  (~epoch 70)
#   BOUNDARY        control_log_std crosses -0.9645                   (~epoch 96)
#
# The boundary crossing is the event the prediction (epoch 89-114) stands or
# falls on, so it exits after reporting it.
set -uo pipefail
D=/workspace/utmist-vc2-phase2/runs/d3_e3_adversarial
CSV="$D/census/sigma_fine.csv"
while true; do
  if [ -f "$CSV" ]; then
    a=$(awk -F, 'NR>1 && $6==1 {print $1}' "$CSV" | sort -u | wc -l)
    b=$(awk -F, 'NR>1 && $7==1 {print $1}' "$CSV" | sort -u | wc -l)
    if [ "$b" -ge 2 ]; then
      echo "BOUNDARY CROSSED (-0.9645) on BOTH control arms -- the pre-registered prediction (epoch 89-114) is now testable:"
      awk -F, 'NR>1 && $7==1' "$CSV"
      exit 0
    fi
    if [ "$a" -ge 2 ] && [ ! -f /tmp/e3_afford_seen ]; then
      touch /tmp/e3_afford_seen
      echo "AFFORDABILITY milestone reached on both control arms: ctrl cost/step is now BELOW the 1.0 survive bonus (projected epoch ~70)"
      awk -F, 'NR>1 && $6==1' "$CSV"
    fi
  fi
  sleep 120
done
