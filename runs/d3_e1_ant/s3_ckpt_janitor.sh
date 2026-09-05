#!/bin/bash
# Archive an E1 seed-3 checkpoint off the (nearly full) /workspace volume once
# BOTH consumers have finished with it -- the topology census JSON and the
# best/median/worst render. Moves, never deletes, so a re-analysis is still
# possible from /root.
#
# Exists because ENOSPC on /workspace killed seed 3 at epoch 39 on 2026-08-30:
# 157 MB per checkpoint x 11 checkpoints x 3 seeds does not fit beside 19 GB of
# earlier hopper/E0 results and D1's periodic videos.
A=/root/e1_ckpt_archive/ant_e1_s3
C=/workspace/utmist-vc2-phase2/runs/d3_e1_ant/census
R=/workspace/utmist-vc2-phase2/runs/d3_e1_ant/renders
M=/workspace/Transform2Act/results/ant_e1_s3/models
mkdir -p "$A"
while true; do
  for ep in 10 20 30 40 50 60 70 80 90; do
    f=$(printf "%s/epoch_%04d.p" "$M" "$ep")
    j=$(printf "%s/ant_e1_s3_e%04d.json" "$C" "$ep")
    v=$(printf "%s/ant_e1_s3_e%04d_bmw.mp4" "$R" "$ep")
    if [ -f "$f" ] && [ -f "$j" ] && [ -f "$v" ]; then
      echo "[janitor] archiving epoch $ep ($(date -Is))"
      mv "$f" "$A"/ 2>/dev/null
    fi
  done
  [ -f "$M/epoch_0100.p" ] && [ -f "$C/ant_e1_s3_e0100.json" ] && break
  sleep 180
done
echo "[janitor] seed 3 complete, stopping"
