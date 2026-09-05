#!/usr/bin/env bash
# D3 M3 E4 wave launcher. One SEED PAIR per wave -- the pair is the smallest
# coherent unit, because the two lineages only mean anything against each
# other. Usage:  launch_e4.sh <seed>        e.g.  launch_e4.sh 1
#
# GPU: a wave is 2 arms at ~6.0 GB each (max observed reserved) = ~12.1 GB of
# 20.5 GB. It fits alongside ONE other design-on arm, not two. Projected before
# wave 1 rather than discovered at 95%; see D3_E4_PREREQ.md.
#
# MPS IS ACTIVE ON THIS BOX. Never kill a CUDA client -- stop by stop-file.
set -euo pipefail
S="${1:?usage: launch_e4.sh <seed>}"
cd /workspace/Transform2Act
source env-gpu.sh
ROOT=/workspace/Transform2Act/results/_e4_snapshots
mkdir -p "$ROOT"
for L in a b; do
  P=$([ "$L" = a ] && echo b || echo a)
  CFG="rtg_e4_s${S}${L}"
  nohup .venv-gpu/bin/python \
    /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/train_e4_gnn.py \
    --cfg "$CFG" --partner-cfg "rtg_e4_s${S}${P}" --snapshot-root "$ROOT" \
    --opp-refresh 10 --curriculum-steps 130208333 \
    --eval-every 5 --eval-episodes 10 \
    --morph-every 1 --morph-episodes 20 \
    --video-every 6 --video-episodes 9 --archive-every 50 \
    --num-threads 10 --wandb --wandb-name "d3_e4_${CFG}" \
    --stop-file "/tmp/stop_e4_s${S}${L}" \
    > "/tmp/e4_s${S}${L}.log" 2>&1 &
  echo "launched $CFG pid $! (partner rtg_e4_s${S}${P}, stop-file /tmp/stop_e4_s${S}${L})"
  sleep 5
done
