#!/usr/bin/env bash
# D3 E4B: launch the three shared-weight ring arms.
# MPS IS ACTIVE. Every arm gets a stop-file; nothing is ever signalled.
set -euo pipefail
cd /workspace/Transform2Act
source env-gpu.sh
SEEDS="${@:-1 2}"   # default: the two concurrent arms; s3 runs afterwards
# WARM START. Only two design-ON competent checkpoints exist -- rtg_e31d_s3body
# ran with force_identity_design so its design head is untrained. s3 therefore
# replicates s2's seed on purpose: s2 vs s3 isolates pure RNG variation, while
# s1 vs s2 carries RNG plus a different starting body.
warm_for() {
  case "$1" in
    1) echo /workspace/Transform2Act/results/rtg_e31_s1/models/epoch_0400.p;;
    2|3) echo /workspace/Transform2Act/results/rtg_e31_s2/models/epoch_0400.p;;
  esac
}
for S in $SEEDS; do
  CFG="rtg_e4r_s${S}"
  WARM="$(warm_for $S)"
  [ -f "$WARM" ] || { echo "MISSING warm-start checkpoint $WARM -- refusing"; exit 1; }
  nohup .venv-gpu/bin/python \
    /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/train_e4r_gnn.py \
    --cfg "$CFG" --warm-start "$WARM" \
    --ring-every 10 --ring-delta 0.0 --ring-persist-every 4 \
    --curriculum-steps 130208333 \
    --eval-every 5 --eval-episodes 10 \
    --mirror-episodes 20 --ladder-episodes 10 --ladder-k 5 \
    --morph-every 1 --morph-episodes 20 \
    --video-every 6 --video-episodes 9 --archive-every 50 \
    --restart-check-epoch 200 \
    --num-threads 10 --wandb --wandb-name "d3_e4b_${CFG}" \
    --stop-file "/tmp/stop_e4b_s${S}" \
    > "/tmp/e4b_s${S}.log" 2>&1 &
  echo "launched $CFG pid $! warm-start $(basename $(dirname $(dirname $WARM)))/$(basename $WARM) (stop-file /tmp/stop_e4b_s${S})"
  sleep 8
done
