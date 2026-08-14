#!/usr/bin/env bash
# Stopgap for a launcher bug: the four section-22 kick arms were started without
# --gcs-bucket (train_kick_warp defaults to None), so nothing is being backed up.
# Restarting them would cost ~15% of an 8-hour experiment, so instead this does
# by hand what the flag would have done, every 20 minutes.
# Checkpoints and config only -- videos are large and reproducible from them.
set -u
export PATH=/workspace/google-cloud-sdk/bin:$PATH   # gcloud is not on the default PATH here
BUCKET=${BUCKET:-vc2-2026-checkpoints}
cd /workspace/utmist-vc2-phase2
while true; do
  for n in kick_e0_control kick_e1_shapeoff kick_e2_upright3 kick_e3_posereset; do
    [ -d "runs_v2/$n" ] || continue
    gcloud storage rsync "runs_v2/$n" "gs://$BUCKET/$n" \
      --exclude='.*\.mp4$' --exclude='.*wandb.*' --quiet 2>&1 | tail -2
  done
  echo "[sync] $(date -u +%H:%M:%S) done"
  sleep 1200
done
