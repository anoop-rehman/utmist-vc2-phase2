#!/usr/bin/env bash
# Back up a training tree that lives OUTSIDE this repo.
#
# Written 2026-08-23, the day a pod replacement destroyed a completed
# 1000-epoch Transform2Act run and CompetEvo's 346-epoch reference run. Neither
# was ever synced anywhere: our own trainers push checkpoints to GCS via
# warp_port/gcs.py, but third-party trainers know nothing about that, and their
# output directories were the only copy.
#
# A sidecar rather than a patch, on purpose: /workspace/Transform2Act and
# /workspace/competevo are upstream checkouts that get re-cloned, and any edit
# we make to them is lost exactly when we need it.
#
#   bash scripts/persistence/sync_external.sh SRC REMOTE_PREFIX [INTERVAL_SECONDS]
#
#   # one shot
#   ... /workspace/Transform2Act/results transform2act
#   # sidecar alongside a run: sync every 30 min until killed
#   ... /workspace/Transform2Act/results transform2act 1800
#
# Restore is the same rsync with the arguments the other way round; see the
# RESTORE note at the bottom.
set -uo pipefail

SRC=${1:?usage: sync_external.sh SRC REMOTE_PREFIX [INTERVAL_SECONDS]}
PREFIX=${2:?usage: sync_external.sh SRC REMOTE_PREFIX [INTERVAL_SECONDS]}
INTERVAL=${3:-0}
BUCKET=${VC2_GCS_BUCKET:-vc2-2026-checkpoints}
export PATH=/workspace/google-cloud-sdk/bin:$PATH

DEST="gs://${BUCKET}/${PREFIX}"

once () {
  if [ ! -d "$SRC" ]; then
    echo "[$(date -u +%FT%TZ)] SKIP: $SRC does not exist"
    return 0
  fi
  # THE GUARD. An empty source is what a fresh pod looks like, and syncing it is
  # at best pointless. It is also one flag away from catastrophe: add
  # --delete-unmatched-destination-objects to an rsync from an empty directory
  # and the backup is gone. This file never passes that flag, and it refuses to
  # run at all when there is nothing to push, so the dangerous case cannot arise
  # by accident later.
  if [ -z "$(find "$SRC" -type f -print -quit 2>/dev/null)" ]; then
    echo "[$(date -u +%FT%TZ)] SKIP: $SRC is empty -- refusing to sync nothing"
    return 0
  fi
  local n bytes
  n=$(find "$SRC" -type f | wc -l)
  bytes=$(du -sb "$SRC" 2>/dev/null | cut -f1)
  echo "[$(date -u +%FT%TZ)] sync $SRC -> $DEST  ($n files, $((bytes / 1024 / 1024)) MB)"
  # Additive: no --delete-unmatched-destination-objects anywhere in this file.
  # A local tree that has lost files must never be able to prune the backup.
  gcloud storage rsync "$SRC" "$DEST" --recursive --quiet 2>&1 \
    | grep -Ev '^(At |Copying|Completed|Average|[. ]*$)' || true
  echo "[$(date -u +%FT%TZ)] done"
}

if [ "$INTERVAL" -gt 0 ] 2>/dev/null; then
  echo "sidecar: every ${INTERVAL}s, kill to stop"
  while true; do
    once
    sleep "$INTERVAL"
  done
else
  once
fi

# RESTORE:
#   gcloud storage rsync gs://vc2-2026-checkpoints/transform2act \
#       /workspace/Transform2Act/results --recursive
