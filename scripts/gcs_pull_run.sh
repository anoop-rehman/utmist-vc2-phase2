#!/usr/bin/env bash
# Pull a run's checkpoint dir down from GCS so --resume can continue it.
#
# This is what makes spot instances safe. When SkyPilot's managed-spot relaunches
# a preempted job on a fresh instance, runs_v2/<run>/ is gone -- so --resume, which
# reads that local dir, would silently start from scratch. This pulls the last
# synced checkpoint back down first. On the very first launch the GCS prefix is
# empty and this is a no-op, so the job starts fresh. Idempotent either way.
#
# Usage: gcs_pull_run.sh <bucket> <run_name>
set -euo pipefail

BUCKET="${1:?usage: gcs_pull_run.sh <bucket> <run_name>}"
RUN="${2:?usage: gcs_pull_run.sh <bucket> <run_name>}"
DEST="runs_v2/${RUN}"
SRC="gs://${BUCKET#gs://}/${RUN}"

mkdir -p "${DEST}"
echo "[gcs_pull] checking ${SRC}/ ..."
# checkpoint.pt is the only file --resume strictly needs; pull the small siblings
# too so provenance and best.pt survive a preemption. `|| true`: a missing object
# (first launch) must not abort the job.
for f in checkpoint.pt config.json best.pt checkpoint_mid.pt latest.pt; do
  if gcloud storage cp "${SRC}/${f}" "${DEST}/${f}" 2>/dev/null; then
    echo "[gcs_pull]   pulled ${f}"
  fi
done

if [ -f "${DEST}/checkpoint.pt" ]; then
  echo "[gcs_pull] resume artifact present -> job will --resume"
else
  echo "[gcs_pull] no prior checkpoint -> fresh start"
fi
