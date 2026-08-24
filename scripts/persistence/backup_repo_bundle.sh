#!/usr/bin/env bash
# Push the whole repository to GCS as a single git bundle.
#
# The gap this closes: commits that have not been pushed to a remote exist on
# exactly one machine. On 2026-08-24 there were 13 of them and no push
# credential on the pod (the user runs `gh auth login` themselves), so a day's
# work was in precisely the position the checkpoints were in when the previous
# pod died.
#
# A bundle is the right shape for this. It is one file, it carries the FULL
# history and every ref -- not a working tree -- and `git clone` reads it
# directly, so a restore needs nothing but the file:
#
#   gcloud storage cp gs://BUCKET/repo_bundles/NAME.bundle .
#   git clone --branch anoop NAME.bundle restored/
#
# This is a safety net, not a substitute for pushing. Push when you have a
# credential; a bundle nobody knows about is not a backup either.
#
#   bash scripts/persistence/backup_repo_bundle.sh            # one shot
#   bash scripts/persistence/backup_repo_bundle.sh 3600       # every hour
set -uo pipefail

INTERVAL=${1:-0}
BUCKET=${VC2_GCS_BUCKET:-vc2-2026-checkpoints}
export PATH=/workspace/google-cloud-sdk/bin:$PATH

REPO=$(git rev-parse --show-toplevel) || { echo "not in a git repo"; exit 1; }
NAME=$(basename "$REPO")
TMP=${TMPDIR:-/tmp}/${NAME}.bundle
DEST="gs://${BUCKET}/repo_bundles/${NAME}.bundle"

once () {
  cd "$REPO" || return 0
  local ahead
  # Informational only -- the bundle goes up regardless. A branch with no
  # upstream reports nothing, which is itself the case worth backing up.
  ahead=$(git log --oneline @{u}..HEAD 2>/dev/null | wc -l || echo "?")
  echo "[$(date -u +%FT%TZ)] bundling $NAME ($(git rev-parse --abbrev-ref HEAD), ${ahead} unpushed)"
  rm -f "$TMP"
  # --all: every ref, so a restore is not silently missing a branch.
  if ! git bundle create "$TMP" --all >/dev/null 2>&1; then
    echo "[$(date -u +%FT%TZ)] FAILED to create the bundle -- nothing uploaded"
    return 1
  fi
  # VERIFY BEFORE UPLOADING. A corrupt backup is worse than a missing one,
  # because it is only discovered at restore time.
  if ! git bundle verify "$TMP" >/dev/null 2>&1; then
    echo "[$(date -u +%FT%TZ)] bundle FAILED verification -- nothing uploaded"
    return 1
  fi
  gcloud storage cp "$TMP" "$DEST" --quiet 2>&1 \
    | grep -Ev '^(Copying|Completed|Average|[. ]*$)' || true
  echo "[$(date -u +%FT%TZ)] done: $DEST ($(du -h "$TMP" | cut -f1))"
  rm -f "$TMP"
}

if [ "$INTERVAL" -gt 0 ] 2>/dev/null; then
  echo "every ${INTERVAL}s, kill to stop"
  while true; do once; sleep "$INTERVAL"; done
else
  once
fi
