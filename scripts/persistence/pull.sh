#!/usr/bin/env bash
# PULL Claude Code chat state <- GCS, onto a fresh machine or pod.
#
#   bash pull.sh
#   cd /workspace && claude --resume
#
# Additive: never deletes local sessions, so running it on a machine that
# already has history is safe.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/claude-gcs.env"

[ -d /workspace/google-cloud-sdk/bin ] && PATH="/workspace/google-cloud-sdk/bin:$PATH"

if [ "${GCS_BUCKET:-REPLACE_ME}" = "REPLACE_ME" ]; then
    echo "ERROR: set GCS_BUCKET in $HERE/claude-gcs.env" >&2; exit 1
fi
command -v gcloud >/dev/null 2>&1 || { echo "ERROR: gcloud not on PATH" >&2; exit 1; }
gcloud auth print-access-token >/dev/null 2>&1 || {
    echo "ERROR: not authenticated. Run:  gcloud auth login" >&2; exit 1; }

# On a RunPod box, land state on the network volume so a pod stop cannot eat
# it. Anywhere else, use the normal config dir.
# CLAUDE_PULL_DEST overrides everything - handy for restoring into a scratch
# dir to inspect before committing to it.
if [ -n "${CLAUDE_PULL_DEST:-}" ]; then
    DEST="$CLAUDE_PULL_DEST"
elif [ -d /workspace ] && [ -w /workspace ]; then
    DEST=/workspace/.claude-persistent
else
    DEST="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
fi
mkdir -p "$DEST"

SRC="gs://${GCS_BUCKET}/${GCS_PREFIX}/config"

# Mirror of sync.sh's allowlist.
SYNC_DIRS=(projects)
SYNC_FILES=(history.jsonl settings.json .claude.json .credentials.json)

echo "source : $SRC"
echo "dest   : $DEST"
echo

for d in "${SYNC_DIRS[@]}"; do
    gcloud storage ls "$SRC/$d" >/dev/null 2>&1 || continue
    echo "  dir  $d/"
    mkdir -p "$DEST/$d"
    gcloud storage rsync "$SRC/$d" "$DEST/$d" \
        --recursive --project="$GCS_PROJECT" 2>&1 | grep -Ev '^(Copying|Completed|Average|$)' || true
done

for f in "${SYNC_FILES[@]}"; do
    gcloud storage cp "$SRC/$f" "$DEST/$f" --project="$GCS_PROJECT" >/dev/null 2>&1 &&
        echo "  file $f" || true
done
chmod 600 "$DEST/.credentials.json" 2>/dev/null || true

# Wire the config dir up if we are on a pod with the bootstrap script present.
if [ -x "$HERE/claude-bootstrap.sh" ] && [ "$DEST" = /workspace/.claude-persistent ]; then
    echo
    bash "$HERE/claude-bootstrap.sh"
else
    echo
    echo "Set this in your shell (and .bashrc/.zshrc) if not already set:"
    echo "  export CLAUDE_CONFIG_DIR=$DEST"
fi

echo
echo "Restored session(s):"
find "$DEST/projects" -name '*.jsonl' -printf '  %f  %s bytes  %TY-%Tm-%Td\n' 2>/dev/null | sort
echo
echo "Resume with:   cd /workspace && claude --resume"
echo "NOTE: --resume only lists sessions recorded from the SAME cwd."
