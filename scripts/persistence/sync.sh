#!/usr/bin/env bash
# PUSH Claude Code chat state -> GCS.
#
#   bash sync.sh
#
# Syncs an explicit allowlist: transcripts, prompt history, settings, memory.
# Skips plugins/ and the caches, which are large and fully regenerable.
#
# Allowlist rather than --exclude on purpose: `gcloud storage rsync --exclude`
# silently honors only part of a "|"-joined regex (verified - a lone pattern
# excluded 395 files, the same pattern OR'd with a second excluded 1), so a
# denylist here would quietly ship everything.
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

# --- pick the source config dir -------------------------------------------
if [ -d /workspace/.claude-persistent ]; then
    SRC=/workspace/.claude-persistent
    # If claude is still writing to the container overlay, rescue that first.
    [ -x "$HERE/claude-snapshot.sh" ] && bash "$HERE/claude-snapshot.sh" >/dev/null 2>&1 || true
else
    SRC="${CLAUDE_CONFIG_DIR:-$HOME/.claude}"
fi
[ -d "$SRC" ] || { echo "ERROR: no config dir at $SRC" >&2; exit 1; }

DEST="gs://${GCS_BUCKET}/${GCS_PREFIX}"

# What counts as "the chat". projects/ carries transcripts AND memory/.
SYNC_DIRS=(projects)
SYNC_FILES=(history.jsonl settings.json)
# NOT in SYNC_FILES: Claude Code writes .claude.json BESIDE the config dir
# (/root/.claude.json), not inside it. It was listed here until 2026-08-14 and
# the `[ -f "$SRC/$f" ] || continue` below skipped it silently on all 75 syncs
# since -- the bucket's copy was dated 2026-07-26 while the live file changed
# daily, and pull.sh would have restored the July one over it. Resolved from an
# explicit candidate list instead, and NOISY when it finds nothing, because a
# silent skip is exactly what hid this.
CLAUDE_JSON=""
for c in "$SRC/.claude.json" "$HOME/.claude.json"; do
    [ -f "$c" ] && { CLAUDE_JSON="$c"; break; }
done
[ "${INCLUDE_CREDENTIALS:-0}" = "1" ] && SYNC_FILES+=(.credentials.json)

echo "source : $SRC"
echo "dest   : $DEST/config/"
echo

for d in "${SYNC_DIRS[@]}"; do
    [ -d "$SRC/$d" ] || continue
    echo "  dir  $d/"
    gcloud storage rsync "$SRC/$d" "$DEST/config/$d" \
        --recursive --project="$GCS_PROJECT" 2>&1 | grep -Ev '^(Copying|Completed|Average|$)' || true
done

for f in "${SYNC_FILES[@]}"; do
    [ -f "$SRC/$f" ] || continue
    echo "  file $f"
    gcloud storage cp "$SRC/$f" "$DEST/config/$f" --project="$GCS_PROJECT" >/dev/null 2>&1
done

if [ -n "$CLAUDE_JSON" ]; then
    echo "  file .claude.json  (from $CLAUDE_JSON)"
    gcloud storage cp "$CLAUDE_JSON" "$DEST/config/.claude.json" \
        --project="$GCS_PROJECT" >/dev/null 2>&1
else
    echo "  WARNING: no .claude.json found at $SRC/ or $HOME/ -- per-project" >&2
    echo "           trust and tool approvals will NOT be restored." >&2
fi

# Ship the scripts too, so a brand-new machine can bootstrap from the bucket.
for f in pull.sh sync.sh claude-gcs.env claude-bootstrap.sh claude-snapshot.sh CLAUDE-PERSISTENCE.md; do
    [ -f "$HERE/$f" ] && gcloud storage cp "$HERE/$f" "$DEST/$f" --project="$GCS_PROJECT" >/dev/null 2>&1 || true
done

N=$(find "$SRC/projects" -name '*.jsonl' 2>/dev/null | wc -l)
BYTES=$(gcloud storage du -s "$DEST/config" --project="$GCS_PROJECT" 2>/dev/null | awk '{print $1}')
echo
echo "Pushed $N session transcript(s). Remote size: $(( ${BYTES:-0} / 1024 )) KB"
[ "${INCLUDE_CREDENTIALS:-0}" = "1" ] ||
    echo "(credentials excluded - run 'claude' and log in on the new machine)"
