#!/usr/bin/env bash
# Copy the live session's transcript + history from the container overlay onto
# the network volume.
#
# You only need this for a session that started BEFORE bootstrap was applied
# (i.e. one still writing to a real /root/.claude directory). Once
# /root/.claude is a symlink to the volume, every write already lands there and
# this script is a no-op.
#
# Run before stopping the pod:   bash /workspace/claude-snapshot.sh
set -euo pipefail

PERSIST=/workspace/.claude-persistent

if [ -L /root/.claude ]; then
    echo "/root/.claude is already a symlink to the volume - nothing to copy."
    exit 0
fi

mkdir -p "$PERSIST/projects"
cp -a /root/.claude/projects/. "$PERSIST/projects/" 2>/dev/null || true

if [ -f /root/.claude/history.jsonl ]; then
    touch "$PERSIST/history.jsonl"
    cat "$PERSIST/history.jsonl" /root/.claude/history.jsonl \
        | awk 'NF && !seen[$0]++' > "$PERSIST/history.jsonl.tmp"
    mv "$PERSIST/history.jsonl.tmp" "$PERSIST/history.jsonl"
fi

for f in settings.json .credentials.json; do
    [ -f "/root/.claude/$f" ] && cp -a "/root/.claude/$f" "$PERSIST/$f"
done

# .claude.json lives beside the config dir, not inside it, when
# CLAUDE_CONFIG_DIR is unset - so grab it from $HOME too.
[ -f /root/.claude.json ] && cp -a /root/.claude.json "$PERSIST/.claude.json"
chmod 600 "$PERSIST/.credentials.json" 2>/dev/null || true

echo "Snapshotted to $PERSIST:"
find "$PERSIST/projects" -name '*.jsonl' -printf '  %f  %s bytes\n' | sort
