#!/usr/bin/env bash
# Restore Claude Code after a RunPod stop/start.
#
# Why this exists: on RunPod, "/" is a container overlay that is reset to the
# base image every time the pod is stopped. /workspace is a network volume and
# survives. Both the claude CLI (/root/.local) and its config+transcripts
# (/root/.claude) live on the overlay by default, so they vanish on restart.
# This script puts both back and repoints the config dir at the volume.
#
# Run once after every pod start:   bash /workspace/claude-bootstrap.sh
set -euo pipefail

PERSIST=/workspace/.claude-persistent
CACHE=/workspace/.claude-cli-cache

echo "[1/3] claude CLI"
if command -v claude >/dev/null 2>&1; then
    echo "      already present: $(command -v claude)"
elif [ -d "$CACHE/versions" ]; then
    mkdir -p /root/.local/share/claude /root/.local/bin
    cp -a "$CACHE/versions" /root/.local/share/claude/
    ln -sfn "/root/.local/share/claude/versions/$(cat "$CACHE/VERSION")" /root/.local/bin/claude
    echo "      restored $(cat "$CACHE/VERSION") from volume cache (offline)"
else
    echo "      no cache; downloading installer"
    curl -fsSL https://claude.ai/install.sh | bash
fi

echo "[2/3] config dir -> volume"
mkdir -p "$PERSIST"
if [ -e /root/.claude ] && [ ! -L /root/.claude ]; then
    # A fresh container may have created a stub dir. Keep anything new, but
    # never let it overwrite what is already on the volume (-n = no clobber).
    cp -an /root/.claude/. "$PERSIST"/ 2>/dev/null || true
    rm -rf /root/.claude
fi
ln -sfn "$PERSIST" /root/.claude
echo "      /root/.claude -> $PERSIST"

# claude reads .claude.json from $CLAUDE_CONFIG_DIR when that is set, but falls
# back to $HOME/.claude.json when it is not (e.g. a non-login shell). Symlink it
# so both paths resolve to the volume and can never disagree.
if [ -e /root/.claude.json ] && [ ! -L /root/.claude.json ]; then
    [ -s "$PERSIST/.claude.json" ] || cp -a /root/.claude.json "$PERSIST/.claude.json"
    rm -f /root/.claude.json
fi
[ -f "$PERSIST/.claude.json" ] || echo '{}' > "$PERSIST/.claude.json"
ln -sfn "$PERSIST/.claude.json" /root/.claude.json
echo "      /root/.claude.json -> $PERSIST/.claude.json"

echo "[3/3] shell env"
LINE_PATH='export PATH="$HOME/.local/bin:$PATH"'
LINE_CFG='export CLAUDE_CONFIG_DIR=/workspace/.claude-persistent'
for L in "$LINE_PATH" "$LINE_CFG"; do
    grep -qxF "$L" /root/.bashrc 2>/dev/null || echo "$L" >> /root/.bashrc
done
export PATH="$HOME/.local/bin:$PATH"
export CLAUDE_CONFIG_DIR="$PERSIST"

N=$(find "$PERSIST/projects" -name '*.jsonl' 2>/dev/null | wc -l)
echo
echo "Done. $N saved session(s) on the volume."
echo "Now run:   cd /workspace && claude --resume"
