#!/usr/bin/env bash
# Put the 2v2 game on the public internet for four known people, for one session.
#
#   scripts/play_online.sh              # random join code, port 8090
#   scripts/play_online.sh 8091 sardine # explicit port and join code
#
# Prints an https URL and a join code. Hand out the URL in one place and the code
# in another; neither is useful alone. Ctrl-C tears down both the tunnel and the
# server, and the URL dies with them -- a quick tunnel is per-process and there is
# no account, no DNS record and nothing left running afterwards.
#
# The tunnel is Cloudflare's `--url` quick tunnel: no account, no TLS of our own,
# and (measured, see docs/PLAY_2V2.md §1b) it carries the MJPEG stream to four
# concurrent viewers at the full 20 fps for as long as you leave it up.
set -euo pipefail

# Derived, not hardcoded: this has to run the checkout it was invoked from
# (a worktree, a clone on a laptop), or you tunnel the wrong code to your friends.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
VENV="${ROWER_VENV:-$REPO/.venv}"
PORT="${1:-8090}"
CODE="${2:-$(head -c 6 /dev/urandom | base32 | tr 'A-Z' 'a-z' | head -c 8)}"
TOOLS="$REPO/.tools"
CFD="$TOOLS/cloudflared"

cd "$REPO"
mkdir -p "$TOOLS" logs

# One 40 MB static binary, no package manager, no root. Pinned to "latest" on
# purpose: quick tunnels are a Cloudflare service and an old client eventually
# stops being able to open one.
if [ ! -x "$CFD" ]; then
    echo "[online] fetching cloudflared -> $CFD"
    curl -sSL -o "$CFD" \
        https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x "$CFD"
fi
"$CFD" --version

# The code goes in the environment, not in argv: `ps aux` is readable by everyone
# with a shell on this box, and this box is shared.
export ROWER_JOIN_CODE="$CODE"
export MUJOCO_GL=egl
export PYTHONPATH="$REPO"

echo "[online] starting the game server on :$PORT (scene compile takes ~15 s)"
"$VENV/bin/python" -m rower_soccer.game.server --port "$PORT" "${@:3}" \
    > logs/game_online.log 2>&1 &
GAME=$!
trap 'kill $GAME $TUN 2>/dev/null || true' EXIT INT TERM

for _ in $(seq 1 120); do
    curl -sf "http://localhost:$PORT/health" > /dev/null && break
    kill -0 $GAME 2>/dev/null || { echo "[online] server died:"; tail -20 logs/game_online.log; exit 1; }
    sleep 1
done

"$CFD" tunnel --url "http://localhost:$PORT" --no-autoupdate \
    > logs/tunnel.log 2>&1 &
TUN=$!
URL=""
for _ in $(seq 1 60); do
    URL=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' logs/tunnel.log | head -1 || true)
    [ -n "$URL" ] && break
    sleep 1
done
[ -n "$URL" ] || { echo "[online] no tunnel URL after 60 s; see logs/tunnel.log"; exit 1; }

cat <<EOF

  ================================================================
   URL        $URL
   join code  $CODE
  ================================================================

  Send the URL and the code to your three friends. Everyone opens the
  URL, types the code when the browser asks, picks a name and taps a
  seat. Nothing else is exposed: without the code the page loads and
  then does nothing at all.

  Ctrl-C here kills the tunnel, the URL and the server.
  Server log: logs/game_online.log   Tunnel log: logs/tunnel.log
EOF

wait $GAME
