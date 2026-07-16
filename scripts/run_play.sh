#!/usr/bin/env bash
# Launch the interactive play server, detached. Usage: run_play.sh [port]
cd /workspace/utmist-vc2-phase2
PORT="${1:-8087}"
mkdir -p logs
exec env MUJOCO_GL=egl .venv/bin/python -m rower_soccer.warp_port.play_server \
    --follow runs_v2/follow_base/latest.pt \
    --dribble runs_v2/cur1_ours_ws/latest.pt \
    --port "$PORT"
