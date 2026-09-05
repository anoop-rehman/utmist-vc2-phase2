#!/bin/bash
# Upload any E1/E1.1 video sidecar within ~5 minutes of it being written, by
# E0's path: `e0_wandb_media.py` -> separate `<name>_media` run, wandb.log with
# NO explicit step, `epoch` declared as the step metric. Renamed to .json.sent
# once uploaded, so this is safe to run on a glob in a loop.
cd /workspace/utmist-vc2-phase2
set -a; . /workspace/.env; set +a
while true; do
  for d in runs/d3_e1_ant/renders runs/d3_e11/renders; do
    ls $d/*.mp4.json >/dev/null 2>&1 || continue
    .venv/bin/python -m rower_soccer.t2a_port.e0_wandb_media $d/*.mp4.json
  done
  sleep 300
done
