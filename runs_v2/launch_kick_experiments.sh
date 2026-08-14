#!/usr/bin/env bash
# Section-22 experiments. All four warm-start from v12's final weights, so the
# ONLY difference between an arm and the control is the change under test.
# --init-from (not --resume) restarts total_steps at 0, which also makes
# --shaping-anneal-steps anchor correctly without --shaping-anneal-from.
set -u
cd /workspace/utmist-vc2-phase2
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log MUJOCO_GL=egl
# WANDB_API_KEY lives in the gitignored .env. `set -a` exports what the file
# defines without echoing any of it; without this every arm dies at startup on
# "No API key configured".
set -a; . ./.env; set +a
INIT=runs_v2/kick_ant_v12_v3_unfrozen/latest.pt
# --gcs-bucket is NOT optional here: train_kick_warp defaults it to None, so
# omitting it silently disables all backup. That is how npmp_rower_v2 was lost.
COMMON="--gcs-bucket vc2-2026-checkpoints
        --creature-xml creature_configs/ant.xml --arena pitch --pitch-scale 0.3125
        --init-from $INIT --worlds 2048 --max-hours 8 --steps 400000000
        --ball-radius 0.15 --ball-mass 0.045
        --reward-kind point --w-arrive 3.0 --w-strike 0.1
        --segment-secs-range 2.0 6.0 --target-dist-range 3.0 6.0
        --first-video-secs 900 --video-secs 1800"

launch () {  # name, extra args...
  local name=$1; shift
  setsid nohup .venv/bin/python -m rower_soccer.warp_port.train_kick_warp \
    --run-name "$name" $COMMON --w-upright 1.0 "$@" \
    > "runs_v2/$name.log" 2>&1 < /dev/null &
  disown; echo "launched $name"
}

launch kick_e0_control
launch kick_e1_shapeoff  --shaping-anneal-steps 150000000
launch kick_e3_posereset --reset-pose-each-segment
# w-upright appears twice for e2; argparse keeps the LAST, which is the point.
setsid nohup .venv/bin/python -m rower_soccer.warp_port.train_kick_warp \
  --run-name kick_e2_upright3 $COMMON --w-upright 3.0 \
  > runs_v2/kick_e2_upright3.log 2>&1 < /dev/null &
disown; echo "launched kick_e2_upright3"
