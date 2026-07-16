#!/usr/bin/env bash
# Curriculum stage-1 diagnostic: 3 architectures on the SIMPLIFIED dribble task
# (worm+ball fixed adjacent, stationary close target). Isolates precise ball control
# from locomotion/reaching/tracking. Tells us: is the task the wall (all learn), or
# the architecture (plain learns, ours doesn't), or the body (none learn)?
# 3 arms + follow_base = 4 runs.
set -u
cd /workspace/utmist-vc2-phase2
mkdir -p logs
BUCKET=vc2-2026-checkpoints
INIT=runs_v2/_init_follow_base.pt
# Stage 1: fixed worm yaw + ball ahead at 0.8 m; target stationary, 0.5-1.5 m away.
STAGE1="--fixed-start --ball-spawn 0.8 0.8 --target-speed 0 0 --target-dist 0.5 1.5"
COMMON="--worlds 2048 --steps 20000000000 --max-hours 10 --ent-floor -1.2 \
  --ent-ceil 0.0 --ent-anneal-steps 400000000 --first-video-secs 90 \
  --video-secs 1200 --ckpt-secs 1800 --gcs-bucket $BUCKET $STAGE1"

launch () {  # run_name extra_args...
  local name=$1; shift
  setsid nohup env MUJOCO_GL=egl .venv/bin/python \
    -m rower_soccer.warp_port.train_dribble_warp \
    --run-name "$name" $COMMON "$@" \
    > "logs/${name}.log" 2>&1 < /dev/null &
  echo "launched $name (pid $!)"
  sleep 8
}

launch cur1_ours_ws       --init-from "$INIT"    # latent+decoder, warm-started
launch cur1_ours_scratch                          # latent+decoder, from scratch
launch cur1_plain         --plain                 # plain MLP, from scratch

echo "all 3 curriculum stage-1 arms launched."
