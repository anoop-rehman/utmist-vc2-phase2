#!/usr/bin/env bash
# Fan out the 12-arm overnight sweep on THIS pod (shared A4000), detached.
# Each run is setsid+nohup so an SSH disconnect can't kill it; all sync to GCS and
# save best/latest/checkpoint, so they resume tomorrow. Staggered 8s to avoid 12
# simultaneous Warp JIT compiles / GCS bursts.
set -u
cd /workspace/utmist-vc2-phase2
mkdir -p logs
BUCKET=vc2-2026-checkpoints
COMMON="--worlds 2048 --max-hours 10 --ent-floor -1.2 --ent-ceil 0.0 \
  --ent-anneal-steps 400000000 --first-video-secs 90 --video-secs 1200 \
  --ckpt-secs 1800 --gcs-bucket $BUCKET"

launch () {  # task run_name extra_args
  local task=$1 name=$2; shift 2
  setsid nohup env MUJOCO_GL=egl .venv/bin/python \
    -m rower_soccer.warp_port.train_${task}_warp \
    --run-name "$name" $COMMON "$@" \
    > "logs/${name}.log" 2>&1 < /dev/null &
  echo "launched $name (pid $!)"
  sleep 8
}

# --- follow (4) ---
launch follow follow_base
launch follow follow_sm05        --smooth-coef 0.05
launch follow follow_sm10        --smooth-coef 0.1
launch follow follow_sm10_en05   --smooth-coef 0.1 --energy-coef 0.05

# --- dribble (8), paper reward, from scratch ---
launch dribble dribble_base
launch dribble dribble_base_s1      --seed 1
launch dribble dribble_sm05         --smooth-coef 0.05
launch dribble dribble_sm10         --smooth-coef 0.1
launch dribble dribble_sm20         --smooth-coef 0.2
launch dribble dribble_sm10_en02    --smooth-coef 0.1 --energy-coef 0.02
launch dribble dribble_sm10_en05    --smooth-coef 0.1 --energy-coef 0.05
launch dribble dribble_sm10_s1      --smooth-coef 0.1 --seed 1

echo "all 12 launched."
