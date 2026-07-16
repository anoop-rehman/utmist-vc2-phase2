#!/usr/bin/env bash
# Warm-started dribble sweep: each arm inits from follow_base (locomotion solved),
# so it only has to learn ball control -- the from-scratch runs sat flat at the
# do-nothing floor for 200M steps because they had to learn locomotion AND ball
# control at once, a chicken-and-egg the reward can't bootstrap.
# 5 arms + the still-running follow_base = 6 runs sharing the GPU (~2x the speed of
# the 12-way split).
set -u
cd /workspace/utmist-vc2-phase2
mkdir -p logs
BUCKET=vc2-2026-checkpoints
INIT=runs_v2/_init_follow_base.pt
COMMON="--worlds 2048 --steps 20000000000 --max-hours 10 --ent-floor -1.2 \
  --ent-ceil 0.0 --ent-anneal-steps 400000000 --first-video-secs 90 \
  --video-secs 1200 --ckpt-secs 1800 --gcs-bucket $BUCKET --init-from $INIT"

launch () {  # run_name extra_args
  local name=$1; shift
  setsid nohup env MUJOCO_GL=egl .venv/bin/python \
    -m rower_soccer.warp_port.train_dribble_warp \
    --run-name "$name" $COMMON "$@" \
    > "logs/${name}.log" 2>&1 < /dev/null &
  echo "launched $name (pid $!)"
  sleep 8
}

launch dribble_ws_base
launch dribble_ws_sm05        --smooth-coef 0.05
launch dribble_ws_sm10        --smooth-coef 0.1
launch dribble_ws_sm20        --smooth-coef 0.2
launch dribble_ws_sm10_en02   --smooth-coef 0.1 --energy-coef 0.02

echo "all 5 warm-started dribble arms launched."
