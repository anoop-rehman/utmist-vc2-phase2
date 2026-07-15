#!/usr/bin/env bash
# Fine-control dribble sweep. Warm-started dribble reaches the ball but can't PARK
# it (fitness flat at 0.19). Two levers, isolated across 5 arms:
#   A fine control : --state-dependent-std (gSDE-flavored) OR --ent-floor -2.5 (cheap)
#   B reward-park  : --shaping-anneal-steps (fade velocity shaping -> pure fitness)
# All warm-start from follow_base and keep entropy annealing (the collapse fix).
# 5 arms + follow_base = 6 runs sharing the GPU.
set -u
cd /workspace/utmist-vc2-phase2
mkdir -p logs
BUCKET=vc2-2026-checkpoints
INIT=runs_v2/_init_follow_base.pt
ANNEAL=400000000
COMMON="--worlds 2048 --steps 20000000000 --max-hours 10 --ent-ceil 0.0 \
  --ent-anneal-steps $ANNEAL --first-video-secs 90 --video-secs 1200 \
  --ckpt-secs 1800 --gcs-bucket $BUCKET --init-from $INIT"

launch () {  # run_name extra_args...
  local name=$1; shift
  setsid nohup env MUJOCO_GL=egl .venv/bin/python \
    -m rower_soccer.warp_port.train_dribble_warp \
    --run-name "$name" $COMMON "$@" \
    > "logs/${name}.log" 2>&1 < /dev/null &
  echo "launched $name (pid $!)"
  sleep 8
}

# A alone
launch drib_sdstd        --state-dependent-std
launch drib_lowent       --ent-floor -2.5
# B alone
launch drib_park         --ent-floor -1.2 --shaping-anneal-steps $ANNEAL
# A + B (the bet)
launch drib_sdstd_park   --state-dependent-std --shaping-anneal-steps $ANNEAL
launch drib_lowent_park  --ent-floor -2.5 --shaping-anneal-steps $ANNEAL

echo "all 5 fine-control dribble arms launched."
