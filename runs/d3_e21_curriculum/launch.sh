#!/bin/bash
# D3 M3 E2.1 launcher -- the curriculum ablation. One arm per invocation:
#   ./launch.sh cur_s1 | cur_s2      CompetEvo's exploration curriculum ON
#   ./launch.sh flat_s1 | flat_s2    E2's flat env reward (the CONTROL)
#
# The two conditions differ in ONE argument, `--curriculum-steps`. Everything
# else -- cfg, seed, batch, minibatch, optim epochs, lr, hdims, log_std,
# sampler threads, eval cadence, video cadence -- is identical between them
# and identical to E2's `mlp_s{1,2}` arm, so the control is a genuine re-run
# and not a comparison against stored numbers.
#
# NEVER kill these. Each takes --stop-file; touch it to end cleanly after the
# epoch in flight. Under NVIDIA MPS a signal to one CUDA client can corrupt
# the live survivors.
#
# --max-epoch 400 x min_batch_size 50,000 = 20.0M env steps per arm, 4x E2.
# --curriculum-steps 4,000,000 = 80 epochs = 20% of the run, which is
# CompetEvo's OWN ratio for this task (`config/run-to-goal-ants-v0.yaml`:
# termination_epoch 200 of max_epoch_num 1000 at min_batch_size 50000).
set -euo pipefail
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
cd /workspace/Transform2Act
source env-gpu.sh
set -a; . /workspace/.env; set +a
P=/workspace/Transform2Act/.venv-gpu/bin/python
T=/workspace/utmist-vc2-phase2/rower_soccer/t2a_port
L=/workspace/utmist-vc2-phase2/runs/d3_e21_curriculum/logs
COMMON="--num-threads 10 --max-epoch 400 --save-interval 10 \
        --eval-every 5 --eval-episodes 10 --video-every 40 --video-episodes 9"
case "$1" in
  cur_s1|cur_s2)
    S=${1#cur_}
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES= setsid nohup $P $T/train_e11_mlp.py --cfg rtg_mlp_$S \
      --tag cur --curriculum-steps 4000000 $COMMON \
      --wandb --wandb-name d3_e21_mlp_cur_$S --stop-file /tmp/stop_e21_cur_$S \
      > $L/train_cur_$S.log 2>&1 & ;;
  flat_s1|flat_s2)
    S=${1#flat_}
    # shellcheck disable=SC2086
    CUDA_VISIBLE_DEVICES= setsid nohup $P $T/train_e11_mlp.py --cfg rtg_mlp_$S \
      --tag flat --curriculum-steps 0 $COMMON \
      --wandb --wandb-name d3_e21_mlp_flat_$S --stop-file /tmp/stop_e21_flat_$S \
      > $L/train_flat_$S.log 2>&1 & ;;
  *) echo "unknown arm $1"; exit 1 ;;
esac
echo "launched $1 pid $!"
