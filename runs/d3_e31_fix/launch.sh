#!/bin/bash
# D3 M3 E3.1 launcher -- the derived fix for E3's actuator deletion.
#
#   ./launch.sh p_s1 | p_s2 | p_s3     PRIMARY: control_log_std -1.5, design ON
#   ./launch.sh f_s1 | f_s2 | f_s3     SECOND:  floor n_motors>=4 AND log_std -1.5
#
# GPU MEMORY CAPS THIS AT THREE DESIGN-ON ARMS AT A TIME. E3's three arms peaked
# at 19.0 GB of 20.475 while their bodies were still ~13 nodes; six would be
# ~38 GB. The primary arms run first because they carry the pre-registered
# falsifier, which fires by epoch 20; the floor arms follow when they finish or
# when the falsifier resolves.
#
# Everything except `control_log_std` (and `min_motors` on the floor arms) is
# identical to E3, so the comparison isolates the fix.
#
# NEVER kill these. Each takes --stop-file. NVIDIA MPS is active and a signal to
# a CUDA client can corrupt the live survivors.
set -euo pipefail
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
cd /workspace/Transform2Act
source env-gpu.sh
set -a; . /workspace/.env; set +a
P=/workspace/Transform2Act/.venv-gpu/bin/python
T=/workspace/utmist-vc2-phase2/rower_soccer/t2a_port
L=/workspace/utmist-vc2-phase2/runs/d3_e31_fix/logs
COMMON="--curriculum-steps 130208333 --eval-every 5 --eval-episodes 10 \
        --morph-every 1 --morph-episodes 20 --video-every 6 --video-episodes 9 \
        --archive-every 50 --num-threads 10"
case "$1" in
  p_s1|p_s2|p_s3)
    S=${1#p_}
    # shellcheck disable=SC2086
    setsid nohup $P $T/train_e3_gnn.py --cfg rtg_e31_$S $COMMON \
      --wandb --wandb-name d3_e31_primary_$S --stop-file /tmp/stop_e31_p_$S \
      > $L/train_p_$S.log 2>&1 & ;;
  f_s1|f_s2|f_s3)
    S=${1#f_}
    # shellcheck disable=SC2086
    setsid nohup $P $T/train_e3_gnn.py --cfg rtg_e31f_$S $COMMON \
      --wandb --wandb-name d3_e31_floor_$S --stop-file /tmp/stop_e31_f_$S \
      > $L/train_f_$S.log 2>&1 & ;;
  *) echo "unknown arm $1"; exit 1 ;;
esac
echo "launched $1 pid $!"
