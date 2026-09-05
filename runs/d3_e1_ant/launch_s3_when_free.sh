#!/bin/bash
# Launch E1 seed 3 as soon as seed 1's trainer exits, so the card never holds
# three reference runs at once (E0 measured 19.2 GB of 20.5 with three).
# Polls; never signals anything.
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
while pgrep -f "train_their_ant.py --cfg ant_e1_s1" > /dev/null; do sleep 60; done
cd /workspace/Transform2Act && source env-gpu.sh
exec .venv-gpu/bin/python \
  /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/train_their_ant.py \
  --cfg ant_e1_s3 --num_threads 15 --stop-file /tmp/stop_ant_e1_s3
