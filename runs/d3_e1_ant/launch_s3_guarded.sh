#!/bin/bash
# E1 seed 3, launched only when there is MEASURED headroom -- never from a
# remembered figure.
#
# The budget inherited from E0 ("three concurrent reference runs = 19.2 GB of
# 20.5") does NOT transfer to our ant. Two of OUR seeds peaked at 19.95 GB and
# OOM-ed the live D1 run off the card at 21:59 on 2026-08-29 (D1 asked for 8 MB
# and could not get it). Our 13-body graph with 500-1000 step episodes makes
# their float64 PPO update far heavier than their 5-body ant's.
#
# So: wait for seed 1 to exit, then require a real free-memory margin sampled
# over 60 s -- the 19.95 GB was a TRANSIENT during the PPO update, not steady
# state (steady is 4-7 GB per seed), so a single instantaneous reading is not
# evidence of anything.
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
NEED_FREE_MIB=8000          # a seed's own transient peak, with room for D1

while pgrep -f "train_their_ant.py --cfg ant_e1_s1" > /dev/null; do sleep 30; done
echo "[s3] seed 1 exited at $(date -Is); sampling free memory for 60 s"

while true; do
  worst=999999
  for i in $(seq 1 12); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    total=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits)
    free=$(( total - used ))
    [ "$free" -lt "$worst" ] && worst=$free
    sleep 5
  done
  echo "[s3] worst free over 60 s: ${worst} MiB (need ${NEED_FREE_MIB})"
  [ "$worst" -ge "$NEED_FREE_MIB" ] && break
  echo "[s3] not enough headroom, waiting"
  sleep 120
done

cd /workspace/Transform2Act && source env-gpu.sh
echo "[s3] launching at $(date -Is)"
exec .venv-gpu/bin/python \
  /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/train_their_ant.py \
  --cfg ant_e1_s3 --num_threads 15 --stop-file /tmp/stop_ant_e1_s3
