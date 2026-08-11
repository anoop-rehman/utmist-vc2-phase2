#!/usr/bin/env bash
# Transform2Act hopper, paper scale, ON THE GPU in unmodified float64.
#
# Measured (docs: GPU_PROFILE.md): the PPO update drops 1911.6 s -> 93.9 s and
# the 1000-epoch ETA 24.5 days -> 3.5 days, a 7.0x speedup from the DEVICE
# alone. float32 buys a further 0.2% and is not worth the precision argument:
# the update is launch- and Python-bound (the policy loops over every state in
# the minibatch in Python), not FLOP-bound, so the card's 1/64 fp64 rate never
# bites. TF32 is irrelevant here for the same reason -- it only applies to fp32
# matmuls, and we are not in fp32.
#
# --num_threads 32, not their default 20 and not the CPU run's 8: once the
# update moves to the GPU, SAMPLING becomes 60% of the epoch, and sampling is
# CPU-bound mujoco on a 48-core box. The drill trainers on this pod are Warp/GPU
# and hold few cores, so 32 leaves comfortable headroom.
#
# Separate cfg name (hopper_gpu) so results/ does not collide with the CPU
# sanity run in results/hopper.
set -e
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"   # env-gpu.sh appends to it under set -u
cd /workspace/Transform2Act
. /workspace/Transform2Act/env-gpu.sh
nohup .venv-gpu/bin/python design_opt/train.py \
  --cfg ant_gpu --num_threads 16 --gpu_index 0 \
  >> results_ant_gpu.log 2>&1 &
echo "launched ant_gpu pid=$!"
