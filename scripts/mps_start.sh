#!/usr/bin/env bash
# Start the NVIDIA MPS daemon. Run ONCE per pod boot, BEFORE launching trainers.
#
# Measured on this pod (RTX 4000 Ada, 5 concurrent drill trainers, 4-minute
# clean-launch A/B): aggregate 30,454 -> 103,465 steps/s, a 3.4x speedup, with
# GPU "utilization" unchanged at 57% -> 58%.
#
# Why it works: without MPS every process gets its own CUDA context and the
# driver TIME-SLICES between them. Our kernels are small (1-2k ants of batched
# physics plus a small MLP), so the context switch is a large fraction of the
# work. MPS funnels all clients through one server context, so the kernels
# interleave instead of taking turns.
#
# Clients must see the same pipe directory, so anything launching a trainer has
# to export these two variables (runs_v2/relaunch_all.sh does).
set -u
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

if pgrep -x nvidia-cuda-mps-control >/dev/null; then
    echo "[mps] already running (pid $(pgrep -x nvidia-cuda-mps-control))"
else
    nvidia-cuda-mps-control -d
    sleep 2
    echo "[mps] started (pid $(pgrep -x nvidia-cuda-mps-control))"
fi
echo -n "[mps] active thread %: "
echo get_default_active_thread_percentage | nvidia-cuda-mps-control
echo "[mps] export these in any shell that launches a trainer:"
echo "      export CUDA_MPS_PIPE_DIRECTORY=$CUDA_MPS_PIPE_DIRECTORY"
echo "      export CUDA_MPS_LOG_DIRECTORY=$CUDA_MPS_LOG_DIRECTORY"
