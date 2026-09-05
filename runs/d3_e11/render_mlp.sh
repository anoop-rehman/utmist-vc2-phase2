#!/bin/bash
# E1.1 MLP arms: render a best/median/worst clip at a spread of checkpoints and
# emit E0's sidecar contract so `e0_wandb_media.py` uploads them into separate
# `<name>_media` runs with no explicit step.
#
# WHAT THE CLIP SHOWS, and it is NOT what E0's and E1's clips show: the body is
# FROZEN here (force_identity_design), so all three panels are the SAME
# creature. The panels differ only in control and reset noise, so the clip is
# showing GAIT QUALITY and episode survival, never design variation.
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
R=/workspace/utmist-vc2-phase2/runs/d3_e11/renders
mkdir -p "$R"
cd /workspace/Transform2Act && source env-gpu.sh
for s in 1 2; do
  for ep in 0000 0399 0799 1199 1599 1999 2399; do
    out="$R/e11_mlp_s${s}_pub_e${ep}_bmw.mp4"
    [ -f "$out" ] && continue
    echo "=== render mlp s$s epoch $ep $(date -Is)"
    nice -n 19 taskset -c 40-47 env MUJOCO_GL=osmesa LP_NUM_THREADS=4 \
      .venv-gpu/bin/python \
      /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/e11_mlp_video.py \
      --cfg ant_e11_mlp_s$s --tag pub --epoch $((10#$ep)) --episodes 9 \
      --out "$out" --wandb-run "d3_e11_mlp_s${s}_pub_media" \
      --step $((10#$ep)) 2>&1 | grep -v "param out of bounds"
  done
done
echo "MLP RENDERS DONE $(date -Is)"
