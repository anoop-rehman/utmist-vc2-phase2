#!/bin/bash
# Fill the matched-MLP clips that the first pass missed. train_e11_mlp saves at
# epoch (n+1)%interval==0, i.e. 0,9,19,...,99 -- the first render pass asked for
# 20/40/60/80, which do not exist, and errored out.
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
cd /workspace/Transform2Act && source env-gpu.sh
R=/workspace/utmist-vc2-phase2/runs/d3_e11/renders
for s in 1 2; do for ep in 19 39 59 79; do
  out=$(printf "%s/e11_mlp_s%s_matched_e%04d_bmw.mp4" "$R" "$s" "$ep")
  [ -f "$out" ] && continue
  nice -n 19 taskset -c 40-47 env MUJOCO_GL=osmesa LP_NUM_THREADS=4 \
    .venv-gpu/bin/python /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/e11_mlp_video.py \
    --cfg ant_e11_mlp_s$s --epoch $ep --episodes 9 --out "$out" \
    --wandb-run "d3_e11_mlp_s${s}_matched_media" --step $ep 2>&1 | grep -v "param out of bounds"
done; done
echo "MATCHED FILL DONE $(date -Is)"
