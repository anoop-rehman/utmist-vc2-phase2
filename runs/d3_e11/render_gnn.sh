#!/bin/bash
# E1.1 GNN arm clips. Same caveat as the MLP arm's: the body is FROZEN
# (force_identity_design), so all three panels are the SAME creature and the
# clip shows gait quality and episode survival, NOT design variation -- unlike
# E0's and E1's clips, where every episode draws its own design.
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
R=/workspace/utmist-vc2-phase2/runs/d3_e11/renders
mkdir -p "$R"; cd /workspace/Transform2Act && source env-gpu.sh
while true; do
  pending=0
  for ep in 0 10 20 30 40 50 60 70 80 90 100; do
    out=$(printf "%s/e11_gnn_s1_e%04d_bmw.mp4" "$R" "$ep")
    [ -f "$out" ] && continue
    if [ "$ep" = "0" ]; then extra="--untrained"; else
      ck=$(printf "/workspace/Transform2Act/results/ant_e11_gnn_s1/models/epoch_%04d.p" "$ep")
      [ -f "$ck" ] || { pending=1; continue; }; extra=""
    fi
    echo "=== render gnn s1 e$ep $(date -Is)"
    nice -n 19 taskset -c 40-47 env MUJOCO_GL=osmesa LP_NUM_THREADS=4 \
      .venv-gpu/bin/python /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/e0_video.py \
      --cfg ant_e11_gnn_s1 --epoch $ep --episodes 9 --out "$out" \
      --wandb-run d3_e11_gnn_s1_media $extra 2>&1 | grep -v "param out of bounds"
  done
  [ "$pending" = "0" ] && break
  sleep 180
done
echo "GNN RENDERS DONE $(date -Is)"
