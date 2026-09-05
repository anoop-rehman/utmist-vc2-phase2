#!/bin/bash
# Census each E1 checkpoint as soon as it lands, so the analysis is not a
# serial tail on a 10-hour run. One core; no CUDA context.
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
cd /workspace/Transform2Act && source env-gpu.sh
OUT=/workspace/utmist-vc2-phase2/runs/d3_e1_ant/census
mkdir -p "$OUT"
while true; do
  pending=0
  for cfg in ant_e1_s1 ant_e1_s2 ant_e1_s3; do
    for ep in 0 10 20 30 40 50 60 70 80 90 100; do
      j=$(printf "%s/%s_e%04d.json" "$OUT" "$cfg" "$ep")
      [ -f "$j" ] && continue
      if [ "$ep" = "0" ]; then
        [ -d "/workspace/Transform2Act/results/$cfg" ] || { pending=1; continue; }
      else
        ck=$(printf "/workspace/Transform2Act/results/%s/models/epoch_%04d.p" "$cfg" "$ep")
        [ -f "$ck" ] || { pending=1; continue; }
      fi
      echo "=== census $cfg epoch $ep  $(date -Is)"
      .venv-gpu/bin/python \
        /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/e0_analyse.py \
        --cfg "$cfg" --epochs "$ep" --episodes 200 --out "$OUT" 2>&1 \
        | grep -v "param out of bounds"
    done
  done
  [ "$pending" = "0" ] && break
  sleep 120
done
echo "ALL CENSUSES DONE $(date -Is)"
