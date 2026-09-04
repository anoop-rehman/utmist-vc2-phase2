#!/bin/bash
# D3 M3 E3: capture a checkpoint of the LIVE policy without waiting for an
# archival one and without touching the trainers.
#
# Why this is needed: best.p is frozen at epochs 0-3 on these arms (exec_R_eps
# plateaus immediately at the blob's survive-bonus return), and the first
# archival checkpoint is epoch 20. The population question -- has p_act4 gone
# on falling past 0.30 -- cannot wait for either.
#
# train_e3_gnn.py writes models/_video_tmp.p every --video-every epochs, hands
# it to the renderer, and deletes it in a `finally`. It exists for the ~60-120 s
# the render takes. Polling at 1 s and copying it out is enough, costs nothing,
# and cannot perturb the trainer: it is a read.
set -uo pipefail
D=/workspace/utmist-vc2-phase2/runs/d3_e3_adversarial
mkdir -p "$D/census/live_ckpt"
declare -A got
while true; do
  n=0
  for c in rtg_e3_s1 rtg_e3_s2 rtg_e3_s3; do
    src="/workspace/Transform2Act/results/$c/models/_video_tmp.p"
    if [ -z "${got[$c]:-}" ] && [ -f "$src" ]; then
      dst="$D/census/live_ckpt/${c}_live.p"
      if cp "$src" "$dst" 2>/dev/null; then
        got[$c]=1
        echo "$(date -Is) captured $c live checkpoint -> $dst"
      fi
    fi
    [ -n "${got[$c]:-}" ] && n=$((n+1))
  done
  [ "$n" -ge 3 ] && { echo "$(date -Is) all three captured"; exit 0; }
  sleep 1
done
