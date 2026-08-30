#!/bin/bash
# D3 M3 E2: the post-hoc pass. Run AFTER every arm has finished.
#   ./collect.sh            all arms
#   ./collect.sh gnn_s1     one arm
# Checkpoint indices differ by trainer: the GNN saves `epoch_%04d.p` at
# epoch+1 (so 100), `train_e11_mlp.py` saves at `epoch` (so 99; the published
# arm at 2399, its last multiple of --save-interval 200).
set -uo pipefail
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
cd /workspace/Transform2Act; source env-gpu.sh
P=/workspace/Transform2Act/.venv-gpu/bin/python
T=/workspace/utmist-vc2-phase2/rower_soccer/t2a_port
O=/workspace/utmist-vc2-phase2/runs/d3_e2_rtg/posthoc
mkdir -p $O
run() {  # arm cfg epoch tag out
  echo "=== $5"
  CUDA_VISIBLE_DEVICES= $P $T/e2_posthoc.py --arm $1 --cfg $2 --epoch $3 \
      ${4:+--tag $4} --episodes 20 --out $O/$5.json 2>&1 | grep -vE "^\s*$"
}
ARMS=${1:-all}
[ "$ARMS" = all ] || [ "$ARMS" = gnn_s1 ] && run gnn rtg_gnn_s1 100 "" gnn_s1
[ "$ARMS" = all ] || [ "$ARMS" = gnn_s2 ] && run gnn rtg_gnn_s2 100 "" gnn_s2
[ "$ARMS" = all ] || [ "$ARMS" = mlp_s1 ] && run mlp rtg_mlp_s1 99 "" mlp_s1
[ "$ARMS" = all ] || [ "$ARMS" = mlp_s2 ] && run mlp rtg_mlp_s2 99 "" mlp_s2
[ "$ARMS" = all ] || [ "$ARMS" = pub_s1 ] && run mlp rtg_mlp_s1 2399 pub pub_s1
[ "$ARMS" = all ] || [ "$ARMS" = pub_s2 ] && run mlp rtg_mlp_s2 2399 pub pub_s2
echo "=== comparison"
/workspace/utmist-vc2-phase2/.venv/bin/python $T/e2_compare.py $O/*.json
