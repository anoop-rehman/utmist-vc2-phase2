#!/bin/bash
# D3 M3 E2.1: the post-hoc pass. Run AFTER every arm has finished.
# `train_e11_mlp.py --save-interval 10` saves at `epoch`, so a 400-epoch run's
# last checkpoint is epoch_0399.
set -uo pipefail
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
cd /workspace/Transform2Act; source env-gpu.sh
P=/workspace/Transform2Act/.venv-gpu/bin/python
T=/workspace/utmist-vc2-phase2/rower_soccer/t2a_port
O=/workspace/utmist-vc2-phase2/runs/d3_e21_curriculum/posthoc
E=${2:-399}
mkdir -p $O
run() {  # arm cfg epoch tag out
  echo "=== $5"
  CUDA_VISIBLE_DEVICES= $P $T/e2_posthoc.py --arm $1 --cfg $2 --epoch $3 \
      ${4:+--tag $4} --episodes 20 --out $O/$5.json 2>&1 | grep -vE "^\s*$"
}
A=${1:-all}
[ "$A" = all ] || [ "$A" = cur_s1 ]  && run mlp rtg_mlp_s1 $E cur  cur_s1
[ "$A" = all ] || [ "$A" = cur_s2 ]  && run mlp rtg_mlp_s2 $E cur  cur_s2
[ "$A" = all ] || [ "$A" = flat_s1 ] && run mlp rtg_mlp_s1 $E flat flat_s1
[ "$A" = all ] || [ "$A" = flat_s2 ] && run mlp rtg_mlp_s2 $E flat flat_s2
[ "$A" = all ] || [ "$A" = d2rep_s1 ] && run mlp rtg_mlp_s1 $E d2rep d2rep_s1
[ "$A" = all ] || [ "$A" = d2rep_s2 ] && run mlp rtg_mlp_s2 $E d2rep d2rep_s2
# the floor, measured through the SAME instrument. `--arm idle` ignores the
# checkpoint and emits zero torque.
[ "$A" = all ] || [ "$A" = idle ]    && run idle rtg_mlp_s1 0 "" idle
echo "=== analysis"
/workspace/utmist-vc2-phase2/.venv/bin/python $T/e21_analyse.py $O/*.json
