#!/bin/bash
# D3 M3 E2 launcher. One arm per invocation:  ./launch.sh <arm>
#   gnn_s1 gnn_s2      Transform2Act GNN, GPU
#   mlp_s1 mlp_s2      plain-MLP PPO, batching MATCHED to the GNN (50,000/2,048), CPU
#   pub_s1 pub_s2      plain-MLP PPO, published PPO-MuJoCo batching (2,048/64), CPU
#
# NEVER kill these. Each takes --stop-file; touch it to end cleanly after the
# epoch in flight. Under NVIDIA MPS a signal to one CUDA client can corrupt the
# live survivors.
set -euo pipefail
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
cd /workspace/Transform2Act
source env-gpu.sh
set -a; . /workspace/.env; set +a
P=/workspace/Transform2Act/.venv-gpu/bin/python
T=/workspace/utmist-vc2-phase2/rower_soccer/t2a_port
L=/workspace/utmist-vc2-phase2/runs/d3_e2_rtg/logs
case "$1" in
  gnn_s1|gnn_s2)
    S=${1#gnn_}
    setsid nohup $P $T/train_e2_gnn.py --cfg rtg_gnn_$S --num_threads 14 \
      --eval-every 5 --eval-episodes 10 --video-every 10 --video-episodes 9 \
      --wandb --wandb-name d3_e2_gnn_$S --stop-file /tmp/stop_e2_gnn_$S \
      > $L/train_gnn_$S.log 2>&1 & ;;
  mlp_s1|mlp_s2)
    S=${1#mlp_}
    setsid nohup $P $T/train_e11_mlp.py --cfg rtg_mlp_$S --num-threads 10 \
      --save-interval 10 --eval-every 5 --eval-episodes 10 \
      --video-every 10 --video-episodes 9 \
      --wandb --wandb-name d3_e2_mlp_$S --stop-file /tmp/stop_e2_mlp_$S \
      > $L/train_mlp_$S.log 2>&1 & ;;
  pub_s1|pub_s2)
    S=${1#pub_}
    # E1.1's published-PPO-MuJoCo configuration, verbatim from the args stored
    # in `results/ant_e11_mlp_s1_pub/epoch_2399.p`: batch 2048, minibatch 64,
    # 10 optim epochs, linear lr anneal, 1 sampler thread. 2441 x 2048 = 5.0M.
    setsid nohup $P $T/train_e11_mlp.py --cfg rtg_mlp_$S --tag pub \
      --num-threads 1 --batch 2048 --mini-batch 64 --optim-epochs 10 \
      --anneal-lr --max-epoch 2441 --save-interval 200 \
      --eval-every 100 --eval-episodes 10 --video-every 800 --video-episodes 9 \
      --wandb --wandb-name d3_e2_mlp_${S}_pub --stop-file /tmp/stop_e2_pub_$S \
      > $L/train_pub_$S.log 2>&1 & ;;
  *) echo "unknown arm $1"; exit 1 ;;
esac
echo "launched $1 pid $!"
