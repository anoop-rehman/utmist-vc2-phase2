#!/bin/bash
# E1.1 MLP arms with batching MATCHED to the GNN arm (batch 50,000, minibatch
# 2048, 10 PPO epochs), 100 epochs = the GNN arm's 5.0M step budget.
#
# These were launched once and stopped at epoch 10 to give CPU back to E1;
# this re-runs them to completion WITH LOGGING AND VIDEO WIRED FROM THE START,
# which is the thing that was missing the first time.
#
# Waits for E1 seed 3 so it cannot slow the last E1 seed (the first attempt
# cost E1 ~40% of its epoch rate). CPU only -- no CUDA context.
cd /workspace/Transform2Act && source env-gpu.sh
while pgrep -f "train_their_ant.py --cfg ant_e1_s3" > /dev/null; do sleep 60; done
echo "[matched] E1 seed 3 done at $(date -Is); starting"
rm -rf /workspace/Transform2Act/results/ant_e11_mlp_s1 /workspace/Transform2Act/results/ant_e11_mlp_s2
for s in 1 2; do
  rm -f /tmp/stop_e11_mlp_s$s
  setsid nohup .venv-gpu/bin/python \
    /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/train_e11_mlp.py \
    --cfg ant_e11_mlp_s$s --num-threads 8 --save-interval 10 \
    --stop-file /tmp/stop_e11_mlp_s$s \
    > /workspace/utmist-vc2-phase2/runs/d3_e1_ant/logs/train_e11_matched_s$s.log 2>&1 &
done
while pgrep -f "train_e11_mlp.py --cfg ant_e11_mlp_s. --num-threads 8" > /dev/null; do sleep 60; done
echo "[matched] training done at $(date -Is); shipping metrics"
cd /workspace/utmist-vc2-phase2
set -a; . /workspace/.env; set +a
for s in 1 2; do
  .venv/bin/python -m rower_soccer.t2a_port.e11_ship_mlp \
    --dir /workspace/Transform2Act/results/ant_e11_mlp_s$s \
    --name d3_e11_mlp_s${s}_matched \
    --notes "E1.1 MLP baseline, batching MATCHED to the GNN arm (batch 50000, minibatch 2048, 10 epochs, lr 3e-4). Design stages run but forced to identity." \
    --config arm=mlp batching=matched seed=$s policy_lr=3e-4 batch=50000 mini_batch=2048 optim_epochs=10 anneal_lr=false net=64,64_tanh
done
echo "[matched] rendering"
cd /workspace/Transform2Act && source env-gpu.sh
for s in 1 2; do
  for ep in 0 20 40 60 80 99; do
    out=/workspace/utmist-vc2-phase2/runs/d3_e11/renders/e11_mlp_s${s}_matched_e$(printf %04d $ep)_bmw.mp4
    [ -f "$out" ] && continue
    nice -n 19 taskset -c 40-47 env MUJOCO_GL=osmesa LP_NUM_THREADS=4 \
      .venv-gpu/bin/python \
      /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/e11_mlp_video.py \
      --cfg ant_e11_mlp_s$s --epoch $ep --episodes 9 --out "$out" \
      --wandb-run "d3_e11_mlp_s${s}_matched_media" --step $ep 2>&1 \
      | grep -v "param out of bounds"
  done
done
echo "[matched] ALL DONE $(date -Is)"
