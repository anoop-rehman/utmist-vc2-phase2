#!/bin/bash
# D3 M3 E3 launcher -- the first adversarial rung with the design stages LIVE,
# plus the frozen-body GNN control that makes an E3 null interpretable.
#
#   ./launch.sh e3_s1 | e3_s2 | e3_s3     design stages ON   -- run FIRST
#   ./launch.sh ctl_s1 | ctl_s2           design stages OFF  -- run AFTER
#
# THE TWO GROUPS RUN SERIALLY, NOT TOGETHER, and that is a measured decision
# rather than a preference. All five at once is 3x10 + 2x8 = 46 sampler
# threads against a cgroup quota of 10.2 CPUs (`/sys/fs/cgroup/cpu.max`
# = 1020000/100000; `nproc`'s 48 is not what this container gets), and it took
# the GPU to 19.0 GB of 20.475 at the update peak. Under that load the two
# CPU-only control arms logged ZERO epochs in 15 minutes while the three E3
# arms degraded from 346 s to ~7 min per epoch.
#
# The control does not need to run BESIDE E3 -- it only needs to exist before
# E3 is INTERPRETED. So E3 takes all three seeds unimpeded first, then the two
# controls get the free card. Total wall clock is lower this way than five
# contended arms, and E3 keeps its third seed.
#
# Every arm: E2.1's `d2rep` reward regime (--curriculum-steps 130208333, alpha
# 1.000 -> 0.846 over 400 epochs, never crossing E2.1's critical 0.739), our
# converted DeepMind ant, E2's scripted opponent, 400 epochs x min_batch_size
# 50,000 = 20.0M env steps. The arms differ in exactly ONE cfg field,
# `env_specs.force_identity_design`, and the control arms differ from E2's own
# `rtg_gnn_s{1,2}` only in budget and checkpoint cadence.
#
# The control arms run on the GPU, because by the time they run it is free.
# Their first launch was CPU-only and it did not work: khrylib's `agent.py`
# sets OMP_NUM_THREADS=1 at import and `env-gpu.sh` sets it again, so the PPO
# update ran SINGLE-THREADED -- still going after 700 s against 150 s for the
# same update on the GPU. `--torch-threads` is kept for anyone who has to run
# an arm on CPU (setting the env var after torch is imported does not move
# torch's thread pool; `torch.set_num_threads` does), but it is not the fix
# here. Running serially is.
#
# NEVER kill these. Each takes --stop-file; touch it to end cleanly after the
# epoch in flight. NVIDIA MPS is active and a signal to one CUDA client can
# corrupt the live survivors -- that has destroyed two runs on this project.
set -euo pipefail
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
cd /workspace/Transform2Act
source env-gpu.sh
set -a; . /workspace/.env; set +a
P=/workspace/Transform2Act/.venv-gpu/bin/python
T=/workspace/utmist-vc2-phase2/rower_soccer/t2a_port
L=/workspace/utmist-vc2-phase2/runs/d3_e3_adversarial/logs
# 130,208,333 is E2.1's d2rep value, read out of
# runs/d3_e21_curriculum/launch.sh rather than reverse-engineered from the
# alpha trajectory: it makes a 20M-step run complete the SAME 0.1536 fraction
# of its anneal that D2's run completed of its own.
COMMON="--curriculum-steps 130208333 --eval-every 5 --eval-episodes 10 \
        --morph-every 1 --morph-episodes 20 --video-every 6 --video-episodes 9 \
        --archive-every 50"
case "$1" in
  e3_s1|e3_s2|e3_s3)
    S=${1#e3_}
    # shellcheck disable=SC2086
    setsid nohup $P $T/train_e3_gnn.py --cfg rtg_e3_$S --num-threads 10 $COMMON \
      --wandb --wandb-name d3_e3_gnn_$S --stop-file /tmp/stop_e3_$S \
      > $L/train_e3_$S.log 2>&1 & ;;
  ctl_s1|ctl_s2)
    S=${1#ctl_}
    # shellcheck disable=SC2086
    setsid nohup $P $T/train_e3_gnn.py --cfg rtg_e3c_$S \
      --num-threads 10 $COMMON \
      --wandb --wandb-name d3_e3_gnnctl_$S --stop-file /tmp/stop_e3c_$S \
      > $L/train_ctl_$S.log 2>&1 & ;;
  *) echo "unknown arm $1"; exit 1 ;;
esac
echo "launched $1 pid $!"
