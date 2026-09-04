#!/bin/bash
# D3 M3 E3 post-hoc. One instrument for every arm plus the idle zero-torque
# floor, both protocols, 20 episodes, identical episode seeds.
#
#   ./collect.sh [EPOCH]      default 400
#
# Writes runs/d3_e3_adversarial/posthoc/*.json -- TRACKED, so the write-up and
# any later artefact source these rather than a training log. Nothing in the
# results table comes from wandb or from a `log_train.txt` line.
set -euo pipefail
E=${1:-400}
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
cd /workspace/Transform2Act
source env-gpu.sh
P=/workspace/Transform2Act/.venv-gpu/bin/python
T=/workspace/utmist-vc2-phase2/rower_soccer/t2a_port
O=/workspace/utmist-vc2-phase2/runs/d3_e3_adversarial/posthoc
C=/workspace/utmist-vc2-phase2/runs/d3_e3_adversarial/census
mkdir -p "$O" "$C"
pids=()
for c in rtg_e3_s1 rtg_e3_s2 rtg_e3_s3 rtg_e3c_s1 rtg_e3c_s2; do
  CUDA_VISIBLE_DEVICES= $P $T/e3_posthoc.py --cfg $c --epoch "$E" --episodes 20 \
      --out "$O/${c}_e$(printf %04d "$E").json" > "$O/${c}_e$(printf %04d "$E").log" 2>&1 &
  pids+=($!)
done
# the negative control: zero torque on every motor, frozen body, same
# instrument, same episode seeds. `D3_E2_RTG.md` measured -523.7 / goal 0.00.
CUDA_VISIBLE_DEVICES= $P $T/e3_posthoc.py --arm idle --cfg rtg_e3c_s1 --epoch 0 \
    --episodes 20 --out "$O/idle.json" > "$O/idle.log" 2>&1 &
pids+=($!)
for p in "${pids[@]}"; do wait "$p"; done
echo "--- post-hoc done, epoch $E ---"

# The design comparison against E1, on E0's OWN census instrument so E0, E1 and
# E3 are three readings of one measurement. E1's rows are copied in rather than
# recomputed -- they are the numbers `D3_E1_ANT.md` reports.
cp -n /workspace/utmist-vc2-phase2/runs/d3_e1_ant/census/ant_e1_s*_e0100.json "$C/" 2>/dev/null || true
for c in rtg_e3_s1 rtg_e3_s2 rtg_e3_s3; do
  CUDA_VISIBLE_DEVICES= $P $T/e0_analyse.py --cfg $c --epochs 100,"$E" \
      --episodes 200 --out "$C" >> "$C/analyse.log" 2>&1 &
done
wait
CUDA_VISIBLE_DEVICES= $P $T/e0_analyse.py --compare --out "$C" \
    --cfgs ant_e1_s1,ant_e1_s2,rtg_e3_s1,rtg_e3_s2,rtg_e3_s3 --epoch 100 \
    | tee "$C/compare_e1_vs_e3_epoch100.txt"
CUDA_VISIBLE_DEVICES= $P $T/e0_analyse.py --compare --out "$C" \
    --cfgs rtg_e3_s1,rtg_e3_s2,rtg_e3_s3 --epoch "$E" \
    | tee "$C/compare_e3_epoch$E.txt"
