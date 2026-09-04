#!/bin/bash
# D3 M3 E3: run the population probe at each archival checkpoint, so the
# epoch-100 decision has the SERIES it needs and not just its endpoint.
#
# Section 3c's third row keys on "p_act4 strictly decreasing over the
# epoch-20/40/60/80/100 checkpoints", which requires a probe at each of them.
# Section 3c's first row keys on p_act4 and the step share AT epoch 100.
# Neither is available from the live JSONL: e3_morph.census gained its motor
# columns after these arms launched, so the running trainers do not log them.
#
# STRICTLY SEQUENTIAL AND NICED. Section 5b measured what concurrent probes do
# to the live arms under this box's 10.2-CPU quota -- and then measured that
# they were NOT the sustained cause, which does not make them free. One probe
# at a time, nice -n 19, and never while another is running.
set -uo pipefail
D=/workspace/utmist-vc2-phase2/runs/d3_e3_adversarial
T=/workspace/utmist-vc2-phase2/rower_soccer/t2a_port
P=/workspace/Transform2Act/.venv-gpu/bin/python
cd /workspace/Transform2Act
source env-gpu.sh
mkdir -p "$D/census"
while true; do
  for c in rtg_e3_s1 rtg_e3_s2 rtg_e3_s3 rtg_e3c_s1 rtg_e3c_s2; do
    for e in 20 40 60 80 100 200 300 400; do
      ck=$(printf "epoch_%04d" "$e")
      src="/workspace/Transform2Act/results/$c/models/$ck.p"
      out="$D/census/pop_${c}_e$(printf %04d "$e").json"
      [ -f "$src" ] || continue
      [ -f "$out" ] && continue
      echo "$(date -Is) probing $c @ $ck"
      CUDA_VISIBLE_DEVICES= nice -n 19 timeout 3600 "$P" \
        "$T/e3_population_probe.py" --cfg "$c" --ckpt "$ck" --designs 200 \
        --out "$out" >> "$D/census/population_watcher_probes.log" 2>&1 \
        || echo "$(date -Is) probe FAILED for $c @ $ck"
    done
  done
  sleep 300
done
