#!/bin/bash
# D3 M3 E1: render a best/median/worst clip at every census epoch, for every
# seed, as soon as its checkpoint lands -- and upload it the way E0 does.
#
# WHY THIS EXISTS: E1 was launched with metrics shipping only. No clip was ever
# rendered and no `_media` run was ever created. That is NOT the wandb
# step-drop bug E0 found (nothing was uploaded at all); it is a gap in E1's
# launch. The upload path below is E0's, unchanged: `e0_wandb_media.py` logs
# into a SEPARATE `<name>_media` run with NO explicit step and `epoch` declared
# as the step metric, which is what makes a late/backfilled row impossible to
# drop.
#
# Renders are `nice -n 19` and strictly sequential so they cannot take sampler
# cores from the live training runs.
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log

# Seed 3's ORIGINAL media run id is unusable: the aborted 39-epoch attempt was
# deleted from wandb, and wandb refuses to recreate a deleted run id
# ("run ... was previously created and deleted; try a new run id"). So the
# clean seed-3 run logs to a v2 id. Seeds 1 and 2 keep their original ids.
media_run_name () {
  if [ "$1" = "3" ]; then echo "d3_e1_ant_seed3_media_v2"; else echo "d3_e1_ant_seed${1}_media"; fi
}

R=/workspace/utmist-vc2-phase2/runs/d3_e1_ant/renders
mkdir -p "$R"
cd /workspace/Transform2Act && source env-gpu.sh

render () {   # cfg seed epoch extra_args...
  local cfg=$1 seed=$2 ep=$3; shift 3
  local out
  out=$(printf "%s/%s_e%04d_bmw.mp4" "$R" "$cfg" "$ep")
  [ -n "$SUFFIX" ] && out=$(printf "%s/%s_e%04d_%s.mp4" "$R" "$cfg" "$ep" "$SUFFIX")
  [ -f "$out" ] && return 0
  echo "=== render $cfg e$ep $* $(date -Is)"
  nice -n 19 taskset -c 40-47 env MUJOCO_GL=osmesa LP_NUM_THREADS=4 .venv-gpu/bin/python \
    /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/e0_video.py \
    --cfg "$cfg" --epoch "$ep" --episodes 9 --out "$out" \
    --wandb-run "$(media_run_name $seed)" "$@" 2>&1 | grep -v "param out of bounds"
}

upload () {
  local n
  n=$(ls "$R"/*.mp4.json 2>/dev/null | wc -l)
  [ "$n" = "0" ] && return 0
  cd /workspace/utmist-vc2-phase2
  set -a; . /workspace/.env; set +a
  .venv/bin/python -m rower_soccer.t2a_port.e0_wandb_media "$R"/*.mp4.json
  cd /workspace/Transform2Act
}

while true; do
  pending=0
  for seed in 1 2 3; do
    cfg=ant_e1_s$seed
    [ -d "/workspace/Transform2Act/results/$cfg" ] || { pending=1; continue; }
    # epoch 0: the untrained policy's clip, plus the INITIAL BODY clip (zero
    # design action). Rendered before the first upload so both land in one
    # log call, as E0 does -- two log calls at the same step is exactly what
    # dropped rows for E0.
    SUFFIX=initial render "$cfg" "$seed" 0 --untrained --initial-body \
      --video-key video/initial_ant
    SUFFIX= render "$cfg" "$seed" 0 --untrained
    for ep in 10 20 30 40 50 60 70 80 90 100; do
      ck=$(printf "/workspace/Transform2Act/results/%s/models/epoch_%04d.p" "$cfg" "$ep")
      if [ -f "$ck" ]; then SUFFIX= render "$cfg" "$seed" "$ep"; else pending=1; fi
    done
  done
  upload
  [ "$pending" = "0" ] && break
  sleep 180
done
echo "ALL E1 RENDERS DONE $(date -Is)"
