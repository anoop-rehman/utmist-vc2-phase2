#!/usr/bin/env bash
# D3 E4B: run seed 3 once s1 and s2 finish. s3 is DEFERRED, not dropped --
# three seeds is the minimum that survives one dead controller.
#
# Why it is deferred at all: three concurrent arms peaked at 18650 MiB (91%),
# with 17% of samples above the 17500 trigger and only 1825 MiB of headroom.
# That is not phase-locking that staggering could fix -- T_update is ~57% of an
# epoch, so three independent arms overlap 0.57^3 ~= 19% of the time by
# arithmetic. Two arms overlap 0.57^2 ~= 32% of the time but peak near 11.6 GB,
# which is comfortable.
set -uo pipefail
LOG=/tmp/e4b_s3_autolaunch.log
LOCK=/tmp/e4b_s3.launched
NEED_FREE=14000
say(){ echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
[ -e "$LOCK" ] && { say "lock present"; exit 0; }
say "armed: waiting for rtg_e4r_s1 (694927) and rtg_e4r_s2 (695070)"
while kill -0 694927 2>/dev/null || kill -0 695070 2>/dev/null; do sleep 120; done
say "s1 and s2 have exited"
worst=999999
for j in 1 2 3 4 5 6; do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$f" -lt "$worst" ] && worst=$f; sleep 10
done
say "worst free over 60s: ${worst} MiB (need >= ${NEED_FREE})"
if [ "$worst" -lt "$NEED_FREE" ]; then
  say "REFUSING TO LAUNCH: nothing killed, nothing started."; exit 1
fi
touch "$LOCK"
rm -f /tmp/stop_e4b_s3
rm -rf /workspace/Transform2Act/results/rtg_e4r_s3
say "launching rtg_e4r_s3 from a clean slate"
cd /workspace/Transform2Act && source env-gpu.sh
nohup .venv-gpu/bin/python \
  /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/train_e4r_gnn.py \
  --cfg rtg_e4r_s3 --ring-every 10 --ring-delta 0.0 \
  --curriculum-steps 130208333 --eval-every 5 --eval-episodes 10 \
  --mirror-episodes 20 --ladder-episodes 10 --ladder-k 5 \
  --morph-every 1 --morph-episodes 20 \
  --video-every 6 --video-episodes 9 --archive-every 50 \
  --restart-check-epoch 150 --num-threads 10 --wandb \
  --wandb-name d3_e4b_rtg_e4r_s3 --stop-file /tmp/stop_e4b_s3 \
  > /tmp/e4b_s3.log 2>&1 &
say "launched rtg_e4r_s3 pid $!"
sleep 300
CFGS="rtg_e4r_s3" WAIT_ROWS=1800 \
  bash /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/assert_e4b_instruments.sh >> "$LOG" 2>&1
say "instrument assertion exit=$?  -- s3 sequence complete"
