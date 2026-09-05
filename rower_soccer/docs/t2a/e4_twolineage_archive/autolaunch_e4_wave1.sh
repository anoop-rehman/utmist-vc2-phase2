#!/usr/bin/env bash
# D3 M3 E4: detached wave-1 launcher.
#
# Written because two of my in-session waiters were reaped mid-wait. If the
# trigger dies, s1 finishes and the card sits idle until someone notices --
# the same instrumentation-at-transition failure E3 hit three times. This runs
# under nohup so it does not depend on the session.
#
# It REFUSES to launch rather than squeeze: MPS is active, and an OOM does not
# stay contained to whoever asked for the memory.
set -uo pipefail
LOG=/tmp/e4_autolaunch.log
LOCK=/tmp/e4_wave1.launched
S1_PID=3426432
NEED_FREE=14000          # a pair peaks ~11.3-12.9 GB; refuse below this
say(){ echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }

[ -e "$LOCK" ] && { say "lock present, already launched; exiting"; exit 0; }
say "armed: waiting for rtg_e31_s1 (pid $S1_PID)"
while kill -0 "$S1_PID" 2>/dev/null; do sleep 60; done
say "rtg_e31_s1 exited"

# 1. the null is the calibration for the verdict: regenerate it to FINAL
say "regenerating cross-seed null"
python3 /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/e4_null_traj.py >> "$LOG" 2>&1

# 2. sustained headroom check -- 6 readings over 60s, use the WORST
say "measuring sustained headroom"
worst=999999
for i in $(seq 1 6); do
  f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
  [ "$f" -lt "$worst" ] && worst=$f
  sleep 10
done
say "worst free over 60s: ${worst} MiB (need >= ${NEED_FREE})"
if [ "$worst" -lt "$NEED_FREE" ]; then
  say "REFUSING TO LAUNCH: not enough headroom. Nothing killed, nothing started."
  say "Stop rtg_e31d_s3body by stop-file (/tmp/stop_e31d_s3body) if its tail is"
  say "worth trading, then re-run this script."
  exit 1
fi

# 3. launch, and take the lock first so a retry cannot double-launch
touch "$LOCK"
say "launching wave 1 (seed pair 1)"
bash /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/launch_e4.sh 1 >> "$LOG" 2>&1
sleep 120
CFGS="rtg_e4_s1a rtg_e4_s1b" \
  bash /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/assert_e4_instruments.sh >> "$LOG" 2>&1
say "instrument assertion exit=$?"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> "$LOG" 2>&1
say "wave 1 launch sequence complete"
