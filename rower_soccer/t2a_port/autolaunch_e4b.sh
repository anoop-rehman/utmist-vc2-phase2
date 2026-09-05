#!/usr/bin/env bash
# D3 E4B detached launcher. Refuses rather than squeezes.
# Runs under setsid so it does not depend on the session (four in-session
# waiters were reaped during E4).
set -uo pipefail
LOG=/tmp/e4b_autolaunch.log
LOCK=/tmp/e4b.launched
NEED_FREE=17000     # 3 arms project to 15871 MiB (measured 5309/arm) + margin
say(){ echo "[$(date +%H:%M:%S)] $*" >> "$LOG"; }
[ -e "$LOCK" ] && { say "lock present, already launched"; exit 0; }

say "armed: waiting for the card to clear (need >= ${NEED_FREE} MiB free)"
for i in $(seq 1 240); do
  worst=999999
  for j in 1 2 3 4 5 6; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    [ "$f" -lt "$worst" ] && worst=$f
    sleep 10
  done
  say "worst free over 60s: ${worst} MiB"
  [ "$worst" -ge "$NEED_FREE" ] && break
done
if [ "$worst" -lt "$NEED_FREE" ]; then
  say "REFUSING TO LAUNCH: only ${worst} MiB free after waiting. Nothing killed, nothing started."
  exit 1
fi

touch "$LOCK"
say "launching E4B, 3 seeds"
bash /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/launch_e4b.sh >> "$LOG" 2>&1
say "waiting for first epochs, then asserting instruments"
CFGS="rtg_e4r_s1 rtg_e4r_s2 rtg_e4r_s3" WAIT_ROWS=1800 \
  bash /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/assert_e4b_instruments.sh >> "$LOG" 2>&1
say "instrument assertion exit=$?"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader >> "$LOG" 2>&1
say "E4B launch sequence complete"
