#!/bin/bash
# D3 M3 E3: launch the two frozen-body GNN control arms once, and only once,
# all three E3 seeds have finished CLEANLY.
#
# "Cleanly" means each arm's log ends with Transform2Act's own `training done!`
# AND no trainer process survives. A watcher that fired on a crash would start
# the controls on a card that still had a wedged client on it, and would also
# hide the crash -- so the two conditions are checked separately and the script
# says which one it is waiting on.
#
#   setsid nohup ./launch_ctl_when_e3_done.sh > logs/ctl_watcher.log 2>&1 &
set -uo pipefail
D=/workspace/utmist-vc2-phase2/runs/d3_e3_adversarial
while true; do
  done_n=0
  for s in 1 2 3; do
    grep -q "training done!" "$D/logs/train_e3_s$s.log" 2>/dev/null && done_n=$((done_n+1))
  done
  alive=$(pgrep -cf "train_e3_gnn.py --cfg rtg_e3_s" || true)
  if [ "$done_n" -eq 3 ] && [ "$alive" -eq 0 ]; then
    echo "$(date -Is) all three E3 seeds finished cleanly -- launching controls"
    "$D/launch.sh" ctl_s1
    sleep 30
    "$D/launch.sh" ctl_s2
    exit 0
  fi
  # A seed that has exited WITHOUT `training done!` is a crash or a stop-file,
  # and the controls must not start on that quietly.
  if [ "$alive" -lt $((3 - done_n)) ]; then
    echo "$(date -Is) STOPPED: $done_n of 3 finished but only $alive alive --"\
         "an E3 seed exited without 'training done!'. Not launching the"\
         "controls; look at the logs."
    exit 1
  fi
  sleep 300
done
