#!/usr/bin/env bash
# D3 M3 E4 watcher. Emits one line per event; silence means nothing tripped.
# Covers BOTH progress and failure, because a watcher that only greps the happy
# path is silent through a crashloop -- and silence looks like "still running".
# MPS IS ACTIVE: this script never kills anything. It only reports.
set -uo pipefail
CFGS="${CFGS:?set CFGS to the space-separated cfg ids that should be live}"
GPU_TRIP="${GPU_TRIP:-17500}"
while true; do
  for c in $CFGS; do
    pid=$(pgrep -f "train_e4_gnn.py --cfg $c --partner" | grep -v "^$$\$" | head -1 || true)
    f="/workspace/Transform2Act/results/$c/e4_epochs.jsonl"
    if [ -z "$pid" ]; then echo "DEAD: $c has no process"; continue; fi
    [ -s "$f" ] || continue
    python3 - "$c" "$f" <<'PY'
import json,sys,os,time
c,f=sys.argv[1],sys.argv[2]
r=[json.loads(l) for l in open(f)]
last=r[-1]; age=(time.time()-os.path.getmtime(f))/60
msgs=[]
if age>25: msgs.append("STALLED %.0f min since last epoch"%age)
cen=last.get("census",{})
if cen.get("p_act4",1.0)<0.5: msgs.append("p_act4 %.2f -- actuator collapse (E3's failure)"%cen["p_act4"])
ev=[x for x in r if "eval" in x]
if ev:
    e=ev[-1]["eval"]; rc=ev[-1].get("race",{})
    if rc.get("draw_rate",0)>0.5: msgs.append("draw_rate %.2f -- coupled channel OFF, verdict would be UNTESTABLE"%rc["draw_rate"])
    msgs.append("e%d goal %.2f loss %.2f fell %.2f fwd %.2f speed %.3f nb %.1f draw %s"%(
        ev[-1]["epoch"],e["goal_rate"],e["loss_rate"],e["fall_rate"],e["max_fwd"],
        e["speed"],e.get("bodies_exec",0),rc.get("draw_rate")))
print("%s: %s"%(c,"; ".join(msgs)) if msgs else "",end="\n" if msgs else "")
PY
  done
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "$used" -gt "$GPU_TRIP" ]; then
    sleep 20; u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$u2" -gt "$GPU_TRIP" ] && echo "GPU SUSTAINED HIGH: ${used} then ${u2} MiB of 20475 -- stop an arm BY STOP-FILE (MPS active, never kill)"
  fi
  sleep 300
done
