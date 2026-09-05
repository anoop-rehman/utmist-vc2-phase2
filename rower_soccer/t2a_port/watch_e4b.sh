#!/usr/bin/env bash
# D3 E4B watcher. Covers progress AND every failure signature worth acting on,
# because a watcher that greps only the happy path is silent through a
# crashloop. MPS IS ACTIVE: this script never kills anything, it only reports.
set -uo pipefail
CFGS="${CFGS:-rtg_e4r_s1 rtg_e4r_s2 rtg_e4r_s3}"
GPU_TRIP="${GPU_TRIP:-17500}"
declare -A PIDS=( [rtg_e4r_s1]=694927 [rtg_e4r_s2]=695070 [rtg_e4r_s3]=695257 )
while true; do
  for c in $CFGS; do
    pid="${PIDS[$c]:-}"
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then echo "DEAD: $c (pid $pid)"; continue; fi
    [ -f "/workspace/Transform2Act/results/$c/RESTART_RECOMMENDED" ] && \
      echo "$c: RESTART RECOMMENDED -- pre-registered dead-controller rule fired"
    f="/workspace/Transform2Act/results/$c/e4r_epochs.jsonl"
    [ -s "$f" ] || continue
    python3 - "$c" "$f" <<'PY'
import json,sys,os,time
c,f=sys.argv[1],sys.argv[2]
r=[json.loads(l) for l in open(f)]
last=r[-1]; age=(time.time()-os.path.getmtime(f))/60
m=[]
if age>30: m.append("STALLED %.0f min"%age)
cen=last.get("census",{})
if cen.get("p_act4",1.0)<0.5: m.append("p_act4 %.2f -- actuator collapse (E3's failure)"%cen["p_act4"])
ev=[x for x in r if "eval" in x]
if ev:
    e=ev[-1]["eval"]; mi=ev[-1].get("mirror",{}); la=ev[-1].get("ladder",{})
    st=mi.get("stalemate_rate")
    if st is not None and st>0.5 and (mi.get("fwd_mean",9)<2.5):
        m.append("DEGENERATE MIRROR: stalemate %.2f at fwd %.2f m"%(st,mi.get("fwd_mean",-1)))
    m.append("e%d goal %.2f fwd %.2f speed %.3f | mirror dec %.2f mut %.2f stale %.2f fwd %.2f | ladder win %s rho %s | ring %d"%(
        ev[-1]["epoch"],e["goal_rate"],e["max_fwd"],e["speed"],
        mi.get("decisive_rate",-1),mi.get("mutual_rate",-1),mi.get("stalemate_rate",-1),mi.get("fwd_mean",-1),
        la.get("mean_win"),la.get("spearman"),last.get("ring",{}).get("size",0)))
if m: print("%s: %s"%(c,"; ".join(m)))
PY
  done
  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "$u" -gt "$GPU_TRIP" ]; then
    sleep 20; u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$u2" -gt "$GPU_TRIP" ] && echo "GPU SUSTAINED HIGH: $u then $u2 MiB -- stop an arm BY STOP-FILE (MPS active)"
  fi
  sleep 600
done
