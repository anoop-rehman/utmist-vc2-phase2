#!/usr/bin/env bash
# D3 E4B watcher. Covers progress AND every failure signature worth acting on,
# because a watcher that greps only the happy path is silent through a
# crashloop. MPS IS ACTIVE: this script never kills anything, it only reports.
set -uo pipefail
CFGS="${CFGS:-rtg_e4r_s1 rtg_e4r_s2 rtg_e4r_s3}"
GPU_TRIP="${GPU_TRIP:-17500}"
# Resolve PIDs by scanning /proc each pass rather than pinning them once:
# a hardcoded map reports a deliberately deferred arm as DEAD forever, and
# misses the new PID when that arm relaunches. Explicit /proc scan, never
# pkill/pgrep -f (three self-matches this session).
resolve_pid() {
  # Return the MAIN process, not one of its sampler workers. khrylib forks
  # workers that inherit the parent's argv verbatim, so a cmdline match alone
  # picks an arbitrary one of ten -- which is how a healthy arm gets reported
  # with a changing pid (and, in E3, how a defunct worker was mistaken for a
  # restart). The main is the one the launcher detached, so its ppid is 1;
  # fall back to the lowest pid if that ever fails to match.
  local want="$1" p c ppid best=""
  for p in $(ps -o pid= -C python); do
    c=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
    case "$c" in *train_e4r_gnn.py*"--cfg $want "*) ;; *) continue;; esac
    ppid=$(ps -o ppid= -p "$p" 2>/dev/null | tr -d ' ')
    [ "$ppid" = "1" ] && { echo "$p"; return; }
    [ -z "$best" ] && best="$p"
  done
  [ -n "$best" ] && echo "$best"
}
declare -A SEEN=()
DISKSEEN=999999
while true; do
  for c in $CFGS; do
    pid=$(resolve_pid "$c")
    [ -n "$pid" ] && SEEN[$c]=running
    if [ -z "$pid" ]; then
      # A stop-file means we stopped it on purpose. Say so instead of crying
      # DEAD every pass -- an alarm that fires for an intended state trains
      # the reader to ignore it.
      if [ -e "/tmp/stop_e4b_${c#rtg_e4r_}" ]; then
        # Report an intended state ONCE, not every pass. A watcher that
        # repeats an unchanged expected condition every 10 minutes is noise,
        # and noise is how a real event gets missed.
        if [ "${SEEN[$c]:-}" != "stopped" ]; then
          echo "$c: stopped by stop-file (intentional; deferred, awaiting relaunch)"
          SEEN[$c]=stopped
        fi
      else
        if [ "${SEEN[$c]:-}" != "dead" ]; then
          echo "DEAD: $c has no process and no stop-file"
          SEEN[$c]=dead
        fi
      fi
      continue
    fi
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
    st=mi.get("stalemate_rate"); ep=ev[-1]["epoch"]
    # DEGENERATE MIRROR is a real condition but only MEANS anything once the
    # agent could plausibly move. At epoch 4 an untrained agent tying itself
    # 0-0 is the expected state, not a warning, and the pre-registration
    # scopes this verdict to epochs 200-400. Fire it inside the verdict
    # window, or earlier only as a REGRESSION -- the agent once scored and has
    # now stopped moving, which is genuinely worth waking someone for.
    ever_scored = max((x["eval"]["goal_rate"] for x in ev), default=0.0) > 0.5
    if st is not None and st>0.5 and mi.get("fwd_mean",9)<2.5 and (ep>=200 or ever_scored):
        m.append("DEGENERATE MIRROR%s: stalemate %.2f at fwd %.2f m"%(
            " (REGRESSION -- this arm previously scored)" if ever_scored and ep<200 else "",
            st,mi.get("fwd_mean",-1)))
    # progress line only when a NEW eval has landed, not on every pass
    stamp="/tmp/.watch_e4b_%s_lasteval"%c
    prev=open(stamp).read().strip() if os.path.exists(stamp) else ""
    if str(ep)!=prev:
        open(stamp,"w").write(str(ep))
        m.append("e%d goal %.2f fwd %.2f speed %.3f | mirror dec %.2f mut %.2f stale %.2f fwd %.2f | ladder win %s rho %s | ring %d"%(
            ep,e["goal_rate"],e["max_fwd"],e["speed"],
            mi.get("decisive_rate",-1),mi.get("mutual_rate",-1),mi.get("stalemate_rate",-1),mi.get("fwd_mean",-1),
            la.get("mean_win"),la.get("spearman"),last.get("ring",{}).get("size",0)))
if m: print("%s: %s"%(c,"; ".join(m)))
PY
  done
  # DISK. The one resource with a precedent for killing a run here: E3.1
  # seed 3 died at epoch 39 on a full disk. Reported once per threshold
  # crossing, not every pass.
  freem=$(df -Pm /workspace | awk 'NR==2{print $4}')
  for lim in 6000 4000 2500 1200; do
    if [ "$freem" -lt "$lim" ] && [ "${DISKSEEN:-999999}" -gt "$lim" ]; then
      echo "DISK: ${freem} MiB free on /workspace (crossed ${lim} MiB). Ring is ~1.6 GB/arm to epoch 400; E3.1 s3 died at epoch 39 on a full disk."
      DISKSEEN=$lim
    fi
  done

  u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  if [ "$u" -gt "$GPU_TRIP" ]; then
    sleep 20; u2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$u2" -gt "$GPU_TRIP" ] && echo "GPU SUSTAINED HIGH: $u then $u2 MiB -- stop an arm BY STOP-FILE (MPS active)"
  fi
  sleep 600
done
