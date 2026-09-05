#!/usr/bin/env bash
# D3 E4B: longitudinal GPU-vs-body-size recorder.
#
# Pairs each arm's per-client GPU MiB with its CURRENT bodies_mean, plus the
# card total. Sampled at 5 s so that every bodies_mean bin accumulates far more
# than one epoch of coverage -- the projection takes the PEAK per bin, and a
# peak is only meaningful if the window spans the update phase that produces
# it. An 80 s window on a ~117 s epoch can miss it entirely, which is exactly
# how the earlier 9428 figure understated the two-arm peak.
#
# Explicit pid resolution via ppid == 1: khrylib forks ten sampler workers per
# arm with identical argv, and matching on cmdline alone returns an arbitrary
# one of them.
set -uo pipefail
OUT="${OUT:-/workspace/utmist-vc2-phase2/runs/d3_e4b_ring/census/gpu_vs_bodies.csv}"
CFGS="${CFGS:-rtg_e4r_s1 rtg_e4r_s2 rtg_e4r_s3}"
DUR="${DUR:-100000}"
[ -f "$OUT" ] || echo "ts,cfg,pid,mib,total_mib,bodies_mean,motors_mean,epoch,bodies_max,bodies_max_running" > "$OUT"
end=$(( $(date +%s) + DUR ))
while [ "$(date +%s)" -lt "$end" ]; do
  total=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
  declare -A MEM=()
  while IFS=, read -r p m; do MEM[$(echo $p|tr -d ' ')]=$(echo $m|tr -d ' MiB'); done \
    < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
  for c in $CFGS; do
    main=""
    for p in $(ps -o pid= -C python 2>/dev/null); do
      cl=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null) || continue
      case "$cl" in *train_e4r_gnn.py*"--cfg $c "*) ;; *) continue;; esac
      [ "$(ps -o ppid= -p $p 2>/dev/null | tr -d ' ')" = "1" ] && { main=$p; break; }
    done
    [ -n "$main" ] || continue
    mib="${MEM[$main]:-}"
    [ -n "$mib" ] || continue
    read -r bm mm ep bmax bmaxrun < <(python3 - "$c" <<'PY'
import json,sys,os
f='/workspace/Transform2Act/results/%s/e4r_epochs.jsonl'%sys.argv[1]
try:
    r=[json.loads(l) for l in open(f)]
    c=r[-1]["census"]; print(c["bodies_mean"], c["motors_mean"], r[-1]["epoch"], c.get("bodies_max",""), max(x["census"]["bodies_max"] for x in r))
except Exception: print('', '', '')
PY
)
    [ -n "$bm" ] && echo "$(date +%s),$c,$main,$mib,$total,$bm,$mm,$ep,$bmax,$bmaxrun" >> "$OUT"
  done
  sleep 5
done
