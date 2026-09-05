#!/usr/bin/env bash
# D3 M3 E4: fail loudly if any launched arm is missing an instrument.
# Written because E3 lost instrumentation three times at transitions -- the
# watchers kept polling arms that had already exited, and a new arm launched
# with nothing watching it.
set -uo pipefail
CFGS="${CFGS:?set CFGS to the space-separated cfg ids that should be live}"
fail=0
for c in $CFGS; do
  pid=$(pgrep -f "train_e4r_gnn.py --cfg $c " | head -1 || true)
  f="/workspace/Transform2Act/results/$c/e4r_epochs.jsonl"
  if [ -z "$pid" ]; then echo "MISSING PROCESS: $c"; fail=1; continue; fi
  # Wait (bounded) for the first epoch rather than passing vacuously. The
  # first version printed "NO JSONL YET" and still exited 0 -- an assertion
  # that succeeds on an empty file verifies nothing, which is the same failure
  # as a gate that passes on a dead opponent. WAIT_ROWS=0 skips the wait.
  waited=0
  while [ ! -s "$f" ] && [ "$waited" -lt "${WAIT_ROWS:-900}" ]; do
    sleep 30; waited=$((waited+30))
  done
  if [ ! -s "$f" ]; then
    echo "NO JSONL after ${waited}s: $c (pid $pid) -- NOT verified"; fail=1; continue
  fi
  n=$(wc -l < "$f")
  miss=$(python3 - "$f" <<'PY'
import json,sys
r=[json.loads(l) for l in open(sys.argv[1])]
last=r[-1]
need_row=["census","ring","control_log_std","alpha"]
need_cen=["p_act4","motors_mean","bodies_mean","top_topology_share"]
m=[k for k in need_row if k not in last]
m+=["census."+k for k in need_cen if k not in last.get("census",{})]
ev=[x for x in r if "eval" in x]
if ev:
    for k in ("race","mirror","ladder","dodge_pooled"):
        if k not in ev[-1]: m.append(k)
    mm=ev[-1].get("mirror",{})
    for k in ("decisive_rate","mutual_rate","stalemate_rate","fwd_mean"):
        if k not in mm: m.append("mirror."+k)
print(",".join(m))
PY
)
  if [ -n "$miss" ]; then echo "MISSING FIELDS:  $c -> $miss"; fail=1
  else echo "OK: $c pid $pid, $n epochs, all instruments present"; fi
done
exit $fail
