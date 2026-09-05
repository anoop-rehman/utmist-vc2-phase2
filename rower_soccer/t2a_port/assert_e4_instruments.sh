#!/usr/bin/env bash
# D3 M3 E4: fail loudly if any launched arm is missing an instrument.
# Written because E3 lost instrumentation three times at transitions -- the
# watchers kept polling arms that had already exited, and a new arm launched
# with nothing watching it.
set -uo pipefail
CFGS="${CFGS:?set CFGS to the space-separated cfg ids that should be live}"
fail=0
for c in $CFGS; do
  pid=$(pgrep -f "train_e4_gnn.py --cfg $c --partner" | head -1 || true)
  f="/workspace/Transform2Act/results/$c/e4_epochs.jsonl"
  if [ -z "$pid" ]; then echo "MISSING PROCESS: $c"; fail=1; continue; fi
  if [ ! -s "$f" ]; then echo "NO JSONL YET:    $c (pid $pid)"; continue; fi
  n=$(wc -l < "$f")
  miss=$(python3 - "$f" <<'PY'
import json,sys
r=[json.loads(l) for l in open(sys.argv[1])]
last=r[-1]
need_row=["census","opponent","control_log_std","alpha"]
need_cen=["p_act4","motors_mean","bodies_mean","top_topology_share"]
m=[k for k in need_row if k not in last]
m+=["census."+k for k in need_cen if k not in last.get("census",{})]
ev=[x for x in r if "eval" in x]
if ev:
    m+=["race" ] if "race" not in ev[-1] else []
    m+=["dodge_pooled"] if "dodge_pooled" not in ev[-1] else []
print(",".join(m))
PY
)
  if [ -n "$miss" ]; then echo "MISSING FIELDS:  $c -> $miss"; fail=1
  else echo "OK: $c pid $pid, $n epochs, all instruments present"; fi
done
exit $fail
