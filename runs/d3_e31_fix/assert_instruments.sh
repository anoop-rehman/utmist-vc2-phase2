#!/bin/bash
# D3 M3 E3.1: assert that every quantity the pre-registered falsifiers depend on
# is actually being COLLECTED for the given cfgs, and fail loudly otherwise.
#
# WHY. Twice in this experiment an instrument looked present and was not
# collecting what the decision needed, both times at a transition to new arms:
# once the sigma resolution was 20 epochs against milestones falling between
# checkpoints, and once four watchers kept polling runs that had ended while
# the new arms had no derived artefacts at all. A falsifier that depends on a
# collector nobody checked is not pre-registered in any useful sense.
#
#   ./assert_instruments.sh rtg_e31_s1 rtg_e31_s2 rtg_e31_s3   [timeout_s]
set -uo pipefail
D=/workspace/utmist-vc2-phase2/runs/d3_e31_fix
CFGS=(); T=900
for a in "$@"; do case "$a" in [0-9]*) T=$a;; *) CFGS+=("$a");; esac; done
[ ${#CFGS[@]} -eq 0 ] && { echo "usage: $0 <cfg>... [timeout_s]"; exit 2; }
echo "asserting instruments for: ${CFGS[*]}  (timeout ${T}s)"
deadline=$(( $(date +%s) + T )); fail=0
for c in "${CFGS[@]}"; do
  jl="/workspace/Transform2Act/results/$c/e3_epochs.jsonl"
  csv="$D/census/${c}_morph.csv"
  while [ "$(date +%s)" -lt "$deadline" ]; do
    [ -s "$jl" ] && [ -s "$csv" ] && break
    sleep 15
  done
  msg=$(/workspace/utmist-vc2-phase2/.venv/bin/python - "$jl" "$csv" "$c" <<'PY'
import csv, json, os, sys
jl, cf, cfg = sys.argv[1], sys.argv[2], sys.argv[3]
bad=[]
if not os.path.exists(jl): bad.append("no e3_epochs.jsonl")
else:
    d=json.loads(open(jl).readline()); c=d.get("census") or {}
    # FALSIFIER 1: control_log_std must be present per epoch
    if d.get("control_log_std") is None: bad.append("control_log_std MISSING from the epoch row")
    # FALSIFIER 2: p_act4 must be present per epoch
    if c.get("p_act4") is None: bad.append("census.p_act4 MISSING -- the PRIMARY falsifier has no collector")
    if c.get("motors_mean") is None: bad.append("census.motors_mean missing")
if not os.path.exists(cf): bad.append("no <cfg>_morph.csv -- nothing is distilling the jsonl")
else:
    r=list(csv.DictReader(open(cf)))
    if not r: bad.append("<cfg>_morph.csv has a header but no rows")
    else:
        for k in ("control_log_std","p_act4"):
            if not r[-1].get(k): bad.append(f"CSV column {k} empty on the latest row")
print(("FAIL: " + "; ".join(bad)) if bad else "ok")
PY
)
  if [ "$msg" = "ok" ]; then echo "  [OK  ] $c: control_log_std and p_act4 both collecting (jsonl + csv)"
  else echo "  [FAIL] $c: $msg"; fail=1; fi
done
if [ "$fail" -ne 0 ]; then
  echo "INSTRUMENT ASSERTION FAILED -- the pre-registered falsifiers have no collector."
  exit 1
fi
echo "all instruments confirmed collecting."
