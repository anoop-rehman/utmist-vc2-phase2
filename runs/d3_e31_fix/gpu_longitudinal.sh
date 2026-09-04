#!/bin/bash
# D3 M3 E3.1: log per-arm GPU PEAK against bodies_mean as bodies grow toward the
# 29-body ceiling, so the memory question is answered with a curve instead of an
# extrapolation from a range too narrow to fit.
#
# The first attempt fitted MiB against bodies_mean across BOTH arms and produced
# a slope of 2085 MiB/body with a -31,896 intercept -- an artifact of fitting
# the between-arm difference (2694 MiB at equal body size, caused by each
# process's own caching-allocator high-water mark) rather than any body effect.
# Within-arm over 17.4-18.9 bodies there was no detectable growth at all.
#
# So: sample each arm's peak over a full 2-minute window (long enough to catch
# an update phase), record it against that arm's CURRENT bodies_mean, and let
# the curve accumulate as bodies rise 18 -> 29.
set -uo pipefail
D=/workspace/utmist-vc2-phase2/runs/d3_e31_fix
OUT="$D/census/gpu_peak_vs_bodies.csv"
[ -f "$OUT" ] || echo "ts,cfg,epoch,bodies_mean,n_bodies,peak_mib,total_peak_mib" > "$OUT"
while true; do
  declare -A pk; tot=0
  for i in $(seq 1 60); do
    t=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    [ "$t" -gt "$tot" ] && tot=$t
    while IFS=', ' read pid mib; do
      for s in 1 2 3; do
        p=$(pgrep -f "cfg rtg_e31_s$s " | head -1)
        [ -n "$p" ] && [ "$pid" = "$p" ] && { cur=${pk[$s]:-0}; [ "$mib" -gt "$cur" ] && pk[$s]=$mib; }
      done
    done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
    sleep 2
  done
  ts=$(date +%s)
  for s in 1 2 3; do
    [ -z "${pk[$s]:-}" ] && continue
    f="$D/census/rtg_e31_s${s}_morph.csv"; [ -f "$f" ] || continue
    row=$(tail -1 "$f")
    echo "$ts,rtg_e31_s$s,$(echo "$row"|cut -d, -f1),$(echo "$row"|cut -d, -f13),$(echo "$row"|cut -d, -f5),${pk[$s]},$tot" >> "$OUT"
  done
  unset pk
done
