"""D3 M3 E3.1: project GPU memory against body size, to the KNOWN ceiling.

Body growth is bounded at 29 (`D3_E31_FIX.md`, "The growth ceiling"), so this
projection terminates rather than running open-ended. The question is only
whether the surviving arms fit at the worst case.

Pairs per-client GPU MiB with that arm's `bodies_mean` at the time, takes the
PEAK per (arm, bodies_mean) bin -- the update phase is what OOMs, not the
sampling trough -- and fits MiB = a + b*bodies_mean. Memory should be roughly
linear in body count because the PPO update holds a fixed 50,000 states whose
graphs have `bodies_mean` nodes each.
"""
import csv, sys
from collections import defaultdict

CSV = "/workspace/utmist-vc2-phase2/runs/d3_e31_fix/census/gpu_vs_bodies.csv"
TOTAL, CEILING = 20475, 29.0

rows = [r for r in csv.DictReader(open(sys.argv[1] if len(sys.argv) > 1 else CSV))
        if r.get("bodies_mean") and r.get("mib")]
peak = defaultdict(float)
for r in rows:
    try:
        b = round(float(r["bodies_mean"]), 1); m = float(r["mib"])
    except ValueError:
        continue
    peak[(r["cfg"], b)] = max(peak[(r["cfg"], b)], m)

print(f"peak per-client MiB by bodies_mean ({len(rows)} samples):")
pts = []
for (cfg, b), m in sorted(peak.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    print(f"  {cfg}  bodies_mean {b:5.1f}  peak {m:8.0f} MiB")
    pts.append((b, m))
if len(pts) < 2:
    print("\n  not enough distinct points to fit yet"); sys.exit(0)

n = len(pts)
mx = sum(p[0] for p in pts)/n; my = sum(p[1] for p in pts)/n
den = sum((p[0]-mx)**2 for p in pts)
if den == 0:
    print("\n  all points at one body size; cannot fit a slope yet"); sys.exit(0)
b1 = sum((p[0]-mx)*(p[1]-my) for p in pts)/den
b0 = my - b1*mx
print(f"\n  fit: MiB_per_arm = {b0:.0f} + {b1:.0f} * bodies_mean   (n={n} points)")
print(f"\n  {'bodies_mean':>12}{'per arm':>10}{'2 arms':>10}{'of 20475':>10}  verdict")
for b in (18.5, 20, 22, 24, 26, CEILING):
    per = b0 + b1*b; two = 2*per
    print(f"  {b:>12.1f}{per:>10.0f}{two:>10.0f}{100*two/TOTAL:>9.0f}%  "
          f"{'FITS' if two < 0.92*TOTAL else 'TIGHT' if two < TOTAL else 'DOES NOT FIT'}")
lim = (0.92*TOTAL/2 - b0)/b1 if b1 else float('inf')
print(f"\n  two arms reach 92% of the card at bodies_mean = {lim:.1f}"
      f"   (ceiling is {CEILING:.0f})")
print(f"  -> {'the ceiling is reached FIRST: two arms fit for the whole run' if lim >= CEILING else 'the card binds BEFORE the ceiling: mitigation needed'}")
