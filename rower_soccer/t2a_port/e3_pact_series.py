"""D3 M3 E3: p_act4 against epoch, as a SERIES, with what the points can and
cannot support stated beside it.

`census/population.csv` accumulates one row per (arm, checkpoint) probe. The
trajectory is the finding; single points are not, and four points drawn from
different seeds are not a trajectory at all. This prints the series per seed,
separates within-seed evidence from cross-seed evidence, and refuses to name a
functional form the data cannot distinguish.

    python3 .../t2a_port/e3_pact_series.py
"""
import csv
import os
import sys

CSV = "/workspace/utmist-vc2-phase2/runs/d3_e3_adversarial/census/population.csv"


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else CSV
    if not os.path.exists(path):
        print(f"no {path}")
        return
    rows = [r for r in csv.DictReader(open(path)) if r.get("p_act4")]
    for r in rows:
        r["epoch"] = int(r["ckpt_epoch"]) if r["ckpt_epoch"] else -1
        r["p_act4"] = float(r["p_act4"])
        r["p_act1"] = float(r["p_act1"])
        r["motors_mean"] = float(r["pop_motors_mean"])
        r["readout"] = int(r["readout_n_motors"]) if r["readout_n_motors"] else None

    by = {}
    for r in rows:
        by.setdefault(r["cfg"], []).append(r)
    for v in by.values():
        v.sort(key=lambda r: r["epoch"])

    print("\n=== p_act4 (fraction of 200 sampled designs with >= 4 motors) "
          "against epoch ===")
    print(f"  {'arm':<12}{'epoch':>7}{'ckpt':>12}{'readout':>9}"
          f"{'motors_mean':>13}{'p_act1':>9}{'p_act4':>9}")
    for cfg in sorted(by):
        for r in by[cfg]:
            ep = "untrained" if r["epoch"] < 0 else str(r["epoch"])
            print(f"  {cfg:<12}{ep:>7}{r['ckpt']:>12}"
                  f"{str(r['readout']):>9}{r['motors_mean']:>13.2f}"
                  f"{r['p_act1']:>9.3f}{r['p_act4']:>9.3f}")

    print("\n=== what these points can and cannot support ===")
    within = {c: v for c, v in by.items() if len(v) >= 2}
    if not within:
        print("  No arm has two probes yet, so there is NO within-seed series.")
        print("  Any comparison across arms is confounded with epoch and must")
        print("  not be read as a trend.")
    for cfg, v in within.items():
        a, b = v[0], v[-1]
        span = b["epoch"] - a["epoch"]
        print(f"  {cfg}: WITHIN-SEED, epoch {a['epoch']} -> {b['epoch']} "
              f"({span} epochs): p_act4 {a['p_act4']:.3f} -> {b['p_act4']:.3f} "
              f"({100*(b['p_act4']-a['p_act4'])/a['p_act4']:+.0f}%), "
              f"motors_mean {a['motors_mean']:.2f} -> {b['motors_mean']:.2f}")
        if len(v) < 4:
            print(f"    {len(v)} points on this seed. A drop is established; "
                  f"its SHAPE is not -- two or three points cannot "
                  f"distinguish decelerating from linear from accelerating, "
                  f"and no functional form is claimed here.")
        else:
            print(f"    {len(v)} points; shape may be assessable -- state the "
                  f"successive differences rather than fitting a curve:")
            d = [(v[i+1]['epoch'], v[i+1]['p_act4']-v[i]['p_act4'],
                  v[i+1]['epoch']-v[i]['epoch']) for i in range(len(v)-1)]
            for e, dp, de in d:
                print(f"      -> epoch {e}: {dp:+.3f} over {de} epochs "
                      f"({dp/de:+.4f}/epoch)")
    cross = [c for c, v in by.items() if len(v) == 1]
    if cross:
        print(f"  {len(cross)} arm(s) have a single probe ({', '.join(cross)}) "
              f"-- those points are CROSS-SEED and confounded with epoch.")
    print("\n  No extrapolation is performed. This project's rule is to run a "
          "curve out\n  rather than project it, and that applies here.")


if __name__ == "__main__":
    main()
