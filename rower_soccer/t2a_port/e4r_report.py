"""D3 E4B: assemble the epoch-200 (and final) report from the run's own data.

One command, so the numbers are produced the same way every time and the
pre-registered criteria are evaluated as written rather than by eye.

Everything here is read from `e4r_epochs.jsonl`; the tournament is a separate
job (`e4r_tournament.py`) because it costs compute.
"""
import argparse, json, os
import numpy as np

RES = "/workspace/Transform2Act/results"
SEED_MATCHED = {"rtg_e4r_s1": 1.579, "rtg_e4r_s2": 3.883, "rtg_e4r_s3": 3.883}
SEED_PUBLISHED = {"rtg_e4r_s1": 4.224, "rtg_e4r_s2": 4.891, "rtg_e4r_s3": 4.891}


def load(cfg):
    p = f"{RES}/{cfg}/e4r_epochs.jsonl"
    if not os.path.exists(p):
        return []
    out = []
    for l in open(p):
        try:
            out.append(json.loads(l))
        except Exception:
            pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", default="rtg_e4r_s1,rtg_e4r_s2,rtg_e4r_s3")
    ap.add_argument("--lo", type=int, default=200)
    ap.add_argument("--hi", type=int, default=400)
    a = ap.parse_args()
    for cfg in [c for c in a.cfgs.split(",") if c.strip()]:
        rows = load(cfg)
        if not rows:
            print("%s: no data\n" % cfg)
            continue
        ev = [(r["epoch"], r["eval"]) for r in rows if "eval" in r]
        print("=" * 72)
        print("%s  -- %d epochs, %d evals" % (cfg, rows[-1]["epoch"] + 1, len(ev)))

        # --- speed against the RIGHT baseline ---------------------------
        last = [v["speed"] for e, v in ev][-5:]
        print("  speed last-5 mean %.3f | matched seed %.3f (%+.0f%%) | "
              "published %.3f (%+.0f%%)"
              % (np.mean(last), SEED_MATCHED.get(cfg, np.nan),
                 100 * (np.mean(last) / SEED_MATCHED[cfg] - 1),
                 SEED_PUBLISHED.get(cfg, np.nan),
                 100 * (np.mean(last) / SEED_PUBLISHED[cfg] - 1)))

        # --- criterion 1: RATCHET, gap-restricted ------------------------
        pts = [(r["epoch"], x["age_gap"], x["win_rate"])
               for r in rows for x in (r.get("ladder") or {}).get("rows", [])]
        win = [w for ep, g, w in pts if a.lo <= ep <= a.hi and g >= 100]
        if win:
            m, n = float(np.mean(win)), len(win)
            se = float(np.std(win, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
            print("  RATCHET  win vs selves >=100 epochs older, epochs %d-%d:"
                  % (a.lo, a.hi))
            print("           %.3f +- %.3f (n=%d)  threshold 0.75 -> %s"
                  % (m, se, n, "MET" if m >= 0.75 else "not met"))
        else:
            print("  RATCHET  no qualifying pairs yet (need age gap >=100 in "
                  "epochs %d-%d)" % (a.lo, a.hi))

        # --- criterion 3: degenerate mirror ------------------------------
        mm = [r["mirror"] for r in rows
              if "mirror" in r and a.lo <= r["epoch"] <= a.hi and r["mirror"]]
        if mm:
            st = float(np.mean([x.get("stalemate_rate", 0) for x in mm]))
            fw = float(np.mean([x.get("fwd_mean", 0) for x in mm]))
            mu = float(np.mean([x.get("mutual_rate", 0) for x in mm]))
            bad = st > 0.5 or fw < 2.5
            print("  MIRROR   stalemate %.3f (fail >0.5) | forward %.2f m "
                  "(fail <2.5) | mutual %.3f -> %s"
                  % (st, fw, mu, "DEGENERATE" if bad else "healthy"))
        else:
            print("  MIRROR   no evals in the window yet")

        # --- context: gap-binned win rate --------------------------------
        print("  win rate by age gap (all epochs):", end="")
        for lo, hi in ((0, 20), (20, 50), (50, 100), (100, 1000)):
            w = [w for ep, g, w in pts if lo <= g < hi]
            if w:
                print("  %d-%d:%.2f(n=%d)" % (lo, hi, np.mean(w), len(w)), end="")
        print()
        print()
    print("Tournament (cyclic triples, signed slot bias) is a separate job:")
    print("  e4r_tournament.py --cfg <cfg> --n-ckpt 12 --episodes 20")


if __name__ == "__main__":
    main()
