"""D3 M3 E3.1: what `control_log_std` actually DID, read straight from the
checkpoints.

`control_log_std` is a LEARNED parameter, so `log_std_crit` = -0.8837
(`D3_E3_ADVERSARIAL.md` 3f) is a **basin boundary, not a precision
requirement**: below it the gradient is self-reinforcing (lower sigma -> lower
cost -> higher return -> lower sigma), above it the same gradient runs the
other way, toward deleting actuators instead. Which side you initialise on is
what matters, and that is a far more robust property than a knife-edge.

But it makes one thing load-bearing that was unmonitored: **sigma must actually
go down**. This reads it out of every checkpoint on disk, for any arm, so the
claim rests on the weights rather than on an argument.

    CUDA_VISIBLE_DEVICES= python3 .../t2a_port/e3_logstd_trace.py \\
        --cfgs rtg_e3_s1,rtg_e3_s2,rtg_e3_s3
"""
import argparse
import glob
import json
import math
import os
import pickle
import re
import sys

CRIT = -0.8837          # log_std_crit, D3_E3_ADVERSARIAL.md 3f


def read(path):
    try:
        d = pickle.load(open(path, "rb"))
    except Exception:
        return None
    pd = d.get("policy_dict", {})
    out = {"epoch": d.get("epoch"), "file": os.path.basename(path)}
    for k in ("control_action_log_std", "attr_action_log_std"):
        if k in pd:
            v = pd[k]
            try:
                out[k] = float(v.mean().item())
            except Exception:
                out[k] = float(v.mean())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfgs", default="rtg_e3_s1,rtg_e3_s2,rtg_e3_s3")
    p.add_argument("--results", default="/workspace/Transform2Act/results")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    allrows = {}
    print(f"  {'arm':<14}{'checkpoint':<16}{'epoch':>6}"
          f"{'control_log_std':>17}{'sigma':>8}{'cost/step':>11}"
          f"{'  vs crit -0.8837':>18}")
    for cfg in a.cfgs.split(","):
        rows = []
        for f in sorted(glob.glob(f"{a.results}/{cfg}/models/*.p")):
            r = read(f)
            if r and "control_action_log_std" in r:
                rows.append(r)
        rows.sort(key=lambda r: (r["epoch"] if r["epoch"] is not None else -1))
        for r in rows:
            ls = r["control_action_log_std"]
            sig = math.exp(ls)
            cost = 0.5 * 8 * sig * sig
            side = "BELOW (good)" if ls < CRIT else "ABOVE (bad)"
            print(f"  {cfg:<14}{r['file']:<16}{str(r['epoch']):>6}"
                  f"{ls:>17.4f}{sig:>8.4f}{cost:>11.4f}{side:>18}")
        allrows[cfg] = rows
        if len(rows) >= 2:
            a0, a1 = rows[0], rows[-1]
            de = (a1["epoch"] or 0) - (a0["epoch"] or 0)
            dl = a1["control_action_log_std"] - a0["control_action_log_std"]
            rate = dl / de if de else float("nan")
            need = (CRIT - a1["control_action_log_std"]) / rate if rate else float("inf")
            print(f"  {'':<14}-> moved {dl:+.4f} over {de} epochs "
                  f"({rate:+.5f}/epoch); at that rate reaching the boundary "
                  f"takes {need:.0f} more epochs")
    if a.out:
        json.dump(allrows, open(a.out, "w"), indent=1)
        print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
