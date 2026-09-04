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

CRIT_ANALYTIC  = -0.8837   # log_std_crit derived in D3_E3_ADVERSARIAL.md 3f-ii
# 3f-iv-c measured the boundary on the simulator and it is STRICTER than the
# analytic one, because the derivation held L_ant at the zero-torque episode
# length and assumed no falls. The empirical value is the operative one.
CRIT = -0.9645             # empirical, D3_E3_ADVERSARIAL.md 3f-iv-c
SURVIVE_CROSS = -0.6931    # where ctrl cost/step first drops below the 1.0
                           # survive bonus (0.5*8*sigma^2 = 1 -> sigma = 0.5)


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
          f"{'  vs -0.9645':>14}")
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
            side = "BELOW (good)" if ls < CRIT else "above"
            print(f"  {cfg:<14}{r['file']:<16}{str(r['epoch']):>6}"
                  f"{ls:>17.4f}{sig:>8.4f}{cost:>11.4f}{side:>14}")
        allrows[cfg] = rows
        if len(rows) >= 2:
            a0, a1 = rows[0], rows[-1]
            de = (a1["epoch"] or 0) - (a0["epoch"] or 0)
            dl = a1["control_action_log_std"] - a0["control_action_log_std"]
            rate = dl / de if de else float("nan")
            ls_now, e_now = a1["control_action_log_std"], a1["epoch"] or 0
            if rate:
                e_sur = e_now + (SURVIVE_CROSS - ls_now) / rate
                e_cri = e_now + (CRIT - ls_now) / rate
                print(f"  {'':<14}-> {dl:+.4f} over {de} epochs "
                      f"({rate:+.5f}/epoch, {len(rows)} points)")
                print(f"  {'':<14}   projected: cost/step < survive bonus at "
                      f"epoch ~{e_sur:.0f}; crosses {CRIT} at epoch ~{e_cri:.0f}"
                      f"; locomotion +18-27 -> ~{e_cri+18:.0f}-{e_cri+27:.0f}")
            # is the decay linear, or slowing as the MLP's did?
            if len(rows) >= 3:
                segs = []
                for i in range(len(rows) - 1):
                    d_e = (rows[i+1]["epoch"] or 0) - (rows[i]["epoch"] or 0)
                    if d_e:
                        segs.append((rows[i]["epoch"], rows[i+1]["epoch"],
                                     (rows[i+1]["control_action_log_std"]
                                      - rows[i]["control_action_log_std"]) / d_e))
                print(f"  {'':<14}   segment rates: " + ", ".join(
                    f"{a}->{b} {r:+.5f}" for a, b, r in segs)
                    + "   (the MLP's SLOWED: -0.0231 over 0-40, "
                      "-0.0043 over 100-399)")
    if a.out:
        json.dump(allrows, open(a.out, "w"), indent=1)
        print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
