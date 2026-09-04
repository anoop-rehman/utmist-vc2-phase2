"""D3 M3 E3: is the GNN's slower sigma decay a real architectural difference,
or an artifact of comparing two runs at different points on a shared curve?

The decay rate is strongly sigma-dependent -- the MLP's own series runs
-0.0231/epoch early and -0.0043/epoch late, a 5x change inside one run. So
comparing "MLP rate at epoch 40" against "GNN rate at epoch 40" compares two
different positions on that curve and proves nothing about architecture.

The sharper comparison is **rate at matched sigma**. This computes the MLP's
LOCAL decay rate at each log_std the GNN arms actually reach, from the MLP's
stored per-epoch series rather than from any interpolation, and puts the two
side by side.

If the gap survives, "the GNN reduces sigma more slowly" is a statement about
the architectures rather than about where each happened to be. If it does not,
the gap was position-on-the-curve all along and the doc's framing needs
softening -- which is the more useful outcome of the two and is reported the
same way.

    python3 .../t2a_port/e3_matched_sigma.py
"""
import json
import sys

MLP = ["/workspace/Transform2Act/results/rtg_mlp_s1_d2rep/log.jsonl",
       "/workspace/Transform2Act/results/rtg_mlp_s2_d2rep/log.jsonl"]
# GNN control arms: (log_std, epoch) from archival checkpoints, and the local
# rate over the segment ENDING at that point.
GNN = {
    "ctl_s1": [(-0.0632, 5, None), (-0.1842, 19, -0.00864), (-0.3794, 39, -0.00976)],
    "ctl_s2": [(-0.1456, 15, None), (-0.1788, 19, -0.00830), (-0.3791, 39, -0.01002)],
}
# E3 design-on arms, two points each
GNN_DESIGN = {
    "e3_s1": [(-0.0298, 3, None), (-0.1215, 18, -0.00611)],
    "e3_s2": [(-0.0152, 1, None), (-0.1096, 21, -0.00472)],
}


def series(path):
    out = []
    for line in open(path):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "log_std" in d and "epoch" in d:
            out.append((int(d["epoch"]), float(d["log_std"])))
    out.sort()
    return out


def local_rate(s, epoch, half=5):
    """d(log_std)/d(epoch) around `epoch`, central difference over 2*half.

    Near epoch 0 a centred window runs off the start of the series, so it
    shrinks to the widest symmetric window available and finally to a forward
    difference. That matters here: the MLP passes the design-on arms' log_std
    values before epoch 5, which is exactly where a fixed +/-5 window fails.
    """
    d = dict(s)
    for h in range(half, 0, -1):
        lo, hi = epoch - h, epoch + h
        if lo in d and hi in d:
            return (d[hi] - d[lo]) / (hi - lo)
    for h in range(half, 0, -1):        # forward difference at the boundary
        if epoch in d and epoch + h in d:
            return (d[epoch + h] - d[epoch]) / h
    return None


def epoch_at(s, target):
    """First epoch at which the MLP's log_std has fallen to `target`."""
    for e, ls in s:
        if ls <= target:
            return e
    return None


def main():
    mlps = [(p.split("/")[-2], series(p)) for p in MLP]
    print(f"\n=== MLP local decay rate, from its stored per-epoch series "
          f"(central difference, +/-5 epochs) ===")
    print(f"  {'arm':<18}" + "".join(f"{'ep'+str(e):>11}" for e in
                                     (10, 20, 30, 40, 60, 100, 200, 300)))
    for name, s in mlps:
        print(f"  {name:<18}" + "".join(
            f"{(local_rate(s, e) or float('nan')):>11.5f}"
            for e in (10, 20, 30, 40, 60, 100, 200, 300)))
    print("  (confirms the rate is strongly sigma-dependent inside one run, "
          "so matched-EPOCH is the wrong comparison)")

    print(f"\n=== THE MATCHED-SIGMA COMPARISON ===")
    print(f"  For each log_std the GNN arms actually reach, the MLP's LOCAL "
          f"rate where IT passed that value.")
    print(f"\n  {'GNN arm':<10}{'log_std':>9}{'GNN ep':>8}{'GNN rate':>11}"
          f"{'| MLP ep':>10}{'MLP rate':>11}{'ratio':>9}")
    rows = []
    for label, pts in list(GNN.items()) + list(GNN_DESIGN.items()):
        for ls, ep, rate in pts:
            if rate is None:
                continue
            cells = []
            for name, s in mlps:
                me = epoch_at(s, ls)
                mr = local_rate(s, me) if me is not None else None
                cells.append((name, me, mr))
            name, me, mr = cells[0]
            ratio = (mr / rate) if (mr and rate) else float("nan")
            rows.append((label, ls, ep, rate, me, mr, ratio))
            print(f"  {label:<10}{ls:>9.4f}{ep:>8}{rate:>11.5f}"
                  f"{(me if me is not None else -1):>10}"
                  f"{(mr if mr is not None else float('nan')):>11.5f}"
                  f"{ratio:>9.2f}x")
    ctl = [r for r in rows if r[0].startswith("ctl")]
    des = [r for r in rows if r[0].startswith("e3_")]
    if ctl and [r[6] for r in ctl if r[6] == r[6]]:
        rr = [r[6] for r in ctl if r[6] == r[6]]
        print(f"\n  CONTROL arms (design OFF), matched-sigma ratio: "
              f"{min(rr):.2f}x - {max(rr):.2f}x  (mean {sum(rr)/len(rr):.2f}x)")
    if des and [r[6] for r in des if r[6] == r[6]]:
        rr = [r[6] for r in des if r[6] == r[6]]
        print(f"  DESIGN-ON arms, matched-sigma ratio:              "
              f"{min(rr):.2f}x - {max(rr):.2f}x  (mean {sum(rr)/len(rr):.2f}x)")
    print(f"\n  A ratio near 1.0 would mean the gap was POSITION ON THE CURVE "
          f"and is explained.\n  A ratio holding near the matched-epoch ~2x "
          f"means it is a real architectural difference.")


if __name__ == "__main__":
    main()
