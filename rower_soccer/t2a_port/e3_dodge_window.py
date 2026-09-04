"""D3 M3 E3: the fall-dodge correlation while it is INFORMATIVE, and whether
the falls are a dodge or a locomotion failure.

Until the controls started moving, `r(fall, R)` was degenerate on them -- fall
rate pinned at 0.00 or 1.00, no variance, nothing to correlate. It now has
variance for the first time (0.20 / 0.30), which makes Readings A and B live.
The window is narrow: once the goal rate saturates the correlations go
degenerate again in the other direction, exactly as they did for `d2rep` in
E2.1. So this is the interval where the statistic carries the most information.

DODGE OR LOCOMOTION FAILURE? A fall rate column cannot tell them apart and the
two mean opposite things. The scripted opponent's x is a function of the step
index alone -- `opp_x(k) = 1.0 - v*dt*k` (`D3_E2_RTG.md` 1) -- so for any
fallen episode the opponent's position AT THE FALL is known exactly, and the
gap between the two bodies at that moment separates:

  * **dodge**      -- the agent goes down as the opponent arrives, gap small;
  * **locomotion** -- the agent goes down under its own control mid-run, with
                      the opponent still far away.

    python3 .../t2a_port/e3_dodge_window.py --cfgs rtg_e3c_s1,rtg_e3c_s2
"""
import argparse
import json
import math

V, DT = 0.68, 0.015
CONTACT = 0.75          # torso-to-torso gap counted as "in contact"; the ant's
                        # own body is ~0.5 m across, so anything under this is
                        # the two creatures meeting.


def corr(x, y):
    n = len(x)
    if n < 3:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    if sx == 0 or sy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / (sx * sy)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cfgs", default="rtg_e3c_s1,rtg_e3c_s2")
    p.add_argument("--results", default="/workspace/Transform2Act/results")
    p.add_argument("--out", default=None)
    a = p.parse_args()
    out = {}
    for cfg in a.cfgs.split(","):
        rows = []
        for line in open(f"{a.results}/{cfg}/e3_epochs.jsonl"):
            try:
                d = json.loads(line)
            except Exception:
                continue
            if d.get("eval_episodes"):
                rows.append(d)
        print(f"\n=== {cfg} ===")
        print(f"  {'epoch':>6}{'fall':>6}{'goal':>6}{'fwd m':>8}"
              f"{'r(fall,R)':>11}{'r(fwd,R)':>10}   falls: step / our x / opp x / gap")
        series = []
        for d in rows:
            eps = d["eval_episodes"]
            fell = [float(e["fell"]) for e in eps]
            R = [e["R"] for e in eps]
            fwd = [e["max_fwd"] for e in eps]
            rf, rw = corr(fell, R), corr(fwd, R)
            fr = sum(fell) / len(fell)
            if fr == 0:
                continue
            det = []
            for e in eps:
                if not e["fell"]:
                    continue
                k = e["n"]
                ourx = e["x0"] + e["net_dx"]
                oppx = 1.0 - V * DT * k
                det.append((k, ourx, oppx, oppx - ourx))
            series.append(dict(epoch=d["epoch"], fall_rate=fr,
                               goal=d["eval"]["goal_rate"],
                               fwd=d["eval"]["max_fwd"],
                               r_fall=rf, r_fwd=rw,
                               falls=[dict(step=k, our_x=ox, opp_x=px, gap=g)
                                      for k, ox, px, g in det]))
            ds = "  ".join(f"{k}/{ox:+.2f}/{px:+.2f}/{g:+.2f}" for k, ox, px, g in det[:4])
            print(f"  {d['epoch']:>6}{fr:>6.2f}{d['eval']['goal_rate']:>6.2f}"
                  f"{d['eval']['max_fwd']:>8.2f}"
                  f"{(f'{rf:+.3f}' if rf is not None else '  --'):>11}"
                  f"{(f'{rw:+.3f}' if rw is not None else '  --'):>10}   {ds}")
        out[cfg] = series
        allf = [f for s in series for f in s["falls"]]
        if allf:
            near = [f for f in allf if f["gap"] < CONTACT]
            print(f"\n  {len(allf)} falls total. At the opponent (gap < "
                  f"{CONTACT} m): {len(near)} = {100*len(near)/len(allf):.0f}%")
            print(f"    fall step: min {min(f['step'] for f in allf)} "
                  f"median {sorted(f['step'] for f in allf)[len(allf)//2]} "
                  f"max {max(f['step'] for f in allf)}   "
                  f"(the opponent scores at step 491)")
            print(f"    gap at the fall: min {min(f['gap'] for f in allf):+.2f} "
                  f"median {sorted(f['gap'] for f in allf)[len(allf)//2]:+.2f} "
                  f"max {max(f['gap'] for f in allf):+.2f} m")
            print(f"    -> {'DODGE-shaped: falls cluster where the opponent is' if len(near) > len(allf)/2 else 'LOCOMOTION-shaped: falls happen with the opponent still far away'}")
    print(f"\n  reference: E2 r(fall,R) = +0.989 / r(fwd,R) = +0.019 "
          f"(return measured falling)")
    print(f"             E2.1 d2rep    = -0.94  / +0.95            "
          f"(return measured competence)")
    if a.out:
        json.dump(out, open(a.out, "w"), indent=1)
        print(f"\n  -> {a.out}")


if __name__ == "__main__":
    main()
