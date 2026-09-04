"""D3 M3 E3: the headline table and the two cross-arm statistics.

Sources ONLY the post-hoc JSONs in `runs/d3_e3_adversarial/posthoc/` and the
per-epoch JSONL each trainer writes -- never a wandb series and never a
training log. `D3_E1_ANT.md` 13-17 records what reading two different
statistics side by side nearly cost, and `D3_E2_RTG.md` 5 records the key
rename that followed.

Two statistics are computed across arms, both of them E2's:

  r(fall rate, return)        E2: +0.989   E2.1: -0.517
  r(forward progress, return) E2: +0.019   E2.1: +0.947

E2's finding was that return measured FALLING; E2.1's `d2rep` inverted it so
that return measured competence. Recomputed here over E3's own arm rows plus
the idle floor, on exactly the same seven-row shape, so the three are
comparable. If E3's has drifted back toward E2's structure, the fall-dodge has
reopened -- and with the design stages live, morphology is the channel.

    .venv-gpu/bin/python .../t2a_port/e3_analyse.py --epoch 400
"""
import argparse
import glob
import json
import os

import numpy as np

POST = "/workspace/utmist-vc2-phase2/runs/d3_e3_adversarial/posthoc"


def corr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def row(d, protocol):
    r = d["results"][protocol]
    m = d.get("morphology", {}).get("mean_action", {}) or {}
    c = d.get("morphology", {}).get("census", {}) or {}
    return dict(
        arm=d["cfg"], design_on=d["design_on"], epoch=d["epoch"],
        R=r["R_mean"], sd=r["R_sd"], goal=r["goal_rate"], lost=r["loss_rate"],
        fell=r["fall_rate"], ep_len=r["ep_len"], fwd=r["max_fwd"],
        frac=r["frac_of_goal"], net_dx=r["net_dx"], speed=r["speed"],
        nop=r["net_over_path"], nb=r["bodies_exec"],
        designfail=r["design_fail_rate"], std=d["action_std"],
        n_changed=len(d.get("changed", [])),
        n_bodies_ma=m.get("n_bodies"), n_motors_ma=m.get("model_nu_ours"),
        mass=m.get("model_mass_ours"),
        limb_len=(m.get("limb_length") or {}).get("mean"),
        limb_sum=(m.get("limb_length") or {}).get("sum"),
        gear=(m.get("gear") or {}).get("mean"),
        distinct=c.get("distinct_topologies"),
        top_share=c.get("top_topology_share"),
        r_fall=(r.get("dodge") or {}).get("r_fall_return"),
        r_fwd=(r.get("dodge") or {}).get("r_fwd_return"),
        premium=(r.get("dodge") or {}).get("fall_premium"),
        episodes=r.get("episodes", []))


def fmt(v, f="{:.2f}", na="--"):
    return na if v is None else f.format(v)


def series(cfg):
    """The per-epoch JSONL each trainer writes, joined on epoch.

    This is what pre-registration reading A's SECOND trigger needs -- "the
    morphology series correlates with fall rate" -- and it has to be computed
    from the same rows for both, so the join happens here rather than by eye
    over two wandb panels. Evaluations run every `--eval-every` epochs while
    morphology runs every epoch, so only the epochs carrying both are used.
    """
    f = f"/workspace/Transform2Act/results/{cfg}/e3_epochs.jsonl"
    if not os.path.exists(f):
        return None
    rows = []
    for line in open(f):
        try:
            d = json.loads(line)
        except Exception:
            continue
        if "eval" not in d or not d.get("mean_action_design"):
            continue
        m = d["mean_action_design"]
        rows.append(dict(
            epoch=d["epoch"], alpha=d.get("alpha"),
            fall=d["eval"]["fall_rate"], goal=d["eval"]["goal_rate"],
            fwd=d["eval"]["max_fwd"], R=d["eval"]["R_mean"],
            designfail=d["eval"].get("design_fail_rate", 0.0),
            n_bodies=m["n_bodies"], n_motors=m["model_nu_ours"],
            mass=m["model_mass_ours"], limb_len=m["limb_length"]["mean"],
            limb_sum=m["limb_length"]["sum"], gear=m["gear"]["mean"],
            distinct=d["census"]["distinct_topologies"],
            top_share=d["census"]["top_topology_share"]))
    return rows


def morph_vs_fall(cfgs):
    """Pre-registration reading A, trigger 2, computed rather than eyeballed."""
    print("\n=== the per-epoch morphology series against the fall rate ===")
    print("  Reading A trigger 2 (D3_E3_ADVERSARIAL.md 3b): if a morphology "
          "column moves\n  monotonically with the fall rate over the run, the "
          "fall rate is being bought\n  with the body.")
    cols = ["n_bodies", "n_motors", "mass", "limb_len", "limb_sum", "gear",
            "distinct", "top_share"]
    print(f"\n  {'arm':<14}{'n':>4}" + "".join(f"{c:>11}" for c in cols)
          + f"{'fall@end':>10}{'goal@end':>10}")
    for cfg in cfgs:
        rows = series(cfg)
        if rows is None:
            print(f"  {cfg:<14}  (no e3_epochs.jsonl)")
            continue
        if not rows:
            print(f"  {cfg:<14}  (no epoch carries BOTH an evaluation and a "
                  f"morphology summary yet)")
            continue
        fall = [r["fall"] for r in rows]
        cells = "".join(f"{fmt(corr([r[c] for r in rows], fall), '{:+.3f}'):>11}"
                        for c in cols)
        print(f"  {cfg:<14}{len(rows):>4}{cells}"
              f"{rows[-1]['fall']:>10.2f}{rows[-1]['goal']:>10.2f}")
    print("\n  r(morphology column, fall rate) across epochs, one row per arm."
          "\n  'distinct'/'top_share' are the sampled census, not the "
          "mean-action design.")

    print("\n=== the fall-dodge pair, as the run proceeded ===")
    print(f"  {'arm':<14}{'epochs':>8}{'r(fall,R)':>12}{'r(fwd,R)':>11}"
          f"{'first goal>0':>14}{'max designfail':>16}")
    for cfg in cfgs:
        rows = series(cfg)
        if not rows:
            continue
        fall = [r["fall"] for r in rows]
        fwd = [r["fwd"] for r in rows]
        R = [r["R"] for r in rows]
        first = next((r["epoch"] for r in rows if r["goal"] > 0), None)
        print(f"  {cfg:<14}{len(rows):>8}"
              f"{fmt(corr(fall, R), '{:+.3f}'):>12}"
              f"{fmt(corr(fwd, R), '{:+.3f}'):>11}"
              f"{(str(first) if first is not None else 'never'):>14}"
              f"{max(r['designfail'] for r in rows):>16.2f}")
    print("  (epoch-level, over the whole run: E2 measured +0.989/+0.019 "
          "across arms, E2.1 -0.517/+0.947)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epoch", default="400")
    p.add_argument("--post", default=POST)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    files = sorted(glob.glob(os.path.join(a.post, f"*_e{int(a.epoch):04d}.json")))
    idle = os.path.join(a.post, "idle.json")
    ds = [json.load(open(f)) for f in files]
    if os.path.exists(idle):
        ds.append(json.load(open(idle)))

    for protocol in ("mean_action", "stochastic"):
        rows = [row(d, protocol) for d in ds]
        rows.sort(key=lambda r: (-r["goal"], -r["fwd"]))
        print(f"\n=== E3, {protocol}, epoch {a.epoch}, 20 episodes per arm ===")
        print(f"{'arm':<14}{'design':<8}{'R':>10}{'sd':>8}{'goal':>6}"
              f"{'lost':>6}{'fell':>6}{'len':>7}{'fwd m':>8}{'of 5m':>7}"
              f"{'speed':>7}{'nb':>6}{'motors':>7}{'mass':>7}{'limb':>7}"
              f"{'gear':>7}{'astd':>7}")
        for r in rows:
            print(f"{r['arm']:<14}"
                  f"{('LIVE' if r['design_on'] else 'frozen'):<8}"
                  f"{r['R']:>10.1f}{r['sd']:>8.1f}{r['goal']:>6.2f}"
                  f"{r['lost']:>6.2f}{r['fell']:>6.2f}{r['ep_len']:>7.1f}"
                  f"{r['fwd']:>8.2f}{100 * r['frac']:>6.1f}%{r['speed']:>7.3f}"
                  f"{r['nb']:>6.1f}{fmt(r['n_motors_ma'], '{:.0f}'):>7}"
                  f"{fmt(r['mass'], '{:.3f}'):>7}"
                  f"{fmt(r['limb_len'], '{:.3f}'):>7}"
                  f"{fmt(r['gear'], '{:.0f}'):>7}{r['std']:>7.3f}")

        # E2's seven-row statistic, recomputed on E3's rows
        fr = [r["fell"] for r in rows]
        R = [r["R"] for r in rows]
        fw = [r["fwd"] for r in rows]
        print(f"\n  across-arm  r(fall rate, return) = {fmt(corr(fr, R), '{:+.3f}')}"
              f"   (E2 +0.989, E2.1 -0.517)")
        print(f"  across-arm  r(forward, return)   = {fmt(corr(fw, R), '{:+.3f}')}"
              f"   (E2 +0.019, E2.1 +0.947)")

        print("\n  per-arm, episode level (20 episodes each):")
        print(f"  {'arm':<14}{'n':>4}{'r(fell,R)':>12}{'r(fwd,R)':>11}"
              f"{'premium':>10}{'designfail':>12}{'topos':>7}{'top%':>7}")
        for r in rows:
            print(f"  {r['arm']:<14}{len(r['episodes']):>4}"
                  f"{fmt(r['r_fall'], '{:+.3f}'):>12}"
                  f"{fmt(r['r_fwd'], '{:+.3f}'):>11}"
                  f"{fmt(r['premium'], '{:+.1f}'):>10}"
                  f"{r['designfail']:>12.2f}"
                  f"{fmt(r['distinct'], '{:.0f}'):>7}"
                  f"{fmt(r['top_share'], '{:.1%}'):>7}")

    # design-on arms: did the design stage change the body under the trained
    # policy? The mirror of E2's "134 arrays identical".
    print("\n=== what the design stages did, under each arm's OWN trained "
          "policy, 20 episodes ===")
    for d in ds:
        tag = ("LIVE" if d["design_on"] else "frozen")
        print(f"  {d['cfg']:<14} {tag:<7} "
              f"arrays changed {len(d.get('changed', [])):>3} of "
              f"{d.get('n_arrays', 0)}   body counts "
              f"{sorted(set(d.get('body_counts', [])))}   "
              f"{d.get('distinct_topologies', 0)} distinct topologies "
              f"(top {100 * d.get('top_topology_share', 0):.0f}%)")

    morph_vs_fall([d["cfg"] for d in ds if d.get("design_on") is not None
                   and d["arm"] != "idle"])

    if a.out:
        json.dump(dict(rows=[row(d, "mean_action") for d in ds],
                       series={d["cfg"]: series(d["cfg"]) for d in ds
                               if d["arm"] != "idle"}),
                  open(a.out, "w"), indent=1, default=float)
        print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
