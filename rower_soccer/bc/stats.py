"""What is actually in a BC dataset — counts, coverage, and action sanity.

    PYTHONPATH=. .venv/bin/python -m rower_soccer.bc.stats demos/*.demo.npz
    PYTHONPATH=. .venv/bin/python -m rower_soccer.bc.stats runs_v2/bc/ant.npz --mirror

Reads either a built dataset (`.npz` written by `BCDataset.save`) or raw demo
files, in which case it builds the dataset first with the same defaults the CLI
in `dataset.py` uses.

The action block is the part worth reading before a training run. A drill
expert running in MODE_MEAN emits a bounded, structured command; an expert
running in MODE_NOISE emits something close to uniform on [-1, 1]. The
per-actuator mean/std and the saturation fraction tell those apart at a glance,
and the LEFT/RIGHT column pair says whether the corpus is already balanced (it
is not: humans and the scripted chase both favour one turn direction, which is
exactly the imbalance the mirror augmentation removes).
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np

from rower_soccer.bc.dataset import BCDataset, SPLIT_NAMES, build_dataset

__all__ = ["summary", "counts", "action_stats"]


def _table(title, rows, headers):
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
         for i, h in enumerate(headers)]
    out = [title, "  " + "  ".join(str(h).ljust(w[i]) for i, h in enumerate(headers))]
    out.append("  " + "  ".join("-" * w[i] for i in range(len(headers))))
    for r in rows:
        out.append("  " + "  ".join(str(c).ljust(w[i]) for i, c in enumerate(r)))
    return "\n".join(out)


def counts(ds: BCDataset) -> dict:
    """Sample counts sliced every way a training decision needs."""
    a = ds.arrays
    split = a["split"]
    def by(names, col):
        out = {}
        for i, n in enumerate(names):
            m = col == i
            out[n] = dict(total=int(m.sum()),
                          train=int((m & (split == 0)).sum()),
                          val=int((m & (split == 1)).sum()))
        return {k: v for k, v in out.items() if v["total"]}
    per_match = {}
    for d in ds.meta["demos"]:
        m = a["demo"] == d["index"]
        per_match[d["file"]] = dict(match_id=d["match_id"], split=d.get("split"),
                                    samples=int(m.sum()),
                                    ticks=d["n_ticks"], goals=d["n_goals"])
    return dict(
        total=len(ds),
        by_split={SPLIT_NAMES[i]: int((split == i).sum()) for i in (0, 1)},
        by_skill=by(ds.skill_vocab, a["skill"]),
        by_controller=by(ds.controller_vocab, a["controller"]),
        by_team=by(tuple(ds.meta["team_vocab"]), a["team"]),
        by_layout={f'{l["id"]}:{l["skill"]}/{l["obs_dim"]}':
                   int((a["layout"] == l["id"]).sum()) for l in ds.meta["layouts"]},
        by_mirrored=dict(original=int((a["mirrored"] == 0).sum()),
                         mirrored=int((a["mirrored"] == 1).sum())),
        by_match=per_match,
        dropped=ds.meta.get("dropped", {}),
        skipped=ds.meta.get("skipped", []),
    )


def action_stats(ds: BCDataset) -> dict:
    """Per-actuator distribution + the checks that catch a broken corpus."""
    act = np.asarray(ds.arrays["action"], np.float64)
    if act.size == 0:
        return {}
    sat = np.abs(act) >= 0.999
    return dict(
        n=int(act.shape[0]), dim=int(act.shape[1]),
        mean=act.mean(0).round(4).tolist(),
        std=act.std(0).round(4).tolist(),
        min=act.min(0).round(4).tolist(),
        max=act.max(0).round(4).tolist(),
        frac_saturated=sat.mean(0).round(4).tolist(),
        frac_saturated_all=float(sat.mean().round(6)),
        frac_exactly_zero=float((np.abs(act) < 1e-12).all(1).mean().round(6)),
        out_of_range=int((np.abs(act) > 1.0 + 1e-6).sum()),
        nonfinite=int((~np.isfinite(act)).sum()),
        l2_mean=float(np.linalg.norm(act, axis=1).mean().round(4)),
    )


def _expert_coverage(ds: BCDataset) -> dict:
    n = np.asarray(ds.arrays["expert_obs_n"])
    eo = ds.arrays["expert_obs"]
    bad = 0
    for w in np.unique(n):
        if w <= 0:
            continue
        m = n == w
        bad += int((~np.isfinite(eo[m, :int(w)])).any(1).sum())
    z = ds.arrays["z"]
    return dict(with_expert_obs=int((n > 0).sum()),
                without_expert_obs=int((n <= 0).sum()),
                widths=dict(Counter(int(x) for x in n)),
                nonfinite_rows=bad,
                with_z=int(np.isfinite(z).all(1).sum()),
                without_z=int((~np.isfinite(z).all(1)).sum()))


def summary(ds: BCDataset) -> str:
    c = counts(ds)
    a = action_stats(ds)
    cov = _expert_coverage(ds)
    L = []
    L.append(f"BC dataset  {len(ds)} samples  "
             f"obs {ds.arrays['obs'].shape[-1]}  expert_obs {ds.arrays['expert_obs'].shape[-1]}"
             f"  action {ds.meta['act_dim']}  z {ds.meta['z_dim']}  "
             f"creature={ds.meta.get('creature')}")
    L.append(f"  filters {json.dumps(ds.meta.get('filters', {}), sort_keys=True)}")
    L.append(f"  split   {json.dumps(ds.meta.get('split', {}), sort_keys=True)}"
             f"  -> train {c['by_split']['train']}  val {c['by_split']['val']}")
    if ds.meta.get("augmentation"):
        L.append(f"  augment {json.dumps(ds.meta['augmentation'], sort_keys=True)}")
    L.append(f"  dropped {json.dumps(c['dropped'], sort_keys=True)}")
    for s in c["skipped"]:
        L.append(f"  SKIPPED {s['path']}: {s['reason']}")
    L.append("")
    L.append(_table("per skill", [[k, v["total"], v["train"], v["val"]]
                                  for k, v in sorted(c["by_skill"].items(),
                                                     key=lambda kv: -kv[1]["total"])],
                    ["skill", "total", "train", "val"]))
    L.append("")
    L.append(_table("per controller", [[k, v["total"], v["train"], v["val"]]
                                       for k, v in c["by_controller"].items()],
                    ["controller", "total", "train", "val"]))
    L.append("")
    L.append(_table("per match", [[k, v["match_id"], v["split"], v["samples"],
                                   v["ticks"], v["goals"]]
                                  for k, v in c["by_match"].items()],
                    ["file", "match", "split", "samples", "ticks", "goals"]))
    L.append("")
    L.append(_table("per obs layout", [[k, v] for k, v in c["by_layout"].items()],
                    ["layout", "samples"]))
    L.append("")
    L.append(f"expert obs: {cov['with_expert_obs']} with / "
             f"{cov['without_expert_obs']} without, widths {cov['widths']}, "
             f"{cov['nonfinite_rows']} rows with a non-finite entry")
    L.append(f"latent z:   {cov['with_z']} finite / {cov['without_z']} NaN "
             f"(mirrored samples carry no z by design)")
    L.append("")
    if a:
        rows = [[i, a["mean"][i], a["std"][i], a["min"][i], a["max"][i],
                 a["frac_saturated"][i]] for i in range(a["dim"])]
        L.append(_table("action distribution (per actuator)", rows,
                        ["act", "mean", "std", "min", "max", "|a|>=.999"]))
        L.append(f"  |a| mean L2 {a['l2_mean']}   saturated {a['frac_saturated_all']:.1%}"
                 f"   all-zero rows {a['frac_exactly_zero']:.1%}"
                 f"   out of range {a['out_of_range']}   non-finite {a['nonfinite']}")
        if a["out_of_range"] or a["nonfinite"]:
            L.append("  !! actions outside [-1, 1] or non-finite: the corpus is broken")
    return "\n".join(L)


def main(argv=None):
    import argparse
    import glob as _glob
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("paths", nargs="+", help="a built dataset .npz, or demo files/globs")
    p.add_argument("--mirror", action="store_true",
                   help="also report the mirror-augmented corpus")
    p.add_argument("--json", action="store_true")
    p.add_argument("--keep-idle", action="store_true")
    a = p.parse_args(argv)

    paths = []
    for pat in a.paths:
        paths.extend(sorted(_glob.glob(pat)) or [pat])
    if len(paths) == 1 and not paths[0].endswith(".demo.npz"):
        try:
            ds = BCDataset.load(paths[0])
        except Exception:                                  # noqa: BLE001
            ds = build_dataset(paths, drop_idle=not a.keep_idle)
    else:
        ds = build_dataset(paths, drop_idle=not a.keep_idle)

    if a.mirror:
        from rower_soccer.bc.augment import mirror_dataset
        ds = mirror_dataset(ds)
    if a.json:
        print(json.dumps(dict(counts=counts(ds), action=action_stats(ds),
                              expert=_expert_coverage(ds)), indent=1, default=str))
    else:
        print(summary(ds))


if __name__ == "__main__":
    main()
