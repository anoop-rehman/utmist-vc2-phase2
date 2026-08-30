"""D3 M3 E2: verify through the wandb API that each run really carries what we
think it carries. An uploader exit code is not evidence -- this check caught a
silently-dropped video twice on this project.

For every named run it asserts the run EXISTS, that its history contains the
metric keys, and that `video/best_median_worst` appears in the history (not
just in the summary). Runs are never deleted: deleting one permanently burns
its id and silently breaks later uploads to that name.

    cd /workspace          # NOT the repo root: its `wandb/` artifact dir
                           # shadows the package
    set -a && . /workspace/.env && set +a
    /workspace/utmist-vc2-phase2/.venv/bin/python \\
      /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/e2_wandb_verify.py \\
      d3_e2_gnn_s1 d3_e2_gnn_s2 d3_e2_mlp_s1 d3_e2_mlp_s2
"""
import argparse
import os
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+")
    p.add_argument("--entity", default=None)
    p.add_argument("--project", default="creature-soccer")
    p.add_argument("--video-key", default="video/best_median_worst")
    p.add_argument("--metric-key", default="e2/eval_R_mean")
    a = p.parse_args()
    os.environ.setdefault("WANDB_SILENT", "true")
    import wandb
    api = wandb.Api()
    ent = a.entity or api.default_entity
    bad = 0
    for name in a.runs:
        path = f"{ent}/{a.project}/{name}"
        try:
            r = api.run(path)
        except Exception as e:
            print(f"[MISSING] {name}: {e!r}")
            bad += 1
            continue
        keys = set(r.summary.keys()) if r.summary else set()
        hist_metric = [row for row in r.scan_history(keys=["epoch",
                                                           a.metric_key])]
        try:
            hist_vid = [row for row in r.scan_history(keys=["epoch",
                                                            a.video_key])]
        except Exception:
            hist_vid = []
        n_vid = sum(1 for row in hist_vid if row.get(a.video_key) is not None)
        ok = len(hist_metric) > 0 and n_vid > 0
        bad += 0 if ok else 1
        print(f"[{'OK' if ok else 'FAIL'}] {name}  state={r.state}  "
              f"metric rows={len(hist_metric)}  {a.video_key} rows={n_vid}  "
              f"video-in-summary={a.video_key in keys}  "
              f"last epoch={r.summary.get('epoch')}")
    print(f"\n{len(a.runs) - bad} of {len(a.runs)} runs verified")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
