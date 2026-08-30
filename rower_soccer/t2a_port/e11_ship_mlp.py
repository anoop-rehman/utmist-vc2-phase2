"""Ship an E1.1 MLP arm's `log.jsonl` to wandb.

`scripts/wandb_ship.py` reads a Transform2Act stdout log or a D2 `log.json`;
`train_e11_mlp.py` writes neither, so this is the third shape. It follows the
same rules E0 established the hard way:

  * every row is logged with **no explicit step**, and `epoch` is declared as
    the step metric. `wandb.log(step=N)` silently DROPS a row whose step is
    behind the run's current one, which is how E0 lost six video uploads; a
    backfill of a finished run is exactly the case that triggers it.
  * the run NAME is its id with `resume="allow"`, so re-running this is
    idempotent rather than creating duplicates.

    set -a && . /workspace/.env && set +a
    .venv/bin/python -m rower_soccer.t2a_port.e11_ship_mlp \
        --dir /workspace/Transform2Act/results/ant_e11_mlp_s1_pub \
        --name d3_e11_mlp_s1_pub
"""
import argparse
import json
import os


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--project", default="creature-soccer")
    p.add_argument("--notes", default=None)
    p.add_argument("--config", nargs="*", default=[], metavar="K=V")
    args = p.parse_args()

    os.environ.setdefault("WANDB_SILENT", "true")
    import wandb
    rows = [json.loads(l) for l in open(os.path.join(args.dir, "log.jsonl"))]
    cfg = dict(kv.split("=", 1) for kv in args.config)
    run = wandb.init(project=args.project, name=args.name, id=args.name,
                     resume="allow", config=cfg, notes=args.notes,
                     tags=["d3", "e11", "mlp"])
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")
    for r in rows:
        wandb.log({f"e11/{k}": v for k, v in r.items() if k != "epoch"}
                  | {"epoch": r["epoch"]})
    run.finish()
    print(f"[ship] {args.name} <- {len(rows)} rows from {args.dir}")


if __name__ == "__main__":
    main()
