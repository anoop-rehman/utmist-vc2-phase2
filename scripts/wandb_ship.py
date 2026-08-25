"""Ship a run's on-disk log into wandb — backfill, then optionally follow.

wandb cannot be attached to a process that is already running: `wandb.init()`
happens at startup and there is no way in afterwards. But every trainer here
already writes its history to disk, so the history can be replayed into a run
and then tailed, which gets to the same place.

Two sources, because the three tracks log differently:

  * **`--json`** — our D2 trainers' `log.json` (`{"args": ..., "iters": [...]}`).
  * **`--tb`** — a TensorBoard event directory, which is what Transform2Act and
    CompetEvo's own code write. `wandb sync --sync-tensorboard` handles those
    natively and is the better tool; this path exists for when you want the
    same run naming, tags and step key as everything else.

    # backfill a finished D2 run
    scripts/wandb_ship.py --json runs/competevo_port/t2v2_role_s44/log.json \
        --name t2v2_role_s44 --tags D2 2v2

    # follow a run that is still training
    scripts/wandb_ship.py --json runs/.../log.json --name ... --follow

Idempotent by construction: the wandb run id is the run name and rows are
logged at their own `step`, so re-shipping the same file overwrites rather than
duplicating. That matters because the natural way to use this is to re-run it.
"""

import argparse
import json
import os
import sys
import time


def flat(row, prefix=""):
    """One log row -> flat scalars. Lists become `name_i`, dicts recurse."""
    out = {}
    for k, v in row.items():
        if isinstance(v, dict):
            out.update(flat(v, f"{prefix}{k}/"))
        elif isinstance(v, (list, tuple)):
            for i, x in enumerate(v):
                if isinstance(x, (int, float)) and not isinstance(x, bool):
                    out[f"{prefix}{k}_{i}"] = float(x)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[f"{prefix}{k}"] = float(v)
    return out


def ship_json(args):
    import wandb

    def read():
        with open(args.json) as f:
            return json.load(f)

    blob = read()
    rows = blob.get("iters", blob if isinstance(blob, list) else [])
    cfg = blob.get("args", {}) if isinstance(blob, dict) else {}
    name = args.name or os.path.basename(os.path.dirname(args.json))
    run = wandb.init(project=args.project, name=name, id=name,
                     config=cfg, tags=args.tags, resume="allow")
    print(f"[ship] {run.url}")

    sent = 0

    def push(rs):
        nonlocal sent
        for r in rs:
            step = r.get(args.step_key)
            wandb.log(flat(r), step=int(step) if step is not None else None)
            sent += 1

    push(rows)
    print(f"[ship] backfilled {sent} rows from {args.json}")

    if args.follow:
        print(f"[ship] following; ctrl-c to stop")
        seen = len(rows)
        idle = 0
        while idle < args.idle_giveup:
            time.sleep(args.poll)
            try:
                rows = read().get("iters", [])
            except (json.JSONDecodeError, OSError):
                continue          # a partial write; try again next tick
            if len(rows) > seen:
                push(rows[seen:])
                seen = len(rows)
                idle = 0
                print(f"[ship] {seen} rows", flush=True)
            else:
                idle += args.poll
        print(f"[ship] no new rows for {args.idle_giveup}s; stopping")
    run.finish()


def ship_t2a(args):
    """Transform2Act's stdout monitor lines -> wandb.

    Their trainer writes TensorBoard, but `wandb sync --sync-tensorboard`
    needs the `tensorboard` package, and installing it drags protobuf into a
    venv pinned around torch 1.8 / warp. Their stdout carries every scalar the
    events file does, in a stable `key value` layout, so parsing that keeps the
    venv untouched.

        902  T_sample 14.53  T_update 55.76  T_eval 5.46  ETA 2:02:28
             train_R 8.71  train_R_eps 8059.62  exec_R 9.32
             exec_R_eps 9324.26  hopper_gpu
    """
    import re
    import wandb
    name = args.name or os.path.basename(args.t2a_log).replace(
        "results_", "t2a_").replace(".log", "")
    run = wandb.init(project=args.project, name=name, id=name,
                     tags=args.tags or ["D3"], resume="allow")
    print(f"[ship] {run.url}")
    # `ETA 2:02:28` is a duration, not a number; everything else is a float.
    pair = re.compile(r"([A-Za-z_]+)\s+(-?\d+\.?\d*)(?=\s|$)")
    # A resume re-logs epochs it has already done; the LAST value for an epoch
    # is the live one, and wandb keeps the last write at a given step anyway.
    seen = set()

    def push(fh):
        n = 0
        for line in fh:
            m = re.match(r"^(\d+)\t(.*)$", line)
            if not m:
                continue
            epoch = int(m.group(1))
            row = {k: float(v) for k, v in pair.findall(m.group(2))}
            if not row:
                continue
            wandb.log({f"t2a/{k}": v for k, v in row.items()}, step=epoch)
            seen.add(epoch)
            n += 1
        return n

    fh = open(args.t2a_log)
    n = push(fh)
    print(f"[ship] backfilled {n} epochs from {args.t2a_log}", flush=True)

    if args.follow:
        # Tail from where the backfill stopped, so a growing log costs one
        # read per poll rather than a full re-parse.
        print("[ship] following; ctrl-c to stop", flush=True)
        idle = 0
        while idle < args.idle_giveup:
            time.sleep(args.poll)
            got = push(fh)
            if got:
                idle = 0
                print(f"[ship] +{got} epochs (through {max(seen)})", flush=True)
            else:
                idle += args.poll
                fh.seek(fh.tell())      # clear EOF so the next read sees growth
        print(f"[ship] no new epochs for {args.idle_giveup}s; stopping")
    run.finish()


def ship_tb(args):
    """Defer to `wandb sync`, which reads tfevents properly."""
    name = args.name or os.path.basename(os.path.dirname(args.tb.rstrip("/")))
    cmd = [sys.executable, "-m", "wandb", "sync", "--sync-tensorboard",
           "--project", args.project, "--id", name, args.tb]
    print("[ship] " + " ".join(cmd))
    os.execv(sys.executable, cmd)


def main():
    p = argparse.ArgumentParser()
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--json", help="a D2 trainer's log.json")
    src.add_argument("--tb", help="a TensorBoard event directory")
    src.add_argument("--t2a-log", help="a Transform2Act stdout log")
    p.add_argument("--name", default=None, help="wandb run name AND id")
    p.add_argument("--project", default="creature-soccer")
    p.add_argument("--tags", nargs="*", default=[])
    p.add_argument("--step-key", default="steps",
                   help="row field to use as the wandb step")
    p.add_argument("--follow", action="store_true")
    p.add_argument("--poll", type=float, default=30.0)
    p.add_argument("--idle-giveup", type=float, default=3600.0,
                   help="stop following after this many seconds with no new "
                        "rows, so a forgotten shipper does not outlive its run")
    args = p.parse_args()
    if not os.environ.get("WANDB_API_KEY"):
        raise SystemExit("WANDB_API_KEY is not set (source .env)")
    if args.tb:
        ship_tb(args)
    elif args.t2a_log:
        ship_t2a(args)
    else:
        ship_json(args)


if __name__ == "__main__":
    main()
