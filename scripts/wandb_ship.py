"""Ship a run's on-disk log into wandb — backfill, then optionally follow.

wandb cannot be attached to a process that is already running: `wandb.init()`
happens at startup and there is no way in afterwards. But every trainer here
already writes its history to disk, so the history can be replayed into a run
and then tailed, which gets to the same place.

Three sources, because the tracks log differently:

  * **`--json`** — our D2 trainers' `log.json` (`{"args": ..., "iters": [...]}`).
  * **`--t2a-log`** — a Transform2Act stdout log, ours or theirs. Steps by
    epoch and derives episode length; see `ship_t2a`.
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


def _scalar(v):
    """A `--config k=v` value -> the narrowest type it plausibly is."""
    if v in ("True", "False"):
        return v == "True"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


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

    Three things this does beyond transcribing that line, all of them needed to
    put the port and the reference on one chart:

    * **Episode length is derived**, `X_R_eps / X_R` -> `t2a/train_ep_len` and
      `t2a/exec_ep_len`. Both codebases print reward-per-episode and
      reward-per-step and neither prints length, but length is the whole D3
      argument: the reference reaches the 1,000-step limit while the port sits
      near 108. Checked against the port's own `train_len`, which it does log:
      epoch 999 of `port_s1` gives 477.30 / 4.40 = 108.5 against a logged 108.4.
    * **The port's JSON sidecar line** (`{"epoch": ..., "train_len": ...}`,
      which `train_t2a.py` writes after each monitor line) is parsed too, under
      a `port/` prefix. The reference has no counterpart, so these keys are
      simply absent on reference runs.
    * **Rows are buffered and flushed in epoch order.** A resumed run re-logs
      epochs it already did, so the epoch column jumps BACKWARDS once mid-file;
      every long reference log here has exactly one such jump. wandb drops a
      row whose step is behind the current one, so streaming these straight
      through silently loses everything after the resume. Buffering also makes
      last-write-wins for a re-logged epoch explicit rather than incidental.

    The step is always the epoch, which is what makes port and reference
    overlay -- `--step-key` is a `--json` option and is ignored here.
    """
    import re
    import wandb
    name = args.name or os.path.basename(args.t2a_log).replace(
        "results_", "t2a_").replace(".log", "")

    # `train_t2a.py` prints this at startup so the arm is recoverable from the
    # log alone. The reference has no equivalent and neither do port runs from
    # before it was added, hence `--config`.
    startup = re.compile(
        r"^run\s+(?P<run>\S+)\s+cfg\s+(?P<cfg>\S+)\s+seed\s+(?P<seed>\d+)"
        r"\s+batch_design\s+(?P<batch_design>True|False)"
        r".*?dtype\s+(?P<dtype>\S+)")

    cfg = {}
    with open(args.t2a_log) as f:
        for line in f:
            m = startup.match(line)
            if m:
                cfg = dict(m.groupdict())
                cfg["seed"] = int(cfg["seed"])
                cfg["batch_design"] = cfg["batch_design"] == "True"
                print(f"[ship] startup line: {cfg}")
                break
    for kv in args.config:
        k, _, v = kv.partition("=")
        cfg[k] = _scalar(v)

    run = wandb.init(project=args.project, name=name, id=name,
                     tags=args.tags or ["D3"], config=cfg, notes=args.notes,
                     resume="allow")
    print(f"[ship] {run.url}")
    # Epoch is logged as a metric as well as being the step, so the workspace
    # can name its x-axis instead of showing an anonymous "Step".
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    # `ETA 2:02:28` is a duration, not a number; everything else is a float.
    # `ETA 1 day, 6:35:40` DOES look like `key value` for its first token, so
    # ETA is dropped by name rather than left to the regex.
    pair = re.compile(r"([A-Za-z_]+)\s+(-?\d+\.?\d*)(?=\s|$)")
    monitor = re.compile(r"^(\d+)\t(.*)$")
    sidecar = re.compile(r'^\s*(\{.*"epoch".*\})\s*$')

    # How many zero-reward design steps their `avg_reward` denominator carries
    # per episode. Set to 0 for a PORT log, whose `train_R` counts execution
    # steps only -- decided below, from whether the file has the port's JSON
    # sidecar, and re-decided on every re-read so `--follow` cannot latch it
    # wrong on an empty first pass.
    design_steps = [args.design_steps]

    def parse(fh, rows):
        """Lines -> {epoch: flat row}. A later line for an epoch wins."""
        for line in fh:
            m = monitor.match(line)
            if m:
                epoch = int(m.group(1))
                row = {k: float(v) for k, v in pair.findall(m.group(2))
                       if k != "ETA"}
                if not row:
                    continue
                d = rows.setdefault(epoch, {})
                d.update({f"t2a/{k}": v for k, v in row.items()})
                # Neither codebase prints episode length; both print the two
                # numbers whose ratio is it.
                #
                # `train_R` is NOT the same denominator on the two sides, and
                # putting the raw ratio on one chart is an apples-to-oranges
                # comparison worth 6 steps per episode. Theirs is
                # `LoggerRL.avg_reward` = total reward over ALL logged steps,
                # and `khrylib/rl/agents/agent.py:70` logs the 5 skeleton and 1
                # attribute step too (reward 0 each), so `train_R_eps/train_R`
                # is `exec_steps + skel_transform_nsteps + 1`. The port's
                # `train_R` divides by execution steps only. `exec_R` is
                # execution-only on BOTH sides
                # (`design_opt/utils/logger.py:22`), so `exec_ep_len` needs no
                # correction. See D3_HANDOFF.md, "Update 2026-08-28 (second)".
                for tag in ("train", "exec"):
                    r, r_eps = row.get(f"{tag}_R"), row.get(f"{tag}_R_eps")
                    if r_eps is not None and r is not None and abs(r) > 1e-9:
                        v = r_eps / r
                        if tag == "train":
                            d["t2a/train_ep_len_all_stages"] = v
                            v -= design_steps[0]
                        d[f"t2a/{tag}_ep_len"] = v
                continue
            m = sidecar.match(line)
            if m:
                try:
                    blob = json.loads(m.group(1))
                except json.JSONDecodeError:
                    continue
                epoch = blob.pop("epoch", None)
                if epoch is None:
                    continue
                rows.setdefault(int(epoch), {}).update(flat(blob, "port/"))
                # Only `train_t2a.py` writes this sidecar, and only the port
                # runs it, so seeing one identifies the log as the port's --
                # and the port needs no correction.
                design_steps[0] = 0
        # A row parsed before the first sidecar line kept the reference
        # correction; redo them now that the file's provenance is known.
        if design_steps[0] == 0:
            for d in rows.values():
                if "t2a/train_ep_len_all_stages" in d:
                    d["t2a/train_ep_len"] = d["t2a/train_ep_len_all_stages"]
        return rows

    last = [-1]

    def flush(rows):
        n = 0
        for epoch in sorted(rows):
            if epoch <= last[0]:
                continue        # already sent; wandb would drop it anyway
            wandb.log({**rows[epoch], "epoch": epoch}, step=epoch)
            last[0] = epoch
            n += 1
        return n

    fh = open(args.t2a_log)
    rows = parse(fh, {})
    n = flush(rows)
    print(f"[ship] backfilled {n} epochs (through {last[0]}) "
          f"from {args.t2a_log}", flush=True)

    if args.follow:
        # Tail from where the backfill stopped, so a growing log costs one
        # read per poll rather than a full re-parse.
        print("[ship] following; ctrl-c to stop", flush=True)
        idle = 0
        while idle < args.idle_giveup:
            time.sleep(args.poll)
            got = flush(parse(fh, {}))
            if got:
                idle = 0
                print(f"[ship] +{got} epochs (through {last[0]})", flush=True)
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
    p.add_argument("--config", nargs="*", default=[], metavar="K=V",
                   help="extra wandb config entries. For --t2a-log these fill "
                        "in the arm for a log written before train_t2a.py "
                        "started printing its startup line, and override it "
                        "where both are present")
    p.add_argument("--notes", default=None, help="wandb run notes")
    p.add_argument("--design-steps", type=int, default=6,
                   help="zero-reward design steps per episode that THEIR "
                        "`train_R` denominator counts and the port's does not "
                        "(skel_transform_nsteps + 1; 6 for every hopper cfg). "
                        "Subtracted from `t2a/train_ep_len` on reference logs "
                        "only. The uncorrected ratio is kept as "
                        "`t2a/train_ep_len_all_stages`.")
    p.add_argument("--step-key", default="steps",
                   help="row field to use as the wandb step (--json only; "
                        "--t2a-log always steps by epoch, which is what makes "
                        "port and reference runs overlay)")
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
