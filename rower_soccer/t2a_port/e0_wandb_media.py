"""Upload the mp4s `e0_video.py` rendered, plus their scalars, to wandb.

A separate script because the two halves need different venvs: rendering needs
mujoco-py, which lives only in `/workspace/Transform2Act/.venv-gpu`, and that
venv has no wandb. Run this one with the repo's `.venv`:

    set -a && . /workspace/.env && set +a
    cd /workspace/utmist-vc2-phase2
    .venv/bin/python -m rower_soccer.t2a_port.e0_wandb_media \
        runs/d3_e0_ant/renders/*.mp4.json

Each sidecar is uploaded once and then renamed `*.json.sent`, so this is safe to
run on a glob from a loop. `wandb.init` uses the run NAME as its id with `resume="allow"`, the same
convention `scripts/wandb_ship.py` uses.

**Media goes to its own run, `<name>_video`, NOT into the metrics run.**
Measured the hard way: `wandb.log(..., step=N)` silently DROPS a row whose step
is behind the run's current one, and the metrics shipper had already walked
`d3_e0_ant_s1` to step 13 by the time a step-0 video was uploaded. The upload
reported success, and `api.run(...).history(keys=["video/best_R"])` came back
empty with no `video/*` key in the summary at all. Two writers cannot share one
step counter when one of them logs epochs in order and the other arrives late
with an older epoch. A separate run per seed gives media its own monotonic
counter, and `epoch` is logged as a metric and declared as the step metric so
its charts line up with the metrics run's.

The same rule bites a second way: two clips for the SAME step -- the initial-ant
clip and the epoch-0 rollout -- cannot be two `wandb.log` calls, because the
second one resumes a run whose step is already 0 and is dropped (measured:
`video/initial_ant` was absent from the history while `video/best_median_worst`
at the same step was present). Sidecars are therefore GROUPED by (project, run,
step) and each group is one log call.
"""

import argparse
import json
import os
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("sidecars", nargs="+")
    p.add_argument("--keep", action="store_true",
                   help="do not rename to .sent (re-uploads on the next run)")
    args = p.parse_args()

    os.environ.setdefault("WANDB_SILENT", "true")
    import wandb
    if not hasattr(wandb, "init"):
        sys.exit("`import wandb` resolved to something without .init -- most "
                 "likely the repo's wandb/ artifact directory shadowing the "
                 "package. Run this from a venv where wandb is installed and "
                 "not from a cwd that shadows it.")

    groups = {}
    for path in args.sidecars:
        if not os.path.exists(path):
            continue
        side = json.load(open(path))
        if not os.path.exists(side["mp4"]):
            print(f"[media] missing {side['mp4']}, skipped")
            continue
        groups.setdefault((side["project"], side["run"], side["step"]),
                          []).append((path, side))

    n = 0
    for (project, name, step) in sorted(groups, key=lambda k: (k[1], k[2])):
        items = groups[(project, name, step)]
        run = wandb.init(project=project, name=name, id=name, resume="allow")
        wandb.define_metric("epoch")
        wandb.define_metric("*", step_metric="epoch")
        payload = {}
        for _, side in items:
            payload.update(side["scalars"])
            payload[side["key"]] = wandb.Video(side["mp4"], fps=side["fps"],
                                               format="mp4")
        wandb.log(payload, step=step)
        run.finish()
        print(f"[media] {name} step {step} <- " +
              ", ".join(os.path.basename(s["mp4"]) for _, s in items))
        n += len(items)
        if not args.keep:
            for path, _ in items:
                os.rename(path, path + ".sent")
    print(f"[media] {n} uploaded in {len(groups)} log calls")


if __name__ == "__main__":
    main()
