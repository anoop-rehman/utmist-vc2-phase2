"""Upload the mp4s `e0_video.py` rendered, plus their scalars, to wandb.

A separate script because the two halves need different venvs: rendering needs
mujoco-py, which lives only in `/workspace/Transform2Act/.venv-gpu`, and that
venv has no wandb. Run this one with the repo's `.venv`:

    set -a && . /workspace/.env && set +a
    cd /workspace/utmist-vc2-phase2
    .venv/bin/python -m rower_soccer.t2a_port.e0_wandb_media \
        runs/d3_e0_ant/renders/*.mp4.json

Each sidecar is uploaded once and then renamed `*.json.sent`, so this is safe to
run on a glob from a loop. `wandb.init` uses the run NAME as its id with
`resume="allow"`, the same convention `scripts/wandb_ship.py` uses, so media and
metrics land in one run.
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

    n = 0
    for path in args.sidecars:
        if not os.path.exists(path):
            continue
        side = json.load(open(path))
        if not os.path.exists(side["mp4"]):
            print(f"[media] missing {side['mp4']}, skipped")
            continue
        run = wandb.init(project=side["project"], name=side["run"],
                         id=side["run"], resume="allow")
        payload = dict(side["scalars"])
        payload[side["key"]] = wandb.Video(side["mp4"], fps=side["fps"],
                                           format="mp4")
        wandb.log(payload, step=side["step"])
        run.finish()
        print(f"[media] {side['run']} step {side['step']} {side['key']} "
              f"<- {os.path.basename(side['mp4'])}")
        n += 1
        if not args.keep:
            os.rename(path, path + ".sent")
    print(f"[media] {n} uploaded")


if __name__ == "__main__":
    main()
