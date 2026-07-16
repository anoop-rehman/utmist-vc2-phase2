---
title: Sweeps (SkyPilot)
---

# Sweeps

Hyperparameter sweeps fan out one GPU per arm. There are two ways to run them:
**local** (several arms on one pod) and **SkyPilot** (managed spot jobs, one GPU
each).

## Local fan-out

Launcher scripts in `sky/` start several arms on the current pod, detached, each
logging to `logs/`:

```bash
bash sky/launch_local_12.sh          # the 12-arm overnight fanout (4 follow / 8 dribble)
bash sky/launch_dribble_ws.sh        # warm-start dribble sweep
bash sky/launch_dribble_finectrl.sh  # fine-control: gSDE × shaping-anneal
bash sky/launch_curriculum1.sh       # curriculum stage-1 arms + plain-MLP baseline
```

The pattern for successive-halving: launch many arms as long runs, watch ~1 h of
`eval/fitness_warp` in wandb, then kill the arms that aren't climbing.

## SkyPilot (spot)

Debug on the interactive pod; run only *validated* configs on spot. Full details:
`sky/README.md`.

```bash
pip install "skypilot[runpod]"   # or skypilot[gcp]
sky check

python sky/sweep.py                                   # print plan + cost, launch nothing
python sky/sweep.py --launch --wandb-key "$WANDB_API_KEY"   # submit the sweep
```

### Why spot is safe here

The trainer syncs checkpoints to GCS every ~30 min, writes `best.pt` on every new
peak, and flushes on exit. On preemption SkyPilot relaunches on fresh capacity,
`scripts/gcs_pull_run.sh` pulls the last checkpoint back, and `--resume`
continues. Worst-case loss to a preemption is ~30 min.

## GPU note

The interactive pod is an **RTX 4000 Ada (20 GB)**; torch must be the cu124
build. For spot on GCP you'll want L4/T4 quota (personal projects default to 0 —
request it first).
