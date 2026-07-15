# SkyPilot sweeps

Launch worm-training runs as managed spot jobs, fanned out one GPU per arm. Debug
on the interactive pod; run *validated* configs here.

## Why this is safe on spot

The trainer already syncs checkpoints to GCS every 30 min, `best.pt` on every new
peak, and flushes blocking on exit. On preemption, SkyPilot relaunches the job on
fresh spot capacity, `scripts/gcs_pull_run.sh` pulls the last checkpoint back, and
`--resume` continues. Worst case lost to a preemption: ~30 min.

## One-time setup (on the launching machine)

```bash
pip install "skypilot[runpod]"        # or skypilot[gcp]
sky check                             # verify a cloud is enabled
# RunPod: set RUNPOD_API_KEY.  GCP: gcloud auth application-default login,
#         and request L4/T4 spot quota first (personal projects default to 0).
```

## Launch

```bash
# see the plan + cost, launch nothing:
python sky/sweep.py

# submit the whole sweep (one spot GPU per arm):
python sky/sweep.py --launch --wandb-key "$WANDB_API_KEY"

# a single arm by hand:
sky jobs launch sky/train.yaml --name follow_smooth \
    --env RUN_NAME=follow_smooth --env TASK=follow \
    --env EXTRA_ARGS="--smooth-coef 0.1 --ent-anneal-steps 400000000" \
    --env WANDB_API_KEY=$WANDB_API_KEY
```

## Watch / manage

```bash
sky jobs queue          # status of all managed jobs
sky jobs logs <id>      # stream logs
sky jobs cancel <name>  # kill an arm early (keep-best already saved the peak)
```

Results land in `gs://vc2-2026-checkpoints/<run>/` and on wandb regardless of how
the job ends. Jobs auto-teardown their instance on completion — no idle billing.

## Cost

A full 4-arm sweep is ~$33 if every arm runs the full 48 h on A4000 spot, but runs
converge in ~1.5–3 h — kill them at convergence and it's ~$2. See `sweep.py` for the
live estimate. The GPU is set in `train.yaml` (`accelerators`); override per launch
with `--gpus L4:1` or `--cloud gcp`.

The menu of experiments lives in `sweep.py::MENU` — edit there, keep in git.
