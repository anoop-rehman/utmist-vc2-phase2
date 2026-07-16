#!/usr/bin/env python3
"""Fan out a sweep of worm-training runs as SkyPilot managed spot jobs.

Each entry in MENU becomes one `sky jobs launch sky/train.yaml` on its own spot
GPU. Managed jobs auto-recover from preemption and resume from the GCS checkpoint,
so the sweep survives instances vanishing under it.

  python sky/sweep.py            # print the plan + cost estimate, launch nothing
  python sky/sweep.py --launch   # actually submit the jobs

The MENU is the single source of truth for "what experiments are running". Edit it
here; keep it in git so a sweep is reproducible.
"""
import argparse
import subprocess
import sys

# --- the experiment menu -----------------------------------------------------
# Each arm: name, task, and the EXTRA_ARGS that make it distinct. The common spine
# (entropy floor/ceil, video/ckpt cadence, gcs sync, worlds, max-hours) lives in
# train.yaml, so only the swept knobs appear here.
ANNEAL = "--ent-anneal-steps 400000000"   # entropy anneal, shared by all arms

MENU = [
    dict(name="follow_baseline",     task="follow",
         args=f"{ANNEAL}"),
    dict(name="follow_smooth",       task="follow",
         args=f"{ANNEAL} --smooth-coef 0.1"),
    dict(name="follow_smooth_energy", task="follow",
         args=f"{ANNEAL} --smooth-coef 0.1 --energy-coef 0.05"),
    dict(name="dribble_smooth",      task="dribble",
         args=f"{ANNEAL} --smooth-coef 0.1 --energy-coef 0.02"),
]

# throughput measured on an A4000 (steps/hour), for the estimate only
SPS_PER_HOUR = {"follow": 343e6, "dribble": 307e6}
USD_PER_GPU_HOUR = 0.17     # RunPod A4000 community/spot


def estimate(max_hours):
    print(f"{'arm':24}{'task':9}{'~steps in 48h':>16}{'$ @48h':>9}")
    total = 0.0
    for m in MENU:
        steps = SPS_PER_HOUR[m["task"]] * max_hours
        cost = max_hours * USD_PER_GPU_HOUR
        total += cost
        print(f"{m['name']:24}{m['task']:9}{steps/1e9:>13.1f} B{cost:>8.2f}")
    print(f"{'':49}{'-'*9}")
    print(f"{'TOTAL (all run full ' + str(int(max_hours)) + 'h)':49}{total:>8.2f}")
    print(f"\nBut runs converge/collapse far sooner -- follow hit its peak by ~500M")
    print(f"steps (~1.5h). With keep-best capturing the peak, watch the early videos")
    print(f"and kill converged arms. Realistic sweep cost if killed at ~3h each: "
          f"${len(MENU)*3*USD_PER_GPU_HOUR:.2f}.")


def launch(max_hours, wandb_key):
    for m in MENU:
        cmd = [
            "sky", "jobs", "launch", "sky/train.yaml",
            "--name", m["name"], "--yes",
            "--env", f"RUN_NAME={m['name']}",
            "--env", f"TASK={m['task']}",
            "--env", f"EXTRA_ARGS={m['args']}",
            "--env", f"MAX_HOURS={int(max_hours)}",
            "--env", f"WANDB_API_KEY={wandb_key}",
        ]
        print("+ " + " ".join(cmd))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--launch", action="store_true", help="actually submit jobs")
    ap.add_argument("--max-hours", type=float, default=48.0)
    ap.add_argument("--wandb-key", default="", help="WANDB_API_KEY for the remote")
    a = ap.parse_args()
    estimate(a.max_hours)
    if a.launch:
        if not a.wandb_key:
            sys.exit("\nRefusing to launch without --wandb-key (runs would not log).")
        print("\nlaunching...\n")
        launch(a.max_hours, a.wandb_key)
    else:
        print("\n(dry run -- pass --launch to submit)")
