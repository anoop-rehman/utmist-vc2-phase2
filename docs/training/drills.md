---
title: Training drills
---

# Training drills

The drills implement the skills the experts learn: **follow** (go to a moving
target) and **dribble** (shepherd the ball to a target). Both train in
[Warp](../architecture/warp-backend.md).

## Follow

Worm on the pitch, moving target; reward is the fitness
`exp(-c · ||x_player − x_target||)`. The target's velocity is sampled per
episode; target speed is calibrated to the worm's achievable speed (measured
with a random policy first — see `warp_port/probe_speed.py`).

```bash
MUJOCO_GL=egl .venv/bin/python -m rower_soccer.warp_port.train_follow_warp \
    --run-name follow_base --num-worlds 2048 --steps 500000000
```

Observation: **33 dims** = `proprio(29)` + `target(4)`. The reference
`follow_base` run was trained to **1,231,421,440 steps**.

## Dribble

Same body and pitch, plus a physics ball. Observation: **39 dims** =
`ball_ego(6)` + `proprio(29)` + `target(4)` — the ball obs is 3D (pos + vel) to
match the game. Dribble is much harder than follow and does **not** learn from
scratch; it needs a [curriculum](curriculum.md) and a warm start.

```bash
MUJOCO_GL=egl .venv/bin/python -m rower_soccer.warp_port.train_dribble_warp \
    --run-name dribble_cur1 --init-from runs_v2/follow_base/latest.pt \
    --fixed-start --target-cone 0.0 --num-worlds 2048
```

## Warm-starting (`--init-from`)

Loading a follow checkpoint into a dribble run transfers the **decoder** (the
shared low-level controller); the task-encoder and critic re-init on shape
mismatch (33 vs 39 dims). The proprio/task index buffers are never copied (see
[the observation contract](../architecture/observation-contract.md)).

## Regularizers and stability flags

Common to both trainers (defaults are sane; these are the knobs):

| Flag | Effect |
|---|---|
| `--energy-coef` | control-cost penalty (less spazzing) |
| `--smooth-coef` | CAPS action-smoothness penalty |
| `--ent-anneal-steps` | linearly decay the entropy coefficient (fixes collapse) |
| `--state-dependent-std` | gSDE-flavored state-dependent action std (fine control) |
| `--shaping-scale` | annealed reward shaping (park-not-velocity) |
| `--seed` | reproducibility across arms |

The env clamps observations, scales the accelerometer, and resets any diverged
world before it can poison a batch — see
[the Warp backend](../architecture/warp-backend.md#calibration-and-stability-the-fights-we-had).

## Checkpoints, logging, video

- Checkpoints sync to **GCS** every ~30 min, plus `best.pt` on every new fitness
  peak and a blocking flush on exit. Pull one back with
  `scripts/gcs_pull_run.sh`.
- Metrics + periodic eval videos log to **wandb** (project `creature-soccer`).
  The eval metric is `eval/fitness_warp` — the Warp fitness of one deterministic
  episode (the action distribution's mean, never a sample).
- The reference monitor (`monitor.py`) prints fps/ETA and periodic videos on the
  CPU path.
