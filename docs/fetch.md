---
title: Fetch (quadruped, worm, rower)
---

# Fetch: reproducing dm_control's quadruped-fetch — then porting it to our bodies

[dm_control's `quadruped.fetch`](https://github.com/google-deepmind/dm_control/blob/main/dm_control/suite/quadruped.py)
task: a walker in a square **walled arena** chases a ball — spawned flying
(dropped from height with a random kick) — and brings it to a **target at the
origin**. We reproduce it as an external, known-solvable benchmark for our
training stack, then adapt it to the worm and the two-arm rower.

## The reward (the part worth stealing)

```
reward = upright × reach × (0.5 + 0.5·fetch)          each term ∈ [0,1]
```

- `upright` — linear in (torso-up · world-up): 1 when upright, 0 upside-down
- `reach` — linear tolerance on the walker→ball distance, margin = the whole
  arena diagonal
- `fetch` — same, on the ball→target distance

Everything is a **linear tolerance with an arena-sized margin**, so there is
gradient *everywhere* — plus walls that keep the ball in play. This is
DeepMind's own answer to the flat-reward desert that stalled our dribble task
(see [Dribble curriculum](training/curriculum.md)).

## Two tracks

- **CPU-faithful** (`rower_soccer/fetch/train_fetch_cpu.py`): the suite's own
  env behind a thin gymnasium adapter, trained with SB3 SAC (the standard
  single-machine baseline; the paper's D4PG/DMPO need actor fleets).
- **Warp port** (`warp_port/fetch_env.py` + `train_fetch_warp.py`): the
  *byte-identical model* — dm_control's own `make_model(walls_and_ball=True)`
  XML, tendon-driven legs and filter actuators included — batched on
  mujoco_warp at ~30k fps. Parity verified on identical states: **reward
  matches to 4e-8**, kinematic obs to ~1e-7; only the accelerometer +
  toe-force blocks differ (constraint-force-derived — the engines' known
  contact-softness gap; the gyro is exact at 1.7e-7, proving the pipeline).
  Trains plain-MLP (`--plain`) or the
  [latent bottleneck](architecture/overview.md).

## Quickstart: `fetch_ant_small` (quadruped, scaled 10×10 arena)

One-time setup: follow [Installation](getting-started/installation.md)
(venv + cu124 torch + `MUJOCO_GL=egl`). Then:

```bash
MUJOCO_GL=egl .venv/bin/python -m rower_soccer.warp_port.train_fetch_warp \
    --run-name fetch_ant_small --plain --worlds 1024 \
    --floor-size 5 --ball-drop-z 1 --ball-kick-std 1.5 \
    --steps 20000000000 --max-hours 10 \
    --first-video-secs 120 --video-secs 1200 --ckpt-secs 1800
```

- `--floor-size 5` shrinks the arena to 10×10 m (the suite's `make_model` only
  resizes the floor geom; our builder moves the hard-coded walls in to match),
  and the ball's drop height / kick scale down with it.
- Omit the three arena flags for the **faithful 30×30** task
  (`fetch_warp_plain`); drop `--plain` for the latent-bottleneck arm.
- Add `--no-wandb` / `--gcs-bucket ""` to run without external services.
- Watch `train/ep_rew` (max 1000 = a perfect episode) and the eval videos.
  Reference: plain PPO passes ep_rew ≈ 700 within ~50M steps on the scaled
  arena.

## Quickstart: `fetch_worm_arena5` (our worm, same 10×10 arena)

The worm needs one extra artifact: a **labeled up-axis + rest pose**, because a
GA-evolved body has no canonical belly. Label it once in the browser:

```bash
MUJOCO_GL=egl .venv/bin/python -m rower_soccer.warp_port.label_up \
    --creature-xml creature_configs/three_seg_worm.xml --port 8096
# forward the port, open http://localhost:8096
# pose the creature (click limbs for joint sliders, 6-DOF sliders for the
# whole body), then Save -> creature_configs/three_seg_worm_up_axis.json
```

The repo already ships a labeled `three_seg_worm_up_axis.json` (and one for
the scaled rower), so this step is optional unless you want a different pose.
Then train:

```bash
MUJOCO_GL=egl .venv/bin/python -m rower_soccer.warp_port.train_worm_fetch_warp \
    --run-name fetch_worm_arena5 --scene arena --worlds 1024 \
    --steps 20000000000 --max-hours 10 \
    --first-video-secs 120 --video-secs 1200 --ckpt-secs 1800
```

- `--scene pitch` runs the same task on the full dm_soccer pitch (goals and
  all), with spawns kept in a central ±4.5 m region so the task stays
  completable.
- `--creature-xml creature_configs/two_arm_rower_scaled.xml
  --up-axis-json creature_configs/two_arm_rower_scaled_up_axis.json` swaps in
  the **two-arm rower** (regenerated at the worm's exact scale factors:
  `unity2mujoco --length-scale 0.1768 --gear-scale 0.03` → 1.1 m / 7.3 kg).
  Obs dims and the proprio/task split adapt to the body automatically.
- `--init-from runs_v2/<run>/latest.pt` warm-starts the decoder from a
  follow/dribble checkpoint (worm only — the 29-dim proprio contract matches).

## How the adaptation differs from the quadruped original

| aspect | quadruped fetch | worm/rower fetch |
|---|---|---|
| arena | 30×30 walled square | 10×10 (≈ the bodies' speed ratio), same tilted walls |
| ball spawn | z=2 drop, 5·randn kick | z=1 drop, 1.5·randn kick |
| upright axis | torso z (canonical) | **human-labeled** via `label_up.py` |
| spawn pose | upright + random azimuth | labeled rest pose + random yaw |
| obs | suite obs (90) | our proprio contract (29/65) + fetch task obs (12) |

Three hard-won correctness notes, all verified with zero-action rollouts:

1. **Spawn right-side-up.** Spawning in the model-default orientation left the
   worm 90° off its labeled up, where `upright` plateaus at 0.48 and a 2-DOF
   body cannot roll itself over. The labeled quat composes with the random
   yaw, at a non-contacting height solved for that orientation.
2. **Clear the walls by body reach.** The root sits at one *end* of the worm;
   a root placed at the naive spawn radius buries the far segments inside a
   wall and the solver fires the body into the sky (measured: root z 8.7 m,
   zero action). The creature's spawn box shrinks by its geom-bound reach.
3. **Soften contacts to 0.010.** Upright spawns rest on capsule edges; at the
   drills' 0.005 solref a few percent of worlds went NaN within steps — the
   same failure the ball once had at 0.005.

## The labeling tool

`warp_port/label_up.py` serves a browser UI (Flask + EGL, headless-safe):
**click a limb** in the live render to select it (MuJoCo's `mjv_select` ray
pick — it highlights yellow) and slide its joints; place the whole body with
6-DOF sliders; orbit the camera; **Save**. The JSON captures the root pose,
every joint angle, and `up_local = Rᵀẑ`. The fetch env spawns the creature in
exactly that pose and scores `upright = (1 + R·up_local · ẑ)/2`.
