# UTMIST Virtual Creatures — Creature Soccer

GA-evolved creatures (from the UTMIST Virtual Creatures Unity project) learn to
play 2v2 soccer in [dm_control](https://github.com/google-deepmind/dm_control)'s
MuJoCo soccer environment.

The method follows DeepMind's humanoid football pipeline
([Liu et al., Science Robotics 2022](https://www.science.org/doi/10.1126/scirobotics.abo0235))
with three strategic cuts that make it tractable on small compute:

1. **No mocap stage** — the low-level controller is learned jointly with the
   four skill experts (*follow → dribble → kick → shoot*) via multitask
   curriculum RL, instead of being distilled from motion capture (our bodies
   have 2–8 DOF, not 56).
2. **Simple bodies first** — the 3-segment worm (2 DOF) validates the whole
   pipeline, then the same recipe retrains for the two-arm rower (8 DOF).
   Headline goal: **heterogeneous teams** (worm defense + rower attack).
3. **Human demos instead of PBT bootstrap** — humans play a LoL-style
   click-to-aim interface that drives the trained skill experts; the experts'
   latent motor intentions are recorded and behavior-cloned, then lightly
   fine-tuned with KL-anchored self-play — replacing ~8×10¹⁰ steps of
   population self-play.

Full design: [`rower_soccer/docs/PIPELINE_V2.md`](rower_soccer/docs/PIPELINE_V2.md).

## Repo layout

```
rower_soccer/            current codebase (everything else at repo root is legacy)
  docs/                  pipeline plan, contracts, benchmarks
  tools/                 .creature (Unity NRBF genotype) -> MuJoCo XML converter
  envs/                  soccer env factory (heterogeneous teams), commands
  drills/                follow/dribble/... drill tasks + gymnasium wrapper
  models/                latent-bottleneck policy (expert -> z -> shared decoder)
  warp_port/             MuJoCo Warp batched GPU envs (~94K steps/s vs ~1K CPU)
  train_drill.py         PPO drill training CLI (wandb + video monitoring)
  monitor.py             fps/ETA console lines + periodic eval videos
  render_video.py        offscreen match/drill rendering
creature_configs/        creature XMLs + original .creature genotype files
trained_creatures/       legacy checkpoints (pre-2026 PPO experiments)
```

## Setup

```bash
python3 -m venv .venv
uv pip install --python .venv/bin/python dm_control mujoco gymnasium \
    "stable-baselines3==2.6.0" wandb imageio[ffmpeg] warp-lang mujoco-warp \
    "torch==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124
```

Headless rendering needs `libegl1 libosmesa6 ffmpeg` and `MUJOCO_GL=egl`.

## Quickstart

```bash
# convert a Unity genotype to MuJoCo XML (validated against the rower)
.venv/bin/python -m rower_soccer.tools.unity2mujoco \
    --input creature_configs/3_SEG_WORM.creature \
    --out creature_configs/three_seg_worm.xml --gear-scale 0.03

# render a random 2v2 (rower+worm per team)
MUJOCO_GL=egl .venv/bin/python -m rower_soccer.render_video \
    --out videos/mixed.mp4 --camera top_down

# train the follow drill (CPU reference path; Warp path in warp_port/)
MUJOCO_GL=egl .venv/bin/python -m rower_soccer.train_drill \
    --drill follow --creature worm --steps 10000000 --procs 6 \
    --run-name follow_worm --device cuda
```

Training logs to wandb (project `creature-soccer`) with eval videos every
5 minutes of wallclock.

## Status

- [x] `.creature` → MuJoCo conversion pipeline (NRBF parser + genotype
      expansion, numerically validated on the two-arm rower)
- [x] Follow + dribble drill envs; latent-bottleneck PPO training with
      monitoring; MuJoCo Warp port of follow (93× throughput)
- [ ] Kick + shoot drills, multitask curriculum, expert→prior distillation
- [ ] Browser play UI, human demos, z-space BC, self-play fine-tune
- [ ] Heterogeneous 2v2 showcase video
