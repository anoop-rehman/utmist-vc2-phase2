---
title: Quickstart
---

# Quickstart

Assumes the [installation](installation.md) is done and `MUJOCO_GL=egl` is set
for any command that renders.

## 1. Convert a Unity genotype to MuJoCo XML

```bash
.venv/bin/python -m rower_soccer.tools.unity2mujoco \
    --input creature_configs/3_SEG_WORM.creature \
    --out creature_configs/three_seg_worm.xml --gear-scale 0.03
```

The converter parses the Unity NRBF genotype and expands it to a MuJoCo XML. It
is numerically validated against the two-arm rower.

## 2. Render a random 2v2 match

```bash
MUJOCO_GL=egl .venv/bin/python -m rower_soccer.render_video \
    --out videos/mixed.mp4 --camera top_down
```

## 3. Train the follow drill (Warp — the recommended path)

The Warp backend trains, scores, **and** renders in the same GPU physics — there
is no CPU sim2sim gap. See [the Warp backend](../architecture/warp-backend.md).

```bash
MUJOCO_GL=egl .venv/bin/python -m rower_soccer.warp_port.train_follow_warp \
    --run-name follow_base --num-worlds 2048 --steps 500000000
```

Dribble warm-starts from a follow checkpoint (the decoder transfers):

```bash
MUJOCO_GL=egl .venv/bin/python -m rower_soccer.warp_port.train_dribble_warp \
    --run-name dribble_cur1 --init-from runs_v2/follow_base/latest.pt \
    --fixed-start --target-cone 0.0 --num-worlds 2048
```

See [Training drills](../training/drills.md) and
[Curriculum](../training/curriculum.md) for the full option set.

### CPU reference path

The original CPU/SB3 drill trainer still exists for reference, but every eval it
produced graded the policy under physics it never trained in — prefer Warp.

```bash
MUJOCO_GL=egl .venv/bin/python -m rower_soccer.train_drill \
    --drill follow --creature worm --steps 10000000 --procs 6 \
    --run-name follow_worm --device cuda
```

Training logs to wandb (project `creature-soccer`) with eval videos on a
wallclock cadence.

## 4. Drive the worm yourself (interactive play)

```bash
bash scripts/run_play.sh 8085
# then forward the port and open http://localhost:8085
```

`Q` = follow, `W` = dribble, click the arena to set a target, `Space` = stop,
`R` = reset. Full details: [the play server](../play-server.md).
