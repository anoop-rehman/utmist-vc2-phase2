---
title: Dribble curriculum
---

# Dribble curriculum

Dribble does not learn from scratch — for hundreds of millions of steps the worm
just sits still, because the reward gradient is flat until it *happens* to touch
the ball toward the target. The fix is a curriculum that starts the worm in a
geometry where accidental success is likely, then relaxes it.

## Stage 1 — colinear

The worm, the ball, and the target are placed **colinear**: the ball sits
directly on the line between the worm and the target, right in front of the worm,
so that simply moving forward pushes the ball toward the goal. Only the target's
distance is at the normal follow range (2–5 m); the worm's heading and the ball
are fixed relative to it.

```bash
MUJOCO_GL=egl .venv/bin/python -m rower_soccer.warp_port.train_dribble_warp \
    --run-name cur1_ours_ws --init-from runs_v2/follow_base/latest.pt \
    --fixed-start --target-cone 0.0 --num-worlds 2048
```

Implementation: a single shared angle `theta = rand·2π` sets the worm yaw, the
ball direction, and the target angle (`theta + offset`, with `offset = 0` at
stage 1). `--target-cone` is the half-angle of the offset cone: `0.0` is
perfectly colinear.

!!! note "A geometry lesson learned the hard way"
    An earlier stage-1 design put the target at 0.5–1.5 m — which *inflated*
    fitness just from the spawn — and fixed the ball at +x with a random target
    angle. That let the policy score without learning to dribble. The corrected
    design keeps the normal target distance (spawn fitness back to ~0.189) and
    makes the geometry genuinely colinear.

## Stage 2 — 2D parking

Widen the target cone so the target moves *off* the colinear line: the worm must
now steer the ball, not just push it straight. Warm-start from the stage-1
checkpoint.

```bash
MUJOCO_GL=egl .venv/bin/python -m rower_soccer.warp_port.train_dribble_warp \
    --run-name cur2_ws --init-from runs_v2/cur1_ours_ws/latest.pt \
    --fixed-start --target-cone 1.5708 --num-worlds 2048
```

## Stage 3 (planned) — reaching

Randomize the ball's position relative to the worm, so the worm must first go to
the ball before dribbling it. Introduce when stage 2 plateaus.

## The plain-MLP baseline (the control experiment)

To prove the wall was the **task structure** and not the
[latent-bottleneck architecture](../architecture/overview.md), a plain-MLP
baseline (`SimpleActorCritic` — two 256-unit ELU trunks, no latent, no shared
decoder) was trained on the same task. It stalled at the fitness floor (~0.191)
while the latent policy climbed to **0.687**. That vindicates the architecture:
both hit the same wall from scratch, but only the curriculum — not a different
network — gets past it.

## What worked

| Run | Result |
|---|---|
| dribble from scratch (either architecture) | stuck at the ~0.19 fitness floor |
| plain-MLP + curriculum | still stuck (~0.191) |
| latent policy + colinear stage-1 + follow warm-start | climbed to **0.687** |

The combination that unlocked dribble: **latent policy + warm-start from follow +
colinear curriculum**.
