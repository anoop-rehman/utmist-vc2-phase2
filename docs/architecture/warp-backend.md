---
title: The Warp backend
---

# The Warp backend (Warp is ground truth)

The drills train, are **scored**, and are **rendered** in
[MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp) — a GPU-batched
port of MuJoCo that steps thousands of worlds in parallel. There is no
dm_control / CPU transfer eval anywhere in the training loop.

Code: `rower_soccer/warp_port/`.

## Why score in Warp

Warp and MuJoCo CPU **do not agree**. On byte-identical parameters,
`mujoco_warp` resolves contacts ~6.7× softer, and the worm topples chaotically,
so two runs from identical states diverge exponentially (~0.4 m apart within
0.6 s, open-loop, same actions). Every eval number ever taken on the CPU drill
was therefore grading a policy under physics it had never trained in, and every
video showed a body behaving differently from the one being optimised.

Scoring in Warp makes the eval report the thing we are actually training. The
render module builds a **separate, render-only** MjModel (same scene + a visible
target marker), copies Warp's `qpos` into it, and takes a picture — no stepping,
no solver, no contacts. What you see is exactly the state that was simulated.

## Throughput

| Path | Steps/s | Notes |
|---|---|---|
| CPU / SB3 | ~1K | reference only |
| Warp, 2048 worlds | ~85–95K | ~93× the CPU path |
| Warp, 1 world + CUDA graph | ~5.5 ms/step | ~17× vs no graph (95 ms), for the play server |

**CUDA graph capture** (`use_graph=True`) records the step once and replays it,
removing per-launch overhead — decisive at 1 world (the play server), negligible
at 2048.

## Calibration and stability (the fights we had)

Getting the worm to train in Warp without diverging to NaN took several fixes,
all recorded here because they are easy to reintroduce:

### Contact stiffness (`solref`)

`scene.py` sets an explicit contact time-constant. Both extremes killed runs:

- `0.005` → NaN at 17.7M steps (too stiff)
- `0.020` → NaN at 106M steps (too soft)

Settled on `WARP_SOLREF_TIMECONST = 0.005` for creature/ground/walls/goals, and
a **separate, softer** `WARP_BALL_SOLREF_TIMECONST = 0.010` for the ball (the
ball has `priority=1`, so its solref governs any contact it is in). The ball was
excluded from the creature stiffening specifically because it was the thing
diverging to NaN.

### The real NaN fix: sanitize at the network input boundary

The decisive guarantee is `_clean()` in `ppo.py`, applied to every observation
before it enters the policy:

```python
torch.nan_to_num(obs, nan=0, posinf=100, neginf=-100).clamp(-100, 100)
```

The failure mode was subtle: **large-but-finite** obs (an accelerometer spike of
8168, a ball divergence) detonated `exp(logp - logp_old)` in the PPO ratio, not
`inf`/`nan` obs. Clamping at the input boundary is what actually holds. The
accelerometer is *also* scaled (`/100`, clamped ±50) at the env, and any world
whose `qvel` exceeds 500 (or goes non-finite) is reset to rest before obs/reward
are computed (`_sanitize()`), so one diverged world can't kill a whole batch.

### noslip

`arena.xml` sets `noslip_iterations 0` (was 5). The CPU noslip solver was the
largest single source of CPU-vs-Warp disagreement; turning it off stopped CPU
evals from lying about Warp policies.

## The arena

Drills run on **dm_soccer's pitch** (96×72 m, walls, goals) — the same geometry
as the eventual 2v2 game, not a flat floor — so nothing about pitch scale or
boundaries changes between drill and game. A fixed `topdown` camera is added for
the play server, with `offwidth/offheight` bumped to 1024 for a large offscreen
framebuffer.

## Regularizers

Beyond raw fitness, the Warp trainers add (all tunable):

- **Energy / control-cost penalty** — discourages the worm's spazzing motion.
- **CAPS action smoothness** (Mysore et al.) — penalizes jerky action changes.
- **Entropy annealing** with a floor/ceiling — fixes policy collapse without
  freezing exploration too early.
- **Reward clipping** and an annealed **shaping scale** (park-not-velocity).
- **Keep-best checkpoint** — `best.pt` written on every new fitness peak.

## Headless rendering caveat (EGL + CUDA)

mujoco's EGL context and Warp's CUDA context (including a captured CUDA graph)
must be created **and** used on the same thread, or `eglMakeCurrent` throws
`EGL_BAD_ACCESS`. The [play server](../play-server.md) is architected entirely
around this constraint.
