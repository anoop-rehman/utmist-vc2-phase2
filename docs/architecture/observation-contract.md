---
title: The observation contract
---

# The observation contract

!!! danger "Read this before touching any observation"
    This is the single most load-bearing invariant in the project, and it was
    silently broken until 2026-07-14. The full, authoritative treatment is
    [STAGE2_MULTITASK.md §0.0](../design/stage2-multitask.md). This page is the
    summary.

Two inputs feed the policy, and they obey **opposite** rules.

## Proprio → the shared decoder (a HARD contract)

The [decoder](overview.md) is the one network reused across every drill, every
expert, and the 2v2 game. Its input must therefore be **identical everywhere**,
which means `proprio` may contain only things that:

1. **exist in the 2v2 game**, and
2. are **invariant to where on the pitch the creature is and which way it
   faces.**

!!! bug "What was wrong"
    Proprio used to carry `absolute_root_mat` (9) + `absolute_root_pos` (3) —
    the creature's absolute world orientation and position. Those violate rule
    2: they change with pitch location and heading, so a decoder trained on them
    learns pitch-position-dependent behavior that means something different at
    game time. The fix replaces them with **`world_zaxis`** (gravity direction
    in the body frame) + **`body_height`** — both invariant, both present in the
    game.

Current worm proprio is **29 dims**: joint pos/vel, `world_zaxis`,
`body_height`, gyro, velocimeter, accelerometer (scaled + clamped), and previous
action. It is position- and heading-invariant by construction.

## Task obs → the expert (free per skill)

Each expert may observe whatever its skill needs — the aim target, ball state,
lookahead — because task obs never reach the shared decoder. **But** anything
that survives distillation into a drill prior (pipeline stage 3) must be a
function of *game* observations, in the game's own form, since the prior is
evaluated on game observations to compute the KL.

!!! warning "Do not zero-pad drills to the game's obs"
    A tempting "fix" is to pad every drill's obs out to the game's 119-dim
    vector. Don't. A weight on an always-zero input gets gradient `δ·x = 0`,
    never leaves its random init, and then fires noise the moment that input goes
    live at game time.

### The ball observation

The ball obs went **2D → 3D** to match the game's `ball_ego_position(3)` +
`ball_ego_linear_velocity(3)`. The dribble env's observation layout is:

| block | dims | indices | routed to |
|---|---|---|---|
| `ball_ego` (pos 3 + vel 3) | 6 | `0:6` | expert (task) |
| `proprio` | 29 | `6:35` | decoder + expert |
| `target` | 4 | `35:39` | expert (task) |
| **total** | **39** | | |

The follow env is the same minus the 6 ball dims (**33 dims**); the play server
drives the follow policy with `obs[:, 6:]` so one env can serve both skills.

## Warm-starting and the index buffers

When dribble warm-starts from follow, the decoder weights transfer and the
task-encoder / critic re-init on shape mismatch. The proprio/task **index
buffers** (`p_idx`, `t_idx`) are *never* copied across a warm-start — a
`skip_buffers` guard prevents it — so the 6-shift between the follow and dribble
layouts never corrupts which observation dims route where. This was verified, not
assumed.

## Warp is a different simulator, not a fast MuJoCo

The observation contract has a physics twin: treat Warp as a **separate
simulator that must be calibrated against MuJoCo**, not as a drop-in fast MuJoCo.
It resolves contacts ~6.7× softer on byte-identical parameters, which a policy
will happily farm as free traction. See [the Warp backend](warp-backend.md).
