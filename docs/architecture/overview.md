---
title: Architecture overview
---

# Architecture overview

The whole system is one idea repeated: **factor motor control from intent**, so
that a single low-level controller can be driven by many skills, and eventually
by a human or a self-play policy — all through the same narrow interface.

## The latent-bottleneck policy

```
  obs_task ───────────┐
  obs_proprio ──┬──► Expert_k        (per skill k ∈ {follow, dribble, kick, shoot})
                │      │               each expert is its own weight set
                │      ▼
                │   z_t ∈ R^d          d ≈ 16 for the worm
                │      │
                └──► Decoder  π(a | proprio, z)   ONE shared weight set
                       │
                       ▼
                 joint torques
```

- **Expert (per skill).** Sees `proprio ⊕ task` and emits a latent `z` each
  timestep. Free to use any task-specific observation (the aim target, ball
  state, lookahead). There are four experts, one per skill; they never share
  weights.
- **Decoder (shared).** Sees `proprio ⊕ z` only — never a task observation — and
  emits the joint torques. This is the *low-level controller*. It is the one
  network reused across every drill, every expert, and the 2v2 game.
- **Latent `z`.** A 16-dim bottleneck (the paper used 60 for a 56-DOF humanoid).
  It is the only channel from intent to action, and it is what humans'
  demonstrations are recorded in (see the [pipeline](../pipeline.md)).

Because the decoder's input must be *identical everywhere* it is deployed, what
goes into `proprio` is a hard, load-bearing contract — read
[the observation contract](observation-contract.md) before touching any
observation.

## Why factor it this way

At our compute budget, end-to-end RL on the full 2v2 game converges to
tactic-less ball-chasing (Samtani et al. 2021 at ~40M steps vs Liu et al. 2019 at
40–80B frames). Factoring the problem:

1. **Low-level control** becomes a supervised-feeling RL problem (follow a
   target, shepherd a ball) that is learnable in tens of millions of steps.
2. **Strategy** moves to a space small enough that human demonstrations and a
   short self-play fine-tune are meaningful — instead of needing population-based
   training to *discover* passing and defending from scratch.

This is DeepMind's own 2022 humanoid-football architecture in miniature.

## The training backend: Warp is ground truth

Drills train, score, **and** render in [MuJoCo Warp](warp-backend.md) — a
GPU-batched physics engine running thousands of worlds in parallel (~94K
steps/s, ~93× the CPU path). Critically, eval and video run in the *same* Warp
physics the policy trained in, closing a class of bug where CPU evals graded
policies under physics they had never seen (Warp resolves contacts ~6.7× softer
than MuJoCo CPU on byte-identical parameters).

## The bodies

| Body | DOF | Role | Notes |
|---|---|---|---|
| 3-segment worm | 2 | defender / goalkeeper | validates the whole pipeline; shrunk to ~1.76 m / 22 kg for ball control |
| two-arm rower | 8 | attacker | same recipe, rerun after the worm |

Both are GA-evolved Unity creatures converted to MuJoCo XML in
`creature_configs/`. The headline result is **heterogeneous** teams: one worm +
one rower per side.

## Where the pieces live

See the [repo layout](../reference/repo-layout.md) for the file-by-file map. The
short version:

- `rower_soccer/warp_port/` — the GPU physics, envs, PPO, and play server (the
  active codebase).
- `rower_soccer/models/`, `drills/`, `envs/` — the latent policy, drill tasks,
  and soccer env factory.
- `rower_soccer/docs/` — the canonical design documents (rendered under
  *Design docs* in this site).
