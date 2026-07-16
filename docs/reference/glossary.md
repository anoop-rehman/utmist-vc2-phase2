---
title: Glossary
---

# Glossary

**Proprio** — the shared decoder's entire input; must be present in the 2v2 game
and invariant to pitch position and heading. A hard contract. See
[the observation contract](../architecture/observation-contract.md).

**Task obs** — a per-expert observation (aim target, ball state); free to differ
per skill because it never reaches the shared decoder.

**Expert** — a per-skill network that maps `proprio ⊕ task` to a latent `z`. Four
of them (follow, dribble, kick, shoot); no shared weights.

**Decoder / low-level controller** — the one shared network mapping
`proprio ⊕ z` to joint torques. Reused across every drill, expert, and the game.

**z** — the latent bottleneck (~16 dims for the worm) between intent and action;
the space human demonstrations are recorded in.

**Fitness** — `exp(−c · ||x_player − x_target||)`; the follow/dribble reward and
eval metric (`eval/fitness_warp` in wandb).

**Warp / mujoco_warp** — GPU-batched MuJoCo; the ground-truth physics backend for
train, eval, and render. See [the Warp backend](../architecture/warp-backend.md).

**CUDA graph** — a recorded-and-replayed GPU step (`use_graph=True`); ~17× faster
at 1 world, used by the play server.

**solref** — MuJoCo's contact time-constant; tuned per object in `scene.py`
(creatures/ground stiff, ball softer). Both extremes caused NaN divergence.

**CAPS** — Conditioning for Action Policy Smoothness (Mysore et al.); the
action-smoothness regularizer.

**gSDE** — generalized State-Dependent Exploration; here, a state-dependent
action std for finer control (`--state-dependent-std`).

**Warm-start (`--init-from`)** — load a prior checkpoint's weights; the decoder
transfers across a task change, task-encoder/critic re-init on shape mismatch,
and index buffers are never copied.

**Curriculum** — the staged dribble geometry: colinear (stage 1) → 2D parking
(stage 2) → reaching (stage 3). See [Dribble curriculum](../training/curriculum.md).

**PBT** — Population-Based Training; the paper's mechanism for discovering
tactics, replaced here by human demos + KL-anchored self-play.

**BC** — Behavior Cloning; supervised imitation of human demonstrations in
z-space.

**Worm / rower** — the two bodies: 3-segment worm (2 DOF, defender) and two-arm
rower (8 DOF, attacker). The goal is heterogeneous teams of both.
