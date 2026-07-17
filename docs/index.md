---
title: Home
---

# Creature Soccer

GA-evolved creatures — from the [UTMIST Virtual Creatures](https://github.com/anoop-rehman/utmist-vc2-phase2)
Unity project — learn to play **2v2 soccer** in
[dm_control](https://github.com/google-deepmind/dm_control)'s MuJoCo soccer
environment. The method follows DeepMind's humanoid-football pipeline
([Liu et al., *Science Robotics* 2022](https://www.science.org/doi/10.1126/scirobotics.abo0235))
with three strategic cuts that make it tractable on a single GPU.

<div class="grid cards" markdown>

-   :material-rocket-launch: **Getting started**

    Install the stack and run your first drill, match render, and interactive
    play session.

    [:octicons-arrow-right-24: Installation](getting-started/installation.md)

-   :material-sitemap: **Architecture**

    The latent-bottleneck policy, the observation contract, and why Warp is
    ground truth.

    [:octicons-arrow-right-24: Overview](architecture/overview.md)

-   :material-map-marker-path: **Pipeline**

    The seven-stage roadmap from a single drill to heterogeneous 2v2 teams.

    [:octicons-arrow-right-24: Pipeline](pipeline.md)

-   :material-gamepad-variant: **Interactive play**

    Drive the worm in a browser, live, on Warp physics.

    [:octicons-arrow-right-24: Play server](play-server.md)

</div>

## The three cuts

The paper's recipe is faithful; three deliberate simplifications keep it small:

1. **No mocap stage.** The low-level controller is learned *jointly* with the
   four skill experts (*follow → dribble → kick → shoot*) via multitask
   curriculum RL, instead of being distilled from motion capture. Our bodies
   have 2–8 DOF, not 56.
2. **Simple bodies first.** The 3-segment worm (2 DOF) validates the whole
   pipeline; the same recipe then retrains for the two-arm rower (8 DOF). The
   headline goal is **heterogeneous teams** — worm defense + rower attack.
3. **Human demos instead of PBT bootstrap.** Humans play a LoL-style
   click-to-aim interface that drives the trained skill experts; the experts'
   latent motor intentions are recorded and behavior-cloned, then lightly
   fine-tuned with KL-anchored self-play — replacing ~8×10¹⁰ steps of
   population self-play.

## The one architecture diagram

```
                     [aim target]        (drill envs / human mouse click)
                          │
  obs_task ───────────┐   │
  obs_proprio ──┬──► Expert_k   (per skill k ∈ {follow, dribble, kick, shoot})
                │      │          4 separate weight sets
                │      ▼
                │   z_t ∈ R^d    d ≈ 16 for the worm
                │      │
                └──► Low-Level Controller  π(a | proprio, z)   ONE shared decoder
                       │
                       ▼
                 joint torques (2 for the worm, 8 for the rower)
```

The **decoder** is the one network reused across every drill, every expert, and
the eventual 2v2 game. What it sees — *proprio* — is therefore a hard contract.
See [the observation contract](architecture/observation-contract.md).

## Project status

- [x] `.creature` → MuJoCo conversion pipeline (NRBF parser + genotype
  expansion, numerically validated on the two-arm rower)
- [x] Follow + dribble drill envs; latent-bottleneck PPO with monitoring
- [x] **MuJoCo Warp** port — train, score, and render all in Warp (~94K steps/s,
  ~93× the CPU path), no sim2sim gap
- [x] Dribble **curriculum** (colinear stage 1 → 2D parking stage 2), warm-started
  from the follow policy
- [x] **Interactive play server** — drive the worm in a browser on Warp physics
- [x] **Fetch benchmark** — dm_control quadruped-fetch reproduced faithfully in
  Warp (reward parity 4e-8), then adapted to the worm and rower with a
  browser-labeled up-axis/rest-pose ([details](fetch.md))
- [ ] Kick + shoot drills, full multitask curriculum, expert→prior distillation
- [ ] Browser play UI for humans, z-space BC, self-play fine-tune
- [ ] Heterogeneous 2v2 showcase video

## How the docs are organised

- **Getting started / Training / Interactive play** — task-oriented guides.
- **Architecture** — the design decisions you must understand before changing
  observations, rewards, or the physics backend.
- **Design docs (canonical)** — the authoritative, in-tree design documents
  (`rower_soccer/docs/`) rendered verbatim. When a guide and a design doc
  disagree, the design doc wins.
