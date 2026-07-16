---
title: Pipeline
---

# The pipeline

The full method is a seven-stage roadmap from a single drill to heterogeneous
2v2 teams. This page is the map; the authoritative specs are
[Pipeline v2](design/pipeline-v2.md) (north star) and
[Stage 1-2 Multitask](design/stage2-multitask.md) (the concrete engineering
plan for stages 1–2, including the freeze-and-retrain step that Pipeline v2
omits but everything downstream assumes).

## Stage roadmap

| # | Stage | Body | Gate |
|---|---|---|---|
| 1 | `follow` task env + joint expert/decoder RL | worm | fitness ≥ threshold + video review |
| 2 | +`dribble`, then +`kick`, then +`shoot` (curriculum) | worm | per-task fitness + video |
| 3 | Distill 4 experts → target-agnostic drill priors | worm | prior reproduces skill sans target |
| 4 | Browser play UI (click-drag + skill keys) + demo recording | worm | playtest feels controllable |
| 5 | Human 2v2 demos → behavior cloning in z-space | worm | BC team plays watchable 2v2 |
| 6 | Self-play fine-tune: KL-to-BC + drill-prior mixture + shaping | worm | passes survive; more dynamic play |
| 7 | Rower: rerun stages 1–3 (same code, new body), then **heterogeneous 2v2** | both | final video |

Each stage is gated by a fitness threshold **and** human video review — the final
checkpoints are selected by *watching footage*, not by a scalar.

## How human play, BC, and the latent space connect

Humans play a LoL-style interface: **click/drag sets an aim target; keys pick the
active skill** (follow / dribble / kick / shoot). The chosen expert runs with
that target and **emits `z` each timestep**. We record `(football_obs_t, z_t)` —
demonstrations live in the *latent motor-intention space*, not in torque or
command space. BC then trains a football policy `π(z | football_obs)` on those
latents. The final agent is BC-initialized, acts in z-space through the same
frozen decoder, and is fine-tuned by self-play RL with a KL-anchor to the BC
policy and an optional KL to the mixture of drill priors.

Because the football policy never observes drill targets, the follow expert's
future-target observation creates no train/deploy mismatch for BC.

## Deliberate divergences from the paper

1. **No mocap / NPMP pre-training** — joint multitask RL instead (justified by
   2–8 DOF bodies vs a 56-DOF humanoid).
2. **PBT replaced** by fixed hyperparameters + curriculum gates (stages 1–2) and
   human-BC bootstrap + self-play (stage 6).
3. **Population of 16 → single agent** (+ a checkpoint opponent pool in stage 6).
4. **Attention over players → plain concatenation** of the three other players'
   features (at n=4 attention buys little).
5. **Nash-averaging fitness → simple win-rate / Elo** vs the checkpoint pool.
6. **Bodies:** worm then rower, and **heterogeneous** teams — beyond the paper's
   homogeneous humanoids.

## Where the project is now

Stages 1–2 are the active work. Follow trains robustly in
[Warp](architecture/warp-backend.md); dribble learns via a
[curriculum](training/curriculum.md) warm-started from follow. The
[interactive play server](play-server.md) is an early, Warp-native version of the
stage-4 play UI (worm only, follow + dribble stage 1). Kick, shoot, distillation,
human BC, and self-play are not yet built.
