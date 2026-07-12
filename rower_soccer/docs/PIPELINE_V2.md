# Pipeline v2 — DeepMind-2021-faithful, with three strategic cuts

Supersedes the earlier WASD plan (PLAN v1 / CONTRACTS.md §1 command scheme).
Reference: Liu et al. 2022, "From Motor Control to Team Play in Simulated
Humanoid Football" (Science Robotics; papers in project_management/).

> **North-star doc.** For stages 1-2 the concrete engineering plan is
> `STAGE2_MULTITASK.md`, which supersedes three things here: it adds the
> **freeze-and-retrain** stage (missing below, but assumed by stages 4-6 —
> they all require a fixed z-space), it corrects "task introduction = config
> change" (line 91: only `follow` exists as a Warp env, and it has no ball —
> dribble/kick/shoot need a real physics ball), and it reconciles the §1-3
> CONTRACTS interface, whose input is now `z`, not `Command`.

## The architecture (one diagram to rule them all)

```
                        [aim target]        (drill envs / human mouse click)
                             │
   obs_task ─────────────┐  │
   obs_proprio ──┬─► Expert_k (per skill k ∈ {follow, dribble, kick, shoot})
                 │        │   4 separate weight sets
                 │        ▼
                 │   z_t ∈ R^d        d ≈ 16 for worm (configurable; 60 in paper)
                 │        │
                 └─► Low-Level Controller π(a | proprio, z)   ONE shared weight set
                          │
                          ▼
                    joint torques (2 for worm, 8 for rower)
```

- **Stage 1 cut (no mocap):** low-level controller is NOT pre-trained from
  motion capture. It is learned **jointly** with the experts via multitask
  curriculum RL: train `follow` alone → add `dribble` → add `kick` → add
  `shoot`. One task at a time, each gated by a fitness threshold + human
  video review. Four expert heads, one shared decoder, exactly one latent
  space z shared by all tasks.
- **Distillation stage (kept, same as paper):** after all four experts train,
  distill each into a target-agnostic **drill prior** (sees only obs available
  in football, no targets) by KL in z-space. Used as regularizers in the final
  RL fine-tune, exactly as the paper's Eq. 5-6.
- **Stage 3 cut (no PBT bootstrap):** the multi-agent stage is initialized by
  **human demonstrations → behavior cloning**, then lightly fine-tuned with
  self-play RL, instead of 8e10 steps of population self-play.

### How human play, BC, and the latent space fit together

Humans play a LoL-style interface: **click/drag sets an aim target, keys pick
the active skill** (follow-to-point / dribble-to-point / kick-to-point /
shoot). The chosen expert runs with that target and **emits z each timestep**.
We record `(football_obs_t, z_t)` — i.e., demonstrations live in the *latent
motor-intention space*, not in torque or command space. BC then trains a
football policy `π(z | football_obs)` directly on those latents. The final
agent is BC-initialized, acts in z-space through the same frozen low-level
controller, and is fine-tuned by self-play RL with (a) KL-anchor to the BC
policy and (b) optional KL to the mixture of drill priors (paper Eq. 5).
Smooth "in-between" behaviour = z-space interpolation, as in the paper.
Because the football policy never observes drill targets, the follow-expert's
future-target observation creates no train/deploy mismatch for BC.

## Stage roadmap

| # | Stage | Body | Gate |
|---|---|---|---|
| 1 | `follow` task env + joint expert/decoder RL | worm | fitness ≥ threshold + video review |
| 2 | +`dribble`, then +`kick`, then +`shoot` (curriculum) | worm | per-task fitness + video |
| 3 | Distill 4 experts → drill priors | worm | prior reproduces skill sans target (video) |
| 4 | Play UI (browser, click-drag + skill keys) + demo recording | worm | playtest feels controllable |
| 5 | Human 2v2 demos → BC in z-space | worm | BC team plays watchable 2v2 (video) |
| 6 | Self-play fine-tune: KL-to-BC + drill-prior mixture + shaping rewards | worm | passes survive; more dynamic play |
| 7 | Rower: rerun stages 1–3 (same code, new body); small homogeneous-worm 2v2 validation run, then **heterogeneous (worm+rower) teams** for the headline run | both | final video |

## Stage-1 concrete spec: `follow` on the 3-segment worm

The drill envs are NOT in dm_control's open-source soccer — we implement them
from the supplementary descriptions (Tables S2, S3):

- **Env:** worm on pitch, moving target. Target velocity sampled at episode
  start ("moves at fixed velocity, in variable directions"). Episode ~10-20s.
  V1: constant velocity vector per episode. V2 (variant to test): direction
  re-sampled every few seconds. Target speed range calibrated to worm's
  achievable speed (measure first with random policy).
- **Observations (egocentric, paper-style):** proprio (joint pos/vel, root
  orientation, gyro/velocimeter/accelerometer) + task context (current target
  position AND future target position(s) — paper gives the agent lookahead;
  we expose target at t and t+~1s, both egocentric).
- **Reward = fitness:** `exp(-0.5 * ||x_player - x_target||)` (Table S3;
  distance in meters — we may need to rescale the exponent for our pitch/body
  scale; treat the coefficient as tunable, fitness = same formula).
- **Nets (scaled from NPMP/MPO papers):** decoder MLP 3 layers (256 units for
  worm; paper used 1024 for 56-DOF), input proprio ⊕ z, Gaussian torque
  output. Expert: 2-layer MLP encoder(s) (+ small LSTM, paper-style) → Gaussian
  over z. Latent d=16 default; try {8, 16, 32}. Optional AR(1)-style temporal
  smoothness regularizer on z (α=0.95, NPMP prior analog) as a variant.
- **Trainer:** see MPO-vs-PPO decision (below). Modular `Trainer` interface
  so the optimizer is swappable without touching envs/architecture.
- **Multitask mechanics (stage 2+):** vectorized envs sampled per-task;
  gradients from all active tasks flow into the shared decoder; each expert
  updates only on its own task's data. Task introduction = config change +
  resume from checkpoint.
- **Deliverables:** training curves, checkpoint, rendered video (target +
  worm trace), and interactive `dm_control.viewer` script for live watching.

## Divergences from the paper (deliberate, recorded)

1. No mocap/NPMP pre-training (joint multitask RL instead) — justified by
   2-DOF/8-DOF bodies vs 56-DOF humanoid.
2. PBT replaced by: fixed hyperparameters + curriculum gates (stages 1-2),
   human-BC bootstrap + self-play (stage 6). PBT may return later if needed.
3. Population of 16 → single agent (+ checkpoint opponent pool in stage 6).
4. 2v2 attention module over player pairs → plain concatenation of the 3
   other players' features (at n=4 the attention buys little; revisit if we
   scale). LSTM policy/critic retained.
5. Nash-averaging fitness → simple win-rate/Elo vs checkpoint pool.
6. Bodies: worm (2 DOF) then rower (8 DOF); heterogeneous teams — beyond
   anything in the paper (their teams are homogeneous humanoids).

## Open items (waiting on user)

- MPO vs PPO decision (analysis in chat; recommendation: PPO for stage-1
  bring-up behind a swappable Trainer interface, MPO as fidelity upgrade).
- Success-threshold values for stage gates (propose after first training run).
