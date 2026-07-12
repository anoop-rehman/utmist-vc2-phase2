# Stage 1-2 Engineering Plan — Multitask Joint Training + Freeze

Concrete engineering plan under PIPELINE_V2.md (the north-star). Covers the
joint multitask phase (PIPELINE_V2 stages 1-2) and the **freeze-and-retrain**
phase, which PIPELINE_V2 omits but which everything from its stage 3 onward
silently assumes.

Reference: Liu et al. 2022, "From Motor Control to Team Play in Simulated
Humanoid Football" (Science Robotics; papers in `project_management/`).

---

## 0. Why multitask, and not the frozen-follow shortcut

The alternative we rejected: train `follow` alone, freeze its decoder, and
train the other three experts against that frozen z-space. That is cheaper and
would probably work *for the worm* — a 2-DoF body's action manifold is tiny
enough that `follow` likely exercises most of it, so a follow-only latent
plausibly spans the skill repertoire.

It does not survive the move to the 8-DoF rower. More joints means more
distinct motions that `kick` and `shoot` need but `follow` never visits, so a
follow-only decoder would fail to span the repertoire and the later experts
would be stuck trying to express skills the controller cannot produce.
Multitask does not make that bet: every skill contributes gradients to the
shared decoder, so the latent space is spanned by construction.

We pay a bit of upfront complexity for something that scales across
morphologies. **Revisit trigger:** at substantially higher DoF, on-policy
multitask exploration itself starts to strain, and the mocap/Unity-motion
warm-start (the "Option C" we deferred) becomes attractive again. The design
below is deliberately agnostic to how the decoder was initialized, so that
route stays open — see §8.

---

## 1. Mapping onto the paper

| This plan | Paper |
|---|---|
| Joint multitask phase (§4-5) | Stage 1 substitute. They obtained a task-agnostic low-level controller by distilling mocap; we obtain one by co-training across the four drills. Different data source, same product. |
| Freeze + retrain per task (§6) | Stage 2, verbatim. Drill experts trained by RL in a frozen z-space. |
| Distill experts → drill priors | Stage 3 (their Eqs. 4-5). Out of scope here; see PIPELINE_V2 stage 3. It is *why* the freeze phase must produce discrete per-task experts. |

---

## 2. Corrections to PIPELINE_V2

Three things in PIPELINE_V2 are stale or missing. This doc supersedes them.

**(a) The freeze stage is missing, and it is load-bearing.** PIPELINE_V2 goes
joint-multitask (stages 1-2) → distill drill priors (stage 3) with the decoder
never explicitly frozen. But stages 4-6 all assume a fixed decoder and a fixed
z-space: human play records `(football_obs, z)`, BC trains `π(z|obs)` on those
latents, and self-play acts in z-space "through the same frozen low-level
controller" (PIPELINE_V2:46-48). If the decoder is merely "whatever joint
training left it as" and keeps drifting through distillation, then the z-space
humans record demos in is not the z-space the final agent deploys into — a
silent train/deploy mismatch, and one that would present as mysteriously bad BC
rather than as an error. Freezing is not a nicety; it is the precondition that
makes everything downstream coherent. It belongs as its own gated stage between
PIPELINE_V2 stages 2 and 3.

**(b) PIPELINE_V2:91-92 massively understates the env work.** It says adding a
task in stage 2 is a "config change + resume from checkpoint." True of the
*trainer*; false of the *env*. Only `follow` exists as a Warp env today, and
follow has no ball — its targets are virtual, computed in torch
(`follow_env.py:80-82`, `scene.py:4-6`: "Drill targets are kinematic
abstractions... they never touch the physics"). `dribble`, `kick` and `shoot`
all require a ball as a **real physics entity the creature contacts**. That
means a ball body in the Warp scene, ball dynamics, contact, and per-task
obs/reward. It is the single biggest chunk of real work in this plan, and
"config flip" hides it entirely. See §3.

**(c) CONTRACTS.md is internally superseded but not updated.** PIPELINE_V2:3
says it supersedes "CONTRACTS.md §1 command scheme" — i.e. the whole WASD
`Command(a_cmd, r_cmd, kick)` interface is replaced by the z-space latent
interface. But CONTRACTS.md §1-3 still presents WASD as a frozen contract, and
§2's `LowLevelController.act(proprio_obs, command)` still takes a `Command`.
The controller's input is now **`z`**, not `Command`. Reconcile before someone
builds against the wrong interface. Concretely, CONTRACTS §2 should become:

```python
class LowLevelController(Protocol):
    creature_kind: str
    def reset(self) -> None: ...
    def act(self, proprio_obs: np.ndarray, z: np.ndarray) -> np.ndarray:
        """proprio + motor intention z -> actuator vector in [-1, 1]."""
```

with the anti-drift `git_sha` rule of §3 extended to a `controller_id`
(see §7). CONTRACTS §4-7 (high-level obs, demo format, HL checkpoint, env
factory) are unaffected.

**Smaller notes.** MPO-vs-PPO (PIPELINE_V2:112) stays open; this plan assumes
PPO since that is what is built, behind the swappable-`Trainer` seam the doc
already wants. The stochastic-z and LSTM-expert "variants" in PIPELINE_V2:82-86
are deferred for worm bring-up (2 DoF does not need them) — but see §9, which
is a real subtlety they interact with.

---

## 3. Env layer — the big piece

### What exists

`WarpFollowEnv` (single task, no ball). One Warp model, `num_worlds` copies,
world-synchronized episodes (global reset every `episode_steps`), physics
captured in one CUDA graph. That graph capture is where the 93× comes from and
we do not want to give it up.

### Recommended design: one scene, world-partitioned by task

Build a single Warp model of **creature + ball**, allocate `N` worlds, and
statically partition them into contiguous per-task groups (`N/4` each at full
curriculum; fewer groups while ramping). One physics batch, one graph capture.
`follow` simply ignores the ball.

- **`scene.py`** — add `build_creature_ball_scene()`: the existing creature XML
  plus a free-joint ball body. Extend `SceneMeta` with `ball_body`,
  `ball_qpos`, `ball_qvel`. Verify contact geometry actually works: the worm
  must be able to *push* the ball (ball mass/radius/friction vs. worm scale is
  a real tuning risk, not a formality — check it before anything else).
- **`multitask_env.py`** — `WarpMultiTaskEnv`, generalizing `WarpFollowEnv`:
  - `task_of_world: LongTensor[N]`, static per world.
  - **Obs.** Proprio stays exactly as-is (indices 0:37) — this is what the
    decoder sees, and keeping it byte-identical to `follow` preserves both
    checkpoint transfer and the CPU parity eval. The task block becomes a
    per-task slice:

    | task | task obs |
    |---|---|
    | follow | `target_ego(2)`, `target_ego_future(2)` |
    | dribble | `ball_ego(2)`, `ball_vel_ego(2)`, `target_ego(2)` |
    | kick | `ball_ego(2)`, `ball_vel_ego(2)`, `target_ego(2)` |
    | shoot | `ball_ego(2)`, `ball_vel_ego(2)`, `goal_ego(2)` |

    Zero-pad to a common `max_task_dim` so obs is one rectangular
    `[N, 37 + max_task_dim]` tensor; each expert indexes only its own task's
    columns. The decoder never touches these, so the padding is harmless.
  - **Reward.** `_reward()` dispatches per task-group, each writing into its
    own world-slice. `follow` reuses the existing modes verbatim. The other
    three follow the paper's Tables S2/S3 in shape — dribble = keep ball close
    + move ball toward target; kick = ball speed gain along facing within a
    window; shoot = ball velocity toward goal — with coefficients treated as
    tunable for our body/pitch scale, exactly as PIPELINE_V2:79-81 already says
    for follow. **Use potential-based (`progress`-mode) shaping wherever the
    task admits it**; the `follow` run already showed the naive `velshape`
    variant is hackable.
  - Keep world-synchronized episodes (global reset) — graph-friendly, unchanged.

### Fallback

If the partitioned scene fights graph capture: four separate
`WarpMultiTaskEnv` instances (one task each, `N/4` worlds), stepped in a loop,
sharing only the model class. More captures, more memory, trivially correct.
Try partitioned first; fall back if it turns fiddly. **Do not** let this
decision block §4-6 — the trainer interface is identical either way.

---

## 4. Model layer — minimal surgery

`LatentExtractor` already separates the expert side (`proprio_enc`, `task_enc`,
`expert`, `z_proj`) from the `decoder` and the `critic`. Refactor along that
existing seam:

- **`SharedDecoder`** = `decoder` + `action_net` + `log_std`. Exactly the slice
  `decoder_state_dict()` (`latent_policy.py:97`) already carves out. **Must
  expose `action_mean(proprio, z) -> a`** as a first-class method — see §9.
- **`TaskHead`** = `{proprio_enc, task_enc, expert, z_proj, critic, value_net}`,
  one per task, each carrying its own task-obs index range. Note the critic is
  per-task and takes `[proprio, task]` directly, bypassing the bottleneck — it
  already is separate in the current code, and that is what makes §5's per-task
  advantage normalization natural.
- **`MultiTaskActorCritic`** = `nn.ModuleList[TaskHead]` + one `SharedDecoder`.
  Forward takes `(obs, task_id)`, routes to the right head for `z` and value,
  shared decoder for the action. For a partitioned batch: run each head on its
  world-slice and scatter back — a Python loop over ≤4 tasks, negligible.

---

## 5. Trainer layer — multitask PPO

New `warp_port/ppo_multitask.py` (or generalize `PPOTrainer`).

**Equal weighting is a loss-level sum, not a reward-level one.** We do *not*
sum the four reward scalars — the four tasks live in different worlds with
different rewards and different critics, and a single creature never follows
*and* shoots in one episode. What we sum is:

```
total_loss = Σ_k β_k · L_k        (β_k = 1 by default)
```

where each `L_k` is the PPO clipped objective over task `k`'s minibatch. The
shared decoder receives the combined gradient from all four; each expert
updates only on its own task's data.

**Per-task advantage normalization is the mechanic that makes "equal" real.**
What determines whether one task captures the shared decoder is not reward
scale but **advantage** scale. A sparse `shoot` reward (zero most steps, one
spike on goal) and a dense `follow` reward (~0.5 every step) can have similar
per-step maxima yet wildly different advantage magnitudes — so normalizing the
rewards to a common range does *not* equalize gradient pull. Compute GAE per
task-group and **normalize advantages within each task** before pooling. (The
current single-task code already normalizes globally, `ppo.py:116` — this is
the same operation, done per group.)

When a task turns out under- or over-served — it will, on some task — the knob
to turn is its **`β_k`, not its reward coefficients.** Rewards define the task;
`β_k` defines its share of the gradient. Keep them separate.

**Decoder LR schedule.** Two Adam param groups: experts/critics at `lr`, the
shared decoder at `lr · decoder_lr_mult`, with `decoder_lr_mult` decaying
across the joint phase (e.g. 1.0 → 0.1). Let the experts move fast while the
shared substrate settles slowly, so the decoder is nearly stable by the time we
freeze it. This softens the moving-target cost and makes the freeze less of a
discontinuity.

**Buffer.** Each transition is tagged with its task id — free, since it is the
world's static task.

**Keep** the `ent_floor`/`ent_ceil` clamp on the shared `log_std`
(`ppo.py:140-142`).

---

## 6. Curriculum control

- Config carries `active_tasks`, starting `["follow"]`.
- **Advancing** = add a task to the list, allocate its world-slice, instantiate
  a **fresh** `TaskHead` for it, and resume: warm-start the decoder and the
  existing heads from the prior checkpoint, new head starts from scratch. This
  is exactly the case the run-dir/resume guard added in `2356c50` exists for —
  each curriculum leg is a `--resume` writing its own `config_resume_N.json`.
- **Gate** = per-task fitness threshold + video review (PIPELINE_V2's gate).
- **Log per-task eval independently, every interval.** With a shared decoder,
  the signature failure is one task silently regressing while another's
  gradients reshape the decoder, and a pooled reward number hides it. This is
  the same discipline as the per-task video evals, but now load-bearing rather
  than nice-to-have.

---

## 7. Freeze and retrain — the phase PIPELINE_V2 is missing

New entrypoint `train_frozen_expert.py`. Load the joint checkpoint, freeze the
`SharedDecoder` (`requires_grad_(False)`), and re-run single-task PPO for each
task, training **only** its `TaskHead` against the now-fixed z-space, to
convergence.

It does three distinct jobs:

1. **Recovers what the joint phase costs.** During multitask every expert was
   chasing a moving decoder, so no expert ever fully converged against a stable
   interface. Freeze it and each expert can be optimized cleanly. You typically
   get back the performance the moving target ate.
2. **Produces the deployable artifacts** in the split form below.
3. **It is the diagnostic.** If freezing the decoder and retraining an expert
   *cannot* recover good per-task performance, that is proof the joint phase
   produced an over-compromised decoder — a mediocre "average gait" serving
   none of the four well. That is the principal failure mode of shared-decoder
   multitask (negative transfer), and this is exactly where it surfaces.
   Escalation if it fires: more decoder capacity, rebalanced `β_k`, or (for
   higher-DoF bodies) the Unity-motion warm-start.

So it earns its keep in the happy path and is the alarm in the unhappy one.

### Checkpoint format

- **`checkpoint.pt`** — monolith, for resume. Unchanged in role; now holds N
  heads + shared decoder + optimizer.
- **`controller.pt`** — the `SharedDecoder`. **Immutable after freeze.** Plus
  `meta.json` with a **`controller_id`** = hash of its weights.
- **`expert_<task>.pt`** — that task's `TaskHead`, stamped with the
  `controller_id` it was trained against. The loader **refuses on mismatch**
  (CONTRACTS §3 anti-drift rule, applied to the z-interface).

Because the decoder is frozen *before* export, `controller_id` is stable by
construction — the version-stamping problem solves itself.

---

## 8. Sequencing

| # | Milestone | Proves |
|---|---|---|
| 1 | Ball in the Warp scene + single-task `dribble` env. Sanity-train it alone. | Ball contact and reward work, before any multitask complexity is layered on. |
| 2 | `MultiTaskActorCritic` + multitask trainer on `follow` + `dribble` only. Equal weight, per-task advantage norm. | The shared-decoder mechanics. Watch **both** eval curves. |
| 3 | Add `kick`, then `shoot` (curriculum, each gated). | Full drill set. |
| 4 | Freeze + per-task retrain; export split checkpoints. | Deployable artifacts + the negative-transfer diagnostic. |
| 5 | *(separate work)* Distill → drill priors — PIPELINE_V2 stage 3. | |

Milestone 1 first, deliberately: ball contact is the highest-uncertainty,
lowest-dependency piece, and everything else is wasted if the worm cannot move
the ball.

**Rower (8 DoF) later.** This pipeline — per-task expert + shared decoder +
freeze-and-retrain — is completely agnostic to *how the decoder got
initialized*. For the rower, keep the entire structure and, if pure on-policy
multitask struggles at 8 DoF, warm-start the decoder from the Unity-evolved
motion (mocap-style distillation) without changing anything downstream. The
joint phase then becomes optional fine-tuning on top of a data-initialized
controller rather than the sole source of it. Starting on-policy does not lock
us out of the mocap route — it builds the harness that accepts either.

---

## 9. Recorded now, acted on later: the stochasticity flip

**Do not act on this yet. Do honor it in the `SharedDecoder` interface.**

During drills, `z` is deterministic and the stochasticity lives in the Gaussian
action head — `Normal(action_net(lat), log_std.exp())`, `ppo.py:32`. PPO
explores by jittering the *torques* around what the decoder suggests. This
keeps PPO log-probs exact and is why `latent_policy.py:11-13` describes z as a
deterministic bottleneck.

Self-play (PIPELINE_V2 stage 6) must work the other way round. The football
policy's decisions *are* intentions — dribble left, turn, shoot — so it outputs
a distribution over **`z`** and samples it, and the frozen decoder maps each
sampled `z` to torques **deterministically**, with no torque jitter on top.
Exploration moves up a level: one random draw is one whole coordinated
behaviour attempt, not joint spasm. That is the entire payoff of having a
decoder, and it is only collected by putting the noise on `z`.

The hazard is stacking both noises: sample `z` *and* let the decoder sample
torques around its mean, and you get z-noise smeared with torque-noise, forcing
a retrain or surgery on a frozen artifact you were counting on never touching
again.

**Therefore:** `SharedDecoder` exposes a deterministic
`action_mean(proprio, z) -> a` from day one. Drills call
`Normal(action_mean(...), log_std)` and sample torques; self-play calls
`action_mean(...)` directly and lets the football policy's own `log_std` (on
`z`) supply all the randomness. The same frozen weights then serve both regimes
with **zero retraining**. This path already exists in spirit — it is what
`cpu_eval_video` uses when it takes `d.mean` — the point is that it must stay a
first-class, callable method on the decoder rather than something entangled
inside a `.dist()` that always samples.

Stages 1-2 do not care which way the noise flows. But they are where the
`SharedDecoder` interface gets designed, and that is why this is written down
now.
