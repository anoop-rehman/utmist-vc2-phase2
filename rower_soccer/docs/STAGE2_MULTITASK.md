# Stage 1-2 Engineering Plan — Multitask Joint Training + Freeze

Concrete engineering plan under PIPELINE_V2.md (the north-star). Covers the
joint multitask phase (PIPELINE_V2 stages 1-2) and the **freeze-and-retrain**
phase, which PIPELINE_V2 omits but which everything from its stage 3 onward
silently assumes.

Reference: Liu et al. 2022, "From Motor Control to Team Play in Simulated
Humanoid Football" (Science Robotics; papers in `project_management/`).

---

## 0.0 The observation contract — read this before touching any obs

The single most load-bearing invariant in the project, and the one that was
silently broken until 2026-07-14. Two inputs, opposite rules:

**Proprio → the shared decoder. This is a HARD contract.** The decoder is the one
network reused across every drill, every expert, and the 2v2 game. Its input must
therefore be *identical everywhere*, which means it may contain only things that

1. **exist in the 2v2 game**, and
2. are **invariant to where on the pitch the creature is and which way it faces.**

Proprio used to carry `absolute_root_mat` (9) + `absolute_root_pos` (3) — the
creature's global yaw and global xy. Both rules violated. It let the decoder learn
gaits that only work at one spot facing one way, and dm_soccer does not supply
those quantities at all: it gives `world_zaxis` (gravity in the body frame) and
`body_height`. So the "reusable low-level controller" — the load-bearing premise of
this entire pipeline — was built on 12 inputs that would not exist at deployment.

Fixed: `world_zaxis` + `body_height` replace them. Proprio **37 → 29**. That is
what a motor controller actually needs (which way is up, how high off the ground)
and nothing more.

**Task obs → the per-task expert. Free to differ.** Four experts, four separate
input layers, no shared weights — nothing forces a common shape. Follow sees a
target; dribble sees ball + target; shoot will see ball + goal.

**But: anything that survives DISTILLATION must be a function of game obs, in the
game's own form.** This is the constraint people miss. The drill experts are
scaffolding — they get distilled into reduced-observation *drill priors* (stage 3),
and the football policy's KL is computed by evaluating those priors **on game
observations**. So every obs that survives into a prior has to be feedable from the
game.

Worked example, and a real bug we caught this way: `ball_ego` was 2-D (ego xy
position + ego xy velocity). The game gives `ball_ego_position` **(3)** +
`ball_ego_linear_velocity` **(3)**. The ball obs is the core of both the dribble and
shoot priors, so it survives distillation — and a 2-D obs cannot be fed from a 3-D
one. Every drill would have trained against a dead-end interface, discovered only at
stage 3. Now 3-D, matching the game. Same reasoning says shoot's goal obs must use
the game's `opponent_goal_mid` (3) and goal-corner representation.

The drill-only obs (`target_ego`, `target_ego_future`) are fine: distillation is
precisely what *removes* them. The follow prior drops the target and becomes
"locomote"; the dribble prior drops it and becomes "move the ball". That is the
bridge from drills to football, and it is why the freeze phase produces priors
rather than shipping the experts.

### Why NOT to zero-pad the drills to the game's 119-dim obs

A tempting "fix": give every drill the full game obs space, zeroing the irrelevant
fields (teammates, opponents, score), so everything matches. **This is worse than
useless.**

The gradient w.r.t. a weight is `δ · x`. If `x` is always 0, the gradient is always
**exactly** 0 — so those weights never move from their random initialisation. Then
at game time the teammate obs goes nonzero, gets multiplied by never-trained random
weights, and injects noise straight into the expert. You would ship a network with
90 dimensions of latent garbage, primed to fire the moment it matters.

**An input that is always zero is not ignored. It is untrained.** Obs spaces are
*supposed* to differ per task; distillation reconciles them.

---

## 0.1 Backend calibration — Warp is not MuJoCo, and must be tuned to match

`mujoco_warp` resolves contacts **~6.7x softer** than MuJoCo CPU on *byte-identical*
parameters — same `solref`/`solimp`/`condim`/friction, same 0.0025 timestep, same 10
substeps, same Newton solver, same 100 iterations. Every `opt` field and every geom
field was compared; nothing was misconfigured.

    mean floor penetration     Warp -2.28 cm      CPU -0.34 cm

It is **not** under-convergence: raising `iterations` 100 → 500 and `ls_iterations`
50 → 200 moves penetration by <5%. At convergence, penetration is set by constraint
**compliance**, not iteration count — so `solref` is the only lever.

This matters more than it sounds, because **the worm propels itself entirely by
pushing against the ground.** Sinking 2 cm into the floor is free traction, and the
policy learns to farm it: stiffening contacts to CPU's level costs the old follow_v2
**175 reward** in Warp (413 → 238). That soft-contact exploit — not noslip — was the
bulk of the sim2sim gap, and it is visibly why old eval videos looked like the worm
was glitching *through* the floor rather than slithering *on* it.

**Fix:** `WARP_SOLREF_TIMECONST = 0.005` in `warp_port/scene.py`, applied post-compile
to every geom. Lands Warp at -0.317 cm against CPU's -0.342 cm. Warp's `solref` now
**deliberately differs** from the CPU drill's 0.02 — the two backends need different
nominal values to produce the same physics. Do not "fix" this back. 0.005 is
2 x timestep, MuJoCo's documented stability floor; verified stable over 600 steps x
256 worlds of random torque, zero NaNs, `max|qvel|` unchanged.

**Generalise the lesson:** treat Warp as a *different simulator that must be
calibrated against MuJoCo*, not as a fast MuJoCo. Any new contact-bearing entity
(the ball, the goal posts, the rower's future limbs) should have its penetration
checked against CPU before a long run is launched on it.

---

## 0.2 The arena — dm_soccer's pitch, in both sims

Both drills train on the **same arena the 2v2 game uses**: dm_soccer's `Pitch`,
half-extents (48, 36) → 96 x 72 m ground, four bounding walls, two goals (posts at
y = ±11.88, crossbar at z = 5.33). `drills/follow.py` uses the real `Pitch`;
`warp_port/scene.py`'s `_BASE_XML` carries the same geometry, transcribed from the
compiled dm_control model rather than hand-typed, so the two scenes are literally
the same world.

At the drills' ±10 m bounds the worm reaches no wall and no goal, and the pitch
ground's contact params are byte-identical to the old `floors.Floor`. So
follow/dribble physics are **unchanged** by the swap. What it buys: no arena shift
at transfer, and `shoot` has a real goal to aim at. Cost, measured: the extra 24
geoms under `mujoco_warp`'s n-by-n broadphase take 2048-world throughput from 268k
to 252k env-steps/s (~6%).

---

## 0.3 Record of corrections — claims that did NOT survive scrutiny

Kept because re-deriving them costs more than reading them.

**`53971b6`'s "+44.5 reward from noslip=0" does not replicate.** On 64 episodes it
is **+18.2 ± 14.0** — not significant. `noslip=0` is still *correct* (Warp does not
implement the noslip solver, so the CPU eval must not either), but it was a minor
contributor, not the fix. The real one was contact stiffness (0.1 above).

**Trajectory-level sim2sim parity is unattainable on this creature, and that is not
a bug.** The worm spawns as an unstable vertical stack and topples chaotically, so
float32 Warp and float64 MuJoCo diverge exponentially from identical initial states
regardless of any physics flag: replaying an *identical action sequence* from an
*identical state*, the two are 0.4 m apart within 0.6 s and 6 m apart at 15 s — with
`noslip=0` no better than `noslip=5`. **Validate sim2sim distributionally (many
episodes, aggregate statistics), never by matching trajectories.** A trajectory-
divergence test here measures the Lyapunov exponent, not the physics.

**The "CPU under-reports Warp by 41 reward" figure was contaminated and is
withdrawn.** That harness forced `qpos_z = spawn_z` into both sims — but CPU's root
body carries a +0.2613 m frame offset while Warp's does not, so it dropped the CPU
worm from 0.26 m too high. The **envs are fine** (both natively spawn the root at
0.2613 m); the test was not. The penetration and stiffening numbers in 0.1 are
unaffected (native resets, no forcing). **The true cross-sim reward gap is currently
unmeasured** and must be re-measured after the retrain, with native resets.

Cheap guard against repeating this: when forcing state across the two sims, force
**`xpos`**, not `qpos`, or assert the root heights match after the reset.

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

## 0.5 Scale — the mass ratio, and why the worm is being lightened

> **OUTCOME (read first).** Mass scaling alone was tried and **was not enough**.
> It leaves the creature's locomotion invariant (F = ma with force and mass both
> scaled by k) and it fixed the mass ratio, but it did **not** stop the ball being
> launched: in a damped collision the ball departs at roughly the contact speed of
> whatever hit it for as long as that segment outweighs it, and even at 12 kg total
> the tail segment is 8.5 kg against a 0.045 kg ball. Launching is set by the tail's
> **tip speed**, and only shrinking the body reduces that (tip speed ~ sqrt(s)).
>
> The worm was therefore **shrunk**, not merely lightened: `unity2mujoco
> --length-scale 0.1768 --gear-scale 0.03` → **1.76 m / 22.01 kg**, ball:worm
> 1:489 (dm_soccer's ratio), footprint radius 4.65 m → 0.82 m. Measured: max ball
> speed 13-23 m/s → 7.0 m/s, max displacement 24-75 m → 7.5 m, ball comes to rest
> in 100% of worlds (2.2 s) vs rarely. The ball is now controllable.
>
> The rest of this section is kept for the reasoning and the measurements, but the
> conclusion it reaches ("lighten, keep the size") was superseded.

### What we tried, and what it cost

The ball is DeepMind's and is internally consistent: `SoccerBall` at 0.35 m /
0.045 kg, sized against the goal (ball:goal-width 0.035, vs 0.030 in real
football). The worm came out of Unity at **9.95 m / 3981 kg**. That is a
ball:player mass ratio of **1:88,485**, against dm_soccer's own 1:489 and real
football's 1:163 — the ball is ~180x lighter, relative to the body, than a
football should be.

We accepted that deliberately, to keep the 691M-step follow checkpoint valid so
dribble could warm-start from it, and ran three dribble configs to find out
empirically whether it mattered. **It does.** Measured with the warm-started
policy on the dribble drill (512 worlds, one 15 s episode):

| | |
|---|---|
| worm gets within 2 m of the ball | **10% of worlds** |
| worlds where the ball moves at all | 12.9% |
| ball displacement | mean 0.85 m, **max 74.9 m** (drill bounds are ±27 m) |

So the ball is mostly untouched, and on the occasions it *is* touched it gets
ejected clean out of the drill. The tell in the training logs was subtle and
worth remembering: **all three runs reported near-identical `fitness` at every
step** (0.113/0.115/0.117 → 0.166/0.166/0.165 → 0.107/0.107/0.107) despite three
different reward functions. Fitness that does not depend on the policy means the
policy is not touching the ball — what was actually being plotted was the target
drifting away from a stationary ball, modulated by episode phase.

### The fix being applied: lighten the worm, keep its size

Scale **mass and torque by the same factor k**, leaving all lengths alone.

This is not a compromise, it is the physically clean lever. With force and mass
both scaled by k, F = ma gives identical accelerations; gravity is an
acceleration and does not care about mass; ground friction scales too (μN ∝ m,
and the force needed to accelerate ∝ m). **The worm's own locomotion is exactly
invariant** — same gait, same speed, same everything. What changes is the
momentum ratio against the ball, which does *not* scale, and that is precisely
the quantity that is broken.

So it fixes the ball interaction at zero cost to the creature, and the follow
policy should transfer nearly intact. (One caveat: touch sensors read contact
*force*, which scales with mass, so that slice of the proprio observation does
change. Everything else — joint angles, velocities, gyro, accelerometer — is
unchanged.)

`tools/unity2mujoco.py --mass-scale` implements it: density ∝ k, gear ∝ k,
armature ∝ k, joint damping ∝ k, stiffness ∝ k. Pick k by measuring, not by
arithmetic — the contacting *segment's* mass is what governs the collision, and
the worm's 22 kg would be spread across three segments, so matching BoxHead's
total mass is not automatically the right target.

Measured on this worm (`warp_port/probe_speed.py`), and unchanged by mass
scaling:

| quantity | value |
|---|---|
| achievable speed | **2.83 m/s** (0.28 body-len/s; best gait 0.75 Hz, amp 1.0, 225° phase) |
| random-policy speed | 0.20 m/s — the floor, not the ceiling |

Follow-drill constants (target speed (0.25, 2.0) m/s, spawn 2–6 m, bounds 27 m)
are catchable against 2.83 m/s, so they stand. They live in
`warp_port/follow_env.py` and `drills/follow.py` and **must stay in step** — the
CPU drill is the transfer/parity eval, so a mismatch reads as a phantom sim2sim
gap.

### The other lever: shrink the worm (`--length-scale`)

Kept available, not currently used. `--length-scale` does a Froude-similar
rescale (mass ∝ s³, torque ∝ s⁴, armature ∝ s⁵, joint damping ∝ s^4.5,
stiffness ∝ s⁴). At `--length-scale 0.1768 --gear-scale 0.03` the worm becomes
1.76 m / 22.01 kg — BoxHead's mass exactly, and dm_soccer's 1:489 ratio to the
digit. Ball physics checks out there (`probe_ball.py`): ball settles at rest,
12.9% of random-action worlds displace it at 0.2–5.4 m/s, and a 5 m/s ball comes
to rest in 2.2 s.

Why we are not using it:

- It **changes the creature**, so every drill constant must be recalibrated with
  `probe_speed.py`. At 1.76 m the worm's achievable speed drops to 1.64 m/s, so
  the existing 2.0 m/s follow target cap becomes *uncatchable* and the drill
  unlearnable no matter how good the policy is — with nothing in the training
  loop to report it. Reward just stays low and reads as "needs more steps".
- It is **not** behaviourally similar. `solref` and the timestep are pinned to
  dm_control's convention (which the ball shares — a creature with a different
  contact timescale from the ball in the same scene would be incoherent) rather
  than Froude-scaled, and worm locomotion *is* ground contact. Measured: the
  small worm runs at 0.93 body-len/s vs the original's 0.28.

Mass scaling has neither problem, which is why it is the one we reach for.

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

*(Updated 2026-07-14 — this section originally said "no ball". Milestone 1 is done.)*

`WarpFollowEnv` (33-dim obs: proprio 29 + task 4) and `WarpDribbleEnv` (39-dim:
ball 6 + proprio 29 + target 4), both on dm_soccer's pitch, both with a
CPU dm_control twin verified at **element-wise obs parity** (max |warp − cpu| =
1.6e-05, float32 rounding). One Warp model, `num_worlds` copies,
world-synchronized episodes (global reset every `episode_steps`), physics
captured in one CUDA graph. That graph capture is where the 93× comes from and
we do not want to give it up.

`scene.py` already builds creature **+ ball** (`BallSpec` = dm_control's
`SoccerBall` verbatim: r=0.35, m=0.045, `condim=6`, `priority=1` — both of those
last two are load-bearing, see the comments there) and carries the full pitch.
`SceneMeta` already has `ball_body` / `ball_qpos` / `ball_qvel`.

So the multitask env work below is no longer "add a ball to the scene" — that is
done and validated. What remains is the world-partitioning and the per-task
reward dispatch.

Still missing: `kick` and `shoot` envs, the multi-expert model, the multitask
trainer, per-task advantage normalisation, and the freeze/retrain entrypoint.

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

| # | Milestone | Status | Proves |
|---|---|---|---|
| 1 | Ball in the Warp scene + single-task `dribble` env. Sanity-train it alone. | **env done; drill unproven** | Ball contact and reward work, before any multitask complexity is layered on. |
| 2 | `MultiTaskActorCritic` + multitask trainer on `follow` + `dribble` only. Equal weight, per-task advantage norm. | not started | The shared-decoder mechanics. Watch **both** eval curves. |
| 3 | Add `kick`, then `shoot` (curriculum, each gated). | not started | Full drill set. |
| 4 | Freeze + per-task retrain; export split checkpoints. | not started | Deployable artifacts + the negative-transfer diagnostic. |
| 5 | *(separate work)* Distill → drill priors — PIPELINE_V2 stage 3. | not started | |

**Milestone 1 status, precisely.** The *env* is done: the ball is in the scene,
physics is validated (`probe_ball.py`: settles, 12.9% of random-action worlds
displace it at 0.2-5.4 m/s, a 5 m/s ball comes to rest in 2.2 s in 100% of worlds),
and Warp/CPU obs parity holds. The *drill* is not yet proven — no dribble run has
ever produced a policy that touches the ball.

**The gate for milestone 1 is ball-contact rate, not reward.** Every dribble
attempt so far failed the same way and it is worth knowing the signature: all three
runs reported *near-identical fitness at every step despite three different reward
functions* (0.113/0.115/0.117 → 0.166/0.166/0.165). **Fitness that does not depend
on the policy means the policy is not touching the ball** — what was being plotted
was the target drifting away from a stationary ball. Three "different" curves lying
on top of each other is a stronger signal than any one of them being flat.

The completed `dribble_v2_progress` run (800M steps) was graded and is **dead**:
0/12 episodes touched the ball, ep_rew 152.7 vs 152.6 for a do-nothing policy,
fitness 0.268 vs 0.267. Statistically indistinguishable from doing nothing.
Do not resume it.

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
