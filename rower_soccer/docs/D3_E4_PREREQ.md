# D3 M3 E4 prerequisite — what can Transform2Act's design head actually see?

*2026-09-05. `PLAN_D3_M3.md`'s E4 rung records the requirement: D2 found a
shared design head cannot condition on role or opponent at all until
`--role-in-design` was added (SMD 0.110 → 0.833), and **"the equivalent question
here is what Transform2Act's design head can see. It must be checked before
this rung, not after."** This is that check. **No E4 code has been written and
nothing has been launched.***

## The answer: it sees nothing about the simulation. At all.

`design_opt/models/transform2act_policy.py` lines **170** (attribute stage) and
**194** (skeleton stage) both begin with the same slice:

```python
obs = torch.cat((obs[:, :self.attr_fixed_dim], obs[:, -self.attr_design_dim:]), dim=-1)
```

and the dimensions confirm what that discards:

```
control_state_dim = attr_fixed_dim + sim_obs_dim + attr_design_dim
skel_state_dim    = attr_fixed_dim +                attr_design_dim
attr_state_dim    = attr_fixed_dim +                attr_design_dim
```

On our run-to-goal ant the node row is **25 columns = 4 attr_fixed + 16 sim_obs
+ 5 attr_design**. The control head receives all 25. **The design heads receive
9** — the 4-column body-depth one-hot and the body's own 5-column attribute
genome. **The entire 16-column `sim_obs` slice is dropped**, and that slice is
where every appended task column lives: `(opp_dx, opp_dy, goal_dx)`, plus all
qpos and qvel.

**Measured, not just read.** Moving the opponent 4 metres (x = +1.0 → −3.0) and
recomputing the observation:

| | max \|Δ\| |
|---|---|
| **design-head input** (the 9 columns) | **0.000e+00** |
| dropped `sim_obs` slice | 4.000e+00 |
| the 3 appended columns | `(+2.00, ~0, +5.00)` → `(−2.00, ~0, +5.00)` |

> **The design head is blind to the opponent — and to the goal, to its own
> joint angles, and to its own velocity. It sees only its own body's structure
> and parameters.**


### This is upstream's architecture, not a port defect

Worth stating plainly, because *"Transform2Act's design head is blind by
design"* and *"our port lost the columns"* are different claims and only the
first is true. `transform2act_policy.py` is **unmodified from upstream**:

```
$ git status --short design_opt/models/transform2act_policy.py   # (empty)
$ git log --oneline -1 -- design_opt/models/transform2act_policy.py
09fc902 initial code release!
```

Our only `design_opt`/`khrylib` edits are `envs/__init__.py`, `envs/ant.py`,
`envs/hopper.py` and `robot/xml_robot.py` — none of them touches the policy.
The class *declares* the asymmetry in its own constructor: execution
normalises the full observation, both design stages drop `sim_obs` by
construction. The slice is intended behaviour of the published method.

## The retrospective consequence — the most important thing here

**E3 and E3.1 evolved their bodies blind to the task.** No knowledge of the
opponent, the goal line, the distance remaining, their own joint angles or
their own velocity. `D3_E2_RTG.md` §2 introduced the three appended columns as
*"CompetEvo's information content in a translation-invariant frame"* and noted
"both arms are fed exactly these columns" — **true of the control head, false
of the design heads**, which had already sliced them away.

The only channel from task to morphology was **PPO's advantage**.

> Through that single channel, and with the design head unable to see the task
> at all, the search still produced **three distinct topologies, every one of
> them beating the fixed ant by 1.7-3.3x** (s2 4.89 m/s / 76 steps, s1
> 3.72 m/s / 91, s3's body 2.58 m/s / 131, against the frozen ant's 1.50 m/s /
> 218). Morphology was shaped entirely by return, never by observation.

That is a stronger result than it looked when written up, and it is also the
direct evidence for what follows.

## What is actually ruled out — and what is not

My first draft said convergence would be *"guaranteed by construction"* and an
arms race *"impossible."* **That is too strong and I withdraw it.**

The design head cannot *observe* the opponent, but it is *trained by*
opponent-dependent returns. Two learners with drifting weights do not produce
identical outputs from identical inputs. Precisely:

* **Ruled out: observation-conditioned specialisation.** No design head can
  detect *"my opponent is fast and low, so I should grow long legs."* Nothing
  about the opponent enters its input. This is real and it is what D2's
  `--role-in-design` fixed.
* **Available: return-mediated divergence.** When one side improves, the
  other's returns fall, and its design head's advantage signal changes with
  them. This is **the identical channel that produced E3.1's three bodies**,
  which were also shaped without observation.

### The coupled channel is strong, and I measured it

Whether return-mediation can do anything depends on how much of the buffer
return actually depends on the opponent. In `run_to_goal.py` the sparse term is
genuinely zero-sum and the episode is a **race** — `done` fires the moment
either side reaches:

```python
n_reached = int(reached) + int(opp_reached)
parse = 0.0
if n_reached == 1:
    parse = GOAL_REWARD if reached else -GOAL_REWARD   # GOAL_REWARD = 1000
```

Measured on E3.1 s2 at epoch 399 (`alpha` = 0.8468):

| term | raw | buffer weight | weighted |
|---|---:|---:|---:|
| `dense` (uncoupled: forward, ctrl cost, survive) | 443.8 | `alpha` | 375.8 |
| `parse` (**coupled**) | 1000.0 | `1-alpha` | 153.2 |
| **weighted buffer return** | | | **529.0** |
| **win → loss swing** = `2000 x (1-alpha)` | | | **306.4 = 58%** |

**Losing the race instead of winning it moves the buffer return by 306 points
against a dense component of 376** — the adversarial channel is nearly as large
as everything else combined, even after d2rep down-weights it to 15%. Return
mediation has plenty to work with.

**And E3.1 never exercised it.** The scripted opponent crossed the line at step
**491**; E3.1's winners finished in **76-131**. The race was never contested,
so `parse` was a near-constant +1000. **E4 is the first rung on which the sign
of `parse` is actually in play.** That makes it a genuinely new experiment
rather than a variant.

## Revised recommendation: run E4 blind and faithful first

The option ranking changes accordingly.

* **Blind self-play is a real question with a real chance of either answer**,
  not a negative control with a predicted "yes". If the bodies diverge, that is
  return-mediated co-evolution and an interesting result on unmodified
  upstream. If they converge, *that* is the earned evidence that sight is
  required — and it costs one experiment instead of two.
* **The sighted head (old Option 1) is the expensive path**: it needs its own
  E4.0 prerequisite rung re-running E3.1's primary arm sighted at 3 seeds,
  or E4 inherits a confound. Better spent *after* we know blind self-play
  converges.

**Plan: build the sighted head behind a cfg flag, leave it unused**, so the
sighted arm is one flag away if the blind run converges.

### The honest caveat on the convergence branch

Run-to-goal is **mirror-symmetric**: both agents start 5 m from their own goal
(∓1 → ±4), and the reward selects on one scalar — *be faster than the other*.
A task with one scalar objective plausibly has one optimum, so **a convergence
result would be ambiguous between "the blind head cannot specialise" and "the
task has only one good body."**

This is partly defused by evidence we already have: **E3.1 found three distinct
topologies all winning**, so the task does *not* have a single morphological
optimum under non-contested selection. But a real race sharpens selection and
could still collapse them. I therefore pre-register the discriminators in §D
below rather than claiming the branch is clean.

## E4 design

> **Why this rung exists, in one line:** `parse`'s sign has never once been in
> play. E3.1's winners finished in 76-131 steps against an opponent arriving at
> 491, so the 306-point coupled term was a near-constant +1000 for the entire
> experiment. **E4 is the first rung on which winning and losing are both
> reachable**, and that is a new regime, not a variant of E3.1.

**Three seeds**, `control_log_std: -1.5`, `min_motors` off — the
exact configuration of E3.1's primary arm, the one that solved 2 of 3.

### A — shape: alternating-snapshot self-play

Two lineages **A** and **B** per seed. Each trains with the existing
single-agent Transform2Act loop, **design head unmodified**. Each faces a
frozen snapshot of the other lineage — body *and* controller — refreshed every
`--opp-refresh` epochs (proposed **10**).

This is chosen over simultaneous two-agent PPO because it needs no rewrite of
`Agent`, and because the two capabilities it *does* need already exist or are
trivial:

| need | status |
|---|---|
| opponent body settable to an arbitrary evolved design | **exists** — `rtg_e31d_s{2,3}body` already freeze evolved bodies |
| opponent driven by a policy | **actuators already in the XML** — `rtg_scene.py:133-141` mirrors every motor as `opp_*`; E2/E3 simply never used them, overwriting `qpos` kinematically in `set_opponent` instead |
| return coupling | **exists and is 58% of the buffer return** (above) |

The new code is: load a snapshot policy, write its outputs into the `opp_*`
`ctrl` slots instead of calling `set_opponent`, and drop the kinematic
prescription. The gate is the mirror of E2.1's and E3's: prove the opponent's
compiled body is the snapshot's body, and that a **non-refreshed** opponent
reproduces E3.1 numbers exactly.

**Known limit — refresh staleness.** Each lineage always faces an opponent up
to `--opp-refresh` epochs old, so **the design bounds the *rate* of divergence
this experiment can resolve, not whether divergence happens.** A slow arms race
is measured faithfully; one whose characteristic timescale is under ~10 epochs
is partly masked, because each side is reacting to a body its rival has already
moved past. Two consequences, both stated in advance:

* A **NULL verdict is weaker than a DIVERGENCE verdict.** It means "no
  divergence at timescales above the refresh interval", not "no divergence".
* **The refresh interval is a measured knob, not a free parameter.** Per-epoch
  Δ is logged, so if divergence appears and is fast, a follow-up at
  `--opp-refresh 1` is the natural next rung; if Δ is flat, staleness is the
  first thing to rule out before accepting NULL.

10 epochs is chosen because E3.1's within-lineage drift over a 10-epoch lag is
**SMD 0.185** — the opponent is meaningfully stale but not a different body
(the 40-epoch lag reaches 0.305, and the between-seed null is 0.89).

### B — budget

E3.1 measured **20.77 M steps per 400-epoch arm**. E4 is 2 lineages x 3 seeds =
**6 arms ≈ 125 M steps**, roughly **2x E3.1's 62 M** and well above the
briefing's "meaningfully more than E2.1's 7.5 M". At the observed 100-118 s per
epoch and the GPU ceiling of 3 design-on arms, a seed's pair runs concurrently:
**3 waves x ~13 h ≈ 40 h**.

### C — the divergence metric, pre-registered

Distances are the two already implemented in `e0_analyse.compare`, both
computed per epoch from the per-epoch `mean_action_design` and standardised by
the pooled `sampled_genome_std` population at that same epoch:

* **SMD** — mean |Δgenome| over shared bodies / sampled-population std. **Primary.**
* **Jaccard** on body-name sets. Secondary: it barely moves (0.63-0.75 all run).

**The null is internal to E4, not borrowed from E3.1.** With 3 seeds x 2
lineages = 6 agents:

* **`D_self(e)`** — distance between the two lineages **within** a run
  (3 pairs), i.e. the pair that co-evolved against each other.
* **`D_null(e)`** — distance between lineages in **different** runs,
  role-matched A-to-A and B-to-B (the scene is mirror-symmetric, so roles are
  not interchangeable; role-crossed pairs reported as a check).

Both are measured at the same epoch, in the same experiment, under identical
conditions. **The only difference is whether the pair co-evolved.** Comparing
instead against E3.1's cross-seed numbers would confound co-evolution with
opponent type (scripted vs learned), so it is not the null.

> **Δ(e) = D_self(e) − D_null(e)**

### D — what counts as divergence (fixed before running)

**Trajectory, not endpoint.** The cross-seed null *rises monotonically through
training* — measured on E3.1's three seeds, pooled SMD by window:

| epochs | 0-49 | 50-99 | 100-199 | 200-299 | 300-399 |
|---|---:|---:|---:|---:|---:|
| cross-seed SMD | 0.17 | 0.52 | 0.74 | 0.88 | 0.93 |

Any fixed threshold applied to a final-epoch number would be measuring
**elapsed training, not divergence** — which is also why on this project an
endpoint has inverted the series three times (the "seed that converged"
retraction, the premature-lock-in retraction, the mass-correlation sign flip).

Over the window **epochs 200-400**, aggregating per-pair distances into the
window mean *before* comparing:

| verdict | criterion |
|---|---|
| **DIVERGENCE** | mean Δ ≥ **+0.15** SMD units **and** Δ(e) > 0 in ≥ **80%** of epochs |
| **CONVERGENCE** | mean Δ ≤ **−0.15** **and** Δ(e) < 0 in ≥ **80%** of epochs |
| **NULL** | anything else — self-play does not measurably reshape morphology |

**Where 0.15 comes from.** It is calibrated from measured spread, not
invented. Over the same 200-400 verdict window, E3.1's three cross-seed pairs
give SMD **0.836 / 0.857 / 0.976** — a pair-to-pair sd of **0.0755**. With 3
within-run pairs and 12 between-run pairs, the standard error of Δ is

```
SE(Δ) = sqrt(0.0755^2/3 + 0.0755^2/12) = 0.049
```

so **0.15 is 3.1 SE**. E4's own `D_null` spread is reported alongside the
verdict so a reader can check the floor held. (Caveat: s1 stopped at epoch 328,
so its pairs contribute a truncated window; full numbers in
`docs/t2a/e4_null/e31_crossseed_null.json`.)

**Regime check** (briefing rule: *check the window holds one regime*). Confirm
no lineage is still improving at epoch 400; if goal rate is still climbing, the
window is extended rather than the verdict taken.

**Two degeneracy guards, both of which would silence the channel rather than
answer the question:**

1. **Draw saturation.** Mirror symmetry means equal-speed lineages reach their
   goals on the same physics step, where `n_reached == 2` and **`parse = 0`**.
   A high draw rate means the coupled channel is *off*, not that divergence is
   absent. Pre-registered: log per-epoch draw rate and the step-gap between
   arrivals; if draws exceed **50%** over the verdict window, Δ is reported as
   **untestable**, not as NULL.
2. **Ceiling saturation.** If both lineages exceed goal 0.95 for >100
   consecutive epochs, returns stop differentiating. Such a run's Δ is reported
   separately and not pooled.

**The discriminator for the ambiguous convergence branch** (§ "honest
caveat"), **with its comparison set and thresholds committed now** — in
`docs/t2a/e4_null/e31_comparison_set.json`, written before any E4 run existed,
so this is a threshold test rather than a judgement made after seeing the
answer.

The set is E3.1's three final bodies:

| seed | topology | bodies/motors | note |
|---|---|---:|---|
| s1 | `025ed15ce09b` | 16 / 7 | **provisional** — s1 was at epoch 329; refresh at 400 |
| s2 | `50271e7f5d26` | 18 / 6 | the 4.89 m/s winner |
| s3 | `901ec8c2e00b` | 12 / 6 | body included on the **frozen-body diagnostic's** evidence, not s3's own controller (which scored goal 0.00) — the set is of **bodies**, not controllers |

Both thresholds are measured, and they do not overlap:

| verdict | rule | basis |
|---|---|---|
| **MATCH** | SMD to nearest set body **≤ 0.44** | pooled **p95 of within-lineage drift** over a 40-epoch lag in E3.1's late window (lag 10 → 0.281, lag 20 → 0.367, lag 40 → 0.437) — the furthest one lineage moves while still being "the same body" |
| **DISTINCT** | SMD **≥ 0.75** | **p05 of the between-seed null** in the same window — as far apart as two independent searches get |
| **AMBIGUOUS** | between | reported as such, never rounded to a verdict |

The 1.7x gap between 0.44 and 0.75 is what makes the test meaningful. **Sanity
check the set passes**: E3.1's three bodies are mutually **0.864 / 0.921 /
0.973** apart — every pair above the DISTINCT line, confirming "three distinct
bodies" under the same rule E4 will be judged by.

### E — carried forward from E3.1, unchanged

* **3 seeds**, `control_log_std: -1.5`, `min_motors` off.
* **Frozen-body diagnostic per seed** — the probe that overturned the
  premature-lock-in story; it separates a controller failure from a morphology
  failure, and at a 1-in-3 controller-failure rate we should expect to need it.
* `p_act4`, action-σ, fall/forward dodge correlations, per-epoch morphology
  census, `--stop-file`, GCS checkpoint archiving.
* wandb with video in the same run; mean-action **and** stochastic evaluation
  side by side, 20+ episodes, plus the idle zero-torque floor.
* **Forward progress, not return, is the primary readout.**

## What this design does not test

* Whether a **sighted** design head would specialise. That is the flagged arm,
  built but unused.
* Whether divergence, if found, is *useful* — a morphological arms race could
  produce different bodies that are no faster. Speed is reported per lineage
  alongside Δ.
* Anything about the **2-3x design-head σ gap** of `D3_E31_FIX.md` §3g-ii,
  which remains unexplained.

## The gate — 11 assertions, all passing (2026-09-05)

`rower_soccer/t2a_port/gate_e4.py`. E3's briefing warned that a design stage
which silently no-ops would give "a clean, boring, completely wrong null that
looks exactly like a real result". E4's version of that failure is an opponent
that is present in the XML but inert, or one acting in the wrong frame. Either
would produce a tidy convergence number that means nothing.

| # | assertion | result |
|---|---|---|
| 1 | `build(src)` with no opponent override is byte-identical to the checked-in `rtg_ant.xml` | 9644 bytes, identical — E2/E3 untouched |
| 2 | a heterogeneous opponent compiles with its source's body/motor counts | s2body 18b/6m, s3body 12b/6m |
| 3a | our `sim_obs` == the opponent's rotated `sim_obs` in mirror-equivalent states | **max\|Δ\| = 0.000e+00** over 5 random states |
| 3b | the two bodies stay mirror-equivalent after 20 steps of equal body-local torque | max\|Δ\| = 5.55e-17 |
| 4 | `ctrl_cost` excludes the opponent's torques (differential **and** absolute) | identical inert vs active; == 0.5·Σa² |
| 5 | `opponent_mode=scripted` reproduces `run_to_goal` bit for bit | ret 31.231474 both, 40 steps both |
| 6 | the snapshot **races**: reaches its goal at the speed it trained at | **4.657 m/s in slot 1 vs 4.891 trained in slot 0 — 4.8% off** |

Gate 6 is the one that matters, and it is end-to-end evidence that the rotation
is right: s2's controller, driving s2's own evolved body, in the *opposite*
slot from the one it trained in, runs its own race at within 5% of its training
speed.

### Six bugs the gate caught, every one of which fails quietly

Recorded because the common thread is that **none of them crashes the run** —
each yields a plausible number.

1. **A free joint's `qvel[3:6]` is BODY-LOCAL**, so a world rotation must
   **not** negate angular velocity. This was *measured* (yaw a box 90° about z,
   set `qvel[3:6] = [1,0,0]`, recover the net rotation axis: it comes out world
   +y, the body's own x) rather than derived. Deriving it would have negated
   those three columns and left the opponent walking slightly wrong forever.
2. **The execution stage flag is `2`, not `0`.** With `0` the policy ran its
   *skeleton* head and every control column came back exactly `0.0`. Measured
   before the fix: the opponent travelled **0.153 m in 200 steps** with
   `max|torque| = 0.000` while its root z fell 0.831 → 0.260. It stood there
   and collapsed, and nothing errored.
3. **The policy consumes a batch.** A flat observation list made `forward()`
   read the node count off the observation matrix's last row; `RunToGoalEnv`'s
   broad `except Exception` converted that into an ordinary "fall".
4. **Quaternion double cover.** Composing the π-z rotation with an
   already-yawed body lands on `w = −1` where slot 0 starts at `w = +1` — the
   same orientation, a different number to a network. Canonicalised to `w ≥ 0`.
5. **The base env maps qpos→qvel with a fixed `−1` offset**, correct only
   behind *one* free joint. The opponent sits behind two. Now the DOF address
   is read off the compiled model (`jnt_dofadr`), which is exact for any number
   of bodies.
6. **Opponent torques must never enter the vector `ctrl_cost` sums over**, or
   the learner is billed 0.5·Σa²_opp for its rival — the exact term that
   deleted every actuator in E3. The opponent's control is injected downstream
   of the cost, in `do_simulation`.

Two of these — 2 and 3 — were caught only because gate 6 was tightened from
"the outcome changed" to "the opponent reaches its goal at its trained speed".
The weaker version **passed while the opponent was standing still**: the small
return difference was the inert body's weight shifting the contacts. A gate
that can pass on a dead opponent is not a gate.

## Measured before launch: cost and GPU fit

**The opponent's inference cost, measured.** Driving agent 1 with a policy
instead of a kinematic write adds a GNN forward pass to every control step:

| per control step | ms |
|---|---:|
| physics (`frame_skip` 5 substeps) | 0.499 |
| opponent observation assembly | 0.814 |
| opponent observation + policy forward | **2.614** |

That is **5.2x the cost of physics** per step, but it parallelises with the
rollout workers: 50 000 steps/epoch x 2.614 ms = 131 CPU-seconds, spread over
10 workers = **~13 s of wall clock per epoch** against E3.1's ~110 s epoch, so
**+12%**. 400 epochs x ~123 s = 13.7 h per wave, **3 waves ≈ 41 h** — the
budget in §B survives the measurement.

**GPU projection, done before wave 1 rather than discovered at 95%.** Max
reserved per E4 arm observed in the smoke run: **6.8 GB** (E3.1 arms ran
2.6-4.6 GB; the spread between arms is caching-allocator reserve, not a stable
requirement, which is why the *maximum* is used for planning).

| configuration | MiB | of 20 475 | |
|---|---:|---:|---|
| wave (2 arms) + `rtg_e31_s1` + `rtg_e31d_s3body` | 20 184 | 99% | **does not fit** |
| wave (2 arms) + `rtg_e31d_s3body` | 15 884 | 78% | fits |
| wave (2 arms) alone | 12 112 | 59% | fits |

So a wave fits alongside **one** other design-on arm, not two. Wave 1 therefore
starts when `rtg_e31_s1` finishes (55 epochs left when this was written, ~1.7 h)
rather than being squeezed in at 99%. Seeds are not shed and the wave is not
split: a seed *pair* is the smallest coherent unit, because each lineage is
only meaningful against the other.

### Correction: the projection above was built from snapshots, and snapshots understate the peak

A watcher tripped at **19 569 MiB of 20 475 (96%)** while the two smoke arms
were still running beside the two E3.1 arms. Confirmed with a sustained
measurement before acting, as the trigger itself demands, and the reading
oscillates:

```
19569 -> 18391 -> 19545 -> 15423 MiB   (15 s apart)
```

so **96% is a transient peak during the PPO update**, not steady state — the
per-process figures summed to 15.0 GB at the same moment. Every number in the
table above came from `nvidia-smi` snapshots taken *between* those peaks, so
**the table understates what a wave actually needs**. The honest planning
figure is the peak, not the snapshot:

| measured | MiB | of 20 475 |
|---|---:|---:|
| peak, 2 E3.1 arms + 2 E4 arms | **19 569** | **96%** |
| same instant, summed per-process snapshots | 15 015 | 73% |

Implication: 2 E4 arms peak at roughly **11.5-13.6 GB together**, so a wave
alongside `rtg_e31d_s3body` should peak near **16-18 GB (78-88%)** rather than
the 78% the snapshot table claimed. That still fits, with less margin than
stated. The plan is unchanged — wave 1 waits for `rtg_e31_s1` — but the wave is
launched with a peak watcher, and if the sustained peak exceeds ~19 GB one arm
is stopped **by stop-file** (MPS is active; nothing is killed) and the wave
re-planned rather than left to OOM.

This is the same failure mode as the retracted GPU-vs-bodies fit in
`D3_E3_ADVERSARIAL.md`: `nvidia-smi` reports the caching allocator's *reserve*,
and reasoning from it without watching how it moves gives a confident wrong
number.

**The opponent policy costs no GPU memory**: it is constructed on CPU and
measured at **+0.0 MiB CUDA** across five constructions, so the ~40 refreshes
over a run do not leak.

**MPS is active on this box** (`nvidia-cuda-mps-control` + server, and the
training arms carry `CUDA_MPS_PIPE_DIRECTORY`). Nothing is killed to make room:
the floor arm was stopped by **stop-file**, and the smoke arms were given
`--max-epoch 3` so they exit on their own.

### Second correction: the smoke pair measured the footprint, and 89% was speculation

The smoke pair ran to completion beside both live arms and pinned the card at
**19 741 MiB (96.4%)** for five consecutive samples — not a transient. That is a
second, independent measurement of a real E4 pair, and it agrees with the
pessimistic reading rather than the snapshot table. **734 MiB of headroom, and
D1 died at 19 950.**

The smoke arms exited on their own 80 s later, each having written all three
rows of `--max-epoch 3`. They were never signalled: they carried no
`--stop-file` (the trainer only checks one when the argument is given), and
they held CUDA contexts under active MPS, where a signal can corrupt the
survivors rather than just the target. Both live arms verified afterwards on
their **original** PIDs — `rtg_e31_s1` 3426432 at epoch 348/400 (goal 1.00,
3.740 m/s, its best speed of the run) and `rtg_e31d_s3body` 3834648 at 219/400.

With the pair gone, sustained total over 8 readings is **7 560-9 714 MiB
(37-47%)**.

**Re-derived from totals, because the total-vs-sum-of-parts gap is not constant**
(1 547 MiB in one sample, 370 in another — the per-process figures and the total
are read at different instants and every process fluctuates):

| derivation of wave + `s3body` | MiB | of 20 475 |
|---|---:|---:|
| measured peak (A) minus s1's contribution in A | 15 439 | 75% |
| pair per-process peak 11 276 + s3body + overhead | 15 448 | 75% |
| pair at its *simultaneous* peak (~12 850) + s3body + overhead | 17 022 | 83% |
| ~~"production arms cost 7 000 each"~~ | ~~18 172~~ | ~~89%~~ |

**The 89% row is struck because it was an assumption, not a measurement**, and
the assumption is false: production differs from the smoke only in video every
6 epochs, 10-episode evals and checkpoint archiving, and **none of those touch
the GPU**. Rendering is `MUJOCO_GL=osmesa` (software, CPU), and every eval,
census and video block is wrapped in `to_cpu(agent.policy_net, ...)`, which
moves the policy *off* the card for the duration. The only GPU work is the PPO
update, and that is identical in the smoke (same `min_batch_size` 50 000, same
`mini_batch_size`). **The smoke therefore already measured the GPU-relevant
footprint of a production arm.**

So the measured band for wave 1 + `s3body` is **15.4-17.0 GB (75-83%)** —
under the 17 500 sustained trigger, with **3.4-5.0 GB of headroom**. That is
real margin, so wave 1 goes when `rtg_e31_s1` finishes (52 epochs, ~1.6 h) and
does **not** need to wait for `s3body` as well. Wave 1 launches with
`watch_e4.sh` armed at the 17 500 trigger; if the sustained peak breaches it,
one arm is stopped **by stop-file** (`launch_e4.sh` gives every E4 arm one) and
the wave re-planned. Nothing is ever killed while MPS is up.

`rtg_e31d_s3body` also carries a stop-file (`/tmp/stop_e31d_s3body`) if its
remaining 181 epochs are ever worth trading for headroom — but on these numbers
that trade is not needed.

## Wave 1 is running — and the budget was wrong by more than 2x

Launched 18:53:36: `rtg_e4_s1a` (386140) and `rtg_e4_s1b` (386224), exchanging
snapshots every 10 epochs. The epoch-0 handshake retry **fired in production**
(`partner appeared after retry` on s1a), so neither arm opened against a
passive body.

### The epoch rate, measured properly (supersedes the panic below)

**The ~90 h figure below is withdrawn: it came from a bad measurement.** It was
taken over three epochs that included process startup, the epoch-0 video, and
CPU contention from `s3body`. Re-measured on the freed machine over epochs
**6-13** (no startup, no epoch-0 video, `s3body` stopped), timestamping each
epoch's arrival:

| | s1a | s1b |
|---|---:|---:|
| per-epoch (non-video mean) | **170 s** | **181 s** |
| per-epoch (all-in) | 168 s | 177 s |
| spread | 150-205 s | 155-205 s |
| trend, first half → second half | 164 → 175 s | 179 → 184 s |

Neither arm is still settling; with n = 7-8 and a 55 s spread those trends are
noise, not drift.

**The E3.1 baseline, measured the identical way** — `log_eval.txt` timestamps a
video checkpoint every 6 epochs, so consecutive intervals divided by 6 give the
epoch rate, median-filtered to drop restart gaps:

| | s1 | s2 | s3 | **mean** |
|---|---:|---:|---:|---:|
| E3.1 s/epoch | 141 | 131 | 149 | **141** |

So **self-play costs +21% to +29%**, and the ~41 h estimate decomposes as:

| | baseline | overhead | s/epoch | 3 waves |
|---|---:|---:|---:|---:|
| predicted | 110 s | +12% | 123 | 41 h |
| **actual** | **141 s** | **+29%** | **181** | **60 h** |

**The overhead prediction was roughly right; the error was my E3.1 baseline.**
I quoted 110 s/epoch from an offhand observation rather than a measurement —
the real figure across three seeds is 141 s. Most of the 41 → 60 h gap is that,
not self-play.

**A wave is 20.1 h** (the slower arm sets it, since the pair must finish
together), so **three waves ≈ 60 h**.

### Will the rate hold as the arms learn?

Worth asking, because episode length collapses during training and the design
stages plus a MuJoCo model recompile run **per episode**, not per step: E3.1 s1
went from ~115 episodes per 50 000-step epoch to ~544, a 4.7x increase in
per-episode work.

**Measured, and it does not matter.** Over that same run E3.1 s1's epoch time
went **144 → 141 → 139 s** (first/middle/last third) while episodes per epoch
went **115 → 157 → 392 → 544**. Flat, in fact slightly falling. The per-episode
recompile is negligible against the 50 000 fixed control steps, so the 170-181 s
measured at epochs 6-13 should hold for the whole run and 60 h is a projection
rather than an early-regime snapshot.

### The earlier (withdrawn) rate estimate

| | E3.1 | **E4 measured** |
|---|---:|---:|
| seconds per epoch | ~110 | **266-275** |
| 400 epochs, per wave | ~12 h | **~30 h** |
| three waves | ~41 h | **~90 h** |

**My "+12%" projection was wrong by a factor of more than two, and the error was
in the reasoning, not the measurement.** The per-step cost (2.614 ms vs 0.499 ms
for physics) was measured correctly; what was wrong was dividing 131 CPU-seconds
by 10 workers to get 13 s of wall clock. That division assumes ten idle CPUs.
The cgroup quota is **10.2 CPUs total**, shared by three arms of ten workers
each, so the machine was already oversubscribed and added CPU work converts to
wall clock at close to 1:1 against *available* parallelism, not against the
worker count. Assuming spare capacity that the cgroup does not have is the same
mistake as reading `nproc` (48) instead of the quota — recorded in
`PLAN_D3_M3.md` §2 — in a new costume.

This is a real change to the plan and it should be a decision, not a fait
accompli: three waves at ~30 h is ~90 h, not the ~41 h the rung was approved
on. It should improve now that `s3body` has stopped and CPU contention has
dropped; the rate will be re-measured on the freed machine before any
projection is quoted again.

### GPU, after acting on the trigger

The 17 500 trigger fired at 19 082 MiB. Measured rather than acted on: 1 Hz
sampling showed a **flat plateau at 17 186-17 192 MiB (84%)** with the 19 082
transient real but rare — plateau headroom 3.3 GB, **peak headroom 1.4 GB**,
and the plateau sitting 300 MiB *under* the trigger. Against D1's death at
19 950 that is not real margin, so `rtg_e31d_s3body` was stopped **by
stop-file** (never signalled; MPS is active). It exited cleanly and both E4
arms were unaffected.

| | MiB | of 20 475 | headroom |
|---|---:|---:|---:|
| plateau, 2 E4 arms + s3body | 17 192 | 84% | 3 283 |
| transient peak, same | 19 082 | 93% | 1 393 |
| **flat, 2 E4 arms alone** | **10 962** | **53.5%** | **9 513** |

81 samples at 1 Hz after the stop: zero excursions above 17 500. Wave 2 will
therefore fit on its own with ~9.5 GB spare, and does not need anything else
stopped.

### Wave 1 at epoch 4 (both arms)

| | s1a | s1b |
|---|---:|---:|
| goal / loss / fell | 0.00 / 0.00 / 0.00 | 0.00 / 0.00 / 0.00 |
| forward | 0.11 m | 0.12 m |
| `p_act4` | 0.95 | 0.85 |
| motors (mean) | 6.65 | 7.05 |
| **draw rate** | **0.00** | **0.00** |
| race margin | 0.06 m | 0.18 m |

Untrained and barely moving, which is what epoch 4 should look like — E3.1 took
~200 epochs to solve. The two things that matter this early both read clean:
**`p_act4` is 0.85-0.95**, nothing like E3's collapse to 0.000 by epoch 17; and
the **draw guard reads 0.00 rather than a spurious 1.00**, because neither side
is reaching the line, so the race is undecided rather than drawn.
