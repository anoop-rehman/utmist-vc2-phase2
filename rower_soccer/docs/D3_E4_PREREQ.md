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
