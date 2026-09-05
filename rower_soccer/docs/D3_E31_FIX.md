# D3 M3 E3.1 — repairing the control-cost economics that made E3 delete its actuators

*2026-09-04. Follows [`D3_E3_ADVERSARIAL.md`](D3_E3_ADVERSARIAL.md), whose §3e
established the failure, whose §3f derived this fix before running, and whose
instrument this reuses unchanged. Every number below names the command that
produced it.*

## The one-paragraph version

E3 ran Transform2Act's design+control loop on an adversarial task and **the
design search deleted every actuator** — `p_act4` = 0.000 on 3 of 3 seeds by
epoch 17, not one design in 600 with four motors. The mechanism is the **dense
control cost**: at `control_log_std = 0` a fresh policy pays ~3.89/step against
a 1.0 survive bonus, and deleting the actuators makes `0.5·Σa²` exactly 0
forever — a faster route than learning small actions, measured at **17 epochs
against ~125**. E3.1 changes the one number that closes that gap.

## What E3 established that this rung depends on

| | |
|---|---|
| the failure | `p_act4` 0.825 (untrained) → **0.000**, 3 of 3 seeds, by epoch 17 |
| it is specific, not general | topological diversity **preserved** — 90-149 distinct topologies of 200 — while actuation went to zero |
| the mechanism | dense control cost; `train_R` −2.4 → +0.78, the 0-motor body's ceiling |
| `d2rep` cannot help | it down-weights `parse`; the control cost is in `dense`, weighted ~1.0 |
| **the controller is not at fault** | the frozen-body GNN control reached **goal 1.00, forward 5.02 m, fell 0.00, R ≈ +1510** — E3's null is the design loop's |

That last row is why E3.1 exists and why it was gated behind the controls
finishing: had they failed, the fix list would have been about the controller
instead.

## The fix, derived before running

Full derivation in [`D3_E3_ADVERSARIAL.md`](D3_E3_ADVERSARIAL.md) §3f. The
constants:

| | |
|---|---|
| `cost_crit` | **0.6831 / step** |
| `log_std_crit`, analytic | **−0.8837** |
| `log_std_crit`, **measured on the simulator** | **−0.9645** |
| chosen | **`control_log_std` = −1.5** (σ 0.223, cost **0.199/step**) |

−1.5 rather than −1.0 because the empirical boundary is stricter than the
analytic one and −1.0 sits **0.036** below it against −1.5's **0.536** — and
because the exploration cost that would justify staying high **does not exist
in this range**: path travelled by noise alone is flat at 3.60-4.36 m across
σ 0.41 → 0.17, with no monotone trend.

**A retraction carried from E3**: "charge the control cost per actuator
*present*" does not work — a 0-motor body has none present, so it pays 0,
unchanged. Normalising by actuator count fails identically. **Any strictly
positive control cost makes actuators worse than none until forward progress
pays**, and at initialisation it does not.

## The two arms

| arm | cfg | change from E3 | grid margin (§3f-iii) |
|---|---|---|---|
| **primary** | `rtg_e31_s{1,2,3}` | `control_log_std` 0 → **−1.5** | STAND +86.2 vs blob +21.2 |
| **second** | `rtg_e31f_s{1,2,3}` | the same **plus** `env_specs.min_motors = 4` | STAND **+210.3**, blob unreachable |

Everything else — task, opponent, `d2rep` regime, budget (400 × 50,000 = 20.0M),
instrument — is E3's, so a diff against `rtg_e3_s{seed}.yml` is the whole
experimental delta.

**The floor alone would not have worked, and that is why it is the second arm
and not the first.** §3f-iii measured it: at `log_std = 0` a 4-motor ant pays
2.0/step, so falling early still beats standing (−20.9 against −582.6) and the
morphology failure converts straight back into **E2's fall-dodge**, reached
through control instead of the body. The floor is free *once σ is fixed*, and
removes the failure structurally rather than pricing it.

**GPU memory caps this at three design-on arms at a time** — E3's three peaked
at 19.0 GB of 20.475 while their bodies were still ~13 nodes, so six would be
~38 GB. The primary seeds run first because they carry the falsifier.

## The gate

`rower_soccer/t2a_port/gate_e31.py` — **7 checks, 0 failed**.

| check | result |
|---|---|
| E3.1 arms init at `log_std` −1.5 | −1.5000 exactly → σ 0.2231, cost **0.1991/step, below the 1.0 survive bonus from step 0** |
| E3's arms unchanged | `log_std` 0.0000, cost 4.0000/step |
| the floor binds | min actuators over 12 all-remove episodes = **4** (floor 4) |
| **NEG: without the floor** | the same actions reach **0 motors** — E3's failure reproduced on demand |
| E3's cfg carries no `min_motors` | confirmed; E0-E3 byte-for-byte unchanged |

The floor is an optional branch in `AntEnv.allow_remove_body` defaulting to 0 =
off, counting actuators on the **current** robot each call because
`apply_skel_action` removes bodies one at a time and the floor must hold at
every step of that loop.

## Instrumentation — and the trap that nearly cost the falsifier

**E3's four watchers all carried hardcoded cfg lists and kept polling
faithfully after every one of those runs ended.** `census_sidecar.sh`,
`population_watcher.sh`, `logstd_watcher.sh` and `sigma_sampler.sh` were all
still running, all pointed at `rtg_e3_s*` / `rtg_e3c_s*`, all dead — so the
instrumentation *looked* healthy while producing nothing for E3.1, and
`runs/d3_e31_fix/census/` was empty. All four are now stopped.

**What was actually at risk, stated precisely.** `p_act4` and
`control_log_std` were never in danger: both are written **per epoch by the
trainer itself** into `results/<cfg>/e3_epochs.jsonl`, because
`e3_morph.census` gained its motor columns *before* these arms launched.
Verified at epoch 0 — `census['p_act4']` = 0.80 / 0.80 / 0.95 on the three
seeds. What was missing was every **derived** artefact: no CSV, no falsifier
check, nothing watching. The raw data was safe; the thing that would have told
us it fired was not.

**Both instrumentation failures in this experiment happened at a TRANSITION** —
new arms, new directory — which is exactly when a watcher keeps pointing at the
old target and nobody notices, because the files it writes keep looking fresh.

Fixed three ways:

1. `runs/d3_e31_fix/watch.sh` distils each arm's JSONL into
   `census/<cfg>_morph.csv` (24 columns including `p_act4` and
   `control_log_std`) and evaluates **both falsifiers** every 120 s, taking its
   cfg list from `$CFGS` so repointing is a variable rather than an edit.
2. `runs/d3_e31_fix/assert_instruments.sh` fails **loudly** if any quantity a
   falsifier depends on is not producing rows for a given cfg.
3. `launch.sh` now runs that assertion against the cfg it just launched. **A
   falsifier that depends on a collector nobody checked is not pre-registered
   in any useful sense.**

Assertion at launch: **3 of 3 arms confirmed collecting.**

## Pre-registered falsifiers

> **Either fires and the fix has failed:**
> 1. `control_log_std` **exceeds −0.9645 at any point in the first 20 epochs**;
> 2. `p_act4` **collapses to 0 by epoch 20**.

The first tests the mechanism directly and fires earlier; the second tests its
consequence and is the statistic that stopped E3. Both are readable per epoch
from `runs/d3_e31_fix/census/` with no wandb.

**Both watch for COLLAPSE only.** The mirror failure — the design head pinning
at the 29-body ceiling — is neither of them, and is recorded separately as
**Outcome C** below: a regime to classify rather than a trigger to halt.
**Every report of the falsifier table must carry forward progress and goal rate
beside the morphology columns**, because a body that is growing *and* walking
and one that is only growing are different results and the falsifiers as
written cannot tell them apart.

## Epoch 0 — the first evidence, and it is not yet a result

| | E3 (`log_std` 0) | **E3.1 s1 / s2 / s3** |
|---|---|---|
| mean-action readout | 6 bodies, **0 motors** | **14b/8m, 12b/6m, 12b/6m** |
| `control_log_std` | 0 | **−1.5040 / −1.5033 / −1.5061** |
| population `motors_mean` | (untrained baseline 5.71) | **5.70 / 5.35 / 6.00** |
| **`p_act4`** | → 0.000 by epoch 17 | **0.800 / 0.800 / 0.950** |
| distinct topologies | — | 20/20 on all three |

**The readout is an actuated ant on all three seeds where E3's was a 0-motor
stump at the same epoch, and the population sits at its untrained baseline.
Neither falsifier has fired.** This is epoch 0 of 400 and the test is epoch 20;
nothing here is a result yet.

**Cost**: T_sample 79-108 s, T_update 180-222 s → ~310 s/epoch, ETA ~1 day 6-10 h
for three concurrent arms.

**Epoch 1 — `p_act4` is rising, not falling:**

| | epoch 0 | **epoch 1** |
|---|---|---|
| s1 `p_act4` / motors | 0.80 / 8 | **0.90 / 8** |
| s2 `p_act4` / motors | 0.80 / 6 | **0.95 / 8** |
| s3 `p_act4` / motors | 0.95 / 6 | — |

E3's ran 0.825 → 0.300 (epoch 3) → 0.000 (epoch 17). Two epochs is not a trend
and this is recorded as a transient, not a result.

**Epochs 0-5, both surviving arms** (`census/rtg_e31_s{2,3}_morph.csv`):

| epoch | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| s2 `log_std` | −1.5033 | −1.5083 | −1.5123 | −1.5173 | −1.5194 | −1.5229 |
| s2 `n_bodies` / motors | 12 / 6 | 13 / 8 | 16 / 8 | 16 / 8 | 21 / 9 | 21 / 9 |
| **s2 `p_act4`** | 0.80 | 0.95 | 0.85 | **1.00** | **1.00** | **1.00** |
| s3 `log_std` | −1.5061 | −1.5094 | −1.5116 | −1.5149 | −1.5190 | — |
| s3 `n_bodies` / motors | 12 / 6 | 12 / 6 | 16 / 8 | 17 / 8 | 18 / 9 | — |
| **s3 `p_act4`** | 0.95 | 0.95 | 0.95 | **1.00** | **1.00** | — |

> **`p_act4` has reached 1.000 on both arms — every one of 20 sampled designs
> carries four or more motors — where E3's had fallen to 0.300 by epoch 3 and
> 0.000 by 17.** Motor count on the mean-action readout is *rising* (6 → 9) and
> `log_std` is falling away from the boundary (−1.503 → −1.523), so neither
> falsifier is approaching, let alone firing.
>
> **This is epoch 5 of 400 and the test is epoch 20.** It is recorded as a
> transient. What it does establish is that the arms are in the opposite
> regime to E3's at the same epochs, which is the minimum the fix had to do.

## The frozen-body control's final numbers — the reference E3.1 is read against

Measured on the shared instrument from the arms' own final checkpoints
(`e3_posthoc.py`, 20 episodes, both protocols), not from training-log evals:

| arm | protocol | R | goal | fell | forward | of 5.0 m | `r(fall,R)` | `r(fwd,R)` |
|---|---|---|---|---|---|---|---|---|
| control s1 (ep 312) | mean-action | **+1452.1** | **0.95** | 0.05 | 4.82 m | 96.3% | **−0.996** | **+0.994** |
| control s1 | stochastic | +1449.4 | 0.95 | 0.05 | **5.00 m** | 100.0% | −0.975 | +0.745 |

Body frozen: **134 mjModel arrays identical**, 1 distinct topology of 50 sampled
designs, 13 bodies / 8 motors throughout. **And the correlation pair has
inverted to E2.1's `d2rep` structure** — −0.996 / +0.994 against E2.1's
−0.94 / +0.95 — so on a frozen body under `d2rep`, once the agent can do the
task, **return measures competence and not falling**. That is the cleanest
confirmation of E2.1's central result this project has produced, on a different
architecture.

## GPU memory became evidence, and cost a seed

**Three E3.1 arms did not fit on the card, and the reason is the fix working.**
Sustained at **19,613 MiB of 20,475 (95.8%)**, 862 MiB of headroom — E1 lost the
live D1 run at 19.95 GB. Per client: 7,228 / 4,602 / 7,032 MiB.

The driver is body size, measured:

| epoch | 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|---|
| s1 `n_bodies` | 14 | 17 | **22** | 17 | — |
| s2 `n_bodies` | — | 13 | 16 | 16 | **21** |
| s3 `n_bodies` | 12 | 12 | 16 | **17** | — |

**E3's three arms fitted comfortably *because their bodies collapsed to 5
nodes*.** E3.1's don't fit because they are holding 12-22. So the GPU headroom
E3 enjoyed was itself a symptom of its failure, and the memory pressure here is
a (crude, indirect) indicator that the design search is doing something. §5a of
[`D3_E3_ADVERSARIAL.md`](D3_E3_ADVERSARIAL.md) predicted exactly this — *"if the
design search grows bodies back the epochs slow down again, and so does the
card"* — for E3, where it never happened.

**`rtg_e31_s1` was stopped by stop-file** (largest client, 7,228 MiB), leaving
s2 + s3 at ~11.6 GB and **8.8 GB of headroom**. Losing one seed deliberately
beats losing three to an OOM, and the arms hold CUDA contexts under MPS so a
signal was never an option. **E3.1's primary arm is n = 2, not n = 3**, and that
is a resource limit rather than a design choice — recorded as such. The third
seed's checkpoint is saved and the arm is resumable with `--epoch`.

**Consequence for the second (floor) arm**: it cannot run three seeds either,
and on current evidence not even alongside these two. It follows when these
finish, at whatever seed count the card allows.

## Not tested / not claimed

* **Nothing yet.** This document records a launch and a gate. The falsifiers
  resolve at epoch 20 and the rung's question — does the design loop produce a
  body that can act, and does it win — at 400.
* **§3f is an incentive-landscape calculation over fixed measured quantities**,
  like E2.1's `a_crit`. It measures what the objective rewards, not what PPO
  does in it. That is exactly what E3.1 tests.
* **n = 2 seeds on the primary arm** after `rtg_e31_s1` was stopped for GPU
  headroom, and the second (floor) arm has not started at all. Both are
  resource limits, not design choices. E3 had n = 3, so the arms are not
  seed-matched and any E3-vs-E3.1 contrast inherits that.
* **The empirical boundary −0.9645 was measured on the FROZEN 13-body ant.**
  An evolved body with a different actuator count has a different threshold;
  §3f's `n` is 8 throughout.
* **The termination rule is unchanged**, as in E3. §3d's separate finding —
  that charging the fall −1000 while keeping the termination dominates on every
  axis — is a different rung and is not part of this one.

## The growth ceiling — read, not assumed

**Body count is bounded at 29, and the bound is tight.** `AntEnv.allow_add_body`
permits a child only when `min_body_depth ≤ depth < max_body_depth − 1` and the
body has fewer than `max_nchild` children. With this cfg's
`min_body_depth 1 / max_body_depth 4 / max_nchild 2` and our ant's initial tree
(depths `{0:1, 1:4, 2:4, 3:4}`), driving every body to ADD every step saturates:

| add-step | bodies | depth histogram | still addable |
|---|---|---|---|
| initial | **13** | `{0:1, 1:4, 2:4, 3:4}` | 4 |
| 1 | 21 | `{0:1, 1:4, 2:8, 3:8}` | 4 |
| 2 | 25 | `{0:1, 1:4, 2:8, 3:12}` | 4 |
| 3 | **29** | `{0:1, 1:4, 2:8, 3:16}` | **0** |
| 4, 5 | 29 | — | 0 |

**29 = 1 root + 4 at depth 1 + 8 at depth 2 + 16 at depth 3**, and the root
itself can never add (depth 0 < `min_body_depth`), so the four original legs are
structurally fixed. This also confirms `D3_E1_ANT.md`'s count from the other
direction: 29 − 13 = **16 possible additions**, of which it measured 12 as
passive dead weight.

**So the memory demand terminates.** `skel_transform_nsteps` is 5, more than the
3 steps needed to saturate, so 29 is reachable within a single episode's design
phase — the ceiling is not merely asymptotic. The projection therefore has a
finite worst case rather than an open-ended one, and the question is only
whether two arms fit *at 29*.

### The memory projection — and why the obvious fit is invalid

**Asked for a projection of GPU bytes against body size. The data does not
support one, and the first fit I produced was an artifact.** Fitting
`MiB = a + b·bodies_mean` across both arms gave a slope of **2,085 MiB per
body** and an intercept of **−31,896 MiB** — physically meaningless. It was
fitting the *between-arm* difference, not body size:

| | `bodies_mean` | peak MiB |
|---|---|---|
| `rtg_e31_s2` | 18.5 | **7,450** |
| `rtg_e31_s3` | 18.1 | **4,756** |

**A 2,694 MiB gap at 0.4 bodies' difference.** Body count cannot explain it —
PyTorch's caching allocator holds each process's own high-water mark and
`nvidia-smi` reports *reserved* memory, not live tensors. And within each arm,
over 17.4 → 18.9 bodies, there is **no detectable growth**; the 2,596-7,450 MiB
spread inside one arm is the sampling-trough-to-update-peak cycle, not body
size.

**What the numbers do support:**

| | arms | `bodies_mean` | total peak | per arm |
|---|---|---|---|---|
| earlier | 3 | ~16 | 18,862 MiB | 6,287 |
| now | 2 | ~18 | 12,206 MiB | 6,103 |

> **Per-arm peak went 6,287 → 6,103 MiB while `bodies_mean` went 16 → 18.** The
> pressure that forced shedding `s1` was **arm count, not body growth**, and
> body growth over the observed range produced no measurable increase. That is
> reassuring for 18 → 29 but it is not proof, because the observed range is
> 1.5 bodies wide and the ceiling is 11 away.

`gpu_longitudinal.sh` now records each arm's peak over a full 2-minute window
against its `bodies_mean` every cycle, so the curve accumulates as bodies rise
toward 29 and the question gets answered by measurement rather than
extrapolation.

### The mitigation ladder, decided in advance

**Ordered by cost, cheapest first. Note that the first rung costs no science at
all**, which is why it comes before shedding anything:

1. **Shrink the update chunk — results-neutral.**
   `transform2act_agent.update_params` and `update_policy` each hold a
   `chunk = 10000` forward pass over the 50,000-state batch. Both run under
   `to_test(*update_modules)` + `torch.no_grad()`, there is no batch norm, and
   `RunningNorm` does not update in test mode — so each element's forward is
   independent and **chunk size changes peak memory and iteration count, not
   numerics.** Dropping to 2,500 cuts that peak ~4x. Cost: one epoch, because
   it needs the arm restarted from its checkpoint with `--epoch N`.
2. **Shed the higher-peak arm** — currently `s2` at 7,450 MiB against `s3`'s
   4,756, re-measured over ≥60 s at the time rather than assumed from these
   numbers, since the two swap places between phases.
3. **Do NOT run a single arm to 400.** n = 1 supports no claim about seed
   variance, and this project has recorded that limitation three times already.
   If rung 2 is reached, **stop both arms and report n = 2 at whatever epoch
   they reached** in preference to a lone seed running on.

**Trigger**: total sustained update-phase peak above **17,500 MiB** (85%),
measured over ≥60 s — not a single sample. That leaves ~3 GB of margin against
the 19.95 GB at which E1 lost D1.

### The third seed is suspended, not abandoned

`rtg_e31_s1` stopped by stop-file at epoch 4 with **`epoch_0005.p` saved**, and
`train_e3_gnn.py --epoch 5` resumes it. If the longitudinal curve shows the
bodies plateauing below the ceiling, or once one arm finishes, **resuming
restores n = 3 for most of the run**. "n = 2, resource-limited" and "n = 2, one
seed suspended and resumable" are different claims about how much this result
can be strengthened later, and it is the second.

## Outcome C — ceiling saturation, the failure neither falsifier watches for

*Added before the epoch-20 verdict, because the pre-registration had a gap and
the arms are walking into it: readout bodies are at 24-26 of a hard ceiling of
29, up from 12 at epoch 0.*

**Both falsifiers watch for collapse** — `p_act4` → 0, `log_std` rising above
−0.9645. **Neither watches for the mirror failure**: the design head pinning at
the ceiling. "Grows to 29 and stays there" is not self-evidently success. It
could be the fix working — E1 already found this creature evolves away from a
quadruped — or it could be a second degenerate optimum pointing the other way:
instead of deleting everything to escape the control cost, adding everything
because bodies are cheap.

**They are cheap, and that is measurable.** `dense = forward − 0.5·Σa² + 1.0`
has **no mass or size term at all**. Only *actuated* bodies cost anything, via
`Σa²`. And `D3_E1_ANT.md` established that **12 of our ant's 16 possible
additions are jointless clones** — passive dead weight that the reward cannot
see. So the sharp sub-signal is not body count but the **passive fraction**,
`passive = n_bodies − 1 − n_motors`:

| | bodies | motors | **passive** | `max_fwd` | goal |
|---|---|---|---|---|---|
| initial ant | 13 | 8 | **4** | — | — |
| s2 epoch 0 | 12 | 6 | 5 | — | — |
| **s2 epoch 9** | **26** | **11** | **14** | **0.12 m** | 0.00 |
| s3 epoch 0 | 12 | 6 | 5 | — | — |
| **s3 epoch 8** | **18** | **9** | **8** | **0.16 m** | 0.00 |

**s2 has added 14 bodies and 5 motors: roughly three passive bodies for every
actuated one.**

### The definition

> **Outcome C fires when** the **sampled population** `sampled_bodies_mean` ≥
> **26** (90% of the 29 ceiling) **sustained over ≥ 5 consecutive epochs** with
> `p_act4` ≥ 0.9. Readout `n_bodies` ≥ 27 is reported beside it as a secondary
> indicator, **not** as the trigger.

*Originally defined on the readout. **Corrected**, and by E3's own lesson: §3c
of [`D3_E3_ADVERSARIAL.md`](D3_E3_ADVERSARIAL.md) had to rewrite E3's stop rule
for exactly this reason — the mean-action design is the MODE of the design
distribution and not the distribution. Keying Outcome C on the readout would
have repeated the mistake in the opposite direction.*

### What distinguishes healthy growth from a second degenerate optimum

The discriminator is whether the body is being *used*, and the honest test is
against the frozen-body control's own timeline, because that arm shows what
this task's learning curve looks like without any design freedom:

* **C1, healthy growth** — saturation with **forward progress rising**:
  `max_fwd` trending up and crossing ~1.0 m by around epoch 84, which is where
  the frozen-body controls' `net_dx` first turned positive, or any goal > 0.
  Bodies growing *while the task gets solved* is the fix working.
* **C2, a second degenerate optimum** — saturation with `max_fwd` **flat below
  ~0.5 m past epoch ~90** (the controls' locomotion onset plus margin) and goal
  0.00, **with the passive fraction still rising**. A body that is only growing.

**At epoch 9 this is not yet distinguishable and must not be read as either.**
The controls sat at `max_fwd` 0.21 / 0.14 m at epoch 9 and did not locomote
until epochs 84 / 79. E3.1's 0.12-0.16 m is squarely on that trajectory, so
flat forward progress now is exactly what the reference arm did.

### What follows in each case

* **C1** — nothing. Continue to 400.
* **C2** — the design head is accumulating bodies the reward cannot see. The
  analogue of §3c's actuator floor is a **size cost, not a reward-mix change**:
  `dense` has no mass or body-count term, so passive bodies are free by
  construction. The candidate next rung is a per-body or per-unit-mass cost, or
  tightening `add_body_condition`. **Not** a change to `d2rep`, and **not** a
  termination-rule change — for the same reason as E3, neither touches what is
  actually being exploited.

**This is recorded as an outcome to classify, not a trigger to halt.** Growth
toward a larger body may simply be correct, and stopping the run on it would
throw away the result that distinguishes C1 from C2.

### C2 is not a free lunch — physics already charges for mass, and it may be enough

The reward has no mass term, which is the mechanism above. **But mass enters the
objective through the door the reward *does* look at**: a heavier body is harder
to accelerate, so `forward_r = ΔCOM_x/dt` falls. The loading is real — s2's
mean-action mass has gone **0.812 → 1.323 kg** against the original ant's
**0.879**, roughly 1.5× — so there is a countervailing pressure already present
and it is not obvious which way the balance tips.

**All four paired points that exist**, plus the frozen-body controls whose mass
is constant at 0.879 by construction:

| arm | epoch | mass | `max_fwd` |
|---|---|---|---|
| s2 | 4 | 1.165 | **0.154** |
| s2 | 9 | 1.363 | **0.123** |
| s3 | 4 | 1.062 | **0.157** |
| s3 | 9 | 1.145 | **0.113** |
| *control s1 (mass fixed)* | 4 → 9 | 0.879 | **0.10 → 0.21** (rising) |
| *control s2 (mass fixed)* | 4 → 9 | 0.879 | 0.13 → 0.14 (flat) |

**Both design-on arms got heavier and slower over epochs 4-9 while both
fixed-mass controls got faster or stayed flat.** That is directionally what the
mass hypothesis predicts.

**It is four points and it establishes nothing causal.** Forward progress at
epochs 4-9 is dominated by "the policy cannot walk yet", not by mass; the arms
differ from the controls in several ways besides mass; and n = 2. **The
consequence for the C2 remedy is the important part: before proposing a
per-body or per-mass cost, measure whether physics is already supplying one.**
If forward progress recovers as the policy learns despite rising mass, the
pressure is insufficient; if it stays suppressed in proportion to mass, a size
cost would be **double-charging** a penalty the simulator already applies. That
check belongs before the remedy, not after.

### The seeds disagree about the READOUT and agree about the POPULATION

With one arm suspended, a two-seed disagreement about the direction of
morphological change would be most of what we know about morphology. So it
matters which quantity is disagreeing.

| | epoch 0 → 10 | peak |
|---|---|---|
| **s2 readout** `n_bodies` | **12 → 25** | 26 |
| **s3 readout** `n_bodies` | **12 → 17** | 21 (epoch 7) |
| s2 population `sampled_bodies_mean` | **14.1 → 18.9** | 19.4 |
| s3 population `sampled_bodies_mean` | **15.0 → 19.1** | 19.4 |

> **The readouts diverge — 25 against 17 — and the populations do not. Both
> populations rose monotonically from ~14-15 to ~19, and at epoch 9 they were
> identical at 19.4.**

**This is E3's lesson applying to E3.1, and it is the second time it has
changed a conclusion.** §3c of [`D3_E3_ADVERSARIAL.md`](D3_E3_ADVERSARIAL.md)
had to rewrite E3's stop rule because the mean-action design is the *mode* of
the design distribution, not the distribution — there, the readout collapsed to
a 0-motor stump at epoch 5 while 30-79% of sampled designs were still actuated.
Here the same gap runs the other way: the readout says the seeds are heading
in opposite directions, and the population says they are doing the same thing.

**What E3.1 can and cannot say about morphological direction:**

* **Can**: the design *distributions* of both seeds are growing, together, from
  ~14-15 to ~19 bodies, and neither collapses. That is what the falsifier test
  needs, and it is answered.
* **Cannot**: anything about the *mode's* direction. One seed's readout is at
  25 and the other's at 17, and **E3.1 has not established a direction of
  morphological change for the mean-action design.** With n = 2 that is not
  dressed up as more.

**This strengthens the case for resuming `rtg_e31_s1`** (suspended at epoch 5,
`epoch_0005.p` saved, `--epoch 5` resumes it). A third seed is worth more when
the two in hand disagree than when they agree — and on the readout they do.

---

# THE EPOCH-20 VERDICT: both falsifiers NOT FIRED, and growth self-limited

*The pre-registered window has closed on both arms (s2 at epoch 24, s3 at 23).*

## The falsifiers

| | `rtg_e31_s2` | `rtg_e31_s3` |
|---|---|---|
| **F1** — `log_std` > −0.9645 within epoch 20 | **NOT FIRED** (worst −1.5033) | **NOT FIRED** (worst −1.5061) |
| **F2** — `p_act4` → 0 by epoch 20 | **NOT FIRED** (min 0.80) | **NOT FIRED** (min 0.95) |
| **Outcome C** — population ≥ 26 for ≥ 5 epochs | not saturated (18.60) | not saturated (16.85) |

**`log_std` never approached the boundary** — it fell monotonically from −1.503
to −1.583, moving *away* the whole time. **`p_act4` has been 1.000 continuously
since epoch 3** on both arms: every one of 20 sampled designs carries four or
more motors, at every epoch, for twenty consecutive epochs.

**Against E3, on the same instrument at the same epochs:**

| | E3 (`log_std` 0) | **E3.1 (`log_std` −1.5)** |
|---|---|---|
| readout motors | **0** from epoch 0 | **7-11** throughout |
| `p_act4` at epoch 3 | 0.300 | **1.000** |
| `p_act4` at epoch 17 | **0.000** | **1.000** |
| population `motors_mean` | → 0.005-0.055 | 5.4-6.0 → ~9 |

> **The fix works.** The failure that stopped E3 at epoch 19 — the design search
> deleting every actuator — does not occur when the control cost is affordable
> from step 0. The derivation in §3f predicted this before the run and the run
> did not contradict it.

## Growth self-limited — the size cost is unnecessary

The mass question raised against Outcome C is answered, and the answer is that
**the remedy would have been a mistake.** Both arms grew, peaked, and receded:

| | readout peak | now | mass peak | now | population peak | now |
|---|---|---|---|---|---|---|
| s2 | **26** @ ep9 | **21** | **1.363** @ ep9 | **1.196** | 19.55 @ ep11 | 18.60 |
| s3 | **21** @ ep7 | **15** | **1.186** @ ep12 | **0.948** | 19.40 @ ep9 | 16.85 |

*(original ant: 0.879 kg, 13 bodies; ceiling 29)*

**Neither arm approached the ceiling.** Both turned around near epoch 9-12 and
have been shedding bodies and mass since — s3's mass is back to 0.948 against
the original 0.879. **Outcome C did not fire and, on this evidence, will not.**

**And the mechanism is visible.** Forward progress bottomed on both arms at
exactly the epoch mass peaked, and recovered as mass came back down:

| | `r(mass, max_fwd)` | n |
|---|---|---|
| s2 | **−0.855** | 5 |
| s3 | **−0.880** | 4 |
| *pooled* | *−0.605* | *9* |

*The pooled figure is weaker than either arm because the two carry different
mass offsets — the same between-group dilution that made the GPU-vs-bodies fit
invalid earlier in this document. **The within-arm correlations are the valid
ones.***

**What this does and does not establish.** It does not establish causation:
n = 4-5 per arm, both series are time-indexed, and forward progress at epochs
4-24 is still dominated by "the policy cannot walk yet". What is robust and
does not depend on the correlation is the **behaviour** — both arms grew,
peaked and receded without any size term in the reward. **Physics is already
charging for mass, the design search found the charge, and adding a per-body or
per-mass cost would double-charge it.** That check was worth doing before
proposing the remedy rather than after.

## Where E3.1 stands against the frozen-body control

At matched epoch 19, `max_fwd`: **s3 0.279 m, s2 0.137 m** against the controls'
**0.19 / 0.12 m**. So E3.1 is tracking the reference arm's early trajectory,
one seed slightly ahead. The controls did not locomote until epochs 79-84 and
did not reach goal 0.5 until 144-149, so **nothing about the task being solved
is knowable yet** — goal rate is 0.00 on both arms, exactly as expected here.

## What is still open

* **The rung's actual question** — does the design+control loop win the task —
  is unanswered and cannot be answered before ~epoch 150.
* **n = 2**, with `rtg_e31_s1` suspended at epoch 5 and resumable.
* **The floor arm** (`rtg_e31f_s*`) has not started.
* The morphology **direction** for the mean-action readout remains
  unestablished: both arms now recede, but from different peaks (26 and 21) to
  different levels (21 and 15).

---

# THE E3.1 RESULT: the design+control loop WINS the adversarial task — on one seed of two

*`rtg_e31_s2` completed all 400 epochs ("training done!"). `e3_posthoc.py` on
its final checkpoint, one instrument, both protocols, 20 episodes each.*

## s2 solved it, and the evolved body is 3.3x faster than the frozen one

| | **E3.1 s2 (design LIVE)** | frozen-body control | E3 (design LIVE, `log_std` 0) |
|---|---|---|---|
| **goal, mean-action** | **1.00** | 0.95 | **0.00** |
| **goal, stochastic** | **1.00** | 0.95 | 0.00 |
| forward | **5.57 m (100.7%)** | 4.82 m | 0.01 m |
| fell | **0.00** | 0.05 | 1.00 |
| **speed** | **4.950 m/s** | 1.498 m/s | 0.000 |
| episode length | **75.3 steps** | 217.6 | 20.8 |
| R | **+1442.0 ± 10.1** | +1452.1 ± 285.4 | +20.8 |
| body | **18 bodies, 6 motors** | 13 bodies, 8 motors | 5 bodies, **0 motors** |
| mass | 1.470 kg | 0.879 | 0.470 |
| limb length | **0.611 m mean / 10.395 total** | 0.377 / 4.525 | 0.279 / 1.118 |

> **This is the first time on this project that a design+control loop has won
> this task.** E3's question — *can Transform2Act's design+control loop win an
> adversarial task?* — is answered **yes**, on one seed of two.

**The evolved body is not merely adequate, it is better than the one it started
from.** 4.95 m/s against the frozen ant's 1.50, finishing in **75 steps where
the frozen ant needs 218** — a third of the time, against an opponent that
needs 491. It got there with **fewer motors (6 against 8)** and **much longer
limbs** (0.611 m mean against 0.377, total 10.4 m against 4.5). Return variance
collapsed to **±10.1** against the control's ±285.4: it is not winning
sometimes, it is winning identically every time.

**And the design converged.** 100 sampled designs give **5 distinct topologies
with the most common at 93%**, against E3's 90-149 distinct of 200. The
skeleton search settled on one body plan and stayed there — which E0 and E1
never observed on this creature, and which is the behaviour the rung was built
to look for.

Body freezing check inverted as it should be: **96 of 134 mjModel arrays change**
under the trained policy, so the design stages were live throughout.

## s3 did not solve it — 1 of 2, and that is the honest headline

| | s2 | **s3** |
|---|---|---|
| first `max_fwd` > 1.0 m | epoch 109 | epoch **179** |
| first `max_fwd` > 2.5 m | epoch 149 | **never** (349 epochs) |
| first goal ≥ 0.5 | epoch 204 | **never** |
| latest | **goal 1.00, 5.58 m** | **goal 0.00, 0.78 m** |
| final body | 18 bodies, 6 motors, 1.470 kg | 12 bodies, 6 motors, 0.938 kg |

Both arms cleared both falsifiers and kept `p_act4` = 1.000 throughout, so
**neither failed the way E3 did**. But only one learned the task. Against the
frozen-body controls — which solved it on **both** seeds — that is a real
difference and it is the central limitation of this result.

**s2 was also slower than the controls to every milestone** (locomotion 109 vs
79-84, goal 0.5 at 204 vs 144-149). Design search costs epochs; it then
overtakes on the final policy.

## A correction to the epoch-20 report: the mass claim reverses over the full run

At epoch 20 I reported `r(mass, max_fwd)` = **−0.855** / **−0.880** and
concluded that *"physics is already charging for mass, the search found the
charge, and a per-body cost would double-charge it."* **Over the full run that
is wrong.**

| | early window (≤ ep 24, n = 5) | **full run (n = 80 / 70)** |
|---|---|---|
| s2 | −0.855 | **+0.800** |
| s3 | −0.881 | **+0.088** |

**s2 grew from 1.165 to 1.470 kg *and* went from 0.15 m to 5.58 m.** Mass and
forward progress rose together for the whole second half of the run; the
negative correlation was an early-training transient in which the policy could
not walk at any mass. The conclusion that survives is narrower and now better
supported: **growth self-limited and never approached the 29-body ceiling**
(Outcome C did not fire), so a size cost is still unnecessary — **but not for
the reason I gave.** It is unnecessary because the search stopped on its own at
a body that works, not because mass is being penalised.

*Seventh correction in this experiment produced by more data, and the same
shape as the others: a small early window supported a clean mechanism that the
full series reversed.*

## Status

* `rtg_e31_s2` **complete**, 400 epochs, solved.
* `rtg_e31_s3` running at epoch ~352, goal 0.00 — will complete for the record.
* `rtg_e31_s1` **resumed** from its suspended `epoch_0005.p` now the card has
  room. With the two completed seeds disagreeing, the third is worth more than
  it would have been.
* The floor arm (`rtg_e31f_s*`) has still not started.

## Both primary seeds are now final — and the difference is whether the design CONVERGED

`rtg_e31_s3` also completed 400 epochs. Its post-hoc, same instrument:

| | **s2 (solved)** | **s3 (did not)** |
|---|---|---|
| goal, mean-action / stochastic | **1.00 / 1.00** | **0.00 / 0.00** |
| forward | **5.57 m (100.7%)** | 0.70 m (13.5%) |
| fell | 0.00 | 0.20 |
| speed | **4.950 m/s** | −0.054 m/s |
| episode length | **75.3** | 488.4 |
| R | **+1442.0 ± 10.1** | −347.2 ± 398.2 |
| final body | 18 bodies, 6 motors, 1.470 kg | 12 bodies, 6 motors, 0.949 kg |
| limb length mean / total | **0.611 / 10.395 m** | 0.473 / 5.201 m |
| **sampled topologies (of 100)** | **5, most common 93%** | **9, most common 49%** |
| `r(fall, R)` | *undefined* (no falls) | **+0.991** |
| fall premium | — | **+961.9** |

> ### RETRACTED: "the seed that solved it is the seed whose design search converged"
>
> **That claim was wrong and is withdrawn.** It read the two final-epoch
> 100-design probes (s2: 5 topologies / mode 93%; s3: 9 / 49%) as if they
> characterised the runs. **The per-epoch series inverts it — s3 was *more*
> converged than s2 in every window measured:**
>
> | epoch | s2 `top_share` | s3 `top_share` |
> |---|---|---|
> | 50 | 0.05 | 0.20 |
> | 100 | 0.15 | 0.40 |
> | 150 | 0.55 | **0.95** |
> | 200 | 0.60 | 0.85 |
> | 250 | 0.85 | **0.95** |
> | 300 | 0.70 | **0.95** |
> | 350 | 0.75 | 0.60 |
> | 399 | **0.90** | 0.60 |
>
> | mean `top_share` | s2 | s3 |
> |---|---|---|
> | epochs 0-150 | 0.171 | **0.326** |
> | epochs 150-399 | 0.739 | **0.809** |
> | epochs 200-399 | 0.738 | **0.805** |
> | whole run | 0.525 | **0.626** |
>
> s3 reached 0.95 by epoch 150 while s2 was still at 0.55, and only fell back in
> the last ~50 epochs. **The endpoint I quoted is the one moment in 400 epochs
> where the comparison runs the other way.**
>
> *Third instance in this experiment of a mechanism inferred from one time point
> in a series that contains a reversal — after the mass correlation
> (−0.86 early, +0.80 over the full run) and the σ "acceleration" (accelerating
> to epoch 59, decelerating after). Same fix each time: **plot the series before
> proposing the mechanism.** These per-epoch numbers were already on disk in
> `census/*_morph.csv` when I wrote the claim.*

### What the series does support — premature lock-in, as a hypothesis

Offered as a hypothesis with its test, not as a finding.

**When did each seed's modal plan stop changing, and what did it lock onto?**

| | modal plan locked at | plan at lock-in | modal changes |
|---|---|---|---|
| **s2** | **epoch 368** | 18 bodies, 6 motors, 1.415 kg, limb 0.601 m | 45 |
| **s3** | **epoch 145** | 12 bodies, 6 motors, 0.891 kg, limb 0.479 m | 52 (all before ep 145) |

**And the ordering against each seed's own performance is the sharp part:**

| | locked | first fwd > 1.0 m | first fwd > 2.5 m | first goal 1.00 |
|---|---|---|---|---|
| **s2** | 368 | 109 | 149 | **249** — *119 epochs before it locked* |
| **s3** | **145** | 179 — *after locking* | **never** | **never** |

> **s2 solved the task and then settled. s3 settled at epoch 145 — before it had
> shown any locomotion at all — and spent the remaining 255 epochs on a plan
> that never exceeded 2.5 m of the 5.0 m required.**

**Neither seed ever visited the other's terminal plan** — s2's `50271e7f5d26`
never appears in s3's modal series and s3's `901ec8c2e00b` never appears in
s2's. They explored disjoint regions, so this is not a case of one seed finding
the good body and the other passing it by.

**On this reading convergence is not what distinguishes success — timing and
target are, and early convergence is a liability.** But it is two runs, the
lock-in epochs are read from the *mean-action* modal topology (the mode, not
the population — §3c's standing caveat), and "locked onto a bad plan" and
"failed for another reason and stopped moving" are not separable from this
data. `rtg_e31_s1` is the tiebreaker and its `top_share` trajectory is now the
most informative thing it will produce: at epoch 53 it is at **0.25 with 30
modal changes**, i.e. still exploratory — closer to s2's trajectory than s3's
at the same stage (s3 was at 0.20 by epoch 50 and 0.95 by 150).

**s3 reproduces E2's correlation structure exactly**: `r(fall, R)` = **+0.991**
with a measured **+961.9** fall premium, because it never scores and so return
is still bimodal by ending. s2 has no falls at all in 20 episodes, so the
statistic is undefined there — the same degenerate-at-both-ends behaviour
E2.1's `d2rep` showed. **Return measures competence only on the arm that has
any.**

**This is a 1-of-2 result and the mechanism of the split is NOT established.**
Both arms had identical hyperparameters, cleared both falsifiers, and held
`p_act4` = 1.000 throughout. **They differed in an outcome we cannot yet
explain.** The premature-lock hypothesis above fits the series; it is not
established by it.

## Status and the instrumentation failure this transition produced (again)

* `rtg_e31_s2` **complete, solved**. `rtg_e31_s3` **complete, not solved**.
* `rtg_e31_s1` resumed from `epoch_0005.p`, now at epoch ~52 of 400.
* **`rtg_e31f_s1` launched** — the floor arm (`min_motors = 4`), which had never
  started. Epoch 0: 16 bodies, 8 motors, `p_act4` 1.000, `log_std` −1.506.

**And the watcher was pointing at the wrong set again.** `watch.sh` was started
with its default `CFGS` covering only the three *primary* arms, so when the
floor arm launched it produced no CSV and no falsifier check — the third time
in this experiment that an instrument silently kept pointing at the previous
set of runs at a transition. It was repointed by **environment variable rather
than an edit**, which is why the file was written that way, and the floor arm's
census appeared within one cycle. The lesson is now three-for-three: **every
time new arms start, the instrumentation must be re-pointed and re-asserted,
and the assertion is the only thing that catches it.**

---

# E3.1-D — why did s3 fail? Freeze its body and train a fresh controller

*Pre-registered before the run, as the falsifiers were.*

## The question

s2 and s3 had identical hyperparameters, cleared both falsifiers, held
`p_act4` = 1.000 — and one solved the task while the other never exceeded 0.70 m.
Two explanations the data so far cannot separate:

* **(a) the body is incapable** — s3 locked at epoch 145 onto a 12-body,
  0.949 kg, 5.201 m-limb plan that cannot run 5 m in 491 control steps;
* **(b) the body is fine and the controller failed** — a training failure that
  merely coincided with an early design lock.

**The frozen-body machinery separates them, and it is already calibrated**: the
E3 controls established that a frozen 13-body ant under `force_identity_design`
reaches goal 0.95 in 400 epochs on this exact task, regime, budget and
instrument.

## The arms

| arm | body | rationale |
|---|---|---|
| `rtg_e31d_s3body` | s3's epoch-400 design, **frozen** | the question |
| `rtg_e31d_s2body` | s2's epoch-400 design, **frozen** | **control** — it should score; without it a null on s3 is uninterpretable |
| *(already done)* | unmodified 13-body ant, frozen | the E3 controls, goal **0.95** |

Each cfg differs from `rtg_e3c_s1.yml` in **one field**, `model_xml_file`.

## Outcomes, written down before the run

* **s3's body SCORES** → the body was capable and the failure was the
  controller or the design/control interaction. **Kills explanation (a)**, and
  makes premature lock-in much less interesting since the plan locked onto was
  fine.
* **s3's body CANNOT SCORE with a fresh controller and the full budget** →
  s3 locked onto a genuinely dead body. **Direct support for the lock-in
  hypothesis**, and the most informative outcome available.
* **s2's body fails to score** → the harness cannot score on *any* evolved body
  and **a null on s3 means nothing**. This is why the control is not optional.

## The caveat, stated before running

**This is not the experiment s3 ran.** A fresh controller here gets a **fixed**
target where s3 was chasing a moving one — strictly easier. So *"it scores"*
proves the body was **not the blocker**; it does **not** prove s3 could have
found the controller. The asymmetry runs one way: a **null** on s3's body is
strong evidence, a **pass** is weaker.

## The gate

`gate_e31d.py` — **10 checks, 0 failed**. Each body loads as the evolved one
(12/6 and 18/6 against the original ant's 13/8, matching topology hashes), mass
and limb totals match the dumped design to 1e-3, the scripted opponent survives
the round-trip with all 13 bodies and 8 motors and still follows
`1 − v·Δt·k` to **0.000e+00**, and each body is **identical across 10 episodes**
of destructive random design actions.

**One tolerance, recorded rather than asserted away.** The gate originally
demanded bit-equality with the exported XML and **failed on s3's body: 15 of 134
arrays moved.** The cause is a `sin`/`arcsin` round-trip in the identity
attribute step that lands **1.273e-08** off in the genome — present on *both*
bodies — which flips the last digit where a value sits near the XML's 6-dp write
precision. It is a **one-time snap**: `reset_robot` rebuilds from
`init_xml_str` every episode, so it re-applies from the same start and cannot
compound (verified identical at episodes 1, 2, 3, 10, 50, 100, 200). The
physical size is **3.483e-08 kg on a 0.949 kg body — 3.7e-06 %**. The right
assertion is that every episode runs the *same* body, which it does; the
constant offset is a documented tolerance.

---

# E3.1 FINDINGS — design+control wins, on 2 of 3 seeds, and the failure was the CONTROLLER

## The headline

> **Transform2Act's design+control loop solves an adversarial task, and the
> evolved bodies beat the fixed one — on 2 of 3 seeds. The one failure was the
> controller, not the morphology: s3's own body scores with a fresh controller,
> and reaches goal 0.5 *earlier* than either winning seed did.**

| arm | epoch | goal | speed | steps to goal | body |
|---|---|---|---|---|---|
| **s2** | 399 | **1.00** | **4.89 m/s** | **76** | 18 bodies / 6 motors |
| **s1** | 321 | **1.00** | 3.72 m/s | 91 | 16 bodies / 7 motors |
| **s3body** (s3's body, frozen, fresh controller) | 186 | **0.60** ↑ | 2.58 m/s | 131 | 12 bodies / 6 motors |
| s3 | 399 | **0.00** | — | — | 12 bodies / 6 motors |
| *frozen 13-body ant (E3 control)* | 400 | 0.95 | 1.50 m/s | 218 | 13 / 8 |
| *E3, `log_std` 0* | 19 | 0.00 | — | — | 5 / **0** |

## Three distinct solutions — the design space has multiple viable optima

The two winning seeds did **not** converge on the same creature, and the
diagnostic adds a third working body:

* **s2 — 18 bodies, 6 motors, 1.470 kg, 10.4 m of limb**, bounding at
  **4.89 m/s**, goal in **76 steps** of the 491 available.
* **s1 — 16 bodies, 7 motors**, at **3.72 m/s**, 91 steps.
* **s3's body — 12 bodies, 6 motors, 0.949 kg, 5.2 m of limb**, flipping at
  **2.58 m/s**, 131 steps.

All three beat the unmodified ant's **1.50 m/s / 218 steps**, by 1.7x to 3.3x.
**This is not one optimum found repeatedly; it is three different bodies, none
of which is the ant we started from, all of which work.** Neither winning seed
ever visited the other's topology.

## RETRACTED: premature lock-in as a causal story

§ above hypothesised that s3 failed because it **locked at epoch 145 onto a
12-body plan that could not do the task**. **The diagnostic kills that.**

> **s3's body reaches goal ≥ 0.5 at epoch 139 with a fresh controller —
> earlier than s1 (199) or s2 (204).** The plan s3 locked onto is not merely
> adequate, it is **easier to train than either body that won.**

The hypothesis is **withdrawn as a causal account**, not softened. What
survives is the *timing fact* — s3's modal plan stopped changing at epoch 145,
before it had shown any locomotion, where s2's kept moving to epoch 368 — and
that fact now has no demonstrated consequence. **s3 failed for a controller
reason we have not identified.**

*The pre-registered caveat runs in our favour here and is restated: the
diagnostic had the easier job (a fixed target where s3 chased a moving one), so
"the body is capable" is the weaker of the two inferences. But goal 0.5 by
epoch 139 clears that bar comfortably.*

## What this isolates: controller-seed sensitivity

Holding the body fixed and re-drawing the controller turns a total failure into
a success **faster than either winner**. So the variance that produced E3.1's
1-of-3 failure lives in the **controller/optimisation**, not in the design
search. Three seeds of design+control gave two wins; the same morphology under
a fresh controller gave a third. **Any future claim about design search on this
task needs enough seeds to survive one dead controller** — E3.1 would have read
as an outright failure at n = 1 had we drawn s3.

## The floor arm is UNINFORMATIVE at n = 1, not negative

`rtg_e31f_s1` (floor + σ) is at **goal 0.00, forward 0.66 m, epoch 193** — and
it carried the **largest** predicted margin in §3f-iii (+210.3 against the σ
fix alone at +86.2).

**That is not evidence against the floor.** Given the controller-seed
sensitivity just demonstrated — one of three primary seeds failed completely on
a body that trains easily — **n = 1 cannot distinguish "floor + σ is a worse
fix" from "drew an s3-like controller".** The base rate of controller failure
we measured is 1 in 3; a single arm failing is entirely consistent with it.

**What would settle it: 3 seeds of the floor arm**, matching the primary arm's
n. With a 1-in-3 controller-failure rate, 3 seeds give ~70% chance of at least
two successes if the floor is as good as the σ fix alone, against ~0% if it is
genuinely broken. Reported as **inconclusive**, and it will stay inconclusive
until it is run at n ≥ 3.

## Late-training watch on s1 — checked, and it is not a regression

Flagged 2026-09-05 when s1's epoch-329 eval read **goal 0.90, down from 1.00**,
with the question of whether the final post-hoc should be taken at the best
checkpoint rather than blindly at 400.

**Aggregated before comparing, per the measurement rule, and the drop is noise
on top of a monotone improvement:**

| epochs | goal (window mean) | speed (window mean) | n evals |
|---|---:|---:|---:|
| 100-199 | 0.030 | 0.424 | 20 |
| 200-249 | 0.740 | 2.601 | 10 |
| 250-299 | 0.910 | 3.074 | 10 |
| 300-400 | **0.983** | **3.566** | 6 |

Goal rate and speed both rise monotonically across every window. The epoch-329
reading is a **single failed episode out of 10** (`fall_rate` 0.10 on the same
row — one fall), which is the smallest quantum a 10-episode eval can move.
Nothing in the series supports a late-training decline: the two highest speeds
of the whole run are epochs 319 (3.725) and 324 (3.631), both after the
supposed onset.

**The best-checkpoint rule is adopted anyway, as policy rather than as a
response to a regression**: the final post-hoc is taken at the best checkpoint
by (goal rate, then speed), reported alongside the epoch-400 number, and the
two are stated separately whenever they differ. Best so far is **epoch 319,
goal 1.00, 3.725 m/s**. If epoch 400 is within noise of it, the epoch-400
number is the headline and the best is reported beside it.
