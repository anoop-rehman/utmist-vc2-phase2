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

> **Outcome C fires when** readout `n_bodies` ≥ **27** (within 2 of the ceiling)
> **sustained over ≥ 5 consecutive epochs** with `p_act4` ≥ 0.9.

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
