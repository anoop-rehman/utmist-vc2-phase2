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
