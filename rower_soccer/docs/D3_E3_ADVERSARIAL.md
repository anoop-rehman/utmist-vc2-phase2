# D3 M3 E3 — Transform2Act's design+control loop on an adversarial task

*Launched 2026-09-04. The experiment is `PLAN_D3_M3.md` section 1, rung E3.
Every number below names the command that produced it; anything not measured
is in "Not tested" at the end.*

**Creature**: our DeepMind ant (`dev_ant_body.xml`, 13 bodies / 9 joints /
8 motors) converted to Transform2Act's `Robot` dialect
([`D3_M3_E1_ANT_CONVERTER.md`](D3_M3_E1_ANT_CONVERTER.md)), on both sides.
**Task**: CompetEvo's `run-to-goal-ants-v0` against E2's scripted opponent
([`D3_E2_RTG.md`](D3_E2_RTG.md) §1). **Reward regime**: E2.1's `d2rep`
([`D3_E21_CURRICULUM.md`](D3_E21_CURRICULUM.md)). **Morphology**: this is the
first D3 rung where it is **not frozen**.

## The question, and why it needed two gates before any compute

> Can Transform2Act's design+control loop win an adversarial task?

Every D3 result so far has a frozen morphology: E2 and E2.1 verified 134
mjModel arrays identical under each arm's own trained policy, and E1.1 forced
the design stages to the identity action. So two things had to be established
before spending anything.

**(a) The design stage must actually change the simulated body.** A design
stage that silently no-ops would reproduce E2.1's frozen-body numbers on
E2.1's own instrument, with the design heads taking gradients the whole time,
and would read as a clean, boring, completely wrong null. `gate_e3.py` proves
the **mirror** of E2's gate, and proves it in two independent places, because
"the design stage wrote it" and "the simulator ran it" are different claims.

**(b) The fall-dodge hazard is not fixed.** `D3_E2_RTG.md` §6 measured a
degenerate ending worth ~+826: a fall ends the episode before the scripted
opponent's certain goal at step 491 and so never pays the −1000. `d2rep`
avoids it by never weighting the sparse term above 15.4%; the rule is
untouched. **Morphology is a far wider channel to that optimum than control
alone** — measured, not asserted: see §3.

---

## 1. Two decisions, stated before the runs

### 1a. The termination rule is KEPT, unchanged

`D3_E2_RTG.md` §6 left E3 the choice of keeping CompetEvo's fall rule, paying
the loser its −1000 on a fall as well, or dropping the sparse term when an
episode ends in a fall. **E3 keeps the rule exactly as CompetEvo defines it**,
for one reason: E2.1's frozen-body result — goal 0.95/1.00 with zero falls in
40 episodes — was obtained under that rule, and it is the control this rung is
read against. Changing the rule at the same time as turning the design stages
on would be two changes and one result, and an E3 null would not be
attributable to either. The hazard is **instrumented instead**, from epoch 0
(§3), and gated as a live property of evolved bodies rather than a
frozen-body inheritance (`gate_e3.py` phase 7).

### 1b. The budget is 400 epochs × 50,000 = 20.0M env steps per arm

Chosen from the design-stage convergence data rather than from the control
budget, and stated with the arithmetic:

* **Control needs far less.** E2.1's `d2rep` reached goal 0.95 at **4.0M
  steps** (epoch 79) on the frozen body — 80% of E2's own budget and 26% of
  D2's. Control is not what sets this number.
* **Design search sets it.** On *their* ant, E0 measured 187/188/187 distinct
  topologies of 200 at epoch 20 falling to 34/20/27 with a 20-41% most-common
  share at epoch 100. On *our* ant, E1 measured 190/187 at epoch 20 and
  **63/101 distinct with a 5.5-7.0% most-common share at epoch 100** — i.e.
  our ant at epoch 100 sits where their ant was at epoch 40-50, and the
  mean-action design had changed at 10 of 11 censuses on both seeds. Neither
  had converged.
* So **400 epochs is ~2x the horizon on which their ant concentrated, ~4x the
  point at which ours was still fully diverse, and 5x the 80 epochs control
  needed.** It is also exactly E2.1's budget, which makes the frozen-body MLP
  number (0.95/1.00), the frozen-body GNN control here, and the E3 arms three
  readings at one matched budget.

**Seeds**: **3** for E3, 2 for the control. Three was kept rather than traded
for concurrency, and the reason is that n = 2 is already this project's
recorded weak point — `D3_E21_CURRICULUM.md` §7 names it as its central
statistical limitation, and its `flat` arm's two seeds differed by a factor of
two on goal rate (0.15 vs 0.35). **E3 has a wider outcome space than E2.1
did**, because morphology varies as well as control, so it needs at least as
many seeds and not fewer. Three still cannot characterise a spread; it can only
show whether the seeds agree in sign.

---

## 2. The two arms, and why the control is not optional

| arm | cfg | design stages | device | seeds | when |
|---|---|---|---|---|---|
| **E3** | `rtg_e3_s{1,2,3}` | **LIVE** | GPU | 1, 2, 3 | first |
| **GNN control** | `rtg_e3c_s{1,2}` | identity-forced | GPU | 1, 2 | after E3 |

**They run serially, and that is a measured decision.** The first launch put
all five up at once. It did not work, for two independent reasons, both
recorded here rather than smoothed over:

* **CPU.** Five arms is 3x10 + 2x8 = **46 sampler threads against a 10.2-CPU
  quota** (§5). Under that load the two CPU-only control arms logged **zero
  epochs in 15 minutes** while the three E3 arms degraded from 346 s to ~7 min
  per epoch.
* **GPU.** Sampled over 90 s the card is cyclical, not leaking: ~8.6 GB during
  sampling but **19.0 GB of 20.475 at the update peak** with three E3 seeds
  alone — 93%. That is the number that killed D1 once already (E1's two seeds
  took the card to 19.95 GB and D1 died asking for 8 MB). Nothing else could
  be added to it.
* **And the control arms' own problem was not contention.** khrylib's
  `agent.py` sets `OMP_NUM_THREADS=1` at import and `env-gpu.sh` sets it
  again, so a CPU-only arm's PPO update runs **single-threaded**: still
  running after 700 s, against 150 s for the same update on the GPU. Setting
  the env var after torch is imported does not move torch's thread pool;
  `torch.set_num_threads` does, and `--torch-threads` exists for that. It is
  not the fix used here.

**The control does not need to run beside E3 — it only needs to exist before
E3 is interpreted.** So E3 takes all three seeds unimpeded and the controls get
the free card afterwards. Total wall clock is lower than five contended arms
and E3 keeps its third seed.

*The two control arms were stopped by signal, not by stop-file, and that needs
saying plainly. The stop-file is only checked at the END of an epoch and these
arms had not completed one in 15 minutes, so it could never have fired. Before
signalling them, `nvidia-smi --query-compute-apps` was read: it listed the MPS
server and the three E3 arms only, and both control processes had an empty
`CUDA_VISIBLE_DEVICES`. They held no CUDA context, so the MPS
never-kill-a-CUDA-client rule did not apply to them. They had logged zero
epochs, so nothing was lost. **The three E3 arms will be ended by stop-file
only.***

They differ in **one cfg field**, `env_specs.force_identity_design`, run
through the **same trainer** (`train_e3_gnn.py`) and the **same instrument**
(`e2_eval.evaluate`, unchanged from E2).

**Why the control exists.** E2.1 supplies the frozen-body **MLP** number under
`d2rep`: goal 0.95/1.00, zero falls in 40 episodes, 5.00 m of the 5.00 m
required. Without the matching **GNN** number an E3 null is ambiguous between
"the design loop failed" and "the GNN controller cannot do this task", and
that ambiguity is not resolvable after the fact. It also answers E1.1's
architecture question in a regime where both controllers *can* do the task,
which E2 could not: at 5.0M steps on the flat reward neither reached the goal
in 40 episodes, so E2's 3.0x return gap was a mixture of fall rates rather
than a comparison of controllers.

`rtg_e3_s1.yml` differs from E2's `rtg_gnn_s1.yml` in exactly four things —
`force_identity_design` removed, `max_epoch_num` 100 → 400,
`save_model_interval`/`additional_saves`, and the seed — and `rtg_e3c_s1.yml`
in the last three only. Every other training and design hyperparameter is
E2's, checked field by field in `gate_e3.py` phase 1.

### What was built

| file | what it is |
|---|---|
| `rower_soccer/t2a_port/train_e3_gnn.py` | the trainer: `train_e2_gnn.py`'s loop plus the curriculum, the per-epoch morphology summary, the correlation instrument, decoupled video and GCS archiving. `train_e2_gnn.py` is untouched, so E2's four arms stay reproducible from the file that produced them |
| `rower_soccer/t2a_port/e3_morph.py` | the morphology and fall-dodge instruments, and `rng_guard` |
| `rower_soccer/t2a_port/e3_video.py` | renders a clip from a checkpoint **name**, so the video cadence and the archival cadence are independent |
| `rower_soccer/t2a_port/e3_posthoc.py` | the headline table: what the design stages did under the trained policy, both protocols, the dodge statistics |
| `rower_soccer/t2a_port/gate_e3.py` | the gate, 8 phases, 56 checks |
| `design_opt/cfg/rtg_e3_s{1,2,3}.yml`, `rtg_e3c_s{1,2}.yml` | the five arms |
| `runs/d3_e3_adversarial/{launch,collect}.sh` | launch and post-hoc |

**Two edits to shared files, both additive**, and both regression-gated:

1. `e2_eval.roll` now also records `bodies_exec` — the body count **after** the
   design stages, where `bodies` was the count before them. Under
   `force_identity_design` the two are equal by construction, so no E2 or E2.1
   number moves; `best_median_worst` appends `nb=` to a panel label only when
   they differ, so E2's clips render byte-identically.
2. `e2_eval.evaluate` now reports `n_requested` and `design_fail_rate`. `roll`
   returns `None` when the *design* stages end an episode — an evolved body
   that fails to compile or to reset. With the body frozen that cannot happen;
   with the design stages live it can, and silently dropping those episodes
   would bias every rate in the dict toward the designs that survive.

**The curriculum enters through `Agent.custom_reward`** — khrylib's own hook,
already wired into `sample_worker` — so the PPO buffer gets
`alpha*dense + (1-alpha)*parse` while `LoggerRL.step` keeps logging the raw
env reward. That is E2.1's invariant ("the curriculum touches the buffer and
nothing else") obtained **without editing a Transform2Act file at all**.

---

## 3. The gate

```
cd /workspace/Transform2Act && source env-gpu.sh
CUDA_VISIBLE_DEVICES= .venv-gpu/bin/python .../t2a_port/gate_e3.py
```

`runs/d3_e3_adversarial/logs/gate_e3.log` — **56 checks, 0 failed**, in eight
phases, each with at least one negative control.

| phase | what it establishes | headline |
|---|---|---|
| 1 cfgs | the arms are the experiment and E2's are untouched | E3 arms design-live, control arms identity-forced, both at 400 x 50,000; every other training/design hyperparameter and the whole task equal E2's GNN arm; the termination rule is `{'max_nsteps': 500}`, unchanged. NEG: E2's own cfgs still force identity and are still at 100 epochs |
| 2 **the mirror gate** | design changes the body MuJoCo integrates | 20 episodes of destructive random design actions change **96 of 134** mjModel arrays including `body_mass`/`geom_size`/`actuator_gear`/`body_pos`, with body counts spanning **10-21** and motor counts 8-17. The compiled model IS the designed body: capsule radius to **4.6e-07**, capsule length to **1.0e-06**, actuator gear to **3.1e-07** (the XML is the channel and `sync_node` writes 6 dp). A targeted add adds exactly one body, a targeted remove removes exactly one, a targeted gear action moves the genome by 1.0000 and the physical gear 150.0 → 390.3 inside the cfg's 20-400. **NEG: the SAME action sequence with `force_identity_design` changes 0 arrays and holds 13 bodies** — E2's result, reproduced as this gate's own negative control. NEG: a `Robot` param the simulator never compiled IS detected |
| 3 opponent | the scripted opponent survives an evolved body | after design, 13 opponent bodies and 8 opponent motors intact; its joints still LAST in qpos/qvel (so `_opp`'s `nq − qposadr` slice stays correct as our body grows); root x still follows `1 − v·Δt·k` to **0.000e+00** over 6 different designs; still crosses x = −4 at step **491**. NEG: at 1.0 m/s the crossing moves to step 334 |
| 4 reward | it still measures OUR agent | over 200+ steps across 3+ different evolved bodies, `dense + parse == reward` to **0.000e+00** and `dense == forward − 0.5·Σa² + 1.0` to **0.000e+00**; the fall test reads our root z while the opponent is held at 0.5347 |
| 5 curriculum | it is E2.1's, on the GNN path | `alpha_at` is **bit-identical** to `train_e11_mlp.Trainer.alpha` over 400 epochs for both `cur` and `d2rep` (max delta **0.000e+00**); d2rep runs 1.000000 → 0.846400 and never crosses E2.1's critical **0.739** (min 0.8464, so the fall-dodge is worth at most **153.6** against +1000 flat). End to end on a real 3,000-step sample: **the logged return is bit-identical with and without the curriculum** (−2170.3519 both), while **the buffer differs on the 5 sparse steps and holds 0 of the ±1000 at alpha = 1**. NEG: `curriculum_steps = 0` returns `None`; alpha = 0 gives the sparse term alone |
| 6 the instrument | it does not perturb what it measures | a probe inside `rng_guard` leaves the numpy/torch/env streams **bit-identical** over 24 sampled values. NEG: the same probe outside it shifts 18 of 24 |
| 7 **the fall-dodge** | still in the task, and WIDER through morphology | on the frozen body the idle control both falls and loses, and the two sparse-term distributions **do not overlap** (0.0 on a fall, −1000 on a loss). **On 12 randomly evolved bodies the same zero-torque policy falls in 12 of 12, mean episode length 21 steps, against 0.10 on the frozen body.** That is the hazard this rung adds, measured before training |
| 8 reporting | the instrument reports what changed, and E2's numbers do not move | frozen body: `bodies_exec == 13`, `design_fail_rate == 0`, so no E2 number moves. `body_summary` excludes the opponent from every aggregate (13 of 27 bodies, 8 of 16 motors, 0.879 kg of 1.757). NEG: after a design step it reports 13 → 17 bodies, 0.879 → 1.127 kg, limb length 4.525 → 9.134 m |

**Regressions re-run in full after the two shared-file edits:**
`gate_e2.py` **41 passed, 0 failed** (`logs/gate_e2_regression.log`) and
`gate_e21.py` **28 passed, 0 failed** (`logs/gate_e21_regression.log`), so
E2's scene, opponent, frozen body, reward, termination, observation, E1.1
regression, and E2.1's whole curriculum apparatus are untouched by this work.

---

## 3a. A result before training: the DESIGN SPACE is tilted toward the dodge

This came out of `gate_e3.py` phase 7 rather than out of a run, but it is a
finding in its own right and not a gate byproduct, so it is stated as one.

| policy | body | episodes | fell | mean episode length |
|---|---|---|---|---|
| zero torque | the unevolved 13-body ant | 10 | **1** (0.10) | — |
| zero torque | 12 randomly evolved bodies | 12 | **12** (1.00) | **21 steps** |

Same policy — none. Same task, same opponent, same termination rule. The only
difference is the body.

> **E2 established that a *controller* can reach the task's degenerate ending
> by learning to tip over. This establishes that a *design* reaches it with no
> controller at all.**

`D3_E2_RTG.md` §6 measured the ending's value: a fall stops the episode before
the scripted opponent's certain goal at step 491, so it never pays the −1000,
and that is worth +750 to +900 inside every arm. E2 then found that ranking its
seven arms by return reproduced the fall-rate ranking exactly
(`r(fall rate, return) = +0.989`, `r(forward progress, return) = +0.019`).
E2.1's `d2rep` inverted that to −0.517 / +0.947 by holding the sparse term
under 15.4% weight for the whole run — **it avoided the dodge, it did not
remove it**.

What the table above adds is that with the design stages live the dodge is
reachable through a channel E2 and E2.1 did not have, and that the *untrained*
distribution over bodies already sits almost entirely inside it. That is why
morphology, fall rate and E2's correlation pair are logged from epoch 0 rather
than reconstructed afterwards (§4), and it is why keeping the termination rule
unchanged (§1a) had to be an explicit decision rather than a default: E3 is
the first rung where the rule is load-bearing on the *design* search and not
only on the control policy.

**What would falsify it, stated with it.** This is 12 episodes on bodies drawn
from *destructive random design actions* — every body told to add or remove,
every attribute kicked over its full range. That is the gate's stimulus, and it
is deliberately not the distribution the trained search visits: the sampled
census at epoch 0 already draws 20 distinct topologies with body counts 9-19,
and a trained policy's distribution will be narrower and different again. So
the claim this table supports is **"the design space contains a large region
where falling is the default"**, not "the search will end up there". The
falsifier is the run itself: if the E3 arms' per-epoch fall rate falls away
from 1.00 while body counts move off the random distribution, the tilt is
real but escapable, which is a different and better result. §3b pre-registers
how that is read.

---

## 3b. Pre-registration — how the two outcomes will be read, fixed BEFORE the data

*Written before any E3 epoch beyond 1 existed. This project has twice had to
retract reasoning built after seeing a number — the mean-speed "arithmetic not
tactics" argument, and the back-agent reinterpretation of `DESIGN_2V2.md` §11 —
and both times the reasoning was constructed post-hoc. The readings below are
committed in advance.*

The two statistics that decide it are E2's own pair, recomputed on E3's arms by
`e3_analyse.py`, plus the per-epoch morphology series:

| | E2 (5.0M, flat) | E2.1 (20.0M, d2rep, frozen body) |
|---|---|---|
| `r(fall rate, return)` | **+0.989** | **−0.517** across arms; −0.94 over the trained arms |
| `r(forward progress, return)` | **+0.019** | **+0.947** |

### Reading A — "the dodge was found through the body"

**Trigger, both parts required:**
1. `r(fall rate, return)` drifts back toward E2's **+0.989** — concretely, the
   pooled per-arm value ends **positive on at least 2 of the 3 seeds** and the
   across-arm value (the 6-row table: 3 E3 arms, 2 control arms, the idle
   floor; E2's was 7 rows) ends above **+0.5**, while
   `r(forward progress, return)` stays below **+0.3**; **and**
2. the per-epoch morphology series **correlates with fall rate**. Stated as a
   number so it cannot be argued afterwards: `e3_analyse.morph_vs_fall`
   reports `r(column, fall rate)` across epochs for body count, motor count,
   mass, mean limb length, limb-length sum, gear, distinct topologies and
   most-common share. The trigger is **|r| > 0.5 on at least one of body
   count, motor count, mass or mean limb length, with the same sign on at
   least 2 of the 3 seeds**. That is the claim that the fall rate is being
   bought with the body rather than merely coinciding with it.

**Conclusion if triggered:** the design loop found the degenerate optimum
through morphology. That is a real and publishable result — it is the first
demonstration on this project that widening the search space widens the set of
degenerate solutions — and it **promotes fixing the termination rule from
optional cleanup to the next rung, E3.1**: pay the loser its −1000 on a fall as
well, or drop the sparse term when an episode ends in a fall, and re-run.

### Reading B — "the design loop works"

**Trigger, both parts required:**
1. the correlations hold near E2.1's structure — pooled per-arm
   `r(fall rate, return)` **negative** and `r(forward progress, return)`
   **above +0.5**, on at least 2 of the 3 seeds; **and**
2. the evolved bodies trend toward locomotion — the mean-action fall rate
   falls toward 0 and forward progress rises toward the 5.00 m the task needs,
   i.e. a non-zero goal rate on the mean-action protocol.

**Conclusion if triggered:** Transform2Act's design+control loop wins an
adversarial task, and **E4 (self-play, both sides evolving) is on**.

### Neither — the ambiguous middle, named in advance so it cannot be filed under a preference

The result is **ambiguous, and will be reported as ambiguous**, if any of:

* **the correlations are indeterminate** — |pooled `r(fall rate, return)`| < 0.3,
  or the three seeds disagree in sign;
* **the fall rate stays high but so does forward progress**, or vice versa, so
  the two triggers of a single reading do not both fire;
* **the goal rate is 0.00 on every seed with a fall rate below ~0.3** — that is
  neither the dodge nor a working loop, it is "the design search made the task
  harder", which is a third outcome and gets its own name;
* **the mean-action design ends with `morph/n_motors` at or near 0** — call
  this outcome **"the search removed the ability to act"**. It is a third
  reading, distinct from both A and B, and it is named here because it is
  already visible at epoch 4 (see below). A body with no actuators cannot
  reach the goal and cannot exploit the dodge deliberately either: it simply
  topples. Reading A would be wrong for it — there is no *optimisation toward*
  falling, only an inability to do anything else — and Reading B is obviously
  wrong. `D3_E1_ANT.md` predicted this reachable state before E3 existed: our
  ant "can erode to a **0-motor** blob theirs cannot reach", because 12 of its
  16 possible additions are passive dead weight and its depth-1 leg stubs are
  jointless. If E3 ends here, the finding is about the **design space of this
  particular creature**, and the next rung is a constrained design space (a
  floor on actuator count), not a change to the termination rule.
* **`design_fail_rate` is materially non-zero** (> 0.05), because then a
  fraction of episodes never reached the execution stage and every rate is
  conditioned on the designs that compiled;
* **the seed spread exceeds the effect** — E2.1's `flat` arm's two seeds
  differed by a factor of two on goal rate (0.15 vs 0.35), and E3 has a
  *wider* outcome space than E2.1 because morphology varies as well as control.

An ambiguous result is reported as an ambiguous result, with the mechanism I
think explains it, and the next rung is chosen to disambiguate rather than to
confirm.

### An observation at epoch 4 of 400, recorded as a transient, not a result

*Logged here the moment it was seen, so that if it persists it cannot be
presented as a prediction and if it resolves it cannot be quietly dropped.
This project's own rule: a result from a still-running run is a transient.*

Seeds 1-3 at epochs 0-4 have `morph/n_motors` = **0** on the mean-action
design, with `morph/n_bodies` falling 6 → 5 and mass 0.51 → 0.47 kg. Fall rate
is 1.00 and forward progress 0.01 m, which a 0-motor body explains completely
without any appeal to the dodge. `design_fail_rate` is 0.00, so these bodies
compile and run; they simply cannot act.

Two reasons not to read anything into it yet. It is **epoch 4 of 400**, and the
policy is barely off its initialisation. And under `d2rep`'s alpha ≈ 0.998 the
objective is almost pure `dense = forward − 0.5·Σa² + 1.0`, which pays **+1.0
per step survived** — so a blob that topples at step 21 banks ~21 where a
standing, actuated ant banks up to 491. The gradient available to the design
head points *away* from this body. Whether it follows it is the experiment.

The curriculum was verified live against its own schedule at the same time:
`e3/alpha` reads 1.000000, 0.999616, 0.999232, 0.998848, 0.998464 at epochs
0-4 against an expected 1.0, 0.999616, 0.999232, 0.998848, 0.998464.

### What the frozen-body GNN control decides

The control arms (`rtg_e3c_s{1,2}`, run after E3 on the freed card) are what
separate "the design loop failed" from "the GNN controller cannot do this
task", and the reading is fixed here too:

* control reaches a **high goal rate** (near E2.1's frozen-body MLP 0.95/1.00)
  and E3 does not → the deficit is the **design loop**;
* control is also near **0.00** → the deficit is the **GNN controller**, E3 is
  uninterpretable as a morphology result, and the rung to fix is the
  controller;
* control lands **between** → the two effects are mixed and the E3-minus-control
  difference is the only quantity that can be attributed to design.

---

## 4. Logging and the measurement protocol

* **wandb, metrics and video in ONE run per arm**, logged inline from the
  training process as each epoch finishes. The first clip is rendered after
  **epoch 0** and then every 6 epochs. Video renders in a subprocess off a
  **transient** checkpoint that is deleted afterwards — a GNN checkpoint here
  is 157 MB and a ~15-minute video cadence off archival checkpoints would cost
  6 GB per seed on a disk with 13 GB free.
* **With the design stages live the three panels are three different
  creatures**, so the clip shows design variation as well as gait, and each
  panel carries its own body count.
* **Evaluation**: `e2_eval.evaluate`, E2's instrument unchanged —
  mean-action as the headline, stochastic beside it, 20 episodes post-hoc
  (10 inline every 5 epochs), identical episode seeds, plus an **idle
  zero-torque floor** through the same code path.
* **Forward progress, not return, is the primary readout.** E2's whole lesson
  was that return tracked falling.
* **Per-epoch morphology**: mean-action design (topology, body/motor count,
  mass, limb lengths, radii, gears, depth histogram, full genome) plus a
  20-design sampled census (distinct topologies, most-common share, body-count
  range), written to `results/<cfg>/e3_epochs.jsonl` and logged to wandb under
  `morph/`.
* **Per-epoch fall-dodge**: `r(fall rate, return)`, `r(forward progress,
  return)` and the measured fall premium over the evaluation episodes, logged
  under `dodge/`, both per-evaluation and **pooled over the last 5
  evaluations** — this project's rule is to aggregate before comparing rates,
  and `r` over 10 episodes is noise.
* **Checkpoints** are archived to
  `gs://vc2-2026-checkpoints/_t2a_archive/<cfg>/` every 50 epochs and pruned
  locally to the two most recent plus `best.p`, only after `gsutil stat`
  confirms the remote size matches.

---

## 5. Cost and the machine

**A correction to `PLAN_D3_M3.md` §2 carried by this experiment: the machine
does NOT have 48 usable cores.** `nproc` reports 48 and the plan's budget
table says "48 cores", but the container's cgroup quota is
`/sys/fs/cgroup/cpu.max = 1020000 100000` — **10.2 CPUs**. `vmstat` under the
five arms shows 25-34 runnable tasks against 20-21% of 48 cores busy, which is
the quota, and the in-container load average is not a reliable reading of it.
This is consistent with E2.1's own recorded slowdown (`T_sample` 31 → 41 → 83 s
as arms were added, "peak load ~38 of 48") — that run was throttled too, and
its "78% CPU idle" headroom estimate was wrong for the same reason it already
records itself as optimistic.

**Placement, measured.** Epochs 0-1 of the E3 arms ran while the two CPU-only
control arms were up: T_sample 152-166 s, T_update 150-162 s, T_eval 10-26 s.
The re-measurement with the CPUs freed is in §5a.

### 5a. Per-epoch cost, measured

Wall clock taken from the **timestamped** `results/<cfg>/log/log_train.txt`
rather than from a stopwatch, so the "instrument" column is a residual and not
an estimate. Epochs 0-2 ran with the two CPU-only control arms up; 3 onward
did not.

| epoch | wall | sample | update | eval | instrument + video | condition |
|---|---|---|---|---|---|---|
| 0 | — | 165.8 | 161.8 | 18.3 | — | 5 arms |
| 1 | 341.8 | 152.4 | 150.3 | 25.0 | 14.1 | 5 arms |
| 2 | 317.4 | 146.7 | 142.4 | 22.9 | 5.5 | 5 arms |
| 3 | 258.2 | 98.7 | 132.3 | 24.2 | 2.9 | **3 arms** |
| 4 | 198.7 | 71.0 | 115.0 | 9.5 | 3.2 | 3 arms |
| 5 | 181.4 | 61.1 | 109.3 | 7.5 | 3.4 | 3 arms |

*(seed 1; seeds 2 and 3 agree to within a few seconds — seed 2 is faster at
132.4 s by epoch 5.)*

* **Shedding the two CPU-only arms cut `T_sample` from 166 s to 61 s** and the
  epoch from 342 s to 181 s. Transform2Act's own ETA went **1 day 14 h ->
  19 h 28 m**.
* **The per-epoch instrumentation this rung adds costs 3-5 s**, video epochs
  included — 2% of the epoch, not the several minutes the first (contended)
  epochs suggested. The morphology census, the inline evaluation and the
  best/median/worst clip are not what makes this run long.
* **Independent cross-check**: 3 epochs in a 600 s wall-clock window over the
  same period = **200 s/epoch**, against the log's 181-199 s. Projected 400
  epochs: **~18-21 h per seed, all three concurrent**.

**The ETA is contingent, and on the thing under test.** `T_sample` is falling
because the *bodies are shrinking* — 13 nodes to 5 — so each GNN forward is
cheaper. If the design search grows bodies back the epochs slow down again,
and so does the card: the GPU peak over the same unimpeded window was
**17,681 MiB of 20,475 (86%)** with 5-6-body designs, against the **19.0 GB
(93%)** measured with three seeds while designs were still near 13 bodies.
**2.8 GB of headroom is what this run has**, which is why nothing else goes on
the card and why a monitor is armed at 19,200 MiB — E1 lost D1 at 19.95 GB.

*(Results section to be filled in when the runs finish.)*

---

## 6. Not tested / not claimed

*(To be completed with the results. Known before the runs:)*

* **The termination rule was not changed** (§1a). The fall-dodge is present in
  E3 exactly as it is in E2 and E2.1; `d2rep` avoids it by weight, not by
  repair. E4 inherits the same choice.
* **The opponent has only ever been run at 0.68 m/s**, and it is scripted, not
  learned and not self-play. §0a's argument that the CompetEvo observation does
  not depend on the opponent's morphology is what makes a fixed opponent safe
  while ours changes — but a *learned* opponent has not been tried at any rung.
* **Only the GNN.** No MLP arm is run with the design stages live, because
  Transform2Act's design heads are the thing under test and a plain MLP has
  none. The frozen-body MLP number comes from E2.1.
* **No hyperparameter was swept.** Everything but the four listed cfg fields
  is E2's, which is itself E1.1's.
* **Engine.** mujoco-py 2.1 with CompetEvo's own PGS/1000, not D1/D2's
  mujoco_warp with Newton/100.
* **n = 3 seeds for E3, n = 2 for the control.** Three seeds cannot
  characterise a spread; they can show whether the seeds agree in sign. E2.1's
  `flat` arm's two seeds differed by a factor of two on goal rate, and E3's
  outcome space is wider than E2.1's because morphology varies as well as
  control (§1b). Any single-condition mean here is to be read with that.
* **§3a's 12/12 is 12 episodes on destructive-random designs**, which is the
  gate's own stimulus and deliberately not the distribution a trained search
  visits. It supports "the design space contains a large region where falling
  is the default", not "the search will end up there". §3a states the
  falsifier; §3b pre-registers how the run decides it.
* **The two protocols have disagreed twice on this project** (E1.1, E2 §7d), so
  both are reported for every arm and the ordering is checked on both. Nothing
  here assumes they agree.
* **No arm was run with the design stages live and the FLAT reward**, so E3
  cannot separate "the design loop needs `d2rep`" from "`d2rep` is simply the
  regime that works on this task". E2.1 established the latter on a frozen
  body; the interaction with design freedom is untested.
