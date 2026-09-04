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
* **the design POPULATION ends with `p_act4` below 0.05 and a step share below
  0.25** — call this outcome **"the search removed the ability to act"**.
  *(Originally written as "the mean-action design ends with `morph/n_motors` at
  or near 0"; corrected in §3c, because the readout collapsed to a 0-motor
  blob at epoch 5 while 30% of sampled designs still carried four or more
  motors and supplied 91% of the training steps.)* It is a third
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

**Superseded in part by §3c: this is the MEAN-ACTION readout, and the
population it is the mode of is 82% actuated.** The note is left standing
because it is what was seen and when, and because naming the outcome is what
prompted the measurement that corrected it.

Two reasons not to read anything into it yet. It is **epoch 4 of 400**, and the
policy is barely off its initialisation. And under `d2rep`'s alpha ≈ 0.998 the
objective is almost pure `dense = forward − 0.5·Σa² + 1.0`, which pays **+1.0
per step survived** — so a blob that topples at step 21 banks ~21 where a
standing, actuated ant banks up to 491. The gradient available to the design
head points *away* from this body. Whether it follows it is the experiment.

The curriculum was verified live against its own schedule at the same time:
`e3/alpha` reads 1.000000, 0.999616, 0.999232, 0.998848, 0.998464 at epochs
0-4 against an expected 1.0, 0.999616, 0.999232, 0.998848, 0.998464.

### 3b-i. The +1.0-per-step argument, MEASURED — and it reverses under the flat reward

The claim in the transient note below was that under `d2rep`'s alpha ~ 0.998
the objective pays +1.0 per step survived, so the gradient available to the
design head points away from the 0-motor blob. That was an argument. Measured
(`e3_blob_probe.py`, seed 1's own `best.p`, alpha 0.997696 = the live value at
epoch 6, 10 mean-action episodes, identical seeds):

| arm | body | steps | env R | = dense | + sparse | **objective at alpha 0.9977** | endings |
|---|---|---|---|---|---|---|---|
| **blob** — the arm's own mean-action design, its own control head | 5 bodies, **0 motors** | 20.9 | **+21.2** | +21.2 | 0.0 | **21.2** | fell 10/10 |
| **ant_pol** — the SAME weights on the frozen 13-body ant | 13 bodies, 8 motors | 491.0 | −677.8 | +322.2 | −1000.0 | **319.2** | lost 10/10 |
| **ant_idle** — the frozen ant at zero torque | 13 bodies, 8 motors | 458.5 | −465.6 | +334.4 | −800.0 | **331.8** | fell 2, lost 8 |

**The prediction holds, and by more than the argument claimed: keeping the body
is worth +310.6 of objective per episode, a factor of 15.7.**

All three seeds, same probe (`posthoc/blob_probe_s{1,2,3}_e0006.json`):

| seed | blob | blob obj | `ant_pol` obj | `ant_idle` obj | gain from keeping the body | **under the FLAT reward** |
|---|---|---|---|---|---|---|
| 1 | 5 bodies, 0 motors | 21.2 | 319.2 | 331.8 | **+310.6 (15.7x)** | blob beats `ant_idle` by **+486.8** |
| 2 | 6 bodies, 0 motors | 21.2 | 331.5 | 331.8 | **+310.6 (15.7x)** | **+486.8** |
| 3 | 6 bodies, 0 motors | 21.2 | 271.8 | 331.8 | **+310.6 (15.7x)** | **+486.8** |

*The blob column being identical to the tenth across three independently
seeded runs is not a copy-paste error, and the arithmetic says why: with no
actuators the control cost is exactly 0 and the policy's output is discarded,
so `dense` collapses to `sum(forward) + 1.0 x steps`, and `sum(forward)`
telescopes to `net_dx / dt` = 0.004 / 0.015 = **+0.27**. The whole +21.2 is
20.9 survive-bonus steps plus 0.27 m/s of drift. `ant_pol` does differ by seed
(319.2 / 331.5 / 271.8), which is the control head, as it should.*

Two things fall out of this that the argument did not contain:

1. **`ant_pol` (319.2) is statistically indistinguishable from `ant_idle`
   (331.8).** The same weights that produce the blob are, on an unevolved body,
   no better than zero torque — which is expected at epoch 6 and is exactly the
   separation the arm was built for: **the blob's deficit is the BODY, not the
   control head.** Same policy, one body evolved and one not, 21.2 against
   319.2.
2. **Under the FLAT reward the ranking REVERSES.** Read the `env R` column: the
   blob scores **+21.2**, the same weights on the frozen ant (`ant_pol`)
   **−677.8**, and the frozen ant at zero torque (`ant_idle`) **−465.6** —
   because both ants survive to the opponent's certain goal at step 491 and pay
   the −1000 while the blob topples at step 21 and dodges it. **On E2's reward
   the 0-motor blob is the optimum: by +699.0 against `ant_pol` and by +486.8
   against `ant_idle`.** *(Two different comparisons; both are stated because
   quoting one figure alone invites it being read against the other row.)*
   Only `d2rep`'s alpha ≈ 0.998 suppresses the sparse term far enough to invert
   it.

That is `D3_E2_RTG.md` §6's fall-dodge appearing through morphology in its
purest available form, and it is a direct, quantitative vindication of taking
E2.1's reward regime into this rung rather than E2's: **had E3 been run on the
flat reward, the design search would have been rewarded for building a body
that cannot act.**

### 3b-ii. Nothing in the design search penalises an unactuated body

Stated plainly because it is load-bearing for §3c's first row. `design_fail_rate
0.00` does **not** mean the search approves of these bodies — it means it has
no opinion about them. `AntEnv.allow_add_body` and `allow_remove_body`
(`design_opt/envs/ant.py`) constrain **depth and child count only**; neither
reads `body.joints` or the actuator list. `apply_skel_action` and
`set_design_params` return `False` only when `reload_sim_model` raises, and
`transit_execution` only when `reset_state` raises. A 0-motor body does
neither: it compiles, it resets, and it runs — measured, 10 of 10 episodes
above with zero design failures.

So **the only thing in the whole loop that penalises an unactuated body is the
return**, and §3b-i is the measurement of how hard it penalises it (15.7x under
`d2rep`; *negatively* under the flat reward). There is no structural floor on
actuator count, which is why §3c's first row prescribes adding one rather than
changing the termination rule.

### 3c. THE DECISION POINT: epoch 100, and what each state there means

*A pre-registration that says what an outcome means but not when it is judged
is a run you watch indefinitely and then rationalise. The epoch and the rule
are fixed here.*

**Epoch 100 (5.0M steps, a quarter of the budget), chosen from our own data:**

* **E1 produced a decisive morphological verdict on this same creature in this
  same machinery at exactly 100 epochs** — "the evolved creature stops being a
  quadruped", torso height halved 0.561 -> 0.270 on both seeds, mass nearly
  doubled, limbs airborne 71-76%. Whatever E3's design search is going to say
  about this ant, E1 says 100 epochs is enough for it to say it.
* **Control has had its measured time by then.** E2.1's `d2rep` reached goal
  0.95 at epoch 79 on the frozen body. A design-on arm at epoch 100 has had
  more than the control side needed.
* **The design side has had a comparable share.** At epoch 100 E0's ant had
  concentrated to 34/20/27 distinct topologies of 200 (20-41% most-common) and
  ours to 63/101 (5.5-7.0%) — not converged, but far from epoch 20's 187-190.
* **It is checkpointed.** `additional_saves: [20, 100]` puts checkpoints at
  20/40/60/80/100, so the decision has the run-up to it, not only the endpoint.

### The rule keys on the POPULATION, not the greedy readout — and the run has already shown why

The first version of this rule keyed on the **mean-action** design's
`morph/n_motors` alone. **That was wrong, and this project's own data says so
without needing E3 to finish.** E1 measured our ant's topology distribution at
epoch 100 — the horizon this decision sits at — as **63 and 101 distinct
topologies of 200 sampled, most-common share 5.5-7.0%**, against their ant's
20-41%. At the epoch we decide at, the distribution is *provably* not
concentrated for this creature, so the mode is not the population and a rule
that reads only the mode is reading the wrong object.

E3's live census says the same in real time: **18-20 distinct topologies of 20
sampled with a 0.05-0.10 top share**, while the mean-action `topo` hash has
been a single value (`9a51d315a8da`) since epoch 5, and `sampled_bodies_mean`
sits at 9.65-12.45 against the mean-action design's 5.

**Definition, fixed before it is judged.** `p_act1` is the fraction of sampled
designs with **≥ 1** motor; **`p_act4` is the fraction with ≥ 4** — one per
original leg, the minimum that could plausibly walk. **`p_act4` is the number
§3c's rule uses**, and every "% actuated" figure below is `p_act4` unless it
says otherwise.

**Measured** (`e3_population_probe.py`, 200 sampled designs per row).
Provenance: `census/pop_rtg_e3_s1_UNTRAINED.json` and
`census/pop_rtg_e3_s{1,2,3}_best.json`, all committed, and aggregated into
`census/population.csv`.

| checkpoint | **epoch it was saved at** | mean-action readout | pop. motors mean (max) | `p_act1` | **`p_act4`** | **step share of ≥4-motor designs** | distinct topos |
|---|---|---|---|---|---|---|---|
| seed 1, untrained | — | 19 bodies, **7 motors** | 5.71 (12) | 0.995 | **0.825** | 0.991 | 199 / 200 |
| seed 3 `best.p` | **0** | 6 bodies, **0 motors** | 5.26 (10) | 0.995 | **0.790** | 0.989 | 198 / 200 |
| seed 2 `best.p` | **1** | 6 bodies, **0 motors** | 4.45 (10) | 0.985 | **0.680** | 0.980 | 196 / 200 |
| seed 1 `best.p` | **3** | 5 bodies, **0 motors** | 2.51 (9) | 0.820 | **0.300** | 0.910 | 186 / 200 |

**A correction to my own reading of this table, and it is not a small one.**
When first written these rows were labelled "seed 1/2/3 @ `best` (~epoch 7)"
and read as a seed spread — "three seeds that disagree by a factor of 2.6 on
`p_act4` while agreeing perfectly on `n_motors`". **That was wrong.**
`best.p` is written whenever `exec_R_eps` improves, and on these arms that
statistic plateaus almost immediately at the blob's survive-bonus return
(~21), so the three checkpoints are frozen at **epochs 0, 1 and 3** —
recovered from the pickles, whose mtimes (00:58, 01:04, 01:14) all predate the
probe runs (01:32-01:34) and are unchanged since.

So the table is **not three seeds at one epoch**; it is three different seeds
at three different, very early epochs, and the apparent "seed disagreement" is
ordered exactly by epoch. What it actually shows is:

* **at epochs 0-3 the readout was already a 0-motor blob while the population
  was still 30-79% actuated.** That claim stands, and it is what makes §3c's
  change from readout to population correct.
* **whether the population has collapsed by now is UNMEASURED.** The three
  arms are past epoch 12 and there is no population measurement after epoch 3.
  My earlier statement to the effect that "the search did not collapse" is
  established **only for epochs 0-3** and I should not have written it without
  that qualifier.
* the four rows are four different policies, so the
  0.825 → 0.790 → 0.680 → 0.300 ordering **across arms** is confounded with
  epoch and is not a trend.

**But one comparison in that table is NOT confounded, and it is the one that
matters.** Seed 1 supplies **both endpoints on its own**: untrained
`p_act4` = **0.825** and its own epoch-3 checkpoint = **0.300**. That is a
**within-seed drop of 64% in three epochs**, with no seed confound at all.
`pop_motors_mean` moves the same way on the same seed, **5.715 → 2.51**, and
the motor histogram's zero-motor bin goes **1/200 → 36/200**.

So the honest reading is stronger and worse than "unmeasured since epoch 3":
**the available evidence points at the population collapsing, and quickly.**
Two points establish that a drop happened; they cannot establish its *shape* —
decelerating, linear and accelerating are indistinguishable on two points, and
no functional form is fitted here or extrapolated from. `e3_pact_series.py`
prints the series with exactly that caveat attached, and refuses to name a
shape the data cannot support.

**`best.p` is therefore useless as a progress tracker on these arms** and the
real series is the archival checkpoints. `population_watcher.sh` probes each of
epochs 20/40/60/80/100/200/300/400 as it appears; **epoch 20 is the first
measurement of the population under a meaningfully trained policy**, and the
first that can support any statement about whether the search is collapsing.

**Step share is the quantity the rule uses**, because the gradient sees
episodes and not designs. Converting a design share `p` with the measured
lengths: `p·491 / (p·491 + (1−p)·20.9)`. At `p` = 0.05 the step share is
already **0.55**; at `p` = 0.015 it is 0.26. A design share that looks
negligible is not negligible in the batch.

## 3e. THE RESULT: the design search deleted the actuators, and the mechanism is the CONTROL COST

**E3 was stopped at epoch ~19 by its own pre-registered rule**, on 3 of 3 seeds,
by stop-file. The rule (§3c row 1) required three conditions on ≥ 2 of 3 seeds;
all three held on all three.

### 3e-i. The measurement

Population probes of the **live** policy at epoch 17, 200 sampled designs each
(`census/pop_rtg_e3_s{1,2,3}_LIVE.json`):

| seed | readout | `pop_motors_mean` (max) | motor histogram | `p_act1` | **`p_act4`** | **step share** |
|---|---|---|---|---|---|---|
| s1 | 5 bodies, 0 motors | 0.055 (2) | 0: **191**, 1: 7, 2: 2 | 0.045 | **0.000** | **0.000** |
| s2 | 6 bodies, 0 motors | 0.015 (1) | 0: **197**, 1: 3 | 0.015 | **0.000** | **0.000** |
| s3 | 7 bodies, 0 motors | 0.015 (1) | 0: **197**, 1: 3 | 0.015 | **0.000** | **0.000** |

**Not one design in 600 has four motors. The most actuated design in 600 has
two.** The untrained baseline was `p_act4` = 0.825 with a mean of 5.71 motors
and a maximum of 12.

**Confirmed on the ARCHIVAL checkpoints, which is what the result actually
rests on.** The live captures above came from scraping the trainer's transient
video checkpoint; the checkpoints each arm saved on its stop-file exit are on
disk, are re-probeable by anyone, and say the same thing:

| checkpoint (saved at epoch) | readout | motors mean (max) | histogram | `p_act1` | **`p_act4`** |
|---|---|---|---|---|---|
| `rtg_e3_s1/epoch_0019.p` (18) | 5 bodies, 0 motors | 0.04 (2) | 0: **194**, 1: 5, 2: 1 | 0.030 | **0.000** |
| `rtg_e3_s2/epoch_0020.p` (19) | — | 0.005 (1) | — | 0.005 | **0.000** |
| `rtg_e3_s2/epoch_0022.p` (21) | 5 bodies, 0 motors | 0.01 (1) | 0: **197**, 1: 3 | 0.015 | **0.000** |
| `rtg_e3_s3/epoch_0020.p` (19) | 7 bodies, 0 motors | 0.01 (1) | 0: **197**, 1: 3 | 0.015 | **0.000** |

Seven independent 200-design probes across three seeds, two checkpoint sources
and four epochs (17-21): **`p_act4` = 0.000 in every one.**

**And two of the three seeds converged on the SAME degenerate body.** The
mean-action topology hash is `9a51d315a8da` on both seed 1 and seed 2 — the
identical 5-body, 0-motor stump, from independent seeds and independent
initialisations. For scale, E0 measured three seeds on *their* ant landing
0.76-0.82 Jaccard apart and E1 measured 0.75 for our ant's one pair; neither
experiment ever produced two seeds sharing a topology hash. Here the search
does not merely degrade — **it converges, and on the same body.**

### 3e-ii. The mechanism: it is the DENSE control cost, not the sparse fall-dodge

This is the part that matters, and it is not the hazard this rung was built to
watch for.

`dense = forward − 0.5·Σa² + 1.0`. At initialisation `control_log_std` is 0, so
a fresh policy pays about **4.0 per step** in control cost against a survive
bonus of 1.0 — `D3_E21_CURRICULUM.md` §1 measured exactly this and recorded that
"the dense reward's first gradient is *quieten down*". Under E3's `d2rep`,
alpha ≈ 0.998, so **the objective is essentially `dense` alone** and that
gradient is the whole signal.

With the design stages live there are two ways to stop paying it:

1. **learn small actions** — slow, and it only pays off once the control head
   can also keep the body upright and run;
2. **delete the actuators** — immediate, and `0.5·Σa²` becomes *exactly* 0
   forever.

**It took route 2, identically on all three seeds.** The trainer's own
per-step reward is the trace:

| epoch | 0 | 2 | 4 | 6 | 8 | 10 | 14 | 17 |
|---|---|---|---|---|---|---|---|---|
| s1 `train_R` | −2.38 | −1.28 | −0.24 | 0.11 | 0.50 | 0.61 | 0.75 | 0.76 |
| s2 `train_R` | −2.53 | −1.31 | −0.26 | 0.24 | 0.54 | 0.68 | 0.76 | 0.78 |
| s3 `train_R` | −2.51 | −1.28 | −0.24 | 0.37 | 0.60 | 0.68 | 0.76 | 0.77 |

Monotone, concave, asymptoting at **+0.78** — which is the 0-motor body's
ceiling: `+1.0` of survive bonus per step, minus the backward drift the
opponent imposes. The reward went up the whole time. **The run optimised its
objective successfully and the objective was the problem.**

> **`d2rep` cannot prevent this, and that is the structural point.** `d2rep`
> down-weights `parse` — the ±1000, the fall-dodge. **The control cost lives in
> `dense`, which `d2rep` weights at ~1.0.** E2.1's protection is orthogonal to
> the failure that actually occurred, and buying it (alpha ≈ 1) *maximises*
> the weight on the term that caused it.

**Why §3b-i's "+310.6 for keeping the body" did not save it.** That figure
compares the blob against a *standing* actuated ant — a body **plus** a policy
that can hold it up. The design head does not have that policy and cannot get
it without first paying the ~4/step it is busy escaping. The comparison
actually available to it at epoch 0 is different and points the other way:
**pay ~4/step now, or bank ~+1/step now.** §3b-i measured a distant optimum;
the search followed the local gradient.

### 3e-iii. Which of the pre-registered readings this is — and it is none of them

Not **Reading A** ("the dodge was found through the body"): there is no
optimisation *toward* falling. A 0-motor body topples because it cannot do
anything else, and the sparse term never enters — the blob's `parse` is 0.0
(§3d), so the fall-dodge is not what it is exploiting.

Not **Reading B**. It is the **third outcome named in §3b before the data**,
*"the search removed the ability to act"* — with the mechanism now identified,
which the naming did not contain.

**Where §3c was still wrong, and it was wrong in my favour.** §3c relocated the
decision from epoch 100 to "every checkpoint from 20". Even that was late: the
condition was already satisfied at **epoch 17**, and only the live-capture trick
(`catch_live_ckpt.sh`) surfaced it before epoch 20. The original epoch-100 rule
would have run **83 epochs — about 22 hours — past a settled outcome.**

### 3e-iv. The collapse rate: what the points can and cannot support

Asked plainly, and answered plainly.

| seed | points | series |
|---|---|---|
| s1 | 3 | untrained **0.825** → epoch 3 **0.300** → epoch 17 **0.000** |
| s2 | 2 | epoch 1 **0.680** → epoch 17 **0.000** |
| s3 | 2 | epoch 0 **0.790** → epoch 17 **0.000** |

**The drop is established; its shape is not, and I will not name one.** Only s1
has three points. Its absolute rate falls from −0.175/epoch (untrained→3) to
−0.021/epoch (3→17), which *looks* decelerating — but `p_act4` is **bounded
below by 0 and reached the bound**, so the deceleration is forced by the
boundary and is not evidence about the process. Three points against a floor
cannot separate exponential decay from linear-then-floor, and two points
separate nothing. `e3_pact_series.py` prints this caveat with the series and
refuses to fit a form.

**The one shape claim that IS supported comes from a different, better-sampled
series**: `train_R`, 18-21 points per seed, is smooth, monotone and concave to
an asymptote (§3e-ii). That is a claim about the reward the search was
climbing, not about `p_act4`, and the two should not be conflated.

### 3e-v. What this says for the ladder

* **The next rung is a constrained design space**, exactly as §3c row 1
  prescribed: a **floor on actuator count** (or a control cost that does not
  reward amputation — e.g. cost per actuator *present* rather than per action
  emitted, so deleting a motor does not reduce the cost to zero).
* **It is NOT a termination-rule change.** §3d's rule (iii) recommendation
  stands on its own merits for the *fall-dodge*, but it would not have
  prevented this: the blob never collects the sparse term at all.
* **E3.1's spec should be revisited in light of this.** As recorded in
  `PLAN_D3_M3.md` it fixes the termination rule and drops `d2rep`. Dropping
  `d2rep` **raises** the sparse weight and leaves `dense`'s control cost
  untouched, so on its own it does not address §3e-ii. An actuator floor is
  the prerequisite for any design-on rung on this creature.
* **The frozen-body GNN control is now the load-bearing arm**, and it is
  unaffected by all of this: with `force_identity_design` the body cannot be
  edited, so the escape route does not exist. It runs next on the freed card.

---

### 3c-0. THE DECISION POINT MOVED: epoch 100 was 83 epochs too late

*Written when the first live-policy measurement came back, before the other two
seeds had been probed, because the relocation is justified by the rate of change
alone and does not depend on how the other seeds land.*

**Seed 2, epoch 17** (`census/pop_rtg_e3_s2_LIVE.json`, captured from the
trainer's own transient video checkpoint — see below):

| | epoch 1 (`best.p`) | **epoch 17 (live)** |
|---|---|---|
| readout | 6 bodies, 0 motors | 6 bodies, 0 motors |
| `pop_motors_mean` | 4.45 | **0.015** |
| motor histogram | 0-motor bin 3/200 | **0-motor bin 197/200**, max 1 |
| `p_act1` | 0.985 | **0.015** |
| **`p_act4`** | **0.680** | **0.000** |
| step share of ≥4-motor designs | 0.980 | **0.000** |
| distinct topologies | 196 / 200 | 90 / 200 |
| sampled bodies (mean) | 13.76 | 7.82 |

**All three of §3c row 1's conditions are already satisfied on this seed at
epoch 17.** The quantity the epoch-100 rule was going to read has gone from
0.825 (untrained) to 0.000 in seventeen epochs. **A rule that fires 83 epochs
after its own outcome is settled is not a decision rule**, so it moves:

> **Revised: the decision is evaluated at EVERY checkpoint from epoch 20
> onward — archival (20/40/60/80/100…) and any live capture — and fires the
> first time §3c's table resolves on ≥ 2 of 3 seeds at a common checkpoint.**

The three conditions are unchanged; only the *when* moves, and it moves from a
fixed epoch to "as soon as the evidence exists on two seeds". The ≥2-of-3
requirement is kept, which is why one seed at 0.000 does not stop the run on
its own.

**How a live-epoch measurement was obtained at all.** `best.p` is frozen at
epochs 0-3 and the first archival checkpoint is epoch 20, so neither could
answer this. `train_e3_gnn.py` writes `models/_video_tmp.p` every 6 epochs,
hands it to the renderer and deletes it in a `finally`;
`runs/d3_e3_adversarial/catch_live_ckpt.sh` polls at 1 s and copies it out
while it exists. That is a read, it cannot perturb the trainer, and it needed
no change to a running arm.

### The decision, at epoch 100 — SUPERSEDED BY §3c-0, kept for the reasoning

Made on `runs/d3_e3_adversarial/census/<cfg>_morph.csv` (the readout, live) and
`census/pop_<cfg>_e0100.json` (the population, from the epoch-100 checkpoint).

| state at epoch 100 | rule | conclusion |
|---|---|---|
| **all three of:** mean-action `n_motors` = 0, **`p_act4` < 0.05**, **step share of ≥4-motor designs < 0.25** — on ≥ 2 of 3 seeds | **STOP the E3 arms at 100** | The population itself has collapsed: *the search removed the ability to act*. Write E3 up as a null with that mechanism. Next rung is a **constrained design space — a floor on actuator count** — **not** a termination-rule change, because an unactuated body is not an exploitation of the fall rule (§3b). |
| mean-action `n_motors` = 0 but **`p_act4` ≥ 0.05 or step share ≥ 0.25** | **run on to 400** | *The readout collapsed, the search did not* — a named outcome in its own right (§3c-i). It also means the **mean-action protocol is the wrong headline for this rung** and the population column is. |
| **`n_motors` ≥ 4 on the readout** on ≥ 2 of 3 seeds | **run on to 400** | Readings A and B (§3b) take over at 400. Goal rate still 0.00 at 400 with motors intact is the third outcome, *"the design search made the task harder"*. |
| **1 ≤ readout `n_motors` < 4, AND `p_act4` strictly decreasing** over the epoch-20/40/60/80/100 checkpoints | **run on to 200, re-decide ONCE** | At 200 the same table applies and there is **no second extension**. |
| **any seed shows mean-action OR stochastic goal rate > 0** | **run on to 400 regardless** | Do not stop a run that is scoring. |

**Thresholds justified before the data, not after.** `p_act4 < 0.05` is a
**16.5x collapse from the measured untrained baseline of 0.825**, not a number
chosen for convenience. `step share < 0.25` is the binding condition — it
implies `p_act4` below ~0.015, i.e. **fewer than 3 of 200 sampled designs can
walk** — and it is stated in the units the gradient actually works in. Both are
reported because `p_act4` is the number a reader looks for and the step share is
the number that decides.

**What would falsify the population rule itself**: if `p_act4` at epoch 100 is
high while every sampled actuated design still scores 0.00 and falls, then the
population is not the limiting factor either and the answer is about control or
about the task, not about the design search. That is why the last row exists
and why the goal rate can override the whole table.

### 3c-i. "The readout collapsed, the search did not" — a third protocol lesson

E1.1 and E2 §7d each recorded the mean-action and stochastic protocols
disagreeing, which is why this project reports both. E3 adds a sharper version
of the same lesson, because with the design stages live **the two protocols
measure two different creatures**:

* **mean-action = the MODE of the design distribution.** One body, the greedy
  readout, currently a 0-motor blob.
* **stochastic = the POPULATION.** Bodies drawn the way training draws them —
  at epochs 0-3, `p_act1` 0.820-0.995 and mean 2.51-5.26 motors, up to 10.
  (Unmeasured since; see §3c.)

For E2 and E2.1 that distinction was empty: the body was frozen, so both
protocols ran the same creature and the only difference was action noise. For
E3 it is the difference between a statistic about one degenerate body and a
statistic about the thing being optimised. **Both columns are reported for
every arm, as always — but for a design-on arm the stochastic column is the
one that describes the search.**

### 3c-ii. `gear_mean 0.0` — the zero-motor state confirmed from a second field

Worth stating because redundancy is what makes a number trustworthy. The
readout's zero-motor state is visible in **two independent columns** of
`<cfg>_morph.csv` that are computed from different objects: `n_motors` counts
entries in the compiled `mjModel`'s actuator list excluding the opponent's,
while `gear_mean` averages `Actuator.gear` over the `Robot`'s own body
objects. `n_motors` = 0 and `gear_mean` = 0.0 agree on every epoch of every
seed. A bug in one would have to be mirrored by an independent bug in the
other, on the model side and the robot side, to produce this.

Stopping is by **stop-file** (`/tmp/stop_e3_s{1,2,3}`), never by signal.

---

### 3d. THREE TERMINATION RULES x TWO REWARD REGIMES, measured

*Prompted by the user's proposal to "remove the penalty for falling". Read from
source, **there is no fall penalty**: a fall contributes exactly 0 to the
reward and appears only in `done` (`run_to_goal.py`), so what a fall costs is
the rest of the episode's `SURVIVE_BONUS`. The real form of the proposal is
removing `fell` from the termination condition — rule (ii) — and it turns out
to be a better idea than it first looked, for a reason that also makes `d2rep`
a liability.*

`e3_termination_grid.py`, 10 mean-action episodes, alpha 0.997696,
`posthoc/termination_grid.json`. **Probe only — the three live seeds ran the
unmodified rule throughout.**

* **rule (i) `current`** — a fall ends the episode and pays nothing.
* **rule (ii) `nofall`** — a fall does not end the episode.
* **rule (iii) `charged`** — a fall ends the episode **and** is charged −1000.

Rule (iii) needs no separate rollout and that is exact, not an approximation:
under rule (i) a fallen episode's `parse` is 0 by construction, so (iii) is
(i)'s trajectory with −1000 added on exactly the fallen episodes.

Arms: the **blob** (each seed's own design and control head), the **competent
ant** — E2.1's trained `d2rep` MLP on the frozen 13-body ant, the only policy
on this project that plays this task — and the **zero-torque floor**.

#### The two gradients that decide it

| rule | regime | **scoring gradient**<br>(competent − blob) | **upright gradient**<br>(idle ant − blob) | episodes per 50k batch | blob's dead steps |
|---|---|---|---|---|---|
| current | flat | +1330.6 | **−486.8** | 2392 | 0% |
| current | **d2rep** | +531.2 | +310.6 | 2392 | 0% |
| nofall | flat | +1983.1 | **−1.9** | **102** | **95.7%** |
| nofall | d2rep | **+186.9** | **−1.9** | **102** | **95.7%** |
| **charged** | **flat** | **+2330.6** | **+313.2** | **2392** | **0%** |
| charged | d2rep | +533.5 | +312.5 | 2392 | 0% |

**The coordinator's central claim is confirmed, and by more than predicted.**
Removing fall-termination under the flat reward gives a scoring gradient of
**+1983** (predicted ~2100), while the same rule under `d2rep` gives **+187**
(predicted ~100). Down-weighting the sparse term to 0.2% does throw away the
scoring incentive once the dodge is removed structurally: **`d2rep` is 10.6x
worse than flat at rewarding a goal under rule (ii).** Two component
corrections to the derivation, both small and both in the same direction: the
lying blob banks **368.7** of dense, not ~491; and the competent ant measures
**+1351.8** here rather than E2.1's stored +1599, on different episode seeds at
goal 0.90 rather than 0.95/1.00.

**The first correction has a mechanism worth recording on its own, because the
same assumption would misprice any future rule that lets a fallen agent lie in
the opponent's path.** The natural derivation assumes a fallen body just sits
there, so `Σforward` ≈ 0 and `dense` ≈ 491 × the survive bonus. It does not sit
there. `D3_E2_RTG.md` §1 established that the scripted opponent is **effectively
infinitely massive** — its entire state is overwritten after every control step,
so contacts push our agent and the reaction on the opponent is discarded — and a
body lying in its lane is therefore **bulldozed backwards**: `dx` −1.6 to −2.0 m
over the episode, i.e. `Σforward` = −1.94/0.015 ≈ **−129**. The measured
`dense` is 491 − 129 ≈ 362, and 368.7 is what the grid reports. Under rule (ii)
a fallen agent is not a neutral object in the arena; it is a puck.

**But rule (ii) has a defect neither of us anticipated, and it is visible only
because the idle floor was in the grid.** Under `nofall` the upright gradient
collapses to **−1.9** — a fallen body and a standing one score within two
points of each other, in *both* regimes (blob dense 368.7 against idle's
366.8). Removing fall-termination does not create a preference for staying up;
**it makes lying down and standing up equivalent.** The scoring gradient
survives, but the gradient toward the *prerequisite* for scoring does not, and
a policy that cannot yet score sees nothing telling it to stay on its feet.

**And it costs 23x the sample efficiency.** Every episode runs to the full 491
steps, so a 50,000-step batch buys **102 episodes against 2,392**. For the blob
**95.7% of every episode is dead state** — 470 of 491 steps after the fall,
paying only the survive bonus into a body that cannot self-right.

#### The recommendation for E3.1: charge the fall, keep the termination, drop `d2rep`

**Rule (iii) with the flat reward dominates on every axis measured** — the
largest scoring gradient (+2330.6), a restored upright gradient (+313.2), full
sample efficiency (2392 episodes per batch), and no dead state. It has rule
(ii)'s incentive structure without rule (ii)'s two costs.

Three further things the grid says:

1. **A competent policy is indifferent to the rule.** The competent ant scores
   1351.8 under all three rules, identically, because its fall rate is 0.00.
   Changing the termination rule costs a working policy exactly nothing, which
   is what makes rule (iii) safe to adopt.
2. **`d2rep` neuters rule (iii).** The −1000 charge enters the objective scaled
   by `(1 − alpha)` = 0.0023, so it is worth −2.3 and `charged` (18.8) is
   indistinguishable from `current` (21.2). The two fixes are alternatives, not
   complements: **whichever one is used, the other should be dropped.**
3. **Fall-termination was a sample-efficiency device doing double duty as a
   broken reward rule**, and the grid separates the two jobs. Keeping the
   termination preserves the 23x efficiency; charging it supplies the incentive
   the termination alone never did. E2's entire null, and E2.1's need for
   `d2rep` at all, trace to those two jobs having been fused.

#### This does not contradict E3's own choice of `d2rep` — the grid confirms it

Read together with §3b-i: **under the rule E3 actually runs, `d2rep` was the
right call and the flat reward would have been a disaster.** The `current`+flat
row's upright gradient is **−486.8** — the design search would have been
actively *rewarded* for building a body that cannot act — while `current`+d2rep
turns it to **+310.6**. That is exactly the choice §3b-i vindicated, and this
grid re-derives it from a different direction.

What the grid adds is that `d2rep` is a **compensation for a rule defect**, not
a fix for it, and that compensating has a price: it scales the sparse term to
0.2%, which costs 2.5x of the scoring gradient (+531 against +1331) and would
cost 10.6x if the rule were repaired. **Fix the rule and the compensation
becomes the thing holding you back** — hence "whichever one is used, drop the
other", and hence rule (iii) + flat for E3.1 rather than rule (iii) + d2rep.

**What this does NOT say.** It is a scoring of fixed policies under six reward
definitions, not six training runs — it measures the *incentive landscape*,
not what PPO does in it. No arm has been trained under rule (ii) or (iii), and
E2.1 established that a curriculum's realised behaviour can differ sharply from
its nominal shape. The blob checkpoints are from epochs 0/1/3 (§3c), the
competent arm is an MLP where E3 is a GNN, and it is 10 episodes per cell.

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
* **The LIVE evaluation series is the MODE, not the population.** The inline
  `e3/eval_*` curve calls `e2_eval.gnn_actor(..., mean_action=True)`, so with
  the design stages live it evaluates the mean-action design — the very
  readout §3c establishes is unrepresentative. That is a real limitation of
  the live curve and it is not fixable without restarting the arms. **The
  post-hoc (`e3_posthoc.py`) runs both protocols**, and for a design-on arm the
  **stochastic column is the population** and is the one the write-up leads
  with. The live curve is a progress indicator, not a result — which was
  already the rule (`D3_E2_RTG.md` §5: no number in the results table comes
  from a training log), and now has a second reason.
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

* **Shedding the two CPU-only arms cut `T_sample` from 166 s to ~110 s** in
  steady state and the epoch from 342 s to ~215 s. *(This bullet originally
  read "to 61 s" and "to 181 s", from epoch 5. That was a de-phased transient
  and §5b is the correction — the steady state is 100-118 s.)*
* **The per-epoch instrumentation this rung adds costs 3-5 s**, video epochs
  included — 2% of the epoch, not the several minutes the first (contended)
  epochs suggested. The morphology census, the inline evaluation and the
  best/median/worst clip are not what makes this run long.
* **Independent cross-check**: 3 epochs in a 600 s wall-clock window over the
  same period = **200 s/epoch**, against the log's 181-199 s. That window
  straddled the transient; the steady-state figure is **~215-220 s/epoch ->
  ~24 h per seed, all three concurrent** (§5b).

**The ETA is contingent, and on the thing under test.** `T_update` is falling
because the *bodies are shrinking* — 161.8 s to 97.0 s as
`sampled_bodies_mean` goes 14.4 to 7.95. If the design search grows bodies back
the epochs slow down again,
and so does the card: the GPU peak over the same unimpeded window was
**17,681 MiB of 20,475 (86%)** with 5-6-body designs, against the **19.0 GB
(93%)** measured with three seeds while designs were still near 13 bodies.
**2.8 GB of headroom is what this run has**, which is why nothing else goes on
the card and why a monitor is armed at 19,200 MiB — E1 lost D1 at 19.95 GB.

*(Results section to be filled in when the runs finish.)*

---

### 5b. Two wrong explanations for the sampling cost, and the measured one

*Recorded in full, including both wrong turns, because the wrong turns are the
instructive part and because each was killed by a falsifier written down before
its result.*

`T_sample` on seed 1 ran 165.7, 152.4, 146.6, 98.7, 71.0, **61.1**, 99.1,
118.3, 118.6, 114.1, 101.5 over epochs 0-10. Three explanations were offered in
order.

**Wrong answer 1 — "the bodies grew back."** §5a's own guess. Refuted by
`r(mean-action body count, T_sample)` = +0.56 / +0.70 / **−0.08** across the
three seeds — no consistent relationship — and, decisively, by
`morph/sampled_bodies_mean` falling **monotonically 14.4 → 7.95** on seed 1
across exactly the epochs where `T_sample` doubled. The bodies got *smaller*.

**Wrong answer 2 — "my own concurrent probes."** Three blob probes ran
01:22-01:27 and four population probes 01:29-01:34:42, covering epochs 6-9, and
the alignment looked convincing. **The falsifier written down with it was: if
the probes caused it, `T_sample` returns toward 61 s from epoch 10.** Epoch 10
sampled entirely after the last probe finished at 01:34:42 and came in at
**101.5 s**. The attribution fails its own test. The probes were real load
inside a 10.2-CPU quota and they are not free — that lesson stands, and
population probes still run one seed at a time and niced — but they are not the
sustained cause.

**The measured answer — the 61 s was the transient, not the 101-118 s.** The
workload per epoch has been *constant* since epoch 6 while `T_sample` swung by
4x:

| seed 1 | e0 | e3 | e5 | e7 | e9 | e10 |
|---|---|---|---|---|---|---|
| `ep_len` | 43.0 | 29.6 | 29.1 | 27.4 | 27.1 | 27.2 |
| `num_episodes` | 1178 | 1718 | 1729 | 1832 | 1849 | 1844 |
| `sampled_bodies_mean` | 14.4 | 12.1 | 9.75 | 9.15 | 8.3 | 7.95 |
| **`T_sample`** | 165.7 | 98.7 | **61.1** | 118.3 | 114.1 | 101.5 |

Identical episode counts, identical episode lengths, shrinking bodies — and a
4x swing in wall time. That can only be contention. Two further facts pin it:

* **The three seeds have converged on the same value to within 3%** — at their
  epoch 10, `T_sample` is 101.5 / 103.8 / 101.0. Three independent processes
  agreeing that closely is the signature of a shared resource limit, not of
  anything about their bodies, which differ (seed 1's population is 30%
  actuated, seed 3's 79%).
* **Their phase offsets have stabilised.** The spread between the first and
  last seed to finish an epoch grew 14.6 s → 231 s over epochs 0-7 and has sat
  at 232-247 s since. Seeds 1 and 3 finish within **0-2 s of each other**;
  seed 2 runs ~4 minutes ahead. The system has settled.

So epochs 4-6 were a **de-phased, under-loaded window** right after the two
control arms were killed, and 29-72 s was never the steady state. **The steady
state for three concurrent E3 arms on this box is `T_sample` ≈ 100-118 s.**

**Consequence for the ETA, corrected.** §5a's "~18-21 h per seed" was
extrapolated from that transient dip and is **too optimistic**. Steady-state
epochs are ~215-220 s (`T_sample` ~105 + `T_update` ~97 + `T_eval` ~20 +
instrument ~3), giving **~24 h per seed, all three concurrent** — which is what
Transform2Act's own ETA now reads (23:48). `T_update` is meanwhile falling
independently (161.8 → 97.0) as the bodies shrink, so the total has been far
flatter than either component.

**The general lesson, which this project already writes down and I still walked
into twice**: *a number from a still-running run is a transient.* I quoted 61 s
as "unimpeded" one epoch after it appeared, and built an ETA on it. The check
that would have caught it immediately is the one that eventually did: hold the
workload columns (`ep_len`, `num_episodes`) next to the timing column and see
whether the work changed before concluding anything about why the time did.

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
* **The population probe is 200 sampled designs at one checkpoint**, not a
  continuous series: `e3_morph.census`'s motor columns were added after the
  arms launched, so the live JSONL carries topology and body-count statistics
  but not motor counts. The population motor distribution therefore exists at
  the checkpoints (`untrained`, `best`, and 20/40/60/80/100/200/300/400) and
  not at every epoch. Any live claim about motor counts before epoch 20 rests
  on the mean-action readout, which §3c is precisely about not trusting.
* **`step share` uses two fixed episode lengths** — the measured 20.9-step blob
  and a 491-step actuated episode — rather than each design's own length. It is
  a first-order correction for the length asymmetry, not a measurement of the
  batch composition; the batch's real composition is `e3/ep_len` and
  `e3/num_episodes`, which are logged.
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
