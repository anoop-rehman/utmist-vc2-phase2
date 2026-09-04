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

**Seeds**: 3 for E3, 2 for the control.

---

## 2. The two arms, and why the control is not optional

| arm | cfg | design stages | device | seeds |
|---|---|---|---|---|
| **E3** | `rtg_e3_s{1,2,3}` | **LIVE** | GPU | 1, 2, 3 |
| **GNN control** | `rtg_e3c_s{1,2}` | identity-forced | CPU only | 1, 2 |

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

### The gate's own finding, before any training

> **A randomly evolved body falls in 12 of 12 episodes at zero torque, mean
> length 21 steps, where the unevolved ant falls in 1 of 10.**

E2 found that a *controller* can reach the degenerate ending by learning to
tip over. This says a *design* reaches it without needing a controller at all.
It is why the fall rate, the morphology and the correlation pair are logged
from epoch 0 rather than reconstructed post-hoc.

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
