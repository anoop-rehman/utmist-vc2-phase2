# D3 M3 E0 — does the skeleton stage explore from a good starting body?

*2026-08-29. The experiment is `PLAN_D3_M3.md` section 1, rung E0. Every number
below names the command that produced it. Anything not measured is in section 7,
"Not tested".*

**Creature: THEIR ant** — `design_opt/cfg/ant.yml`, `design_opt/envs/ant.py`,
`assets/mujoco_envs/ant.xml`. A torso plus four single-segment limbs, 4 motors.
It is **not** the DeepMind ant D1 and D2 use (13 bodies / 10 joints / 8 motors).
Per `PLAN_D3_M3.md` section 0c they are different creatures and nothing here
transfers to ours by assumption.

---

## 1. Which implementation, and why

Checked rather than assumed, because `PLAN_D3_M3.md` left it open.

**Our GPU port (`rower_soccer/t2a_port/train_t2a.py`) has no ant path.** Not a
flag — four separate places are hopper-only:

| | port | their ant |
|---|---|---|
| initial XML | `hopper.xml`, hardcoded (`train_t2a.py:295`) | `ant.xml` |
| `sim_obs_dim` | `5`, hardcoded (`train_t2a.py:297`) | **13** (`ant.py:41`) |
| root in `sim_obs` / `terms` | planar: `qpos[1], qpos[2]` are `(height, ang)` (`batched_exec_env.py:216-219, 236`) | free joint; tilt from a quaternion (`ant.py:167-169`) |
| `index_base` | `max_nchild + 1` = **3** on `ant.yml` (`design_stage.py:83`) | **5**, hardcoded (`ant.py:32`) |

`/workspace/Transform2Act/results/` contains only `hopper_gpu*` runs: the port
has never been run on an ant, and neither has their reference.

**So E0 runs on their CPU reference**, which ships the ant task unmodified. The
48 mostly-idle cores are what their sampler wants anyway. The cost of the
decision is wall clock: ~200 s/epoch against the port's hopper rate.

### The gate that licensed it

```
cd /workspace/Transform2Act && source env-gpu.sh
.venv-gpu/bin/python rower_soccer/t2a_port/gate_their_ant.py     # GATE PASSED
```

Twelve checks, all passing: the starting body is `ant.xml`'s 5 bodies / 4
motors, `njnt 5 / nq 11 / nv 10`; the observation dims are the ant's (13, not
the hopper's 5); 20 sampled designs all reach execution without the XML
round-trip failing; and the execution reward equals `ant.py`'s
`(dx/dt) - 1e-4*mean(ctrl^2)` recomputed independently, with **no alive bonus**.

**One correction to `PLAN_D3_M3.md` section 0c**: their ant has **5 joints**
(a free root plus four hinges), not 6. `nq 11 / nv 10`. Measured by the gate.

## 2. What was run

```
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
cd /workspace/Transform2Act && source env-gpu.sh
for s in 1 2 3; do
  setsid nohup .venv-gpu/bin/python \
    rower_soccer/t2a_port/train_their_ant.py \
    --cfg ant_e0_s$s --num_threads 15 --stop-file /tmp/stop_ant_e0_s$s &
done
```

`train_their_ant.py` is their `design_opt/train.py` epoch loop with one
addition: a between-epoch `--stop-file` check. Their script can only be stopped
with a signal, and under MPS killing a CUDA client can corrupt the survivors.

`rower_soccer/t2a_port/cfg/ant_e0_s{1,2,3}.yml` are their `ant.yml` with
**exactly three lines changed** — `seed`, `max_epoch_num: 100`,
`save_model_interval: 10` — so `diff design_opt/cfg/ant.yml <cfg>` is the whole
provenance. Copies live in the repo and in `design_opt/cfg/`.

100 epochs answers the topology-convergence and seed-variability questions. **It
does not produce a converged body** and no claim below says it does.

## 3. The hopper control, re-measured — and a correction to `D3_HANDOFF.md`

E0 only means something against the number it is being compared to, so that
number was re-derived with the same code path rather than quoted.

```
.venv-gpu/bin/python rower_soccer/t2a_port/e0_analyse.py \
    --cfg hopper_gpu --epochs 0,50,100,200,400,1000 --episodes 200
.venv-gpu/bin/python rower_soccer/t2a_port/e0_analyse.py \
    --cfg hopper_gpu_s2 --epochs 0,50,100 --episodes 200
```

| run | epoch | distinct topologies / 200 sampled | share of the most common | mean-action design |
|---|---|---|---|---|
| `hopper_gpu` | 0 | 21 | 30.5% | 3 bodies |
| `hopper_gpu` | 50 | 7 | 56.0% | 7 bodies |
| `hopper_gpu` | **100** | **3** | **89.0%** | 8 bodies |
| `hopper_gpu` | 200 | 2 | 98.0% | 8 bodies |
| `hopper_gpu` | 400 | 3 | 96.5% | 8 bodies |
| `hopper_gpu` | 1000 | 3 | 99.0% | **7 bodies** |
| `hopper_gpu_s2` | 0 | 19 | 35.5% | 2 bodies |
| `hopper_gpu_s2` | 50 | 9 | 49.5% | 6 bodies |
| `hopper_gpu_s2` | **100** | **5** | **91.0%** | 7 bodies |

**Correction.** `D3_HANDOFF.md` says "by epoch 100 the skeleton stage has
stopped exploring — 199 of 200 sampled designs share one topology". At epoch
100 the measured figure is **178 of 200 (89%)** on `hopper_gpu` and 182 of 200
(91%) on `hopper_gpu_s2`. 199/200 is the *epoch-1000* regime (198/200 measured).
The direction of the claim survives — the skeleton search is 89-91% converged by
epoch 100 and essentially frozen by 200 — but the number was attached to the
wrong epoch and should not be requoted as an epoch-100 figure.

A second nuance the original claim missed: the mean-action design is **not**
frozen after epoch 100. `hopper_gpu`'s is an 8-body plan at epochs 100, 200 and
400 and a **7**-body plan at epoch 1000 — it drops a limb somewhere in the last
600 epochs. "Stopped exploring" is true of the sampled distribution's width, not
of the design the run would actually deploy.

### A bug in the instrument, found and fixed while doing this

`topology_census.py` sampled with the policy in **train mode**. Their own
sampler wraps everything in `to_test(*self.sample_modules)`
(`khrylib/rl/agents/agent.py:111`), and the policy's three `RunningNorm` layers
mutate their mean/var buffers on every forward while training is true
(`running_norm.py:32-34`). So each design in a 200-design census was normalised
against statistics the previous designs had just moved.

Measured impact at a trained checkpoint: **none** — `hopper_gpu` epoch 100 is
178 / 21 / 1 either way, because by then the running `n` is millions of rows.
At an untrained checkpoint it is not nil: `n` starts at zero, where eval mode
passes the observation through unchanged and train mode normalises it by the
batch. `ant_e0_s1`'s epoch-0 mean-action design is **17 bodies in test mode and
16 in train mode**. Fixed in `topology_census.census` (commit `2a00fd7`); every
number in this document is post-fix.

## 4. The verdict: the skeleton stage keeps exploring on the ant

```
.venv-gpu/bin/python rower_soccer/t2a_port/e0_analyse.py \
    --cfg ant_e0_s{1,2,3} --epochs 10,20,...,100 --episodes 200
```

200 designs sampled the way a training epoch samples them, at every tenth epoch.
`*` marks a census whose **mean-action** design has a different topology from the
previous one.

| epoch | s1 distinct/200 | s1 top share | s1 mean-action bodies | s2 distinct/200 | s2 top share | s2 mean-action bodies |
|---|---|---|---|---|---|---|
| 0 | 182 | 2.0% | 17 * | 176 | 1.5% | 7 * |
| 10 | 178 | 2.0% | 16 * | 191 | 1.0% | 18 * |
| 20 | 187 | 1.5% | 18 * | 188 | 1.5% | 18 * |
| 30 | 175 | 2.0% | 14 * | 165 | 2.5% | 17 * |
| 40 | 120 | 4.0% | 13 * | 106 | 6.5% | 16 * |
| 50 | 91 | 7.5% | 13 | 80 | 8.5% | 16 |
| 60 | 74 | 6.5% | 15 * | 54 | 14.0% | 15 * |
| 70 | 61 | 7.0% | 15 | 32 | 23.0% | 15 |
| 80 | 45 | 12.5% | 14 * | 23 | 25.0% | 14 * |
| 90 | 38 | 15.5% | 16 * | 16 | 39.0% | 14 |
| 100 | **34** | **20.0%** | 15 * | **20** | **41.0%** | 14 |

**The falsification condition did not fire.** E0's stated falsifier was "all
three seeds pin one topology by epoch ~20 and never move". The opposite
happened:

* At epoch 20 the ant is at **187 / 188 distinct topologies out of 200** —
  essentially every sampled design is unique. The hopper is at 21 distinct
  before training has started at all.
* Convergence does not begin until **epoch 30-40** and is nowhere near finished
  at 100: 34 and 20 distinct topologies, most-common share 20% and 41%, against
  the hopper's **3 distinct and 89%** at the same epoch.
* The **mean-action design never settles**. Its topology changed at 9 of the 11
  censuses on seed 1 and 7 of 11 on seed 2, including between epochs 90 and 100.
  On the hopper the mean-action design is the same 8-body plan at epochs 100,
  200 and 400.

So on a starting body that is already a quadruped, the skeleton stage explores
**more** and for **longer**, not less. The worry E0 was run to test — that M3's
headline comparison would be measuring attribute tuning because the body plan
pins immediately — is not supported on their ant.

### The honest caveat, which cuts the other way

Two things make "more exploration" less impressive than it sounds, and both
should be stated before this is used to justify anything:

1. **The reachable design space is much bigger from the ant.** The hopper starts
   with 2 bodies; the ant starts with 5, each of which may take up to 2 children
   to depth 3 (`ant.yml: add_body_condition.max_nchild: 2`, `max_body_depth: 4`).
   A wider sampled distribution at epoch 0 is partly a property of the starting
   tree, not of the search. This is why the table reports the **most-common
   share** as well as the distinct count: concentration is the comparable
   quantity, and on that measure the ant at epoch 100 (20-41%) is roughly where
   the hopper was around epoch 40-50 (est. from 30.5% at epoch 0 -> 89% at 100).
2. **The starting body is good but the controller is random.** "Start from a
   competent quadruped" describes the morphology only; the policy is freshly
   initialised, so early returns are near zero for the ant exactly as for the
   hopper (`exec_R_eps` 4.42 and 6.08 at epoch 0). E0 does **not** test the
   different question "does the skeleton stage stop exploring once the
   controller is already competent" — that would need a pretrained controller
   and has not been run.

## 5. Cross-seed divergence

```
.venv-gpu/bin/python rower_soccer/t2a_port/e0_analyse.py --compare \
    --cfgs ant_e0_s1,ant_e0_s2 --epoch 100
.venv-gpu/bin/python rower_soccer/t2a_port/e0_analyse.py --compare \
    --cfgs hopper_gpu,hopper_gpu_s2 --epoch 100
```

Two distances, because binary SAME/DIFF is nearly useless when every seed starts
from the same creature — two designs differing by one leaf limb read DIFF:

* **Jaccard** on the body-name sets. A body's name encodes its path from the
  root (`xml_robot.py:317-321`), so the name set *is* the tree.
* **SMD** on the attribute genome over shared bodies, standardised by the
  per-column spread of the ~2,900 sampled body rows that seed drew at that
  epoch. A range-fraction (`|delta| / 2`) is printed beside it because it can
  never be degenerate.

| pair | topology | Jaccard | attribute SMD | range fraction |
|---|---|---|---|---|
| ant seed 1 vs seed 2 | DIFF | **0.81** | **0.50** | 0.051 |
| hopper `gpu` vs `gpu_s2` | DIFF | 0.88 | 0.70 | 0.081 |

**The bodies barely diverge; the behaviour diverges enormously.** The two ant
seeds share 13 of the 16 body positions in their union and differ by two leaf
limbs (seed 1 grew `22` and `23`, seed 2 grew `24`). Their attribute genomes are
half a population standard deviation apart, which is *less* than the hopper's two
seeds. And yet:

| | seed 1 | seed 2 |
|---|---|---|
| `exec_R_eps`, mean of final 10 epochs | **800** | **256** |
| net displacement, one mean-action episode | **37.0 m** | **5.9 m** |
| airborne (zero floor contacts) | **47.4%** of steps | 3.6% |
| floor contacts per step | 1.18 | 2.86 |

That is the answer to the variability question, and it is not the answer the
question expected: **on their ant at 100 epochs, seeds converge on nearly the
same body and on completely different gaits.** Whatever produces a 3x reward
spread here lives in the controller, not the body plan. If M3 wants
"genuinely different bodies for different roles", nothing in E0 suggests seed
noise will supply them.

## 6. What the bodies actually look like

Renders in `runs/d3_e0_ant/renders/`, and in wandb under
`d3_e0_ant_seed{1,2,3}_media` (`video/initial_ant`, `video/best_median_worst`).

**The starting ant is not a standing quadruped.** `ant_initial_body.mp4`
(`render_checkpoint.py --cfg ant --untrained --initial-body`, a flag added for
this, which applies a zero design action so the body is exactly the task's
starting XML): a sphere torso of radius 0.25 with four single capsules radiating
flat in the xy plane. **No knees.** It settles onto the floor and lies there —
`e0_body_probe.py --cfg ant --untrained --initial-body` measures 4.68 floor
contacts per step, 0.3% of steps airborne, and **0.11 m of net travel in 1,000
steps**. "Competent quadruped" describes its topology, not its capability.

**Growth is collinear by default.** `add_child_to_body` clones the parent body
and copies its `bone_offset` (`xml_robot.py:519-540`), so a new child extends the
parent's spoke outward rather than adding a knee. The untrained mean-action
designs show exactly that: longer flat spokes, still lying on the floor.

**By epoch 100 both seeds have built jointed legs and stood up.**
`final_meanaction_s{1,2}.mp4`:

* **Seed 2 is a walker.** A symmetric arched stance, torso lifted clear of the
  ground on four multi-segment legs with visible knees. 3.6% airborne, 2.86 floor
  contacts per step, 5.9 m travelled.
* **Seed 1 is a leaper.** Asymmetric, torso low, and it bounds off one long
  jointed limb — **47.4% of its steps have no ground contact at all** — covering
  37.0 m. Six times seed 2's distance from a body 0.81 Jaccard away from it.

Both **locomote** rather than oscillate: `net/path` is 0.987 and 0.991
(`displacement_probe.py`'s definition, the statistic that caught
`hopper_gpu_t32` sprawling).

### Contact exploitation: present, but far milder than the hopper's

`D3_HANDOFF.md` records the reference hopper's limbs going 0.24-0.41 m through
the floor. That is a different contact regime and must not be carried across:
`hopper.xml` sets `solref=".02 1"`, `solimp=".8 .8 .01"` and geom
`density="1000"`; `ant.xml` sets **no** solref/solimp override and
`density="5.0"`. Measured on the ant with the same probe:

| run | deepest below floor | mean depth | >2 cm | >10 cm | capsule radii | airborne |
|---|---|---|---|---|---|---|
| `ant_e0_s1` e100 | 0.110 m | 0.0024 m | 4.3% | 0.1% | 0.069-0.093 | 47.4% |
| `ant_e0_s2` e100 | 0.087 m | 0.0002 m | 0.3% | 0.0% | 0.074-0.090 | 3.6% |
| `hopper_gpu` e1000 | 0.267 m | 0.0104 m | 15.5% | 2.3% | 0.030-0.055 | 80.1% |

The ant's peak penetration is ~1.5 capsule radii against the hopper's 5-9, and
its mean depth is 4-50x smaller.

**And the ant does not press on its design bounds at all.** Measured the same
way `D3_HANDOFF.md` measured the hopper's -- 40 SAMPLED designs, not one
mean-action design, "at the bound" meaning within 1% of the parameter's range
(`rower_soccer/t2a_port/e0_bounds_probe.py`):

| run | capsules at min radius | at max radius | actuators at min gear | at max gear |
|---|---|---|---|---|
| `ant_e0_s1` e100 (585 capsules) | **0.0%** | 0.0% | **0.0%** | **0.0%** |
| `ant_e0_s2` e100 (550 capsules) | **0.0%** | 0.0% | **0.0%** | **0.0%** |
| `hopper_gpu` e1000 (279 capsules) | 44.4% | 0.0% | 13.0% | 33.9% |
| `hopper_gpu_s2` e1000 (280 capsules) | 39.6% | 0.0% | 5.8% | 14.2% |

Observed ranges: ant radii 0.057-0.098 inside [0.03, 0.10] and gears 31-240
inside [20, 400]; hopper radii bottom out at exactly 0.0300 and gears at exactly
20.0 and 400.0. `D3_HANDOFF.md` records 32% of hopper capsules at the minimum
radius and 18% of gears at the maximum; those are reproduced here in direction
and rough size (44%/34% on `hopper_gpu`, 40%/14% on `hopper_gpu_s2`) -- the
difference is a looser tolerance here and possibly a different sampled set, so
treat the handoff's exact percentages as approximate rather than wrong.

**This is the single biggest environmental difference E0 found.** The hopper's
optimum is "as light and as strongly actuated as the bounds allow" and the
bounds are what stops it; the ant's optimum at 100 epochs is in the interior of
its box on every parameter. Whatever makes the hopper degenerate -- no energy
cost, a soft contact model, a 1,000-step alive bonus -- the ant task does not
reproduce it. That makes their ant a **better** starting point for M3 than the
hopper, on the one axis D3 was most worried about.

Seed 1's 47% airborne is the one number here that resembles the hopper's mode
and is worth watching, but it is bounding on real contacts, not falling through
the floor.
