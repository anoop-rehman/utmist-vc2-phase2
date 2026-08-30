# D3 M3 E1 — our ant inside Transform2Act

*2026-08-29. The experiment is `PLAN_D3_M3.md` section 1, rung E1, run with the
converter of [`D3_M3_E1_ANT_CONVERTER.md`](D3_M3_E1_ANT_CONVERTER.md). Every
number below names the command that produced it. Anything not measured is in
the "Not tested" section at the end.*

**Creature: OUR ant** — `rower_soccer/competevo_port/assets/dev_ant_body.xml`,
the DeepMind/gym ant D1 and D2 train: 13 bodies, 9 joints (1 free + 8 hinge),
8 motors, four legs of three links. **Task: THEIR ant task**, unchanged —
`design_opt/envs/ant.py`, `done_condition`, reward and annealing exactly as
`ant.yml` has them. E0 ran the same task on **their** 5-body / 4-motor ant, and
the two are directly comparable because only the starting body differs.

---

## 1. Which implementation, and why — re-checked, not inherited

E0 chose their CPU reference because our GPU port has no ant path. That is
still true, and the four places were re-read rather than requoted (E0's doc
cites `train_t2a.py:295/297`; the file has moved since and the current lines
are 455/459):

| | port | the ant task |
|---|---|---|
| initial XML | `hopper.xml`, hardcoded (`rower_soccer/t2a_port/train_t2a.py:455`) | `ant_competevo.xml` |
| `sim_obs_dim` | `5`, hardcoded (`train_t2a.py:459`) | **13** (`design_opt/envs/ant.py:41`) |
| root in `sim_obs` | planar `(height, ang)` from `qpos[1:3]` (`batched_exec_env.py:214-220`) | free joint, tilt from a quaternion (`ant.py:170-172`) |
| `index_base` | `max_nchild + 1` = **3** (`design_stage.py:83`) | **5**, hardcoded (`ant.py:35`) |

`/workspace/Transform2Act/results/` still contains only hopper runs from the
port. **So E1, like E0, runs on their CPU reference**, which is also what makes
the two comparable: same code, same sampler, same optimiser, different starting
body.

## 2. The converter gate, re-run before building on it

```
cd /workspace/utmist-vc2-phase2
PYTHONPATH=. .venv/bin/python -m rower_soccer.t2a_port.competevo_to_t2a \
    --out /workspace/Transform2Act/assets/mujoco_envs/ant_competevo.regen.xml \
    --require-name-noop            # byte-identical to the checked-in asset
PYTHONPATH=. .venv/bin/python -m rower_soccer.t2a_port.gate_competevo_ant --ours
cd /workspace/Transform2Act && source env-gpu.sh && .venv-gpu/bin/python \
  /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/gate_competevo_ant.py --theirs
cd /workspace/utmist-vc2-phase2 && \
  PYTHONPATH=. .venv/bin/python -m rower_soccer.t2a_port.gate_competevo_ant --cross
```

`runs/d3_e1_ant/logs/gate_ours.log`, `gate_theirs_cross.log`. All four phases
**PASSED**, and the asset regenerates byte-identical (`REGEN IDENTICAL`). The
headline numbers reproduce exactly: 95 indexed `mjModel` arrays equal against
the model D1/D2 compile with largest residual `0.000e+00`; 500 steps of physics
with `max|dqpos| 0.000e+00`; `Robot` adds a limb (13→14 bodies, nu 8→9), removes
one (13→12, nu 8→7) and `AntEnv` runs skeleton → attribute → execution; the
cross-engine mass ratio is 0.96468 and the two engines agree to `1.155e-14`
until the first floor contact at step 18. Nine negative controls all rejected.

## 3. Two constraints checked BEFORE running, because they bound the answer

### 3a. `max_nchild: 2` — and the constraint is bigger than that

The worry the plan flagged: `ant_competevo.yml` inherits
`add_body_condition.max_nchild: 2` and our torso already has four children, so
the torso can never gain a fifth leg. **Confirmed, and it is not the binding
constraint.** Measured with `rower_soccer/t2a_port/e1_designspace_probe.py`,
which asks the env's own `allow_add_body` / `allow_remove_body`
(`design_opt/envs/ant.py:47-59`) about every body and then saturates and erodes
the space with the env's own mutators:

```
cd /workspace/Transform2Act && source env-gpu.sh
.venv-gpu/bin/python .../t2a_port/e1_designspace_probe.py --cfg ant_e1_s1
.venv-gpu/bin/python .../t2a_port/e1_designspace_probe.py --cfg ant_e0_s1
```

| our ant (`ant_e1_s1`) | depth | children | hinge | allow_add | allow_remove |
|---|---|---|---|---|---|
| `0` torso | 0 | 4 | yes | **False** | False |
| `1 2 3 4` leg stubs | 1 | 1 | **no** | True | False |
| `11 12 13 14` hips | 2 | 1 | yes | True | False |
| `111 112 113 114` shins | 3 | 0 | yes | False | True |

Three findings, all of which bound E1 and none of which is the converter's:

1. **The torso is blocked twice over, and the second reason is the stronger
   one.** `allow_add_body` requires `body.depth >= cfg.min_body_depth`, and
   `min_body_depth` defaults to **1** (`design_opt/utils/config.py:66`) — no cfg
   in the repo sets it. The root is at depth 0, so *no* Transform2Act robot can
   ever gain a child at the root, whatever `max_nchild` is. **Their ant in E0
   was under exactly the same rule**, so this is not a handicap that separates
   E1 from E0.
2. **Almost every limb our ant could grow is PASSIVE.** `add_child_to_body`
   (`khrylib/robot/xml_robot.py:530-546`) clones the body it is attached to,
   joints included. Our depth-1 stubs are **jointless**, so any child of a stub
   — and any child of that child — has no hinge and no motor. Saturating the
   space:

   | | start | saturated | bodies added | motors added | passive bodies |
   |---|---|---|---|---|---|
   | **our ant** | 13 bodies / 8 motors | **29 / 12** | 16 | **4** | **16** |
   | **their ant** | 5 bodies / 4 motors | **29 / 28** | 24 | **24** | 0 |

   Both saturate at 29 bodies (1 + 4 + 8 + 16 over four depths), but the
   skeleton stage can add at most **four** actuators to our ant, all of them as
   a second child of a hip. Twelve of the sixteen additions it can make are
   unactuated dead weight.
3. **Ours can lose its whole actuator set; theirs cannot lose any.**
   `allow_remove_body` needs `depth >= min_body_depth + 1 = 2` and no children.
   Their ant's four limbs are at depth 1, so **their initial four limbs are
   unremovable** and erosion terminates in 0 rounds. Ours erodes in 2 rounds to
   **5 bodies / 0 motors** — the torso plus four jointless stubs, an inert blob.
   That is a reachable absorbing state for our ant and not for theirs.

**Does this materially bound the experiment? Yes, and it is stated before the
run rather than after.** E1 cannot answer "what quadruped does Transform2Act
design"; it can only answer "what does it do to *this* quadruped, given that it
may lengthen or thicken any bone, branch one passive spur off any stub, add one
actuated second shin to any hip, and delete leg links". The interesting
comparison with E0 is therefore about *behaviour of the search*, not about the
space it searches, and the two spaces are not the same size.

### 3b. `done_condition.max_ang: 60` — the flag was right, but it bites THEIR ant

`D3_M3_E1_ANT_CONVERTER.md` §6 flagged `max_ang: 60` as inherited unchecked and
possibly terminating a rolling quadruped absurdly early. Measured at epoch 0
(untrained policy, sampled actions, 60 episodes) with
`rower_soccer/t2a_port/e1_eplen_probe.py`:

```
.venv-gpu/bin/python .../t2a_port/e1_eplen_probe.py --cfg ant_e1_s1 \
    --untrained --episodes 60      # and again with --cfg ant_e0_s1
```

| | execution episode length | histogram | terminated by |
|---|---|---|---|
| **our ant** (`ant_e1_s1`) | mean **509**, median 263, min 11, max 1000 | `[10-24]:5 [25-49]:6 [50-99]:7 [100-199]:8 [200-399]:6 [400-699]:2 [1000]:26` | tilt≥60° **57%**, reached max_nsteps **43%** |
| **their ant** (`ant_e0_s1`) | mean **26**, median 22, min 5, max 73 | `[0-9]:7 [10-24]:30 [25-49]:16 [50-99]:7` | tilt≥60° **97%**, height≥2.0 3% |

Torso tilt over all execution steps: ours mean 15.0° / p90 25.7°; theirs mean
30.3° / p90 51.4°.

**So `max_ang: 60` is not terminating our quadruped absurdly early — it is
terminating theirs.** Our ant has knees and stands, so 43% of untrained
episodes run the full 1,000 steps; their flat-limbed ant lies down and 97% of
its untrained episodes end on tilt inside 73 steps. This is a correction to the
direction the converter doc guessed, and it is the reason E1's epochs are ~2x
slower than E0's: the same 50,000-step batch buys far fewer episodes.

`max_ang` is left at 60, unchanged, because changing the task at the same time
as the creature would make E1 uninterpretable.

### 3c. What the starting body actually does

```
.venv-gpu/bin/python .../t2a_port/e0_body_probe.py --cfg ant_e1_s1 \
    --untrained --initial-body
MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m \
    rower_soccer.t2a_port.e1_render_designs --xmls .../ant_e1_s1_e0000_initial.xml \
    --out runs/d3_e1_ant/renders/initial_body.png
```

| | our ant | their ant (E0 §6) |
|---|---|---|
| stance | **stands**, torso z 0.561, lowest geom +0.0197 | lies on the floor |
| floor contacts / step | 3.80 (four feet) | 4.68 |
| airborne | 0.5% | 0.3% |
| net travel, 1,000 steps, zero-ish control | 0.01 m | 0.11 m |
| deepest below floor | 0.0096 m | — |

`runs/d3_e1_ant/renders/initial_body.png` — looked at: a torso sphere with four
three-link legs on the diagonals, knees up and feet down, nothing through the
floor, identical to `D3_M3_E1_ANT_CONVERTER.md`'s render. **E0's caveat 2
("the starting body is good but the controller is random") applies here in a
stronger form: our starting body is not merely a good *topology*, it is a body
that already holds itself up.** It still does not locomote — with a randomly
initialised controller it travels 1 cm in 1,000 steps.

## 4. GPU cost — and a budget correction that was learned expensively

**E0's budget does not transfer to our ant, and assuming it did cost a live
run.** `D3_E0_ANT.md` §7 records "three concurrent reference runs peaked at
19.2 GB of 20.5", and `PLAN_D3_M3.md` §2 records "two concurrent is the
practical ceiling". Both are true of **their** 5-body ant. Measured on ours:

| | E0, their ant | E1, our ant |
|---|---|---|
| concurrent reference seeds | 3 | **2** |
| peak GPU, all clients | 19.2 GB / 20.5 | **19.95 GB / 20.475** |
| per-seed steady state | 2.2-5.6 GB | 2.6-6.6 GB |
| per-seed transient peak | — | **up to 10.1 GB** |

**Two of ours peak worse than three of theirs.** The cause is the batch, not
the policy: our 13-body graph gives ~2.6x the node rows per state, and §3b's
episode-length finding compounds it — our episodes run to 500-1,000 steps where
theirs end at ~26, so a 50,000-step batch of ours is a much larger float64
tensor in `num_optim_epoch: 10` of PPO.

**What it cost.** At 21:59 on 2026-08-29, with E1 seeds 1 and 2 holding 7.7 GB
and 10.1 GB, the live D1 run `soccer2v2_1f_walls` asked for 8 MB and could not
get it:

```
File ".../mujoco_warp/_src/solver.py", line 108, in _create_solver_context
RuntimeError: Failed to allocate 8388608 bytes on device 'cuda:0'
```

It was **not killed** — no `STOP` file was written and no signal was sent; it
OOM-ed because E1 had taken the card. It resumed from its 21:50 checkpoint
(`step 4,950,982,656`) losing ~9 minutes, but it was **down for 50 minutes
before anyone noticed**, and that is the more useful failure: the GPU monitor
in place was watching for *high memory*, which is the cause, and nothing was
watching whether the other workload was still *alive*, which is the
consequence. A liveness watch on every co-tenant process is now part of the
run harness (`runs/d3_e1_ant/` monitors), not an afterthought.

**Operating rule for anything E1-class from here**: one heavy reference seed at
a time alongside D1, and check free memory by *measurement immediately before
launching* — `runs/d3_e1_ant/launch_s3_guarded.sh` samples the worst free
figure over 60 s and requires 8 GB, because the 19.95 GB figure is a
**transient during the PPO update**, not steady occupancy: within minutes of it
the same two seeds were back at 4.6 and 2.6 GB. A single instantaneous reading,
high or low, is not evidence either way.

## 5. What was run — and why the result is TWO seeds, not three

```
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
cd /workspace/Transform2Act && source env-gpu.sh
setsid nohup .venv-gpu/bin/python \
  rower_soccer/t2a_port/train_their_ant.py \
  --cfg ant_e1_s$S --num_threads 15 --stop-file /tmp/stop_ant_e1_s$S &
```

`rower_soccer/t2a_port/cfg/ant_e1_s{1,2,3}.yml` are `design_opt/cfg/ant_competevo.yml`
with **three lines changed** — `seed`, `max_epoch_num: 100`, `save_model_interval: 10`
— so `diff design_opt/cfg/ant_competevo.yml <cfg>` is the whole provenance, and
`ant_competevo.yml` is itself `ant.yml` plus `env_specs` only. Same trainer,
same 100 epochs and same census cadence as E0, so the two are comparable.

| seed | epochs completed | status |
|---|---|---|
| `ant_e1_s1` | **100** | complete |
| `ant_e1_s2` | **100** | complete |
| `ant_e1_s3` | **62** | **PARTIAL — excluded from every number below** |

**Seed 3 is not part of E1's result.** It was stopped cleanly via its stop-file
on 2026-08-30 by user decision, to give the GPU to E1.1. Its wandb runs
(`d3_e1_ant_seed3`, `d3_e1_ant_seed3_media_v2`) are tagged `partial`,
`stopped-epoch-62`, `excluded-from-E1-result` and are kept, not deleted. It also
had an **earlier aborted attempt of 39 epochs**, killed by ENOSPC (§4), whose
artefacts are quarantined in `runs/d3_e1_ant/aborted_s3_39epochs/`. Neither
appears in any table here. **Anyone extending this must not average seed 3 into
a three-seed claim.**

**n=2 is the central limitation of this experiment and it is not a small one.**
Every cross-seed statement below rests on a single pair. Two points cannot
establish a spread, a variance or a distribution; they can only show that two
particular runs landed where they landed. Where E1's pair is contrasted with
E0's triple, the contrast is *suggestive and nothing more*.

## 6. The census: the skeleton stage explores here too, and converges faster

```
.venv-gpu/bin/python .../t2a_port/e0_analyse.py --cfg ant_e1_s$S \
    --epochs 0,10,...,100 --episodes 200 --out runs/d3_e1_ant/census
```

200 designs sampled the way a training epoch samples them, at every tenth
epoch. `*` marks a census whose **mean-action** design has a different topology
from the previous one.

| epoch | s1 distinct/200 | s1 top | s1 MA bodies/motors | s2 distinct/200 | s2 top | s2 MA bodies/motors |
|---|---|---|---|---|---|---|
| 0 | 198 | 1.0% | 19 / 7 * | 195 | 1.0% | 11 / 4 * |
| 10 | 193 | 1.5% | 17 / 9 * | 195 | 1.0% | 15 / 7 * |
| 20 | **190** | 1.5% | 16 / 10 * | **187** | 1.5% | 18 / 10 * |
| 30 | 183 | 2.0% | 17 / 10 * | 190 | 1.5% | 16 / 10 * |
| 40 | 171 | 1.5% | 16 / 9 * | 183 | 2.0% | 20 / 9 * |
| 50 | 150 | 2.5% | 19 / 10 * | 179 | 1.5% | 22 / 11 * |
| 60 | 117 | 6.0% | 21 / 10 * | 169 | 2.0% | 23 / 9 * |
| 70 | 108 | 6.5% | 21 / 10 | 151 | 2.5% | 22 / 11 * |
| 80 | 101 | 5.0% | 18 / 9 * | 152 | 3.0% | 22 / 11 |
| 90 | 75 | 10.0% | 19 / 10 * | 131 | 3.0% | 20 / 8 * |
| 100 | **63** | **7.0%** | 20 / 10 * | **101** | **5.5%** | 22 / 9 * |

### Beside E0's three seeds on THEIR ant

| | E1, our ant (n=2) | E0, their ant (n=3) | hopper |
|---|---|---|---|
| distinct / 200 at epoch 20 | **190, 187** | 187, 188, 187 | — |
| distinct / 200 at epoch 100 | **63, 101** | 34, 20, 27 | 3 |
| most-common share at epoch 100 | **7.0%, 5.5%** | 20%, 41%, 26% | 89% |
| mean-action topology changed | 10 and 10 of 11 censuses | 9, 7, 9 of 11 | frozen after ~100 |

**E0's falsifier does not fire here either, and it fires even less.** At epoch
20 our ant is at 190 and 187 distinct of 200, indistinguishable from their
ant's 187/188/187. By epoch 100 ours is at **63 and 101 distinct with the most
common topology holding only 5.5-7.0%**, where their ant is at 20-34 distinct
and 20-41%, and the hopper is at 3 and 89%. On the concentration measure —
the comparable one, since the reachable space differs — **our ant at epoch 100
is roughly where their ant was at epoch 40-50, and where the hopper was before
training started.**

The mean-action design never settles on either seed: its topology changed at 10
of 11 censuses on both, including between epochs 90 and 100. Body count wanders
between 16 and 23 and motor count between 4 and 11 with no trend to a fixed
plan.

**This has to be read against §3a, and the reading cuts both ways.** Our ant
explores longer in part because it has *less* to gain: the skeleton stage can
add at most four actuators and twelve of its sixteen possible additions are
passive dead weight, so there may simply be no strongly-preferred body plan for
it to converge onto. A wide sampled distribution is evidence that the search is
still moving; it is **not** evidence that the search is finding anything.

## 7. Cross-seed divergence, and the behaviour

```
.venv-gpu/bin/python .../t2a_port/e0_analyse.py --compare \
    --cfgs ant_e1_s1,ant_e1_s2 --epoch 100 --out runs/d3_e1_ant/census
.venv-gpu/bin/python .../t2a_port/e0_body_probe.py --cfg ant_e1_s$S --epoch 100
```

Same two distances E0 used: **Jaccard** on the body-name sets (a body's name
encodes its path from the root, so the name set is the tree) and **SMD** on the
attribute genome over shared bodies, standardised by the per-column spread of
the ~3,700 sampled body rows that seed drew at that epoch.

| pair | Jaccard | attribute SMD |
|---|---|---|
| **E1 ours, s1 vs s2** | **0.75** | **0.58** |
| E0 theirs, s1 vs s2 | 0.81 | 0.50 |
| E0 theirs, s1 vs s3 | 0.82 | 0.39 |
| E0 theirs, s2 vs s3 | 0.76 | 0.41 |
| hopper `gpu` vs `gpu_s2` | 0.88 | 0.70 |

Our one pair is marginally *more* divergent than any of E0's three pairs on
both measures. **With n=2 this is one number and cannot support a claim about
variance.** It is consistent with E0's finding — seeds land on broadly the same
body — and it does not strengthen it.

Behaviour at epoch 100, one mean-action episode each:

| | E1 s1 | E1 s2 | E0 s1 | E0 s2 | E0 s3 |
|---|---|---|---|---|---|
| `exec_R_eps`, final-10 mean | **3192** | **2721** | 800 | 256 | 331 |
| `exec_R_eps`, final epoch | 3346 | 2704 | — | — | — |
| net displacement | **33.3 m** in 222 steps | **119.5 m** in 1000 steps | 37.0 m | 5.9 m | 15.6 m |
| airborne (zero floor contacts) | 76.1% | 71.0% | 47.4% | 3.6% | 25.1% |
| floor contacts per step | 0.32 | 0.34 | 1.18 | 2.86 | 1.75 |
| net/path | 0.999 | 1.000 | 0.987 | 0.991 | 0.928 |
| mean-action design | 20 bodies / 10 motors | 22 bodies / 9 motors | — | — | — |

**The return spread across our two seeds is 1.24x (3192 vs 2721) against E0's
3.1x across three (800 / 256 / 331).** State this carefully: **a 1.24x spread
measured from two runs is not evidence of low variance.** Two points have no
spread in any statistical sense. What can be said is the negative: E1 did not
reproduce E0's headline surprise of near-identical bodies producing a 3x return
difference — but E1 has not run enough seeds to have had a fair chance to.

The two seeds also differ in a way the return hides: seed 1's mean-action
episode **ends at 222 steps** while seed 2's **runs the full 1000**. Per step
they travel 0.150 and 0.119 m, so seed 1 is the faster creature and the shorter
episode is what costs it. Both are far outside E0's regime: 71-76% airborne and
~0.33 floor contacts per step against E0's 4-47% and 1.2-2.9.

## 8. What the bodies look like

Renders: `runs/d3_e1_ant/renders/` (PNG montages, and 29 mp4 clips at every
census epoch); wandb `d3_e1_ant_seed{1,2}_media`, `d3_e1_ant_seed3_media_v2`
(partial). The montages are produced by dumping the mean-action design to XML in
their stack and rendering it in **ours** (`e1_dump_design.py` →
`e1_render_designs.py`), which incidentally re-confirms the converter's claim
that every design descended from the converted ant compiles under modern MuJoCo
with no conversion step — the epoch-100 designs load in mujoco 3.12 directly.

**The starting body stands. Both evolved bodies do not.**

| | initial | s1 e100 | s2 e100 |
|---|---|---|---|
| bodies / motors | 13 / 8 | 20 / 10 | 22 / 9 |
| mass | 0.911 kg | 1.685 kg | 1.708 kg |
| settled torso z | **0.561** | **0.270** | **0.270** |

`initial_body.png`, `s1_initial_vs_e100.png`, `s2_e100.png`, looked at. The
starting ant is the DeepMind quadruped: torso sphere, four three-link legs on
the diagonals, knees up, feet down, torso well clear of the floor. **Both
evolved bodies have abandoned that posture.** Torso height drops by half to
0.270 on both seeds, mass nearly doubles, and the four compact knee-up legs
become long, thin, splayed limbs radiating almost flat, with extra branches —
closer to a spider or a sprawling lizard than to a walking ant. Seed 2's limbs
are longer and thinner still (radius down to 0.054 m against the uniform 0.08 m
it started from).

Read with §7, the picture is coherent: these are not walkers. At 71-76%
airborne and ~0.33 floor contacts per step they spend most of the episode off
the ground, bounding on long compliant limbs, and travel in a near-perfectly
straight line (net/path 0.999-1.000). The reward is `dx/dt` with no energy
term, so nothing pushes back on a heavier body with longer levers.

## 9. Design-box health

```
.venv-gpu/bin/python .../t2a_port/e0_bounds_probe.py ant_e1_s$S 100 40
```

40 SAMPLED designs, "at the bound" = within 1% of the parameter's range — the
same instrument and definition E0 used, so the rows are comparable.

| run | capsules | at min radius | at max radius | at min gear | at max gear |
|---|---|---|---|---|---|
| **`ant_e1_s1` e100** | 732 | 0.3% | **7.2%** | **2.0%** | 0.0% |
| **`ant_e1_s2` e100** | 844 | 0.0% | 0.0% | **7.5%** | 0.0% |
| `ant_e0_s1` e100 | 585 | 0.0% | 0.0% | 0.0% | 0.0% |
| `ant_e0_s2` e100 | 550 | 0.0% | 0.0% | 0.0% | 0.0% |
| `ant_e0_s3` e100 | 610 | 0.0% | 0.0% | 0.0% | 0.0% |
| `hopper_gpu` e1000 | 279 | 44.4% | 0.0% | 13.0% | 33.9% |

Observed ranges: s1 radii 0.0302-0.1000 and gears 22.2-263.3; s2 radii
0.0360-0.0992 and gears 20.0-337.7, inside [0.03, 0.10] and [20, 400].

**Our ant does press on its bounds, where their ant pressed on nothing.** It is
mild — 7.2% of s1's capsules at the *maximum* radius, 7.5% of s2's gears at the
*minimum* — and it is nowhere near the hopper's degeneracy (44% at minimum
radius, 34% at maximum gear). But E0's clean headline, "zero parameters at any
bound on any seed", does **not** carry over to our creature, and the direction
is the opposite of the hopper's on radius: ours wants *thicker* capsules, the
hopper wanted thinner ones. Consistent with §8's near-doubling of mass.

Floor penetration, one mean-action episode each:

| run | deepest below floor | mean depth | >2 cm | >10 cm | airborne |
|---|---|---|---|---|---|
| `ant_e1_s1` e100 | 0.049 m | 0.0019 m | 2.7% | 0.0% | 76.1% |
| `ant_e1_s2` e100 | 0.053 m | 0.0016 m | 2.2% | 0.0% | 71.0% |
| `ant_e0_s1` e100 | 0.110 m | 0.0024 m | 4.3% | 0.1% | 47.4% |
| `hopper_gpu` e1000 | 0.267 m | 0.0104 m | 15.5% | 2.3% | 80.1% |

Penetration is *better* than E0's ant on every column and far better than the
hopper's — deepest 0.049-0.053 m against 0.110 and 0.267, and nothing beyond
10 cm on either seed. The contact model is not being exploited here; the
airborne fraction is real flight between real contacts.

## 10. What E1 answers

**The question was: does our ant behave like their ant here, or differently?**

*Same*: the skeleton stage keeps exploring from a competent starting body, on
both creatures. E0's falsifier ("all seeds pin one topology by epoch ~20 and
never move") fails on ours as clearly as on theirs — 190 and 187 distinct
designs of 200 at epoch 20, and a mean-action design still changing between
epochs 90 and 100.

*Different, in four ways that matter for M3*:

1. **The reachable space is far smaller** (§3a). At most 4 added actuators
   against their 24, twelve of sixteen possible additions passive, and a
   0-motor absorbing state their ant cannot reach.
2. **Convergence is slower, not faster** — 5.5-7.0% top share at epoch 100
   against their 20-41%. Partly because there may be less to find.
3. **The design box is no longer pristine** (§9). Mild but non-zero bound
   pressure where their ant had none.
4. **The evolved creature stops being a quadruped** (§8). Torso height halves,
   mass doubles, and it converts four walking legs into long splayed limbs that
   are airborne 71-76% of the time. Their ant *gained* a stance; ours *lost*
   one.

Point 4 is the one to carry into E3-E6. The soccer creature has to be the
creature D1 and D2 train, and 100 epochs of this task turns that creature into
something that bounds in a straight line with its belly near the floor. On a
pitch — where turning, stopping and ball contact matter — an optimiser that is
free to do this to the body will do it, because `dx/dt` with no energy term is
exactly the pressure that rewards it. **The task, not the design space, is what
needs changing before morphology search is pointed at soccer.**

## 11. Cost

| | |
|---|---|
| wall clock, 2 seeds concurrent | ~145-215 s/epoch, ~4 h per seed |
| seed 3 alone on the card | ~100-125 s/epoch |
| GPU, 2 concurrent + D1 | **peak 19.95 GB / 20.475** (see §4) |
| GPU, 1 seed + D1 | 6.6-9.3 GB |
| CPU | 15 sampler workers per seed |
| disk | **157 MB per checkpoint**, 11 per seed — this filled the volume (§4) |

## 12. Not tested / not claimed

* **Three seeds.** E1's result is **two**. Seed 3 stopped at epoch 62 and is
  excluded; an earlier attempt aborted at 39 epochs on ENOSPC and is
  quarantined. Every cross-seed number here is a single pair.
* **Any claim about variance.** A 1.24x return spread and a 0.75 Jaccard from
  n=2 are two points, not a distribution.
* **Whether these results hold past 100 epochs.** Both seeds were still moving:
  the sampled distribution was still narrowing and the mean-action topology
  still changing at the last census.
* **Whether the evolved body is better *as a body*.** `exec_R_eps` confounds
  morphology and control; E1 ran design+control jointly and cannot separate
  them. That separation is E1.1's job, and E1.1 tests the *controller* on a
  frozen body, not the body on a frozen controller. **The experiment "is the
  evolved body better than the DeepMind ant under an equally-trained
  controller" has not been run.**
* **Whether any of this transfers back to D1/D2's stack.** E1 trained under
  mujoco-py 2.1, whose capsules are 3.5% lighter and whose contact solver is
  not 3.12's, on a floor with `margin="0.01"` where CompetEvo's has 0
  (`D3_M3_E1_ANT_CONVERTER.md` §1).
* **`max_ang: 60`, `robot_param_scale`, the annealing and the batch size** are
  all theirs, unchanged and untuned for our creature — deliberately, so E1's
  answer is about the creature and not about a re-tuned optimiser.
* The **render perturbation** documented in `e0_video.py` is larger here than on
  their ant: our episodes are 200-1000 steps rather than ~26, so pass-1 and
  pass-2 returns can diverge substantially. Panels are labelled with their own
  pass-2 return, so on-screen numbers are right, but the best/median/worst
  *ordering* can be marginal.
