# CompetEvo -> mujoco_warp port: status (unit 2d, stages 0-2)

*Worktree `competevo-port`. Reference: `/workspace/competevo` (read-only).
Plan: `rower_soccer/docs/repro/COMPETEVO_PORT_MAP.md`, section 5.5.
This file records gate results, INCLUDING the ones that did not pass and the
things that pass only because of a quirk in their code.*

Scope: **`run-to-goal-ants-v0`** (fixed morphology, stages 0-1) and
**`run-to-goal-devants-v0`** (stage 2: per-world morphology, the 52-dim dev
observation, the 28-dim design+motor action, the two-head dev policy). Faithful
opponent sampling and their two-learner co-evolution loop are stage 3 and are
NOT here yet.

Numbering: "stage 0/1" is the port map's Stage 0 (fixed-morph harness), split
into an env-parity gate and a PPO smoke. "Stage 2" is the port map's Stage 1
(the design -> model-fields writer) plus the dev env and policy around it.

## What exists

| file | what |
|---|---|
| `scene.py` | the merged 2-ant MJCF -- fixed AND dev ants -- emitted from leg tables; the dev genome's targets as a table (`DEV_GENOME_TABLE`); per-agent index plumbing (`SceneMeta`/`AgentSlices`) |
| `backend.py` | `warp_port` backends + `subtree_com`/`cfrc_ext`; **dev** variants with a PER-WORLD batched Model (Warp) and one MjModel per world (CPU mirror) |
| `run_to_goal_env.py` | batched fixed-morph env: obs, their three reward/termination layers, per-world auto-reset, win counters |
| `design.py` | **stage 2**: genome -> model fields (geoms, body pos, gears, and analytic capsule mass/inertia/ipos), plus exact `mj_setConst` constants |
| `dev_env.py` | **stage 2**: `run-to-goal-devants-v0` batched -- stage flags, the design step, the 52-dim obs, the dev standing band |
| `dev_ppo.py`, `train_dev.py` | **stage 2**: their `DevPolicy`/`DevValue` as one stage-masked module, and the smoke trainer |
| `parity.py`, `their_env_driver.py`, `their_dev_driver.py` | JSON-over-subprocess harnesses driving their CPU envs in their venv |
| `tests/test_parity.py`, `tests/test_design_parity.py` | the stage-0/1 and stage-2 gates |
| `ppo.py`, `train_run_to_goal.py`, `render.py` | shared-policy PPO with their hyperparameters, eval pass, video |

## Gate 0a -- model equivalence (PASS)

Our generated MJCF vs their merged `world_body.ant_body.ant_body.xml`, compiled
at *their* physics options so geometry is isolated from the option deviation:

* `nq/nv/nu = 30/28/16` on both;
* 28 model arrays bit-equal (max abs diff 0 at 1e-12): `body_mass`,
  `body_inertia`, `body_ipos`, `body_pos`, `body_quat`, `body_parentid`,
  `dof_damping`, `dof_armature`, `dof_bodyid`, `jnt_type/range/axis/pos`,
  `geom_type/size/pos/quat/friction/margin/condim/contype/conaffinity/solref/solimp`,
  `actuator_gear/ctrlrange/trnid`, `qpos0`;
* actuator -> joint order identical (this IS the action layout: their block is
  `hip_4, ankle_4, hip_1, ankle_1, hip_2, ankle_2, hip_3, ankle_3`, per agent,
  *not* leg order);
* derived slices match their `Agent._set_body/_set_joint`: agent 0 qpos `[0,15)`
  qvel `[0,14)`, agent 1 qpos `[15,30)` qvel `[14,28)`; goals `+4 / -4`;
  `move_left = False / True`.

## Gate 0b -- observation + reward parity (PASS, at machine epsilon)

Method: 48 hand-set states (plus 24 more forced into floor contact), each a
(prev-state, state, action) triple. Their env is driven through the identical
door -- `before_step()` at the prev state, `set_state()` in place of
`simulate()`, then `after_step()` / `goal_rewards()` / `_get_done()` -- inside
their venv over a JSON pipe. Ours runs the production `RunToGoalEnv.terms()` on
a float64 CPU backend. States span airborne / standing / fallen / past-the-goal
so every reward and termination branch is actually taken (coverage counters are
printed with the report).

Max |ours - theirs| over 48 states (`--cases 48 --seed 0`):

| field group | max abs diff |
|---|---|
| obs / own root pos (qpos 0:3) | 0.0 |
| obs / own root quat (3:7) | 2.22e-16 |
| obs / own joint pos (7:15) | 0.0 |
| obs / own root linvel (qvel 0:3) | 0.0 |
| obs / own root angvel (3:6) | 0.0 |
| obs / own joint vel (6:14) | 0.0 |
| obs / opponent root xy | 0.0 |
| **obs / all 31 dims** | **2.22e-16** |
| reward / torso subtree-COM x | 1.78e-15 |
| reward / forward progress | 1.19e-13 |
| reward / ctrl cost | 4.44e-16 |
| reward / contact cost | 0.0 (both identically zero -- see below) |
| reward / dense total | 1.19e-13 |
| reward / sparse (goal) | 0.0 |
| reward / total | 1.14e-13 |
| flags: reached_goal, winner, fell, terminated | 0 mismatches / 48 |

Branch coverage in that set: 46/48 states in contact (max 32 contacts), 12/48
crossed a goal line, 12/48 contained a fall, 24/48 terminated. The
forced-contact set (24 states, seed 7) reports the same numbers.

The gate asked for ~1e-6 on kinematic fields; the measured worst is 1.14e-13
(forward progress, which divides a COM difference by dt=0.015 and so amplifies
double-rounding by 67x). Nothing is over tolerance, and nothing is excluded.

### The contact-cost term matches for an uncomfortable reason

Their dense reward includes `0.5e-3 * sum(clip(cfrc_ext, -1, 1)^2)`. `cfrc_ext`
is only filled by `mj_rnePostConstraint`, which MuJoCo runs only for
acceleration-stage sensors -- and their scene declares none. Measured directly:
`|cfrc_ext|.max() == 0` on both mujoco 2.3.5 (their venv) and 3.11 (ours), with
up to 32 active contacts. **Their ant's contact cost is a constant 0 through
their entire training run.** So the port reproduces it as 0
(`contact_cost_from_cfrc=False`) rather than introducing a force term their
policies never felt. `RunToGoalEnv(contact_cost_from_cfrc=True)` computes the
real term from the backend's `cfrc_ext` (mujoco_warp does populate it); that
switch is the one place where "faithful to their code" and "faithful to what
their code meant" disagree, and the test asserts the zero so the day it stops
being true is loud.

This is also the answer to the port map's worry that contact-derived fields
would not match: for run-to-goal there are none in the observation, and the only
one in the reward is dead.

## Gate 0c -- solver divergence (diagnostic, not a gate)

`--diverge` runs 40 identical open-loop control steps (200 physics steps) from
an identical start state and reports root-position drift:

| step | 1 | 5 | 20 | 40 |
|---|---|---|---|---|
| max abs drift (m) | 2.9e-15 | 1.4e-14 | 4.4e-09 | 1.5e-07 |

That is *ours on CPU MuJoCo with Newton* vs *theirs with PGS*: the solver swap
alone costs ~1e-7 m over 0.6 s on this scene, i.e. it is nearly free. The
remaining, larger deviation is mujoco_warp itself, measured separately:

| quantity | warp fp32 vs our CPU fp64, same hand-set states |
|---|---|
| obs (31 dims) | 1.9e-07 |
| torso subtree-COM x | 5.4e-07 |
| forward-progress reward | 2.4e-05 |
| total reward | 3.0e-05 |

Those are fp32 representation error (the reward figure is the COM figure times
1/dt), not a modelling difference. Trajectory-level Warp-vs-CPU agreement is NOT
claimed and should not be expected; the repo's own `warp_port/render.py` already
documents that Warp and CPU MuJoCo resolve contacts differently.

## Deviations taken (all deliberate, all reversible)

1. **`solver="PGS" iterations=1000` -> `solver="Newton" iterations=100`.**
   mujoco_warp 1.16 raises `NotImplementedError: mjSOL_PGS is unsupported` at
   `put_model`. Newton is its lane. `iterations` was cut because mujoco_warp
   unrolls the solver loop into kernel launches; measured, 100 vs 20 vs 10
   changes throughput by <7% at 512 worlds but 100 costs ~2 min of one-time
   kernel compilation, and Newton reaches tolerance in single digits here.
   *Correction to the port map:* section 6.3 assumed RK4 would also have to go.
   It does not -- mujoco_warp supports RK4, so the integrator and the 0.003
   timestep are theirs, unchanged, and the deviation is solver-only.
2. **Contact cost = 0** (above) -- reproducing observed behaviour, not the
   written formula.
3. **Per-world auto-reset** instead of a whole-process reset. Standard batching
   change; worlds run out of phase.
4. **Reset noise drawn once**, not twice. Their `_reset` randomizes qpos, calls
   `reset_model` which randomizes again, then `set_xyz` zeroes qvel; only the
   last draw survives, so `qpos = qpos0 + U(-0.1, 0.1)` per element and
   `qvel = 0` is exactly their post-reset state. Their stray
   `np_random.integers(nv) * 0.1` qvel offset is discarded by `set_xyz` before
   any step and is not reproduced.
5. **fp32 on GPU** vs their float64 everywhere. Numbers above quantify it.
6. **Foot bodies are named** `agent{i}/foot_{k}` instead of their
   `agent{i}/anon<random>`. Nothing reads those names.
7. **Their scene file is not touched.** Constructing their env rewrites
   `gym_compete/new_envs/assets/world_body.ant_body.ant_body.xml` with fresh
   random `anon` names; the harness redirects that output to `/tmp` so the
   read-only reference tree stays clean.

## Stage 1 -- PPO smoke

One shared policy plays both ants (run-to-goal is symmetric). Their
hyperparameters: clip 0.2, gamma 0.995, GAE lambda 0.95, 10 optimizer epochs,
minibatch 2048, Adam 5e-5 policy / 3e-4 value, grad-clip 40, actor MLP
[128,128] tanh, critic [512,256] tanh, learned state-independent log_std init 0,
`RunningNorm` advancing only during the update pass, globally standardized
advantages, mask=1 on truncation (their GAE bootstraps across draws).

NOT reproduced: their fixed-morph `Learner` leaves the critic update commented
out (`custom/learners/learner.py:218-229`), so its advantages come from a frozen
random critic. We train the critic. Port map risk 8 says do that only if a curve
refuses to line up.

Throughput on the shared RTX 4000 Ada (four production trainers also resident),
1024 worlds, each control step = 5 substeps of RK4:

* rollout only, measured in isolation: **~20k world-steps/s** (40k
  agent-transitions/s), i.e. 3.9 s for a 64-step rollout;
* end-to-end training, measured over the runs below: **~5k
  agent-transitions/s**, because the PPO update and GPU contention dominate.
  Iteration wall time swung between 6 s and 94 s on an otherwise unchanged
  configuration -- the other trainers, not us.

Solver iterations barely matter (26.0 / 27.8 / 26.5 control-steps/s at 512 worlds
for `iterations` = 100 / 20 / 10), but the FIRST build at `iterations=100` spends
~122 s compiling Warp kernels; it is cached after that.

### Baselines (measured, `--eval-worlds 64`, full episodes)

| policy | per-agent episode return | ep length | win rate |
|---|---|---|---|
| uniform-random actions in [-1,1] | -163.6 / -164.5 | 455 | 0.00 / 0.00 |
| untrained net, stochastic (log_std 0) | ~ -1.02 / step | ~360 | 0.00 |
| untrained net, MEAN actions, torch default head init | 440.7 / 460.7 | 500 | 0.00 / 0.00 |
| **untrained net, MEAN actions, THEIR head init** | **501.9 / 493.9** | **500** | **0.00 / 0.00** |
| **their measured iter-0 eval** (REPRO_NOTES, `smoke-run-to-goal-ants-v0`) | **498.8 / 488.5** | 500 | **0.00** |

The mean-action baseline is the one to compare -- same task, same eval protocol
-- and this is the port map's `iter-0 eval ~= 490-510 per agent, win rate 0.00`
gate. **PASS: 501.9 / 493.9 at win rate 0.00, against their measured
498.8 / 488.5.**

Getting there required finding a real difference rather than shrugging at a 10%
gap. The first attempt read 440.7 / 460.7. Cause: their output heads run
`init_fc_weights` (weights x0.1, bias 0, `custom/utils/tools.py:19-21`), so an
untrained net emits near-zero mean actions and the ant stands still for 500 steps
collecting the +1 survive bonus -- exactly 500 minus a little control cost. Ours
used torch's default init, whose ~0.1 mean actions are real torque at gear 150,
so the untrained ant drifted and lost forward reward. Reproducing their init
closed the gap. Run A below predates the fix and its numbers are the pre-fix
ones, reported as measured.

Note also the two random baselines are not the same thing: uniform noise costs
0.5*E[a^2]*8 = 1.33/step in control cost, a unit-variance Gaussian clipped to
[-1,1] costs ~2.1/step. Episode RETURN is a poor early signal for another reason
too: it is dominated by +1/step survival, so a policy that learns to run and
falls at t=200 scores *worse* than one that stands still for 500 steps. Mean
forward-progress reward per step is the honest early metric and is now logged.

### Run A -- 17 min, raw env reward (NOT the reward they train)

1024 worlds, rollout 64, 4 optimizer epochs, minibatch 8192, Adam 3e-4/1e-3.
31 iterations, 4.06M agent-transitions, ~978 s.

| iteration | eval return (mean actions) | eval win rate | eval ep length |
|---|---|---|---|
| 0 | 461.3 / 478.6 | 0.00 / 0.00 | 500 |
| 15 | 210.9 / 192.1 | **0.016** / 0.000 | 207 |
| 30 | 92.7 / 190.5 | 0.000 / **0.040** | 148 |

Read honestly: **eval return went DOWN and win rate left zero.** Both are the
same fact. PPO gave up the trivial 500-step standing policy for one that moves,
the ants now fall (episode length 500 -> 148), and in exchange some episodes end
with an actual goal crossing -- 1.6% then 4% of games, from a floor of 0.00. The
"reward improves over the random-policy baseline" gate passes on the -164
random-action baseline and on goal-reaching, and fails on eval return against the
stand-still baseline. That failure is a property of the metric at this budget,
not evidence that training worked; it is why run B logs forward progress per
step.

Deviations from their hyperparameters, forced by a 17-minute budget on a GPU
shared with four production trainers: 4 optimizer epochs instead of 10, minibatch
8192 instead of 2048, Adam 3e-4/1e-3 instead of 5e-5/3e-4. Measured cost of their
values at this scale: the update alone takes 27 s per iteration (640 minibatch
launches) against 2.2 s, i.e. their settings would have bought 20 iterations
instead of 31 while learning ~6x slower per sample. Validation-grade runs must
use their values.

**Run A also optimized the wrong objective**, discovered while it was running:
their runner never hands the env reward to the learner. Both runners apply an
exploration curriculum, `r = alpha*dense + (1-alpha)*parse` with alpha annealing
1 -> 0 over `termination_epoch = 200` epochs
(`runner/multi_agent_runner.py:150-167`, `config/run-to-goal-ants-v0.yaml`), so
the +/-1000 goal term FADES IN and is absent at the start. Run A trained on
`parse + dense` throughout. The curriculum is now implemented
(`ppo.CURRICULUM_STEPS`, expressed in agent-steps -- 200 epochs x 50,000 steps --
because our iteration is a different size than theirs) and is the default.

### Run B -- 15 min, their exploration curriculum (the recorded stage-1 result)

Same settings as run A plus their curriculum (alpha 1 -> 0 over 10M agent-steps)
and their head init. 29 iterations, 3.80M agent-transitions, ~930 s, `--eval-every
10`. Artifacts: `runs/competevo_port/smoke_B_curriculum/`
(`log.json`, `train.log`, `eval.mp4`, `policy.pt`); run A's are in
`runs/competevo_port/smoke_A_envreward/`.

| iteration | alpha | **fwd reward / step** | eval return | eval win rate | eval ep len |
|---|---|---|---|---|---|
| baseline (untrained) | 1.000 | -- | 501.9 / 493.9 | 0.00 / 0.00 | 500 |
| 0 | 0.993 | **-0.012** | 517.4 / 519.8 | 0.00 / 0.00 | 500 |
| 10 | 0.928 | **+0.118** | 381.6 / 422.5 | 0.000 / 0.014 | 388 |
| 20 | 0.862 | **+0.257** | 195.5 / 164.9 | 0.00 / 0.00 | 189 |
| 28 | 0.834 | **+0.278** | 145.7 / 140.2 | 0.00 / 0.00 | 137 |

**Gate: PASS on locomotion, with a caveat stated plainly.** Mean forward-progress
reward per agent-step goes from -0.012 (an untrained net drifts slightly the
wrong way) to +0.278 and is still climbing when the budget ends -- monotone
across all 29 iterations. In task units that is the ants' centre of mass moving
toward their own goal line at ~0.28 m/s on average, up from zero, learned from
scratch in 15 minutes on a shared GPU. Both agents improve together, as they must
under a shared policy.

The caveat: **eval episode RETURN falls, 502 -> 146, and the win rate does not
hold above zero.** Both follow from the same thing. Standing still pays 500
(the +1/step survive bonus x 500 steps); running pays ~1.28/step but the ants
fall, so episodes end at 137 steps and collect ~176. PPO is correctly climbing
the reward it was given -- the curriculum reward is ~97% dense forward progress
at this point -- and the policy has not yet learned to run *without falling*,
which is what would let episode length recover and the goal line at x=+/-4 be
reached with any regularity. One goal crossing shows up at iteration 10
(win rate 0.014) and none later; at 4M agent-steps against their 50M this is a
15-minute smoke, not a result. The port map's real Stage-0 gate -- "win rate
leaves 0 and eval reward trend matches a their-code CPU run over the first ~50
epochs" -- is NOT met and is not claimed. What is established is that the loop
runs end to end, the numbers start where theirs start, and the policy learns the
thing the reward asks for.

Also honest: `alpha` only fell from 1.00 to 0.83 in this run, so the +/-1000 goal
term was still almost entirely faded out. Their schedule is 10M agent-steps and
we covered 3.8M. Nothing here says anything about how the sparse term behaves.

Video: `runs/competevo_port/smoke_B_curriculum/eval.mp4` -- 1000 frames at 67 fps (their
`render_fps`), a fixed window spanning 7 consecutive deterministic episodes of
world 0, camera above the halfway line with both goal lines in frame. Episode
lengths 99 / 171 / 98 / 87 / 296 / 92 / 84, no winner in any of them. Watching it
is the point: the ants do drive toward each other's goals and collide near the
halfway line -- which is the actual dynamic of run-to-goal, since each agent's
goal is behind the other -- and then topple. Nothing in the video contradicts the
table; it shows why episode length is 137.


## Stage 2 -- per-world morphology (`run-to-goal-devants-v0`)

Their dev agents have a FIXED topology whose geometry is a flat scale vector
`s in [-1,1]^20`, emitted by the policy as the step-0 action of every episode.
Their env applies it by mutating two lxml trees, re-merging them and calling
`MjModel.from_xml_string` **twice per episode per worker**
(`multi_dev_agent_env.py:274-316`). Here there is ONE compiled model and a
design is a write of that world's row of the batched model arrays. Nothing is
recompiled, ever.

### Which mujoco_warp Model arrays batch per world (answering the stage-1 note)

Measured, not assumed. In mujoco_warp 3.11 every `Model` array whose type spec
starts with `*` -- **105 of them**, including all sixteen this port needs --
defaults to a leading dimension of **1, shared across all worlds**, and every
kernel reads it as `field[worldid % field.shape[0]]`. The supported way to get a
per-world axis is

```python
mjw.put_model(mjm, batch_sizes={"body_mass": nworld, ...})
```

which allocates the listed fields `[nworld, ...]` with the compiled value tiled
into every row. It has to be done at `put_model` time: the leading dimension is
baked into the kernels (`collision_driver.py:307` specializes on
`wp.static(ngeom_rbound > 1)`), so an array cannot be widened afterwards.
`WarpBackend._put_model` is now an overridable hook and
`CompeteWarpDevBackend` passes `batch_sizes`.

Confirmed batched `[nworld, ...]` and written: `geom_size`, `geom_pos`,
`geom_rbound`, `geom_aabb`, `body_pos`, `body_mass`, `body_inertia`,
`body_ipos`, `body_subtreemass`, `actuator_gear`, `body_invweight0`,
`dof_invweight0`, `actuator_acc0`. Also batched but constant under this genome
(it only scales along fixed directions): `geom_quat`, `body_quat`, `body_iquat`.

Two fields worth naming because they are easy to miss:

* **`body_subtreemass`** is what `subtree_com` is divided by -- i.e. it is in the
  forward-progress REWARD. Leave it stale and the reward is wrong, not the
  physics.
* **`geom_rbound` / `geom_aabb`** are the broadphase bounds. Leave them stale and
  a design with longer legs silently loses contacts.

**`bvh_aabb` does not exist in mujoco_warp** and is not in the list above.
MuJoCo's compile-time body BVH is a CPU-MuJoCo structure (mujoco_warp's `bvh_*`
are mesh/hfield/flex only, and it broadphases from `geom_aabb`/`geom_rbound`).
It still had to be handled, because the CPU mirror used by the gate does descend
it -- see the trajectory gate below, where a stale one was the entire residual.

### The CUDA graph does NOT need re-capturing

The stage-1 note warned that it might. It does not, and the test says so
(`test_design_parity.py::mujoco_warp batches the model per world; graph
survives`): a graph captured BEFORE any design write replays correctly after
one, because a design write is an in-place write into device buffers the graph
already points at. Only a SHAPE change would invalidate it, and the shapes are
fixed at `put_model`. Measured on 8 worlds, graph captured at construction, then
designs written and 20 graph replays: worlds given identical designs stay
together to 1.2e-09 (fp32), worlds given different designs separate by 0.56 m.

The thing that IS shape-sensitive is kernel compilation: switching the model
arrays from shared to per-world is a new specialization, so the first build after
this change recompiles the Warp kernels (~2 min, cached afterwards).

## Gate 2a -- design -> model fields vs THEIR compiler (PASS)

*Honest note on how this gate stood: the numbers below are real and reproduce
exactly, but for a stretch the gate was not actually running them. The host
round-trip work changed `HostConstants.__init__` to take a `spec` and
`compute()` to take the genome, and this call site was left on the old
`HostConstants(model).compute(fields, n)` signature -- so the single most
load-bearing gate in the stage was erroring out with a `TypeError` and reporting
FAIL while the rest of the suite went green. Fixed, and the fix makes the gate
stricter: it now hands `compute` the genome and lets `HostConstants` derive the
fields itself, which is the real production entry point rather than a
pre-computed dict. Every number below is from the repaired gate.*

`tests/test_design_parity.py`. Method, per the stage-1 advice: reuse the
model-equivalence check, parameterized by a design vector, and include the mass
block. For 10 random genomes `s ~ U(-1,1)^20` per agent, their code is driven
through its real entry points in their venv (`env.reset()`, then
`env.step([design0, design1])`, which runs `set_design_params` ->
`load_tmp_mujoco_env`), and the merged MJCF it emits is returned along with the
MjModel it compiled.

Two references, because they answer different questions -- (a) their emitted
MJCF compiled by OUR mujoco 3.11 ("is the writer right?"), (b) their MjModel as
compiled by mujoco 2.3.5 in their venv ("is anything else different?").

| field | max abs \|ours - their compiler\| |
|---|---|
| `geom_size` | 1.1e-16 |
| `geom_pos` | 0.0 |
| `geom_rbound` | 1.1e-16 |
| `geom_aabb` | 1.1e-16 |
| `body_pos` | 0.0 |
| **`body_mass`** | **5.6e-17** |
| **`body_inertia`** | **3.5e-18** |
| **`body_ipos`** | **0.0** |
| `body_subtreemass` | 4.4e-16 |
| `actuator_gear` | 0.0 |
| `body_invweight0` (rel) | 3.7e-16 |
| `dof_invweight0` (rel) | 2.9e-16 |
| `actuator_acc0` (rel) | 3.2e-16 |

Those designs are not cosmetic: leg masses vary **4.7x** across the set.

The (b) reference agrees with (a) to 0.0 on every field except `geom_aabb`,
where it differs by exactly 1.0e-2 on all 26 ant geoms. That is `geom_margin`:
**mujoco 2.3.5 pads the broadphase AABB by the contact margin and 3.11 does
not.** It is a compiler-version artifact, not a port bug -- it applies to the
base scene at `s = 0` too -- and the writer matches 3.11 because 3.11 is the
compiler mujoco_warp's model comes from. (This one cost time: matching (b)
naively put a margin into the writer that made it disagree with our own base
model.)

### The mass block, and why it is not just `geom_size`

Every body in this robot carries exactly one geom, so `body_ipos` is the capsule
midpoint, `body_iquat` is the geom frame (unchanged), and `body_inertia` is the
capsule's principal inertia. The formula, verified against MuJoCo's compiler at
2.5e-16 relative on mass and 3.8e-16 on inertia over random (r, h):

```
m_cyl = rho pi r^2 2h                m_sph = rho 4/3 pi r^3
I_axial      = m_cyl r^2/2 + m_sph 2/5 r^2
I_transverse = m_cyl (r^2/4 + h^2/3) + m_sph (2/5 r^2 + h^2 + 3/4 h r)
```

At `s = 0` this reproduces the numbers the stage-1 notes recorded: torso 0.327,
legs 0.0392 / 0.0392 / 0.0676.

Two traps found while building it, both now asserted rather than assumed:

1. **MuJoCo sorts the principal moments** and rotates `body_iquat` to match, so
   which slot of `body_inertia` holds `I_axial` is a property of the compiled
   model. The spec reads the slot off the base model (the entry that differs
   from the median -- a capsule's other two are equal), and
   `_assert_inertia_ordering_is_stable` proves at build time that no design in
   `[-1,1]^20` can reorder them, by checking the corner that squashes a capsule
   hardest (shortest allowed, fattest allowed). Guessing the slot instead
   produced a 92% relative error on `body_inertia` that nothing else caught.
2. **The genome does not scale everything you would expect.** Per leg the five
   parameters are upper-capsule LENGTH, mid RADIUS, mid LENGTH, foot RADIUS,
   foot LENGTH; the upper capsule's radius and the torso sphere are never
   scaled, and the child body `pos` is scaled by the PARENT capsule's length
   factor (which is what keeps links attached). Gears scale by
   `b = 1 + 0.15 s` using the RADIUS parameter of the capsule the motor drives,
   not the length. This is now `DEV_GENOME_TABLE` in `scene.py`, resolved against
   the compiled model by name.

### `mj_setConst`: the constants with no closed form (SOLVED, not deferred)

`body_invweight0` / `dof_invweight0` / `actuator_acc0` are not functions of the
design in any form that can be written down -- MuJoCo derives them from
`inv(M)` at qpos0. The first implementation left them at the base ant's values.
Measured, that is **up to 46% wrong on `dof_invweight0`**, and since
`*_invweight0` sets constraint impedance it is not bookkeeping: over 40 control
steps (0.6 s) from qpos0 under identical open-loop torques it moved the
trajectory by **7.1e-2** (against 1.5e-7 for the entire PGS -> Newton solver
swap -- five orders of magnitude bigger).

The fix is that `mj_setConst` is a NUMERIC routine over an existing model, not a
compile. `design.HostConstants` keeps one host scratch `MjModel`, pushes the
design fields into it, calls `mj_setConst`, and writes the three arrays back --
**0.23 ms per world** for the `mj_setConst` loop itself, machine-epsilon
agreement with a freshly compiled model. Only the worlds that reset on a given
step pay it. `exact_constants=False` restores the stale behaviour, which is what
the gate measures against.

(An earlier revision of this file claimed 0.093 ms per world. Re-measured on the
shared card it is 0.23 ms; the original number timed the bare `mj_setConst` call
and not the field push around it. The cost that actually mattered turned out to
be somewhere else entirely -- see the host round-trip section below.)

### The host round-trip: one fused D2H, not a second `design_fields`

The write path is `design_fields(spec, scale)` -> scatter ten arrays into the
batched Model -> `mj_setConst` for the same worlds. `mj_setConst` runs on the
host, so the five fields it reads (`body_mass`, `body_inertia`, `body_ipos`,
`body_pos`, `actuator_gear`) have to get there.

The first implementation pulled them field by field: ten transfers, each its own
device sync. The second (the one this stage inherited, mid-edit) avoided the
syncs by giving `HostConstants` a float64 CPU twin of the spec and **evaluating
`design_fields` a second time on the host**. That trade is backwards.
`design_fields` is launch-bound -- tens of small kernels over full-width
`[M, ngeom, ...]` tensors -- and it is the expensive half of a write, so paying
for it twice costs more than the sync ever did:

| worlds resetting | fused D2H (now) | host re-derivation (before) |
|---|---|---|
| 1 | 3.71 ms | 6.47 ms |
| 4 | 3.82 ms | 5.27 ms |
| 16 | 5.15 ms | 7.39 ms |
| 64 | 9.90 ms | 82.16 ms |
| 256 | 80.5 ms | 101.0 ms |
| 1024 | 208 ms | 306 ms |

(medians of 15, interleaved A/B in one process; the card is shared with six
other trainers, so single-shot timings on it swing by 5x and only medians of an
interleaved comparison mean anything -- an unmedianed first pass of this same
benchmark read 19 ms and 65 ms for the *same* configuration.)

`HostConstants.from_fields` now takes the fields the writer already computed,
flattens the five it needs into one `[M, 236]` float64 buffer and does **a single
`.cpu()`** -- one sync, 236 doubles per world. `compute(scale)`, the float64
host-side path, is kept as the reference the gate measures.

One consequence, recorded because it is a real (tiny) semantic change: in
production the spec is float32 on device, so `mj_setConst` now sees float32
fields where it used to see a float64 re-derivation. Measured against the
float64 path over 32 random designs, the constants differ by **1e-7 relative**
(`body_invweight0` 9.4e-08, `dof_invweight0` 9.8e-08, `actuator_acc0` 1.0e-07)
-- fp32 epsilon, and *more* consistent than before, since the model fields
actually being simulated are fp32 too. Against the 30% error from leaving these
constants stale it is nothing. The gate is unaffected: its spec is float64 CPU.

## Gate 2b -- whole model, stepped (PASS)

Field equality is necessary, not sufficient. 40 control steps (0.6 s) from
qpos0 under identical open-loop torques, our written model vs a fresh compile of
their emitted MJCF, 6 random designs:

| model | max \|dqpos\| |
|---|---|
| **writer + `mj_setConst` + body BVH** | **3.1e-15** |
| writer, `mj_setConst` skipped | 7.1e-2 |
| writer, body BVH left stale | 4.9e-2 |

The middle row is why `HostConstants` exists; the last is why `bvh_aabb` is in
`CPU_EXTRA_FIELDS`. With both, the per-world write is *the same model* as a
recompile, to machine precision, on a stepped trajectory -- not just field by
field.

## Gate 2c -- dev observation + reward parity (PASS, at machine epsilon)

Same discipline as gate 0b: hand-set states, never a shared rollout, driven
through their identical door -- but now each case also carries a random genome
that is applied first, so the reward is being computed on a differently-shaped
ant on both sides. 24 states, worst over all of them:

| field group | max abs diff |
|---|---|
| obs / stage flag | 0.0 |
| obs / scale vector (20) | 0.0 |
| obs / sim block -- own qpos, qvel, opponent xy (31) | 2.2e-16 |
| **obs / all 52 dims** | **2.2e-16** |
| **reward / torso subtree-COM x** | **8.9e-16** |
| reward / forward progress | 5.9e-14 |
| reward / ctrl cost | 4.4e-16 |
| reward / dense total | 5.9e-14 |
| reward / sparse (goal) | 0.0 |
| reward / total | 1.1e-13 |
| flags: fell, reached_goal, winner, terminated | 0 mismatches / 24 |

Coverage: 22/24 in contact, 6/24 with a fall, 6/24 with a goal crossed.

The `subtree_com` row is the one that matters for this stage. It is a
mass-weighted quantity, so it is only right if `body_mass`, `body_ipos` AND
`body_subtreemass` are all right for that world's design -- it is the check that
a geometry-only writer would fail.

### What the dev env does differently from the fixed-morph one

All four differences are theirs, not artifacts of batching:

1. **obs is 52** = `[stage flag (1) | scale vector (20) | sim obs (31)]`, their
   `DevAnt._get_obs` list flattened in order; **action is 28** =
   `[design (20) | motor (8)]`.
2. **Termination has an upper bound.** `DevAnt.after_step` requires
   `0.28 <= z <= 1.2`; the fixed ant has no ceiling. A dev ant launched upward
   dies and a fixed one does not.
3. **No reset noise survives.** Their `reset` produces `qpos0 + U(-0.1,0.1)`,
   which the step-0 rebuild then throws away by allocating a fresh `MjData`.
   Measured against their env: after the design step our qpos equals theirs to
   0.0 and qvel is exactly 0. The pre-design observation still carries the noisy
   state (the critic sees it), so both are reproduced.
4. **The dev merger's contact bitmask IS live** (unlike the gym_compete one --
   stage-1 note 8 flagged this and it checks out): `conaffinity=i,
   contype=1-i`, so the dev ants collide with each other and the floor but NOT
   with themselves. Asserted in the gate. It is a two-agent-only trick and 2v2
   will need a real bitmask.

## Gate 2d -- episode shape (PASS)

The design step pays reward 0, terminates nothing, flips the stage flag 0 -> 1,
writes the emitted genome into the observation's scale block, does not advance
`_elapsed_steps` (their `_step` is the only thing that does), and leaves the
world at their fresh-`MjData` state. The next step is a normal one and collects
the survive bonus. All measured against their env, not asserted from reading.

Batching note: worlds are asynchronous, so at any wall step a few are in the
design stage and the rest are executing -- exactly the mixed batch their
`DevPolicy.forward` partitions on. Physics is stepped for every world and the
design-stage worlds' step is DISCARDED (their state is overwritten). That wastes
~1/200 of the physics and keeps one CUDA-graph launch per step, which is worth
far more.

## The dev policy

`dev_ppo.DevActorCritic` is their `DevPolicy` + `DevValue` as one stage-masked
module: their `forward` loops over the batch bucketing samples by stage flag and
`get_log_prob` scatters the two log-prob columns back together, which is a
masked computation with a Python loop in front of it. Both heads run on the whole
batch here and the mask picks the answer.

Reproduced from `config/run-to-goal-devants-v0.yaml` and `dev_actor.py`:
scale head `RunningNorm(20) -> MLP[64,64] tanh -> Linear(20)` on ONLY the scale
vector (the design policy never sees the sim state), control head
`RunningNorm(31) -> MLP[64,128,64] tanh -> Linear(8)` on ONLY sim_obs
(`use_entire_obs: false`), critic `MLP[64,64,64]` on the full 52.

Three details that are easy to lose and all change behaviour:

* the scale head's output weights are scaled by **1.0**, not the 0.1 the control
  and value heads get (`dev_actor.py:29-30` vs `50-51`);
* the scale distribution is built with **std / 5** (`dev_actor.py:91`), so with
  `log_std` init 0 a fresh design policy explores at sigma 0.2, not 1.0;
* the dev curriculum is **`termination_epoch: 1000`**, five times the fixed-morph
  ants' 200 -- 50M agent-steps, so `alpha` barely moves in any smoke.

## Stage-2 smoke -- random genomes train without NaNs (PASS, and that is ALL it shows)

`runs/competevo_port/dev_smoke_v2/`. 1024 worlds, 12 min of training loop
(~16 min wall -- the Warp kernel build and the untrained baseline eval are not
in the budget), 27 iterations, **3.54M agent-steps**, 4,666 steps/s on a card
shared with six other trainers.

The claim being gated is narrow and it holds: **0 diverged worlds over 3.5M
steps** with a freshly drawn `s ~ U(-1,1)^20` per agent per episode -- so every
episode is a different pair of robots, and the per-world write, the
`mj_setConst` round-trip and the CUDA graph all survive being driven by a policy
rather than by a test. `nan_worlds 0` throughout, KL bounded (0.078 on iteration
0, ~4e-3 after), no loss blowup.

| | iter 0 | iter 10 | iter 20 | iter 26 |
|---|---|---|---|---|
| eval return | 511.2 / 512.0 | 487.8 / 505.4 | 450.6 / 468.7 | 282.5 / 233.6 |
| eval ep len | 500 | 500 | 460 | 255 |
| eval win rate | 0 / 0 | 0 / 0 | 0 / 0 | 0.011 / 0 |
| `design_std` | 0.233 | 0.317 | 0.330 | 0.337 |
| mean total mass | 1.819 | 1.832 | 1.834 | 1.830 |

What that table does and does not say:

* The return curve **falls**, and that is the same shape stage-1 smoke B had:
  the survive bonus dominates early, the ants learn to drive forward, and
  driving forward makes them topple, so episodes shorten from 500 to 255 and the
  accumulated bonus goes with them. It is not a divergence and it is not
  progress -- it is the exploration curriculum with `alpha` still at 0.965.
* `design_std` drifting 0.233 -> 0.337 says the **design head is moving** (it
  initialises at sigma 0.2 through the `std / 5`). It does not say it is moving
  anywhere good.
* Mean total mass is flat to 0.6% (1.819 -> 1.830), i.e. after 27 iterations the
  design policy has not committed to bigger or smaller ants. Expected:
  `termination_epoch` is 1000 and this is 27.
* The single win (0.011 = 1 game in 90, one agent, one eval) is noise. Nothing
  here is a win-rate result.
* **This is not the port map's stage-2 reference gate** (section 5.5: their M1
  curves, iter-0 eval ~428-440). Reproducing that is stage 3's job, because it
  needs their two-learner loop and opponent sampling. What is claimed here is
  that the machinery underneath is exact (gates 2a-2d) and that it runs.

An earlier 12-min smoke on the pre-fix write path is kept at
`runs/competevo_port/dev_smoke/` (31 iters, 4.06M steps, also 0 diverged). The
two runs track each other closely on episode length per iteration
(0.4 / 4.9 / 23 / 43 / 68 / 99 / 122 / 377 vs 0.4 / 4.8 / 24 / 41 / 71 / 103 /
124 / 377), which is the useful part of the comparison: the fp32 `mj_setConst`
inputs did not visibly move the training trajectory. Their wall-clock per
iteration is **not** a clean A/B -- six other trainers share the card and the
load differed between the runs -- but the direction agrees with the isolated
benchmark above, the early reset-heavy iterations being roughly 2x cheaper
(17.7 / 22.3 / 24.9 s -> 8.4 / 10.1 / 10.1 s for iterations 0, 2, 3).

## Stage-1 notes: what became of each

The stage-1 list of "what stage 2 needs, learned the hard way", with verdicts.

1. **`inertiafromgeom` is not a footnote** -- correct, and the suggested method
   (parameterize `test_model_matches_theirs` by a design vector rather than
   invent a trajectory gate) was the right order of work. Gate 2a. The advice
   was incomplete in one way: field equality on the fields you thought of is not
   the same as model equality, and the two things it missed (`mj_setConst`
   constants, the body BVH) were together worth 7 cm of trajectory. Gate 2b
   exists because of that.
2. **Check the per-world batching first** -- correct and worth the time. The
   answer is above: 105 fields are batchable, all default to shared, and
   `put_model(batch_sizes=...)` is the supported switch. Two corrections to the
   note: there is no `dof_M0` in mujoco_warp's Model, and **the CUDA graph does
   NOT need re-capturing** after a design write (only a shape change would, and
   shapes are fixed at `put_model`). The note's instinct that *derived* mass
   matrices are fine was right -- `qM` is rebuilt every step from
   `body_mass`/`body_inertia` -- but `body_subtreemass`, `geom_rbound` and
   `geom_aabb` are precomputed and are NOT rebuilt, so they do have to be
   written.
3. **The parity harness generalizes for free** -- correct.
   `their_dev_driver.py` is `their_env_driver.py` with a different `build_env()`
   and a design step, and it also returns the merged MJCF their code emitted,
   which turned out to be the reference that matters (it separates "our writer
   is wrong" from "mujoco 2.3.5 and 3.11 disagree").
4. **The dev obs is not the fixed obs** -- correct in every particular (52 =
   1 + 20 + 31, action 28 = 20 + 8, design step stored with reward 0 and
   mask 1, buffer shape unchanged, stage flag selects the head).
5. **One-time kernel compilation dominates short runs** -- still true, and
   switching the model arrays to per-world is a new specialization, so it is
   paid once more (~2 min) the first time.
6. **`nconmax`/`njmax` are per world** -- unchanged; the dev ants generate no
   more contacts than the fixed ones at these designs.
7. **Their goal rods collide** -- unchanged, still theirs.
8. **Self-collision is ON for the fixed ants, and the dev merger had to be
   checked separately** -- checked, and the note was right: the trick IS live in
   `evo_utils.create_multiagent_xml_str`, so the dev ants do NOT self-collide
   while the fixed ants do. Same task, two different robots. Asserted in gate 2a.

## What stage 3 needs

Stage 3 is their actual co-evolution loop: two learners and opponent sampling
(port map section 4.3), which is the part the paper's result actually rests on.

1. **Two learners, not one shared policy.** `dev_ppo` still trains ONE
   `DevActorCritic` playing both ants, as stage 1 did. Theirs holds two
   independent policy+critic pairs and updates them in order (agent 0, then
   agent 1, `optimize_policy`:91-92). The env, the buffer layout
   `[T, worlds, agents, ...]` and the design plumbing are already per-agent, so
   this is a trainer change only.
2. **The opponent checkpoint ring.** Per iteration they run TWO worker fleets:
   in fleet `idx`, ego agent `idx` uses its current weights and the opponent uses
   a checkpoint sampled uniformly from `[max(1, floor(0.5*epoch)), epoch]`, and
   only ego's half of each fleet's data is kept. On GPU: an in-memory ring of
   state_dicts, the world batch split into K opponent-blocks with one sampled
   checkpoint per block per iteration, and the ego role swapped between halves of
   the batch. Note `delta=0.5` for dev (the fixed-morph ants use `delta: 0`,
   full history), and that the dev runner does NOT make the opponent
   deterministic (the fixed-morph one does).
3. **The eval win-rate quotient.** Already implemented the way they count it
   (truncated draws in the denominator), but it has not been compared against
   their curves yet, because nothing has trained long enough to have a win rate.
   The stage-2 smoke's 0.011 is one game in ninety.
3b. **The design write is the next thing that will hurt, and it is host-bound.**
   After the fused-D2H fix a write costs 3.7 ms for one world and 208 ms for all
   1024, against a ~54 ms step -- i.e. it is dominated by a fixed per-call cost
   (`design_fields` is tens of small kernels) plus a serial per-world
   `mj_setConst` loop on the CPU. Steady state is fine, because only the worlds
   that reset pay it, but two stage-3 changes push on exactly this: opponent
   blocks make resets bunch up, and any curriculum that shortens episodes drives
   the reset rate up (the smoke's first iterations, where every world resets
   every step, are 5-15x slower per step than its last ones). If it becomes the
   bottleneck the two levers, in order, are: batch the `mj_setConst` loop (it is
   an `inv(M)` at qpos0 -- mujoco_warp can already do that on device for all
   worlds at once), and fuse `design_fields` into one Warp kernel instead of ~40
   torch ops.
4. **The reference curve exists and should be used.** Their M1 sanity run of this
   exact config (`tmp/run-to-goal-devants-v0/...`) gives iter-0 eval ~428-440 at
   win rate 0.00 and TB curves for the first ~50-100 epochs. That is the stage-2
   gate in the port map's numbering (section 5.5) and it is NOT claimed here --
   what is claimed is that the machinery under it is exact.
5. **Design-parameter CSVs.** Their runner logs 10 random designs per epoch to
   `{run_dir}/{0,1}.csv`; the qualitative convergence of those is one of the
   paper-level comparisons. `train_dev.py` logs `design_mean`/`design_std` only.
6. **Sumo (`robo-sumo-devants-v0`) needs more than a config swap:** per-world
   arena radius (another `geom_size` write, which this writer already supports),
   the `|cfrc_ext|` + torso `xmat` observation block, the win/lose/draw
   structure, and their transform-step obs off-by-one (which the port map says
   not to reproduce).
7. **2v2 will break the contact bitmask.** `conaffinity=i, contype=1-i` only
   works for exactly two agents; it is live in the dev merger, so this is now a
   real constraint on the port and not just a note about their code.
