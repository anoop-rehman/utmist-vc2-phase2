# CompetEvo -> mujoco_warp port: status (unit 2d, stages 0-1)

*Worktree `competevo-port`. Reference: `/workspace/competevo` (read-only).
Plan: `rower_soccer/docs/repro/COMPETEVO_PORT_MAP.md`, section 5.5.
This file records gate results, INCLUDING the ones that did not pass and the
things that pass only because of a quirk in their code.*

Scope so far: **`run-to-goal-ants-v0`** (fixed morphology, two ants, one merged
scene, batched over worlds) plus a shared-policy PPO smoke run. Evolution
(`run-to-goal-devants-v0`), the design-action stage flag, the per-world
model-field writer and faithful opponent sampling are NOT here yet.

Numbering: this covers the port map's **Stage 0** (fixed-morph harness), split
into an env-parity gate and a PPO smoke. The port map's Stage 1 (the design ->
model-fields writer) is the next unit and is referred to below as "stage 2"
where the task's numbering is used.

## What exists

| file | what |
|---|---|
| `scene.py` | the merged 2-ant MJCF, emitted from a leg table; per-agent index plumbing (`SceneMeta`/`AgentSlices`) |
| `backend.py` | `warp_port` backends + `subtree_com`/`cfrc_ext`; CPU backend runs float64 for the gate |
| `run_to_goal_env.py` | batched env: obs, their three reward/termination layers, per-world auto-reset, win counters |
| `parity.py`, `their_env_driver.py` | JSON-over-subprocess harness driving their CPU env in their venv |
| `tests/test_parity.py` | the gate (model equivalence, obs+reward parity, solver-divergence diagnostic) |
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

### Run B -- 15 min, their exploration curriculum

<!--RUN B-->


## What stage 2 needs, learned the hard way

1. **`inertiafromgeom` is not a footnote, and the numbers to match are already
   available.** The compiled fixed ant has torso mass 0.327 (sphere r=0.25 at
   density 5) and per-leg masses 0.0392 / 0.0392 / 0.0676 for the upper, mid and
   foot capsules. Any per-world design write must reproduce MuJoCo's capsule
   formula exactly -- mass = density * (pi r^2 L + 4/3 pi r^3), with the
   cylinder+two-hemisphere inertia about the capsule frame, then `body_ipos` at
   the capsule midpoint. The cheapest possible gate: for 10 random design
   vectors, emit their XML with `set_design_params`, compile it with MuJoCo on
   the CPU, and diff `body_mass`/`body_inertia`/`body_ipos`/`geom_*`/
   `actuator_gear` against our per-world writer. That is exactly the
   `test_model_matches_theirs` check already written here, parameterized by a
   design vector -- reuse it rather than inventing a trajectory gate first.
2. **Check which model fields mujoco_warp actually batches per world BEFORE
   writing the design code.** `put_model` produces a `Model` whose arrays are
   mostly unbatched (shared across worlds). Confirm the per-world axis exists
   for `geom_size`, `geom_pos`, `geom_quat`, `body_pos`, `body_mass`,
   `body_inertia`, `body_ipos`, `dof_M0`-style derived quantities and
   `actuator_gear`; anything unbatched needs expanding, and note that *derived*
   mass matrices are recomputed each step from `body_mass/body_inertia`, so
   writing those is enough only if nothing was baked at `put_model` time.
   Re-capture the CUDA graph after any model-shape change.
3. **The parity harness generalizes for free, and should be used the same way.**
   `their_env_driver.py` already speaks (prev-state, state, action) and returns
   obs/reward field by field; the dev env needs only a different `build_env()`,
   a `set_design_params` call before the state is set, and the 3-array obs
   (`[stage_flag(1) | scale_vector(20) | sim_obs(31)]`) flattened in that order.
   Keep the same "hand-set states, never a shared rollout" discipline: with a
   different solver, a trajectory diff is a solver measurement.
4. **The dev env's obs is NOT this obs.** Stage 2's sim_obs block is the same 31
   numbers, but the policy input is 52 = stage flag + 20 scale + 31 sim, and the
   design action is step 0 of the trajectory with reward 0 and mask 1. The PPO
   buffer here is already `[T, worlds, agents, ...]`; the stage flag becomes a
   per-world boolean selecting the policy head, so the buffer shape does not
   change -- only the head selection and the action width (28 = 20 + 8) do.
5. **One-time kernel compilation dominates short runs.** The first build of this
   scene at `iterations=100` took ~122 s of Warp kernel compilation (cached
   afterwards). Budget it; do not mistake it for a hang, and do not let a
   parameter sweep silently recompile per configuration.
6. **`nconmax`/`njmax` are per world** in this mujoco_warp (64/512 defaults are
   ample here: 32 contacts observed for two ants, a floor and two goal rods).
7. **Their goal rods collide.** `rightgoal`/`leftgoal` are cylinders with default
   contype/conaffinity, so ants can hit them, and mujoco_warp warns that
   CAPSULE-CYLINDER pairs get at most one contact under MULTICCD. Kept as-is
   (it is their physics); worth remembering if a stage-2 ant ever wedges itself
   against a goal line.
8. **Self-collision is ON for these ants.** The port map's section 1.2
   contype/conaffinity trick is from `evo_utils.py` and does *not* apply to the
   gym_compete merger: because the ant's `<default>` already declares a `<geom>`,
   their `color_set` flag skips that branch entirely. Do not "fix" it, and do
   check the dev merger separately at stage 2 -- there the trick IS live, and it
   only works for exactly two agents.
