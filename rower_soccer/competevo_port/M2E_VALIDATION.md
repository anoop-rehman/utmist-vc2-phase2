# Milestone 2e -- paper-number validation of the GPU port

*Worktree `competevo-port`. Reference run: `/workspace/competevo` (read-only),
their exact `config/run-to-goal-devants-v0.yaml` on CPU, launched 2026-08-10 and
**still running** as PID 2846036.*

The claim this milestone is allowed to make is narrow and it is a comparison, not
a smoke: our port, configured to be their config, should produce the same
`eval reward` / `win rate` trajectory *per epoch* as their code does. Everything
below is either a measured number from this session or a statement that a number
could not be measured.

## 0. The reference is better than the plan assumed

`PLAN_D2_COMPETEVO.md` and the brief say to compare against "paper Fig. curves".
There is a much stronger reference available: **their own CPU run of this exact
config is alive on this pod and has reached epoch 159**, with a per-epoch
`Agent_i gets eval reward` / `Agent_i gets win rate` line for every epoch from 0.
`compare_curves.py` in this directory parses it. Two corrections to
`REPRO_NOTES.md` fall out of reading it:

1. **`termination_epoch` for this config is 1000, not 200.** REPRO_NOTES line 100
   says the curriculum is `termination_epoch: 200`; that is
   `run-to-goal-ants-v0.yaml` (the FIXED-morph ants). The devants config the
   sanity run actually uses says `termination_epoch: 1000`
   (`run-to-goal-devants-v0.yaml:51`), so alpha falls 5x more slowly than the
   note implies: at epoch 100 the sparse goal term still carries only 10% weight.
   The port already had this right (`dev_ppo.DEV_CURRICULUM_STEPS = 1000 *
   50_000`); the note is what is wrong.
2. **Their measured iteration cost is now 6-7.5 min, not 4.5.** Epochs 156-158
   took 193+297+17, 146+255+15 and 158+256+22 seconds. The 4.5 min in
   REPRO_NOTES was the first iteration; the policy update grew from 95 s to
   ~255-300 s as episodes lengthened.

Their win rate first leaves 0 at **epoch 89** and is intermittently nonzero from
epoch 92 on. That is the single number this milestone is trying to reproduce.

## 1. Config mapping -- their yaml to our flags

Their file is `config/run-to-goal-devants-v0.yaml`. "Exact" below means the
quantity is numerically identical, not that the code path is a transcription.

| their key | their value | our flag / constant | exact? |
|---|---|---|---|
| `env_name: run-to-goal-devants-v0` | -- | `dev_env.RunToGoalDevEnv` | yes -- gates 2a-2d, machine epsilon |
| `runner_type: multi-evo-agent-runner` | -- | `selfplay.CoEvoPPO` | structurally; see 2 |
| `min_batch_size` | 50,000 ego transitions per learner per epoch | `--worlds 1000 --rollout 100` -> `n_ego = 500`, `500 x 100 = 50,000` | **exact** |
| (implied) env steps per epoch | 2 fleets x 50,000 = 100,000 | 1000 worlds x 100 steps = 100,000 | **exact** |
| `mini_batch_size` | 2048 | `--minibatch 2048` | exact |
| `num_optim_epoch` | 10 | `--epochs 10` | exact |
| `gamma` | 0.995 | default | exact |
| `tau` (GAE lambda) | 0.95 | default | exact |
| `clip_epsilon` | 0.2 | default | exact |
| `policy_lr` | 5e-5 (Adam, wd 0) | `--policy-lr 5e-5` | exact |
| `value_lr` | 3e-4 (Adam) | `--value-lr 3e-4` | exact |
| `l2_reg` | 1e-3, added to the value loss as `sum(p^2) * l2_reg` | Adam `weight_decay=1e-3` on the value optimizer | **NO -- half strength**, see 3 |
| policy grad clip | 40, on `policy_net.parameters()` only | 40, on **all** parameters of the joint backward | **NO**, see 3 |
| `termination_epoch` | 1000 | `DEV_CURRICULUM_STEPS = 1000 * 50_000`; `alpha = 1 - total_steps/50e6 = 1 - iter/1000` | **exact**, epoch for epoch |
| `use_exploration_curriculum` | true | on by default | exact |
| `use_parse_reward` | true | the `parse` term | exact |
| `use_reward_scaling` | false | not implemented | exact (nothing to do) |
| `use_opponent_sample` | true | default (do NOT pass `--no-opponent-sample`) | exact |
| `delta` | 0.5 | `--delta 0.5` | exact rule (`OpponentRing.sample_epoch`), see 2 |
| `save_model_interval` | 1 | `--checkpoint-every 1` | exact |
| `max_epoch_num` | 1000 | `--iters` / `--minutes` | partial -- we will not reach 1000 |
| `eval_batch_size` | 10,000 steps over 10 workers | `--eval-worlds 64`, full episodes | **NO**, see 3 |
| `seed` | 42 | `--seed 42` | nominal only -- different RNG streams entirely |
| `use_entire_obs` | false | ported (control head sees only `sim_obs`) | exact |
| `dev_policy_specs` | scale `[64,64]` tanh, control `[64,128,64]` tanh, `log_std 0` | ported, incl. `std/5` on the scale head and the `x1.0` (not `x0.1`) scale-head output init | exact |
| `dev_value_specs` | `[64,64,64]` tanh | ported | exact |
| `agent_specs.batch_design: true` | -- | the design step | exact |
| `--num_threads 24` | 24 sampler processes | batched worlds | n/a |

**The critical line is the first one.** With `--worlds 1000 --rollout 100` one of
our iterations is exactly one of their epochs -- same 50,000 trained ego
transitions per learner, same 100,000 simulated agent-transitions -- so the x
axes line up with no rescaling, and `alpha` matches epoch for epoch without
touching `--curriculum-steps`. Any other `(worlds, rollout)` pair silently
rescales the curriculum against the epoch axis. **Do not pass
`--curriculum-steps` for this run**; the default is already theirs.

## 2. Structural deviations carried in from stages 0-3

These are recorded in `PORT_STATUS.md` and are unchanged; they are repeated here
because they are the candidate explanations if a curve does not line up.

1. `solver="PGS", iterations=1000` -> `Newton, 100` (mujoco_warp has no PGS).
   Measured cost, isolated: 1.5e-7 m of trajectory drift over 0.6 s.
2. fp32 on GPU vs their float64. Measured: 1.9e-7 on the observation, 3.0e-5 on
   the reward.
3. Contact cost is a constant 0 -- on **both** sides (their `cfrc_ext` is never
   filled; measured, not assumed).
4. Per-world auto-reset; worlds run out of phase. Their env resets wholesale.
5. Reset noise drawn once instead of twice (only their last draw survives
   anyway, and for the dev env the design step discards it entirely).
6. **`blocks = 4` opponent slots per side.** Theirs draws a fresh checkpoint per
   episode per worker, i.e. effectively dozens of distinct opponents live per
   epoch. Ours has 4 per side, redrawn every iteration; a world redraws its slot
   at its own episode reset. The marginal distribution is theirs, the diversity
   within one epoch is narrower, and a world's opponent can change mid-episode
   at an iteration boundary. **This is the most likely place for a real
   behavioural difference in co-evolution** and nobody has measured how wide is
   wide enough.
7. **Ego-split worlds** (500 ego worlds per learner) instead of two full fleets
   of 1000. Same ego data, half the physics; but a given world only ever trains
   one learner.
8. **Epoch 0 plays the learners' current weights.** Theirs plays freshly
   constructed `DevSampler` nets, because its epoch-0 checkpoint load is inside
   a bare `try/except: pass` and the file does not exist. One epoch.
9. Bounded 512-entry opponent ring instead of unbounded pickles. `delta = 0.5`
   never clamps below epoch ~1024; `ring_clamped` in the log is 0 or the run is
   no longer sampling their distribution.

## 3. Deviations found while building THIS milestone

Four, and two of them were fixed here.

### 3a. Their eval reward is the CURRICULUM reward (fixed: now logged)

`multi_evo_agent_runner.sample_worker` applies `custom_reward` whether or not
`mean_action` is set, and `LoggerRL.step` accumulates the reward it was handed
(`lib/rl/core/logger_rl.py:25-32`). So `Agent_i gets eval reward` is
`sum_t [alpha * dense + (1 - alpha) * parse]` at the **current epoch's alpha**,
not the env reward. Our `evaluate_pair` reported the env return
`parse + dense`. The two coincide only while no goal is ever crossed -- i.e.
exactly up to their epoch 89 -- and then diverge by up to +/-1000 per game.

`evaluate_pair(..., alpha=)` now accumulates the curriculum return as well and
the trainer logs it as `eval_ret_curriculum`; `eval_ret` (env return) is kept.
**`eval_ret_curriculum` is the column to compare against theirs.**

### 3b. The design step leaked a reward into the curriculum (fixed)

Their `MultiDevAgentEnv.step` returns `reward_parse: 0, reward_dense: 0` for the
`attribute_transform` stage (`multi_dev_agent_env.py:286-289`). Our `dev_env`
zeroed the env `reward` for design-stage worlds but returned the *executed*
`dense`/`parse` in the info dict, which is what the curriculum trainer reads.
So every episode paid one spurious ~+1 (the survive bonus) that their runner
never pays, and the value target at the stage boundary was wrong. ~1 in 400
steps; small, but it is a fidelity bug and it is now fixed (`dev_env.step`
masks `dense`/`parse` with the same `keep` mask as `reward`). `terms()` is
untouched, so gates 0b/2c are unaffected.

### 3c. The critic's L2 is half theirs, and the grad clip is wider (NOT fixed)

Theirs adds `sum(p^2) * 1e-3` to the value loss, whose gradient contribution is
`2e-3 * p`; ours passes `weight_decay=1e-3` to Adam, contributing `1e-3 * p`.
And theirs clips the gradient norm at 40 over the **policy** parameters only,
after a separate value backward+step; ours does one `(pi_loss + vf_loss)
.backward()` and clips 40 over all parameters. Both are pre-existing port
choices. Not changed for this run: changing the optimizer path invalidates the
stage-1/2/3 numbers everything else is calibrated against, and neither should
bind unless gradients are large. Recorded so that it is on the list if a curve
misses.

### 3d. The control cost was billed on the CLIPPED action (fixed -- this one was worth the run)

Their `DevAnt.after_step(action)` is handed `actions[i][-8:]` straight off the
policy (`multi_dev_agent_env.py:311`) and charges
`ctrl_cost = .5 * np.square(action).sum()`. MuJoCo clamps `ctrl` to
`ctrlrange` inside the step, so the **torque** is clipped and the **cost is
not**. Our `dev_env.step` clamped the motor action to `[-1, 1]` and then handed
the clamped value to `terms()`, so it billed the clipped action.

At `log_std = 0` this is not a rounding detail, it is the dominant term of the
early reward. Eight independent unit Gaussians cost `0.5 * 8 * E[a^2] = 4.0` per
step raw and `0.5 * 8 * E[clip(a,-1,1)^2] ~= 2.1` clipped. With `alpha = 1` the
reward is `1 (survive) - ctrl + forward` and `forward` is ~0 for an untrained
ant, so the whole optimization landscape early on is the control cost.

Measured, before the fix (the 4-iteration smoke of the mapped config):
our sampled episodes ran at **-1.10 reward per step** (`train_ret -117.5` over
`train_len 106`). Their logged `train_R_eps_avg_0` is **-1182.5** at epoch 0 and
-1046 at epoch 5, which at their ~400-step sampled episodes is **~-3.0 per
step**. `1 - 2.07 = -1.07` and `1 - 4.0 = -3.0`: the two numbers are exactly the
two clipping conventions.

Fixed in both envs (`dev_env.step`, `run_to_goal_env.step`): the clamped action
drives the actuators, the raw action is billed. **`terms()` is untouched and was
always faithful** -- which is exactly why every parity gate passed: they call
`terms()` directly with the action they want billed and never go through
`step()`'s clamp. The validation run was relaunched after this fix; the run
started before it (13:50) was killed at 4 iterations and is not reported.

### 3e. Eval sample size and protocol (cannot be matched exactly)

Theirs fills 10,000 steps across 10 workers with mean actions, so the number of
games depends on how long episodes are -- roughly 20-100, and their printed win
rates (0.33, 0.50, 0.60, 0.20) show the granularity. Ours runs `--eval-worlds`
worlds to completion. At 64 worlds we get ~64+ games, i.e. a *less* noisy
estimate of the same quantity. Their win rate is `wins_i / games` with truncated
draws in the denominator (`multi_evo_agent_runner.py:369-372`); ours is the
same. The eval envs also differ in the usual way (fp32, Newton, per-world
reset).

One further honest point about **epoch 0**: with mean actions the design action
is deterministic, so the epoch-0 eval is essentially one episode replayed, and
its value is a function of the random initialization of the scale head -- whose
output weights are scaled by 1.0, not 0.1, so a fresh design policy emits a
non-trivial body plan. Their 428.0 / 428.5 is therefore a **draw from a
seed-dependent distribution**, not a constant, and comparing a single number
against it is only meaningful next to the spread of that distribution. Measured
spread over our seeds is in section 5.

## 4. The smoke, and the run

**Smoke** (`runs/competevo_port/m2e_smoke/`, 4 iterations, 547 s): the mapped
config starts, `alpha` reads 1.0000 / 0.9990 / 0.9980 / 0.9970 -- i.e. exactly
`1 - epoch/1000`, their schedule, epoch for epoch -- the eval fires and reports
`eval_ret_curriculum` = `alpha x eval_ret` to 4 digits (correct, because `parse`
is identically zero while nobody scores), the ring fills 1/1 -> 4/4 with 0
clamps, `opp_lag` climbs 0 -> 1.5 as history accumulates, and `nan_worlds` is 0.
This smoke is what exposed deviation 3d.

**Validation run**, launched 2026-08-11 13:55 UTC, PID 3379808:

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
cd /workspace/vc2-competevo-port
PYTHONPATH=. MUJOCO_GL=egl nohup /workspace/utmist-vc2-phase2/.venv/bin/python \
  -m rower_soccer.competevo_port.train_selfplay \
  --worlds 1000 --rollout 100 --epochs 10 --minibatch 2048 \
  --policy-lr 5e-5 --value-lr 3e-4 --delta 0.5 --blocks 4 \
  --ring-capacity 512 --checkpoint-every 1 --seed 42 \
  --eval-worlds 64 --eval-every 2 --iters 1000 --minutes 258 \
  --out runs/competevo_port/m2e_validation \
  > runs/competevo_port/m2e_validation/train.log 2>&1 &
```

Run directory `runs/competevo_port/m2e_validation/`; `log.json` is the
authoritative record (`train.log` for this run has interleaved bytes near the
start -- see the note at the end of section 5).

## 5. Measured

### The reward per step now matches theirs

The first thing to check after deviation 3d, because it is the term the early
curriculum is made of. Sampled (stochastic) rollouts:

| | reward per agent-step, `alpha = 1` |
|---|---|
| theirs, `train_R_eps_avg_0` -1182.5 at epoch 0 over ~400-step episodes | **~-3.0** |
| ours, before the 3d fix | -1.10 (iterations 0-4, flat) |
| **ours, after the 3d fix** | **-3.11 / -3.03 / -2.99 / -3.00 / -2.99** (iterations 0-4) |

Isolated check on a CPU env with `a ~ N(0,1)`: `ctrl_cost` = 4.0056 per
agent-step against the analytic `0.5 x 8 x 1 = 4.0`, total reward -2.99.

### Epoch-0 eval: we do NOT reproduce their 428, and it is not the seed

`baseline_spread.py`, 8 seeds, 64 eval worlds, untrained pair, mean actions:

| | per-agent epoch-0 eval reward | eval episode length |
|---|---|---|
| ours, 8 seeds | 501.3, sd 10.0, range [483.9, 519.4] | **500** in every seed |
| theirs (their log + their TB `episode_length`) | 428.0 / 428.5 | **501** (= 500 sim + 1 design step) |

So this is **not** seed noise -- their value is 7.3 sd below our seed mean -- and
it is **not** an episode-length difference, which was the obvious hypothesis:
their epoch-0 eval episodes run the full 500 steps, same as ours.

That pins the difference down to one term. With `alpha = 1` and no goal ever
crossed, the eval return is `500 x survive + sum(forward) - sum(ctrl)`; at mean
actions the control head's `x0.1` output init makes `ctrl` negligible on both
sides. So

* ours: `sum(forward) ~= +1`, i.e. the ant's torso COM ends where it started;
* theirs: `sum(forward) ~= -72`, i.e. **their standing ant slides ~1.08 m away
  from its own goal over the 7.5 s episode** (`forward` is
  `move_sign x dCOM_x / 0.015`, so -72 is exactly -1.08 m).

Both of their agents show it symmetrically (428.0 and 428.5), which is what a
body-frame-symmetric slide looks like in a mirror-symmetric scene, and it
persists through their epochs 1-40 (eval 403-450, episode length 501). Ours does
not slide. The leading candidate is the one deviation that is known to be
trajectory-level rather than field-level -- fp32 mujoco_warp + Newton against
float64 MuJoCo 2.3.5 + PGS, which gate 0c already declined to claim agreement
for over 40 control steps, let alone 500. **This is stated as a measured
difference with a named suspect, not as a proven cause: nothing in this session
drove their env for 500 steps to confirm the slide directly.**

Consequence for the gate: the port map's "iter-0 eval ~428-440" gate for the dev
env is **NOT met**, at 501 +/- 10. What IS met is that both sides are a
standing-still policy collecting a survive bonus for 500 steps, and that our
number sits where our own stage-1/2 runs put it (PORT_STATUS records 511/512 and
508/484 for the same quantity).

*Note on `train.log`:* the pre-fix run launched at 13:50 was killed by PID at
13:53, but only its bash wrapper died -- the python child (PID 3378640) survived
and kept writing into the same run directory until 14:03, when it was killed
properly. Both processes rewrote `log.json` wholesale each iteration, so the
file has been authoritative for the surviving run since 14:03; `train.log` has
interleaved bytes from before then. Recorded because a reader of `train.log`
would otherwise see two runs' iteration 0.

## 6. Three things a reader of `log.json` will get wrong

### `"curriculum_steps": null` in `args` does NOT mean the curriculum is off

It is the CLI *override* flag, and `None` means "do not override the default".
`train_selfplay.py` only forwards it when it is set:

```python
kw = ({} if args.curriculum_steps is None
      else {"curriculum_steps": args.curriculum_steps})
```

so the trainer uses `dev_ppo.DEV_CURRICULUM_STEPS = 1000 * 50_000`, which is
their `termination_epoch: 1000` expressed in agent-steps. **Passing the flag
would have been the way to get this WRONG**, not the way to get it right.

The proof is in the log rather than in the code: every row carries the `alpha`
the iteration sampled at, and it reads

```
iter   0 1 2 3 4 5 6 7 8 9
alpha  1.000 0.999 0.998 0.997 0.996 0.995 0.994 0.993 0.992 0.991
```

i.e. exactly `1 - epoch/1000`, their schedule, epoch for epoch. If the
curriculum were disabled the trainer would log `alpha: null` and would optimize
`parse + dense` instead.

And `termination_epoch` is **1000** for this config
(`run-to-goal-devants-v0.yaml:51`). The `200` in REPRO_NOTES' protocol sentence
is `run-to-goal-ants-v0.yaml:43`, the FIXED-morph ants, which is not the config
their sanity run is running. This matters for what the run can conclude: at
`termination_epoch: 1000`, alpha is still **0.90** at epoch 100, so the +/-1000
goal term carries 10% weight there -- and **their** win rate leaves 0 at epoch
89 anyway. So in their run the win rate does not leave 0 because the sparse term
started to bite; it leaves 0 because the ants learned to run far enough to cross
a goal line while the reward was still ~90% dense. Reproducing that is a
statement about locomotion learning, not about the curriculum schedule.

### One of our iterations is exactly one of their epochs -- in every unit

| unit | theirs, per epoch | ours, per iteration |
|---|---|---|
| ego transitions **per learner** (their `min_batch_size`) | 50,000 | 500 ego worlds x 100 rollout = **50,000** |
| ego transitions, both learners | 100,000 | **100,000** (`trainer.total_steps`) |
| simulated world-steps | 2 fleets x 50,000 = 100,000 | 1000 worlds x 100 = **100,000** |
| simulated agent-transitions (2 agents) | 200,000 | **200,000** |
| discarded (opponent-lane) transitions | 100,000 | 100,000 |

The 2x between "env-steps" and "agent-transitions" is the easy mistake and it is
worth restating that **theirs pays it too**: their two fleets each simulate
50,000 env steps and the merge keeps only the ego half of each
(`multi_evo_agent_runner.py:457`). `compare_curves.py` recomputes
`(worlds/2) * rollout` from `log.json` and prints a MISMATCH banner if it is not
50,000, so the axes cannot be silently misaligned by a later run.

### `train_ret` diving to ~-1100 is the metric warming up, not a collapse

`train_ret` is the mean over worlds of the **last completed episode's** return,
and it starts at 0 for every world that has not finished an episode yet. So its
early trajectory is dominated by episode LENGTH, not by reward quality:

| iter | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| `train_len` | 2.4 | 27.0 | 63.9 | 110.5 | 156.2 | 367.0 | 369.2 | 367.2 | 364.2 | 368.2 |
| `train_ret` | -7.5 | -82.0 | -191.3 | -332.2 | -467.5 | -1089.5 | -1085.3 | -1065.6 | -1041.0 | -1033.9 |
| **per step** | -3.11 | -3.03 | -2.99 | -3.00 | -2.99 | -2.97 | -2.94 | -2.90 | -2.86 | **-2.81** |

Per step it is flat at -3.0 and then **improving monotonically** from iteration
5 once episode length saturates. The dive is `-3.0 x length` with length growing
from 2 to 367.

Against theirs, which has no warm-up because their logger only reports completed
episodes: `train_R_eps_avg_0` = **-1182.5** at epoch 0, -1046 at 5, -929 at 10,
-658 at 20, -301 at 40, -177 at 50, -5 at 89, +240 at 159. Ours plateaus at
**-1090** at iteration 5 and is at -1034 by iteration 9. **That is the closest
agreement anything in this port has produced against their curves**, and it is
the direct consequence of the section-3d fix -- before it, ours would have
plateaued near -400.

## 7. Is a ~110-epoch run long enough to test the win-rate criterion? Yes.

The worry is reasonable and the answer is measurable, because their run is the
same config: **their win rate leaves 0 at epoch 89, where `alpha = 0.911`.**

| their epoch | alpha | win rate 0 / 1 | their eval reward |
|---|---|---|---|
| 89 | 0.911 | 0.00 / **0.17** | 126 / 191 |
| 92 | 0.908 | 0.00 / 0.25 | 199 / 310 |
| 94 | 0.906 | 0.00 / 0.40 | 195 / 361 |
| 97 | 0.903 | 0.00 / 0.25 | 222 / 381 |

So in their run the win rate does **not** leave zero because the sparse term
started to bite -- at alpha 0.91 the +/-1000 goal reward carries 9% weight and
has carried roughly that much since epoch 50. It leaves zero because the ants
have learned to run far enough to cross a goal line. A ~110-epoch run reaches
`alpha = 0.89` and therefore **brackets their transition with ~20 epochs of
margin**. The criterion is testable in this budget.

Two honest qualifications on how much a null result would prove:

* their win-rate signal there is **noisy and one-sided**: 0.17-0.40 for agent 1
  and a flat 0.00 for agent 0 all the way to epoch 149, on an eval of roughly
  4-12 games. It is a real departure from zero, but it is not a clean number.
* our learning is currently running **slower per epoch than theirs** on the one
  quantity that is directly comparable, `train_R_eps_avg` (see below), so a
  0.00 at epoch 110 would be evidence that the port learns more slowly, not
  proof that it cannot cross.

### The other landmark this budget definitely reaches

Their eval reward does something much less ambiguous than the win rate before
epoch 89: it **collapses**, 377/446 at epoch 40 to -6/-3 at epoch 50, stays at
about -10 through epoch 70, and recovers from 75.

| their epoch | 40 | 42 | 44 | 46 | 48 | 50 | 60 | 70 | 75 | 80 |
|---|---|---|---|---|---|---|---|---|---|---|
| eval reward, agent 0 | 377.8 | 203.6 | 128.3 | 48.2 | 8.6 | -5.9 | -12.5 | -8.3 | 100.1 | 134.9 |

That is the same dive-then-recover PORT_STATUS records for stage-1 smoke B and
`dev_smoke_v2` -- a policy that stops standing still, starts running, falls, and
loses the survive bonus -- and it lands squarely inside a 110-epoch budget. It
is a sharper shape test than the win rate, because it is a 400-point excursion
rather than a 0.2 on twelve games.

### Cost, corrected

Measured on this run: **~133-136 s per iteration** (15 iterations in 1,988 s;
18 in 2,454 s), on a card shared with six drill trainers and a Transform2Act
run. So

* a full 1000-epoch dev run at this rate is **~37-38 hours**, not 4;
* but their CPU run of the same config is currently taking **6-7.5 min per
  epoch**, i.e. ~110 hours for 1000 epochs. The port is **~3x** faster
  end-to-end here, not the ~19x the raw env-step ratio suggests, because this
  configuration pays their PPO settings -- 10 optimizer epochs at minibatch
  2048 is 250 optimizer steps per learner per iteration, 500 in total, against
  the 16 the stage-3 smokes took -- and because two learners cost ~1.65x one.

## 8. The comparison, epochs 0-40 (run in progress)

Both columns are the same quantity computed the same way: `train` is the mean
sampled-episode return under the curriculum reward (their
`train_R_eps_avg_0` / our `train_ret`), `eval` is the mean-action episode return
under the curriculum reward (their `Agent_i gets eval reward` / our
`eval_ret_curriculum`), `len` is the mean eval episode length. Their epoch axis
and our iteration axis are the same axis (section 6).

| epoch | OURS train | OURS eval 0/1 | OURS len | OURS win | THEIRS train | THEIRS eval 0/1 | THEIRS len |
|---|---|---|---|---|---|---|---|
| 0 | -7* | 510 / 496 | 500 | 0.00 | -1183 | 428 / 429 | 501 |
| 6 | -1085 | 499 / 513 | 500 | 0.00 | -1038 | 425 / 424 | 501 |
| 12 | -967 | 499 / 470 | 500 | 0.00 | -819 | 412 / 435 | 501 |
| 18 | -868 | 488 / 514 | 500 | 0.00 | -715 | 412 / 413 | 501 |
| 24 | -745 | 484 / 524 | 500 | 0.00 | -632 | 406 / 434 | 501 |
| 30 | -573 | 457 / 548 | 497 | 0.00 | -501 | 426 / 453 | 501 |
| 34 | -503 | 386 / 490 | **427** | 0.00 | -435 | 412 / 451 | 501 |
| 38 | -436 | 317 / 336 | **318** | 0.00 | -340 | 441 / 413 | 501 |
| 40 | -395 | 219 / 264 | **228** | 0.00 | -302 | 378 / 446 | 501 |

\* our `train_ret` at epochs 0-4 is the metric warming up (section 6), not a
reward difference; it reaches its true level at epoch 5-6.

What lines up:

* **The training-reward curve has the same shape and nearly the same level.**
  Both climb monotonically from about -1100 toward 0; ours trails theirs by
  roughly 5-8 epochs throughout (ours -1041 at 8, -633 at 28, -395 at 40;
  theirs -972 at 8, -530 at 28, -302 at 40). This is the port's first
  quantitative agreement with one of their training curves, and it exists only
  because of the section-3d fix.
* **The eval-reward collapse happens, and at a comparable epoch.** Theirs
  falls off a cliff between epochs 41 and 50 (378 -> 204 -> 128 -> 9 -> -6) as
  the ants stop standing still, start running and begin to fall; the eval
  episode length goes with it. Ours starts the same collapse at **epoch ~34**
  (eval length 500 -> 427 -> 318 -> 228 by epoch 40). Same landmark, ours
  arriving ~7 epochs earlier.

What does not line up:

* **The absolute eval level is ~80-100 points higher on our side for the whole
  plateau** (our 484-511 against their 403-446 through epoch 30). That is the
  epoch-0 gap of section 5 persisting, not a new effect: their standing ant
  slides backwards and ours does not, which is worth a constant ~-72 per
  episode to them for as long as both policies are standing still.
* Ours trails on `train_ret` yet reaches the collapse EARLIER. Both facts are
  consistent with our policy committing to locomotion sooner, and neither is
  explained here.

Health through epoch 42: **0 diverged worlds, 0 ring clamps, 0 NaNs**, ring
42/42 both sides, `opp_lag` 9.4-11.5 against the `epoch/4` their delta=0.5 rule
predicts (10.5 at epoch 42), KL ~2e-3 per iteration, mean total mass 1.824 and
`design_std` 0.292 (the design head is moving and has committed to nothing --
unchanged from stages 2-3, and expected at `termination_epoch: 1000`).

## 9. Result: 107 epochs, finished

The run completed its budget: **107 epochs (0-106) in 257.3 min, 144 s/epoch,
693 ego-transitions/s**, 0 diverged worlds, 0 NaNs, 0 ring clamps, ring 107/107
on both sides, `opp_lag` 29.1 against the 26.5 their delta=0.5 rule predicts at
epoch 106. Their CPU reference reached epoch 198 in the same wall clock window,
having started 16 h earlier.

### What matches

**1. The training-reward curve, end to end.** Over the 107 epochs both climb
from about -1100 to just above zero, and they arrive at the same place:

| | epoch 5 | epoch 106 |
|---|---|---|
| ours `train_ret` | -1089 | **+27** |
| theirs `train_R_eps_avg_0` | -1047 | **+38** |

**2. The eval-reward curve, quantitatively.** Over the 54 epochs where both
sides have an eval, the correlation between their `eval_R_eps_avg_i` and our
`eval_ret_curriculum_i` is **+0.936 (agent 0) and +0.965 (agent 1)**. Both
curves execute the same three-phase shape at the same epochs: a flat plateau to
~epoch 32-40, a collapse to a trough at epochs 50-70, and a recovery from ~75.

| epoch | 0 | 20 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | 106 |
|---|---|---|---|---|---|---|---|---|---|---|
| **ours** eval 0 | 510 | 484 | 219 | 86 | 57 | 73 | 136 | 275 | 365 | 420 |
| **theirs** eval 0 | 428 | 403 | 378 | -6 | -13 | -8 | 135 | 271 | 263 | 235 |

**3. The eval episode length, which is the behavioural signature under it.**
Through the trough the two are within a few steps of each other -- ours
43 / 43 / 48 at epochs 60 / 64 / 70 against theirs 43 / 44 / 46 -- so both
policies are producing ants that run and fall after the same number of steps.

**4. The asymmetry.** In both runs **agent 0's win rate is 0.00 at every single
eval** and agent 1 is the only one that ever scores. That was not designed in:
the task is mirror-symmetric and the two learners are independent.

### What does not match: the win rate is an order of magnitude too low

| | first epoch with a nonzero eval win rate | mean win rate, epochs 78-106 (agent 0 / agent 1) |
|---|---|---|
| ours | **46** (1 win in 176 games) | **0.0019 / 0.0143** (6 and 20 wins in 2,017 games) |
| theirs | **89** | **0.0000 / 0.1667** |

Read carefully, because the two halves of that table say opposite things:

* our win rate **leaves zero earlier than theirs** (epoch 46 vs 89), and is
  consistently nonzero from epoch 78 on;
* but it saturates around **1-6%** where theirs reaches **17-60%** -- a **12x**
  gap on the mean over the comparable window, and the gap is widening at the
  end (epoch 106: ours 0.058, theirs 0.50).

The early-departure difference is a resolution artifact and should not be
claimed as an advantage: our eval runs 69-704 games per epoch against their
estimated 30-60, so we can resolve a 0.6% win rate and they cannot. The
**magnitude** gap is not an artifact -- 20 wins in 2,017 games against a rate
that would predict ~300 is far outside sampling noise.

So: **the port reproduces their learning curve and fails to reproduce their
win rate.** Our agents learn to run at the same rate they do, survive for the
same number of steps, and recover from the same collapse at the same epoch --
and then cross the goal line about a twelfth as often.

### Candidates for the win-rate gap, none of them tested here

Listed in the order I would test them, with what each predicts.

1. **The eval-reward level offset, which is the same shape as the gap.** Ours
   has run ~60-90 points above theirs since epoch 0 with identical episode
   lengths, i.e. our ants collect more survive-and-forward reward per step and
   fewer goal crossings, while theirs collect less and score more. Section 5
   measured the epoch-0 version of this: with both ants standing still for 500
   steps, theirs slides 1.08 m backwards and ours does not. If the same physics
   difference makes our ants slower over the ground, they simply reach x = +/-4
   less often within an episode. **This is my leading candidate and it is
   testable cheaply**: log mean |COM x| at episode end, or the fraction of
   episodes ending by goal versus by fall, on both sides.
2. **`blocks = 4`.** A learner facing only 4 distinct opponents per epoch,
   where theirs faces dozens, may be learning a narrower, more defensive
   policy. Stage 3b made raising `blocks` nearly free and nobody has measured
   what value is enough.
3. **The two declared optimizer mismatches** (section 3c): our critic's L2 is
   half theirs and our grad clip covers the value parameters too. Both would
   act on how fast the critic sharpens, which is what an agent needs to value
   the +/-1000 goal term at alpha 0.9.
4. **fp32 + Newton at 100 iterations.** Everything in candidate 1 could equally
   be solver behaviour rather than a modelling difference.

### What 2e establishes, stated narrowly

* The config mapping is exact where it can be and declared where it cannot
  (section 1), and one of our iterations is one of their epochs in every unit.
* The two reward-fidelity bugs found here (sections 3b, 3d) were real and the
  control-cost one was worth the whole exercise: it moved the sampled reward
  from -1.10 to -3.00 per step, which is their number, and it is the reason the
  training curves now agree.
* The two-learner loop is stable at their hyperparameters and their batch size
  for 107 epochs: 0 diverged worlds over 10.7M ego transitions, ring exact.
* At **144 s/epoch against their 6-7.5 min**, the port is **~3x** faster
  end-to-end than their 24-worker CPU run -- not the ~19x the raw env-step
  ratio suggests, because this configuration pays their PPO settings. A full
  1000-epoch run is ~43 h here against ~110 h for them.
* **The paper-number gate is NOT met.** The curve shape is reproduced (r =
  0.94-0.97 on eval reward, matching training curve, matching collapse and
  recovery, matching episode lengths, matching agent asymmetry); the win rate
  is 12x too low. Nothing here should be read as "the port reproduces their
  result".

## 6. Correction, 2026-08-12: the non-scoring endings are mostly TIMEOUTS, not falls

`NIGHT_2026-08-11.md:342` reasons "episodes end at 303 of 500 with a 1.3% goal
rate, so ~98.7% terminate early WITHOUT scoring -- **falls**". The premise is
right and the last word is not: "did not score" was silently equated with
"fell", when the env has three exits, and the third one was never counted.

`render_dev_rollout.py` counts them directly, driving the saved 2e pair
(`runs/competevo_port/m2e_validation/policies.pt`, end of the 107-epoch run)
with mean actions over 64 worlds until 381 games finish:

| ending | share | how it is detected |
|---|---|---|
| goal | 6.6% (25) | `info["winner"].any(-1)` |
| fell | 31.8% (121) | `info["fell"].any(-1)`, no winner |
| timeout | **61.7% (235)** | `info["truncated"]`, 500 steps elapsed |

Mean episode length 381.5 of 500. No episode both scored and fell, so the
precedence choice does not move any number.

**The dominant failure is not falling over, it is not travelling.** Nearly
two-thirds of games are two ants still upright at the buzzer, neither having
crossed a goal line 4 m away. The rendered clip (`dev_pair.mp4`, five episodes
of world 0) shows the same thing without a metric: the pair stays clustered
around the halfway line for most of every episode.

This does not refute the physics hypothesis -- it re-points it. A
contact/friction discrepancy that makes our ants *slower* explains a timeout
majority; one that makes them *less stable* would have predicted the fall
majority we assumed and do not observe. The epoch-0 observation (their ant
slides 1.08 m backwards over 500 idle steps, ours does not) is a difference in
how the feet grip, which is on the "slower" side of that split.

Two numbers here also update section 5's framing, and both cut the same way:

* Win rate at this checkpoint is **0.066 summed** (agent 0: 0.0026, agent 1:
  0.0630) against the 0.0143 quoted for epochs 78-106 -- it improved by the end
  of the run, so the gap to their 0.1667 is ~2.5x here, not 12x. **These are
  not the same measurement** (different epoch windows; ours is 381 mean-action
  games at one checkpoint, theirs is their per-epoch logged rate), so this
  narrows the headline claim without replacing it. The 12x stands for the
  window it was measured in.
* The agent asymmetry survives at the end of training: agent 1 wins 24x more
  often than agent 0. Their run has the same asymmetry, so this is reproduced
  behaviour, not our bug.

Still missing, and still the gate for task #25: the same three-way breakdown on
THEIR side. Their runner logs win rate only, so it needs a short instrumented
reference run. Until that exists, "ours time out more than theirs" is an
inference from our side alone.

## 7. The reference's endings, and what the win-rate gap actually is

Section 6 measured our side and said the comparison was still missing. It is no
longer missing. `/workspace/competevo/endings_eval.py` drives THEIR runner,
THEIR env and THEIR `epoch_0107.p` checkpoints through their `mean_action=True`
eval branch and classifies each episode the same three ways. 7 seeds x 48
episodes = 336 games. Nothing in `competevo/` was modified; the script subclasses
their runner to skip the `render_mode="human"` that `training=False` forces, and
the branch it substitutes is the one TRAINING uses.

| ending | theirs (336 games) | ours (381 games) |
|---|---|---|
| reached the goal | **42.6%** | **6.6%** |
| fell over | **32.1%** | **31.8%** |
| ran out of time | **25.3%** | **61.7%** |
| mean episode length | 303.9 / 500 | 381.5 / 500 |

**The fall rates agree to 0.3 percentage points.** Our ants fall over exactly as
often as theirs do. Every point of the gap sits between "goal" and "timeout":
theirs score 42.6% of games, ours time out instead.

That kills the stability reading of the physics hypothesis. Section 6 guessed
"a contact/friction discrepancy that makes our ants slower" over "one that makes
them less stable", and the measurement takes the first and rules out the second
about as cleanly as a measurement can. **Our ants are not more fragile. They are
slower.** They do not cover the 4 m to the goal line inside 500 steps.

Sample sizes make this unambiguous: 143/336 = 42.6% +/- 2.7% against 25/381 =
6.6% +/- 1.3%. Per-seed goal rates on their side run 29.2-54.2%, so the spread
never approaches ours.

Two things this does NOT settle, and neither should be glossed:

* **These are two different training runs**, not one policy evaluated twice —
  their reference run and our port's 2e run, each at their own epoch-107
  checkpoint. So the comparison answers "does our port reach their result", not
  "does our env behave like theirs given the same policy". A cross-evaluation
  (their weights in our env) would separate those, and it is the obvious next
  probe.
* **The agent asymmetry is far starker on their side.** Their win rate splits
  [0.0000, 0.4256] — agent 0 never wins a single one of 336 games. Ours splits
  [0.0026, 0.0630]. Both runs have agent 1 dominant, which is reproduced
  behaviour, but their winner is much more dominant than ours.

Their 0.4256 summed win rate here is also well above the 0.1667 their runner
logged over epochs 78-106. Same caveat as section 6: their logged rate is a
per-epoch training-eval number and this is 336 mean-action games at one
checkpoint. The two are not interchangeable and neither replaces the other.

### The next probe, now well-posed

"Slower" is measurable directly and cheaply: per-episode com_x displacement and
peak forward velocity, on both sides, at the same checkpoint. If ours travels
less far per unit of control effort, the candidates narrow to friction,
actuator gear, or the dense forward-reward scale — and the first two are
gateable against their model fields without training anything.

### A landmine in running their env in parallel

The run above launched 8 seeds and 7 finished. Seed 1 died during env
construction, not during evaluation:

```
ValueError: mjParseXML: empty file
'competevo/evo_envs/assets/world_body.dev_ant_body.dev_ant_body.xml'
```

Their env loader merges the world and agent XMLs into a **fixed path inside the
source tree** and reads it back, so concurrent processes truncate each other's
file. Nothing is wrong with the surviving seeds — a process either gets past
construction or dies there — but the failure is at startup and silent in
aggregate, so a script that globs the outputs will quietly report 7/8 as though
8 had been asked for. Count the files. If more parallelism is wanted, give each
process its own copy of `competevo/evo_envs/assets/`.

## 8. Cross-evaluation: the environment is largely exonerated, the training is not

Section 7 compared two different training runs and so could not separate "our
env is slower" from "our training produced a worse gait". `cross_eval.py`
separates them by holding the policy fixed and changing only the simulator: their
`epoch_0107` weights are remapped into our `DevActorCritic` and run in our
batched env.

The remap is one-to-one and every tensor matches in shape — itself a check on
the port, since a network that had drifted structurally could not be loaded at
all. `remap()` raises on any unmapped tensor so a silent drop cannot pass as a
successful load.

| | their policy, their env | their policy, **our env** | our policy, our env |
|---|---|---|---|
| reached the goal | **41.3%** | **34.8%** | **5.5%** |
| fell over | 34.0% | 34.0% | 39.3% |
| ran out of time | 24.7% | 31.2% | 55.2% |
| mean episode length | 308.5 | 332.6 | 364.9 |
| travel toward goal, mean | 3.15 m | 3.89 m | 2.69 m |
| travel, median | 3.39 m | 4.03 m | 2.45 m |
| win rate, summed | 0.413 | 0.348 | 0.055 |
| games | 288 (6 seeds) | 385 | 384 |

**Their policy keeps 84% of its goal rate in our simulator.** Moving it from
their env to ours costs 6.5 points. Moving from their policy to ours, in the
same env, costs 29.3 points — 4.5x more. Their policy scores 6.3x what ours does
in *our own* environment, and travels 45% further per episode.

So the physics hypothesis that survived section 7 is now bounded rather than
confirmed: there IS a residual environment difference worth 6.5 points, but it
is a minority of the gap. The dominant term is that **our training produced a
much worse gait.** Falls sit at 34.0% in all three columns, which retires
stability as an explanation for anything.

This is a better position than it sounds. The env was the expensive thing to be
wrong about; the learning setup is cheap to iterate on, and section 1 already
lists the places where the port knowingly deviates.

### The prime suspects, now that physics is demoted

All four are in the optimizer or the normalizer, not the simulator. Three were
documented in section 1 as known deviations; the fourth was found by this
cross-evaluation.

1. **`RunningNorm` clipped at 10, theirs clips at 5.** New. Their
   `lib/rl/core/running_norm.py:14` defaults to `clip=5.0` and not one of their
   call sites overrides it; ours defaulted to `10.0`. Every parity gate drives
   the ENV, and this lives inside the policy, so nothing could have caught it.
   Measured under their weights, **0.46% of control-observation components land
   beyond 5 sigma** — about one input in 200 is treated differently. Fixed in
   `ppo.py` as of this commit.
2. **`l2_reg` at half strength.** Their `sum(p^2) * 1e-3` added to the value
   loss versus our Adam `weight_decay=1e-3`.
3. **Grad clip covers the value parameters too.** Theirs clips
   `policy_net.parameters()` only.
4. **Eval protocol differs** (`eval_batch_size` 10,000 steps over 10 workers
   versus our 64 full episodes).

Suspects 2 and 3 both act on how fast the critic sharpens, which is exactly what
an agent needs in order to value the +/-1000 goal term while `alpha` is still
0.9 — i.e. exactly the mechanism that would produce a policy that walks but does
not commit to crossing the line.

### A measurement bug worth recording

The first version of the travel numbers was wrong in both directions and I
nearly reported it. Both envs re-pose the robot AFTER the design stage — ours in
`_apply_designs` (`qpos = qpos0`), theirs in `transit_execution` (which calls
`reset_state(True)`) — so a start position read at `reset()` is a pose the
rollout never occupied. It gave 1.22 m of travel alongside a 33.6% goal rate,
which is self-contradictory, and that contradiction is the only reason it was
caught. Both scripts now latch the start position on the first execution step.
The ending percentages never depended on it.

## 9. The reference SOLVES the task by epoch 125, and epoch 107 caught it mid-climb

Everything in sections 5-8 compares the two runs at **epoch 107**, because that
is where our 2e run stopped. Measuring their later checkpoints shows that was an
unlucky place to stop.

Their `epoch_0200` checkpoint, same protocol as section 7 (6 seeds x 48
mean-action games, their env):

| ending | theirs @ epoch 107 | theirs @ **epoch 200** |
|---|---|---|
| reached the goal | 41.3% | **96.9%** |
| fell over | 34.0% | 1.0% |
| ran out of time | 24.7% | 0.0% |
| mean episode length | 308.5 | 170.0 |
| win rate, summed | 0.413 | **0.969** |
| win rate split | [0.000, 0.413] | [0.399, 0.569] |

Their own logged per-epoch win rate tells the same story: 0.00 through epoch 88,
first nonzero at 89, 0.33 by epoch 100, 0.67 at 107, and **1.00 from epoch 125
onward**, holding there through epoch 346 where the run was stopped.

Three consequences, and the first two are corrections to this document.

**The agent asymmetry is a transient, not a reproduced property.** Sections 7 and
8 record "agent 0 never wins" as behaviour that appears in both runs. At epoch
107 that is true of theirs; by epoch 200 their split is [0.399, 0.569] and by
epoch 225 it is [0.57, 0.43]. Whichever agent is ahead swaps repeatedly. It is a
feature of the middle of training, and treating it as a reproduced property was
reading a snapshot as a fact.

**The reference's logged win rate is a small-sample estimate.** Every value it
ever prints is a fraction with a denominator between 3 and 8 (0.12, 0.14, 0.17,
0.20, 0.25, 0.29, 0.33, 0.38, 0.40, 0.43, 0.50, 0.57, 0.60, 0.62, 0.67, 0.71,
0.75, 0.80, 0.83, 0.86, 0.88, 1.00 — sixths, sevenths and eighths). So the
"~10 sigma" attached to the section-5 win-rate gap assumed a precision the
reference number does not have; the gap is real and large, but that particular
sigma should not be quoted. The numbers to use are the direct measurements here,
288 games each.

**The gap is bigger than epoch 107 suggested, not smaller.** At 107 the
comparison was 41.3% against our 5.5%. The reference's actual converged
behaviour is 96.9%. Our port is not 6.5x off a partially-trained reference — it
is far from a task the reference **solves**.

That raises the value of the corrected re-run rather than lowering it: it runs to
200 epochs, which is exactly where the reference is at 96.9%, so the comparison
is against converged behaviour instead of a mid-climb snapshot. `--iters 200`
was chosen before any of this was known and turns out to be the right axis.

## 10. Result: the three optimizer fixes took the win rate from 0.06 to 0.84

`runs/competevo_port/m2e_fixed` ran to completion, 200 epochs. The comparison
that isolates the fix from the extra epochs is **matched epoch 101**, where both
runs have data:

| epoch | ORIGINAL, summed win | FIXED, summed win |
|---|---|---|
| 1–81 | 0.00 | 0.00 |
| **101** | **0.000** | **0.395** |
| 121 | — | 0.685 |
| 141 | — | 0.872 |
| 199 | — | 0.730 |

And measured directly at 384 mean-action games, against the reference's own
converged behaviour:

| | goal | fell | timeout | win, summed | ep length |
|---|---|---|---|---|---|
| theirs @200, their env | **96.9%** | 1.0% | 0.0% | 0.969 | 170.0 |
| **ours @200, FIXED** | **83.9%** | 15.6% | 0.0% | **0.839** | 175.7 |
| ours @107, original | 5.5% | 39.3% | 55.2% | 0.055 | 364.9 |

**The paper-number gate is close to met.** Our port reaches 87% of the
reference's converged goal rate, on the same task, from the same config, with
episode lengths that agree to 3%. Section 5's headline — "the win rate is 12x too
low" — was a consequence of three optimizer infidelities, not of the physics.

### What is left, and it is a different question from before

Timeouts are **gone** (55.2% -> 0.0%): our ants now travel. Mean travel is 4.71 m
against their 4.02 m, and episodes end in 176 steps against their 170. The whole
"our ants are slower" finding of sections 7 and 8 was downstream of the optimizer.

What remains is **falls: 15.6% against their 1.0%**, and that gap of 14.6 points
almost exactly accounts for the 13-point goal-rate gap. So the stability question
that sections 7-8 retired is back — but as a well-posed 13-point residual rather
than as the explanation for everything, and now with a policy good enough that
the comparison means something.

### Measurement honesty

Two runs of "their policy in our env" gave 34.8% and 30.7% goal rate on 384-385
games. Binomial SEM is ~2.4 points, so run-to-run variation is real and the
estimate is ~32 +/- 3, not 34.8 exactly. Single-run figures in sections 7-9
should be read with that width. It does not move any conclusion here — 83.9% vs
5.5% is not a 3-point question — but the earlier text quoted four significant
figures it had not earned.

## 11. What kind of fall? (2026-08-24)

Section 10 closed with the residual: **15.6% of our episodes end in a fall
against the reference's 1.0%**, and that 14.6-point gap almost exactly accounts
for the 13-point goal-rate gap. That is a count. Four different causes produce
it and they want different fixes, so `fall_analysis.py` separates them.

Measured on `policies_ep0140.pt` of the re-run, 512 worlds x 2 agents = 1,024
agent-episodes, mean actions, 97-99 fallers depending on seed:

```
PYTHONPATH=. MUJOCO_GL=osmesa .venv/bin/python \
  -m rower_soccer.competevo_port.fall_analysis \
  --policies runs/competevo_port/m2e_fixed/policies_ep0140.pt --worlds 512
```

| question | answer |
|---|---|
| **when** | 0% before step 30, 12% by 100, **78% between 100 and 300**, median step 171 |
| **which bound** | **75.8% collapse** (z < 0.28), 24.2% launched (z > 1.2) |
| **which body** | largest \|SMD\| 0.401 over 20 genome dims vs a null max of 0.259 — **1.55x, suggestive only** |
| **contact** | opponent 3.24 m away at the fall vs a 3.54 m all-steps baseline; 11-12% within 1 m — **not collisions** |
| **speed** | agents that fall later were **slower** over steps 0-100 (+0.884 vs +0.980, 1.9 SE) |

**What this rules out.** Not the spawn and not the design stage — nothing falls
in the first 30 steps. Not collisions — falls happen at slightly *more* than the
average opponent distance. Not a speed/stability trade, which was the obvious
reading of "we travel 4.71 m and they travel 4.02 m": the fallers are the slower
ants, not the faster ones.

**What is left.** A gait that walks for ~170 steps and then collapses, and it
happens to the ants that were already moving worse at the start. That is a
control-stability question, and it is now a specific one.

**One quarter of the falls are LAUNCHES**, which is worth separating out: the
dev ant terminates on an upper bound too (`dev_ant.py:291`, z > 1.2) and the
fixed-morph ant has no ceiling. This does not by itself explain the gap against
their run — their reference is a dev ant with the same ceiling — but 24% of a
15.6% rate is 3.7 points of episodes ending because an ant went *up*, which is
a contact-impulse question rather than a balance one.

### A correction, because the first version of this said the opposite

The probe initially reported **100% launches, 0% collapses** — exactly
inverted. It read torso z *before* `env.step()` and classified a fall that
`terms()` detected *after* it, so an ant that collapsed during that frame still
had `z >= 0.28` on the reading and was booked as a launch. `terms()` judges
post-step z and `step()` then auto-resets the world, so with `auto_reset=True`
there is no moment at which the deciding height is readable at all.

Fixed by running with `auto_reset=False` and reading z after the step. The
probe now also counts any fall whose z lands *inside* the band as a probe bug
and says so in its own output rather than reporting a number.

### Not measured

Whether the reference shows the same profile. That needs their checkpoints,
which the pod destroyed; their run-to-goal reference is training again as of
2026-08-24 and reaches the comparable epoch in roughly 15 hours. Until then
every number here describes our port without a control.

## 12. The re-run meets the paper-number gate (2026-08-24)

`runs/competevo_port/m2e_fixed` was re-run from scratch on a new pod, same
config, same seed 42, 200 epochs, 93 minutes. Scored with `score_policies.py`,
384 worlds, mean actions, **three independent eval seeds**:

| eval seed | games | goal | fell | timeout | win, summed |
|---|---|---|---|---|---|
| 1234 | 861 | 97.1% | 2.3% | 0.0% | 0.971 |
| 7 | 875 | 96.8% | 2.3% | 0.0% | 0.968 |
| 99 | 849 | 96.9% | 2.6% | 0.0% | 0.969 |

Against the reference, measured in §9 at its epoch_0200 checkpoint:

| | goal | fell | timeout | length | win, summed |
|---|---|---|---|---|---|
| **theirs @200** | **96.9%** | 1.0% | 0.0% | 170.0 | **0.969** |
| **ours @200, re-run** | **96.9%** | 2.4% | 0.0% | 179.5 | **0.969** |
| ours @200, old pod | 83.9% | 15.6% | 0.0% | 175.7 | 0.839 |

**The paper-number gate is met.** Section 10 said it was "close to met" at 87%
of the reference's goal rate; this run is at 100% of it, with episode lengths
agreeing to 6%.

### The fall residual is closed, and section 11's leads died with it

The 14.6-point fall gap that §10 and §11 were chasing is **1.4 points** here
(2.4% against 1.0%). `fall_analysis.py` on the converged policy finds 11
fallers in 1,024 agent-episodes, and at that sample both hypotheses §11 raised
evaporate:

* the genome signal is now **0.95x the null maximum** — below the noise floor,
  where §11's 1.55x was already only "suggestive". It was noise.
* early speed differs by **0.3 SE** (was 1.9). The speed/stability trade is
  dead in both directions.
* still not collisions: 0 of 11 fallers had an opponent within 1 m.

What survives is only the shape: falls are mid-run (median step 178), and
63.6% are collapses against 36.4% launches.

### What this does NOT establish, and it matters

**The same code, the same seed and the same config produced 83.9% on the
previous pod and 96.9% here.** The only difference is the GPU, and through it
warp's kernel scheduling, so the trajectories were never going to be identical
— but a 13-point spread between two runs of one configuration is a large
spread, and it is the honest headline number of this section:

* **Established:** the port is *capable* of the reference's converged
  behaviour. 83.9% was not a ceiling imposed by the port, and nothing further
  needs fixing to reach 96.9%.
* **NOT established:** that the port *reliably* reaches 96.9%. That is one run.
  Quoting 96.9% as the port's number, without the 83.9% beside it, would be
  picking the better of two samples.
* Consequently the §10 and §11 fall investigations were chasing a property of
  one run, not of the port. That is worth remembering before the next residual
  gets a workstream.

Three or more seeds at 200 epochs (~90 min each here) would settle it and are
the obvious next measurement.

### The reference number is also single-source

96.9% / 1.0% comes from one measurement of their `epoch_0200` checkpoint (§9,
288 games), and those checkpoints no longer exist. Their run-to-goal reference
is training again as of 2026-08-24 and reaches epoch 200 in roughly 15 hours,
which will give a like-for-like their-side number measured by the same tool
rather than a remembered one.
