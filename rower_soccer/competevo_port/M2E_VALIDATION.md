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
