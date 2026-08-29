# D1 unit 1f — 2v2 self-play RL fine-tune (PIPELINE_V2 stage 6, no BC anchor)

*Written 2026-08-27. Builds on unit 1e (`soccer2v2_env.py`, `tests/test_soccer2v2.py`
12/12) and stage 3 (the four distilled drill priors in `runs_v2/_priors/`).*

Files:

| file | what it is |
|---|---|
| `rower_soccer/warp_port/train_soccer2v2_warp.py` | the trainer |
| `rower_soccer/warp_port/gate_soccer2v2_train.py` | its gate, 23/23 |
| `runs/soccer2v2_1f/warmstart_sheet.png`, `.mp4` | the policy before training |
| `rower_soccer/warp_port/eval_soccer2v2.py` | the post-hoc evaluation + video (6) |
| `runs_v2/soccer2v2_1f_base/final_eval.mp4`, `.json` | the finished policy, evaluated |

---

## 0. What this run is, and why it is not a stopgap

PIPELINE_V2 stage 6 is *"self-play fine-tune: KL-to-BC + drill-prior mixture +
shaping rewards"*. **This run has no KL-to-BC anchor**, because stage 5 (human
2v2 demos → BC in z-space) has not happened — the demos do not exist yet.

That is deliberate and the run is worth its GPU-hours as it stands:

1. **It is the paper's own arrangement, minus one stage.** Liu et al. 2022 used
   motion capture for the low-level controller and then *pure self-play RL* for
   team play. Their team-play stage had no behavioural anchor either. What runs
   here IS that stage: a frozen z-space motor controller (ours came from drill
   RL instead of mocap — PIPELINE_V2's recorded "stage 1 cut") with a high-level
   policy optimised by self-play.
2. **It is the control that makes the human BC data measurable.** When the demos
   arrive, *"did BC help?"* is only answerable against a no-BC baseline trained
   in the same env, from the same warm start, with the same budget. This is that
   baseline. Adding the anchor later is a flag and a KL term, not a rewrite.

Both statements are also in the trainer's module docstring, so they travel with
the code.

---

## 1. Design decisions, and the measurement behind each

### 1.1 The policy acts in z-space through the frozen decoder

`ActorCritic` already *is* the z-space contract: `dist(obs)` computes
`action_net(decoder(cat[proprio, z(obs)]))`, and `--freeze-decoder` (default ON)
holds `decoder` + `action_net` fixed. Nothing new was written for this; the
point is that the trainer uses that path and no other.

**Measured** (gate 2a): `ac.dist(obs).mean` is **bitwise equal** to
`gate_drill_priors.action_from_z(ac, obs, ac.z(obs))` — the same function the
drill-prior gate drives. (gate 2c) adding 3.0 to all 34 task columns with `z`
held fixed changes the action by **exactly 0.0**: the low-level controller's
input is proprio + z and nothing else, which is what makes it shareable with the
drills and with the future BC corpus.

A policy emitting raw torques would train and score and transfer nothing. That
is the failure this gate exists to make impossible.

### 1.2 Warm start from shoot, with the task encoder SPLICED

`soccer2v2`'s first 13 task dims are `shoot`'s task block verbatim
(`ball_ego6 + opp_goal_mid3 + post_l2 + post_r2`), so shoot warm-starts more than
the decoder. `ppo.load_pretrained` would **drop** `task_enc.0` (13→34) and
`critic.0` (78→99) on the width mismatch. `load_warm_start` splices them:
shoot's columns land on the matching football columns, the 21 new columns (own
goal, team-mate, two opponents) start at zero. **At init the policy is exactly
the shoot policy evaluated on football observations.**

The source's own widths are *derived* from its tensors (`T_src` from
`task_enc.0`, `P_src = critic.0.in_features − T_src`) and `P_src` is asserted
equal to this env's proprio width. A decoder loaded against a different proprio
layout is the silent-failure mode here, and it now raises instead.

Zero-init is **not** PIPELINE_V2's zero-padding anti-pattern. That warning is
about a weight whose *input* is always zero — gradient `δ·x = 0`, so it never
leaves its random init and fires noise the moment the input goes live. Here the
input is live from step 1. **Measured** (gate 6b): `task_enc.0` has grad norm
2.9e-01 after six iterations — the spliced-in columns are learning.

**Measured** (gate 1a/1b): 25 tensors bit-identical to `runs_v2/s5_c_all/best.pt`
(3 decoder layers + `action_net` + `proprio_enc` + `expert` + `z_proj` +
`value_net` + `log_std`), 0 mismatched, 0 absent, 0 unexpected;
`task_enc.0[:, :13]` and `critic.0`'s proprio+task prefix bit-identical, the 21
new columns exactly zero.

### 1.3 The shoot VALUE HEAD is thrown away, and the critic is warmed up first

**Measured**: the warm-started shoot value head predicts **V = 21.95 ± 7.99** on
football states whose actual returns are **≈ 0.4**. Shoot's reward scale (goal
bonus 5, strike term, dense aim) is nothing like dm_soccer's ±1. A miscalibrated
critic makes every early advantage a readout of its own bias, and those updates
would be spent damaging the one thing the warm start bought.

So, by default: the critic **trunk** is kept, the final linear layer of
`value_net` is zeroed (`--keep-value-head` restores the old behaviour), and the
first `--critic-warmup-iters 10` iterations optimise **only** the value loss —
the policy is held exactly still while the critic calibrates.

### 1.4 Drill-prior mixture — the paper's Eq. 5 with a deterministic z

Eq. 5 regularises the football policy toward a mixture of the four distilled
drill priors. Our `z` is *deterministic* (`LatentExtractor.z` is a projection;
the stochasticity lives in the action head, which is what keeps PPO's log-probs
exact), so the KL from a point mass to the mixture is, up to a constant
independent of the policy, the mixture's negative log-density at that point:

```
L_prior = -log sum_k alpha_k * N( z ; mu_k(o), sigma_k(o) )      (logsumexp, alpha uniform)
```

Each prior reads the football observation through its **own** column map
(`task_cols` from its own checkpoint), which is well-defined only because
soccer2v2's task block opens with shoot's 13 dims, whose first 6 are `ball_ego`
— exactly what `drill_prior.FOOTBALL_TASK_COLS` maps dribble and kick onto.
follow's prior is proprio-only.

**`--w-prior 0.001` was sized, not guessed.** On a real warm-started rollout:

| term | gradient norm on the trainable parameters |
|---|---|
| PPO policy-gradient loss | 3.10 |
| `0.5 × value loss` (before the value-head fix) | 181.9 |
| `-log p_mixture(z)` | 989.0 |

So `w_prior = 0.001` makes the regulariser ≈ **0.32×** the policy gradient — a
constraint, not a whisper, and not a second objective. `0.003` would make it
0.96×; `0.01` (the first guess) 3.2×, i.e. the run would have been distillation
wearing PPO's clothes.

`--w-prior 0` removes the term with an `if`, not a multiply-by-zero, and **keeps
the diagnostic**: the mixture responsibilities are logged either way, so the ON
and OFF runs are comparable on the same axis. That makes the ablation
`--w-prior 0` and nothing else.

### 1.5 Reward: the sparse team reward is what is optimised

The task is `SoccerReward`'s unshaped term — dm_soccer's `Task.get_reward`
verbatim: +1 to every player on the scoring team, −1 to the conceding team.
Shaping is behind flags, is multiplied by `env.shaping_scale`, and the trainer
anneals that linearly to zero.

Both weights were **measured on a warm-started rollout**, not guessed:

| term | first guess | measured per 45 s match | shipped default | measured per match |
|---|---|---|---|---|
| `--w-player-to-ball` | 0.002 | **3.54** (3.5× a goal) | **0.0005** | ≈ 0.9 |
| `--w-ball-to-goal` | 0.05 | ≈ 0 at kickoff; ≤ 0.05×26 m = 1.3 on a full-pitch drive | 0.05 | ≤ 1.3 |

i.e. shaping is *comparable to* one goal over a whole match, never dominant, and
decays to zero over `--shaping-anneal-steps`.

### 1.6 Truncation and the bootstrap

`D3_HANDOFF.md` ("This inverts the port's problem") records the class of bug: a
fixed-`T` batched sampler truncates every world at the rollout boundary, so GAE
must bootstrap `V(s_T)` at cuts an episode-complete sampler never makes. There
are **two** cuts here and `warp_port/ppo.py` gets the second one wrong:

* **rollout boundary** (`t = T−1`): bootstrap `V(self._obs)`. ppo.py does this.
* **match clock** (`done`, every 45 s): the env is reset *inside* `collect`, so
  `val_buf[t+1]` is `V(the kickoff of the NEXT match)`. ppo.py bootstraps that —
  it discounts an unrelated state into the last transition of every match.

`SelfPlayPPO` records `V(s_T)` into `boot_buf` **before** the reset, bootstraps
that, and **cuts** the GAE recursion at the boundary so no advantage leaks across
matches. The time limit is a truncation, not a failure, so it bootstraps rather
than zeroing — D3_HANDOFF's Transform2Act convention is about a genuine terminal;
a 45 s slice of a running match is not one.

**Measured** (gate 4): on a stub env whose observation *is* its value, the three
candidate conventions give three different answers, and the trainer produces the
right one — see §2.

### 1.7 gamma = 0.995, not 0.99

Control dt is 0.025 s. `gamma = 0.99` is a 100-step, **2.5 s** horizon — too
short to connect a shot to the goal it scores. `0.995` is 5 s. This is a
reasoned choice, **not** a measured one; it has not been ablated.

### 1.8 Self-play: one shared policy, all four slots

The env's kickoff is invariant under the 180° mirror that swaps the teams and
every observation is egocentric (1e's TEAM SYMMETRY section), so the symmetric
self-play match is fair by construction. Mean fitness is *identically zero* in
this setting (goal difference is zero-sum over four players), so it cannot rank
anything — which is why the logged metrics are goals-per-match and the
**distribution** of per-world goal totals (`p_0_goals`, `p_1_goal`,
`p_2plus_goals`), not a win rate. A win rate with no goals behind it is a
wipeout artifact.

A checkpoint opponent pool exists behind `--opponent-pool` (default **OFF**):
the away rows of a random share of worlds are driven by a frozen past snapshot
and are masked out of the PPO update. **It is implemented but not gated and not
run.** See §5.

---

## 2. What the gate proves

`PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -m rower_soccer.warp_port.gate_soccer2v2_train`
→ **GATE 23/23**. Every check carries a negative control; a check that cannot
fail is not a check.

| # | measured |
|---|---|
| 1a | 25 tensors bit-identical to `runs_v2/s5_c_all/best.pt`; 0 mismatched, 0 absent from source, 0 unexpected |
| 1b | `task_enc.0[:, :13]` and `critic.0`'s proprio+task prefix bit-identical to shoot; the 21 new task columns exactly zero |
| 1c | **NEG** perturbing `decoder.0.weight[0,0]` by 1e-7 is detected, and the restore reads clean — the equality test is not vacuous |
| 1d | **NEG** `--no-splice` re-inits exactly those two layers and `task_enc.0` then does *not* match shoot — the splice does work |
| 2a | `ac.dist(obs).mean` == `gate_drill_priors.action_from_z(ac, obs, ac.z(obs))` **bitwise** on every row |
| 2b | perturbing `z` by 0.01 moves the action by 5.0e-03 — the decoder is not ignoring its latent |
| 2c | **NEG** +3.0 on all 34 task columns with `z` fixed changes the action by exactly 0.0 — the decoder is blind to task obs |
| 3a | `obs.view(n, 4, -1)[w, k]` bit-identical to `env._player_obs(k)[w]` for every (world, slot) |
| 3b | all 4 slots' `teammate_ego[81:84]` match `to_ego3(own frame, mate root)` and **none** match a neighbouring slot's root |
| 3c | model actuators are 8-per-slot contiguous and prefixed `p0-`…`p3-`, and a per-slot constant action lands in the matching `ctrl` block |
| 3d | **NEG** swapping the two players within each team changes the observation batch, so 3a/3b would fail on a permuted env |
| 4a | advantages `[4.535, 2.3, 7.1]` == hand-computed, with `V(s_T)=7` read before the reset (γ 0.9, λ 0.5) |
| 4b | **NEG** ppo.py's convention would give `[2.51, −2.2, 7.1]`, a terminal cut `[1.7, −4, 7.1]` — three distinct answers, and the trainer gave the right one |
| 4c | the `t = T−1` transition uses `V(self._obs)`; dropping that bootstrap would give −1.0 instead of 7.1 |
| 4d | `adv[t=1]` equals its own delta exactly — the recursion is cut at the match boundary, not merely re-based |
| 5a | `-log p_mix(z)` is per-row, finite, mean 117.75, range [42.9, 172.7] |
| 5b | at `w_prior=0` the penalty is never computed; at 1e-3 it is finite — the ablation is an `if` |
| 5c | descending the term takes `-log p` 117.75 → 79.08 and `z+50` scores 3826.63 — correctly signed, it pulls z *toward* the mixture |
| 5d | mean responsibilities are not one-hot, so Eq. 5 has something to mix |
| 6a | after 6 PPO iterations all 8 decoder + `action_net` tensors are **bit-identical** and every `.grad` slot is `None` |
| 6b | nonzero grads on `z_proj` 2.0e-01, `task_enc.0` 2.9e-01, `proprio_enc.0` 2.2e-01, `critic.0` 4.2e-04, `value_net` 2.1e-01 |
| 6c | 6 iters: every parameter finite, 0 obs-diverged, 0 sim-diverged, 0 non-finite gradients |
| 6d | the warmup iterations add no policy-gradient or prior term to the objective |

---

## 3. The launch

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
cd /workspace/utmist-vc2-phase2
PYTHONPATH=. MUJOCO_GL=egl setsid nohup .venv/bin/python \
  -m rower_soccer.warp_port.train_soccer2v2_warp \
    --run-name soccer2v2_1f_base \
    --worlds 512 \
    --minutes 720 \
    --init-from runs_v2/s5_c_all/best.pt \
    --freeze-decoder \
    --w-prior 0.001 \
    --shaping-anneal-steps 500000000 \
    --ent-ceil -0.3 --ent-anneal-steps 500000000 \
    --log-every 20 --ckpt-secs 900 --video-secs 1800 \
  > runs_v2/soccer2v2_1f_base/train.log 2>&1 &
```

Logs: `runs_v2/soccer2v2_1f_base/train.log` (human) and
`runs_v2/soccer2v2_1f_base/log.json` (machine — one record per logged iteration,
rewritten atomically). Checkpoints every 15 min to `checkpoint.pt` (resumable,
model + optimizer + step count) and `latest.pt` (SB3-compatible export).
In-loop videos every 30 min to `videos/step_*.mp4`.

The ablation is one flag:

```bash
    --run-name soccer2v2_1f_noprior --w-prior 0     # same everything else
```

**Sizing, measured** on the RTX 4000 Ada shared with D3 and the user's
rendering:

| worlds | agent rows | steps/iter | fps | GPU total (incl. ~0.8 GB others) |
|---|---|---|---|---|
| 128 | 512 | 32,768 | 19,662 | 1.62 GB |
| 256 | 1,024 | 65,536 | 33,572 | 1.81 GB |
| 512 | 2,048 | 131,072 | 51,219 | 2.20 GB |

512 worlds was chosen: it is the throughput knee and costs **1.14 GB of our
own** (measured live, `nvidia-smi --query-compute-apps`), far under the ~12 GB
budget. Torch's own peak allocation is 0.51 GB; the rest is mujoco_warp's Data
and the CUDA graph.

**Memory is not the binding constraint -- SM time is.** At 512 worlds the run
sits at ~95% GPU utilisation, so it is taking most of the device's compute
from whatever D3 and the user's rendering want. If that becomes a problem the
knob is `--worlds`: 256 halves the SM demand and costs ~35% of the throughput
(33.6k vs 51.2k steps/s). That is a call for whoever owns the box, not one
this unit should make silently.

`--ent-ceil -0.3` (std ≤ 0.74) pins exploration at the level `shoot` converged
to rather than letting the entropy bonus inflate it: `DRILL_V4_NOTES`' follow_v5
collapse was exactly a converged policy drowning in its own exploration noise
once the fixed entropy bonus became the loudest term left. `--ent-anneal-steps`
removes that pressure on the same schedule as the shaping.

## 4. Smoke run and what the video actually shows

### 4.1 Smoke (finished; these are not mid-flight readings)

512 worlds, 6 minutes, `--shaping-anneal-steps 500000000`, everything else at
defaults. **147 iterations, 19,267,584 env steps, exit 0.**

| | |
|---|---|
| throughput | 54,500 env-steps/s steady (72k on the first, graph-warm iteration) |
| diverged (obs guard) | **0** |
| diverged (sim guard) | **0** |
| non-finite gradients | **0** |
| completed matches | 2,048 (four cohorts of 512) |
| reward | +2.5e-4 to +4.9e-4 per agent-step, i.e. ~0.5 per 45 s match, all of it shaping |
| critic | V 0.00 -> 0.08, tracking returns 0.01 -> 0.08 after the 10-iteration warmup |
| `prior_nll` | 119 -> 68, i.e. z is being pulled onto the drill manifold |
| responsibilities | shoot 0.85 +/- 0.03, dribble ~0.07, kick ~0.07, follow ~0.001 |

Last completed cohort (512 matches, aggregated before reporting):

| | |
|---|---|
| goals per match | **1.76** |
| home / away goals | 0.88 / 0.89 -- balanced, which is what the mirror symmetry predicts |
| **distribution** of per-world goal totals | 0 goals **21%**, exactly 1 **27%**, 2 or more **53%** |
| throw-ins per match | 2.4 |
| mean ball distance from the centre spot | 8.9 m |
| mean uprightness | 0.58 |

Reading the distribution rather than the headline matters here: a fifth of
matches end goalless, so "1.76 goals per match" is not a uniform trickle, it is
a bimodal mix of matches where somebody got hold of the ball and matches where
nobody did.

The 0.58 uprightness is the one number that looks wrong, and it is **not
explained** -- see §5.10.

### 4.2 The warm-started policy, looked at

`runs/soccer2v2_1f/warmstart_sheet.png` and `warmstart.mp4`. Rendered from the
**median-activity** world of 16 (ball path length: min 0.0 m, median 38.1 m,
max 84.3 m over 20 s), deliberately -- see the note below about the first
attempt.

What I see, frame by frame: a correct mirror-symmetric kickoff with the ball on
the centre spot; all four ants converging on the ball within ~4 s; a contested
scrum in the centre circle; the ball struck clear and travelling several metres
upfield; and, in the close-up, an ant **standing on its legs** (not splayed
flat) with the ball at leg height about two body-lengths ahead. The ball/ant
proportion is right, the pitch is the scaled 30 x 22.5 m one, both goals are
present with the right posts, and the teams are correctly coloured and
correctly placed (blue defends -x).

Honestly: this is **four copies of a shoot policy all chasing the same ball**.
There is no passing, no spacing, no defending -- which is exactly what a
warm start with no notion of a team-mate should look like, and is the thing
self-play has to change. Over the 16 worlds it scored 10 goals in 20 s of play
(5 home, 5 away).

**Method note, because it nearly produced a wrong conclusion.** The first
render was one world, one seed, 15 s. In that world nobody reached the ball,
`ball_dist` was 0.00 and the ants looked inert -- I was about to record "the
warm start does not move". Aggregating over 64 worlds instead showed the
opposite: nearest-ant-to-ball closes from 5.8 m to 3.1 m in 10 s, the ball
travels 6.8 m, and with the ball placed 2 m in front of a player it is struck
7.4 m. Gait speed is 1.0 m/s in the match against 2.25 m/s for the same
checkpoint in its own shoot drill -- slower, because the match state is off
shoot's training distribution, but far from inert. One world is an anecdote.

## 5. What I did NOT test

Stated plainly, because an untested thing that is not labelled untested becomes
a recorded mechanism someone else propagates.

1. **The opponent pool (`--opponent-pool`) is implemented and never run.** The
   masking path (away rows of selected worlds driven by a frozen snapshot,
   excluded from the update and from the advantage normalisation) has no gate
   check and no smoke run behind it. Treat it as a sketch until someone gates
   it. The baseline does not use it.
2. **`--no-freeze-decoder` was never run.** It exists as an ablation flag and
   would break the z-space contract; nothing measures what it does.
3. **`gamma = 0.995` is reasoned, not measured.** No sweep over {0.99, 0.995,
   0.998}. The reasoning is only "0.99's 2.5 s horizon is shorter than a shot".
4. **The prior weight was sized once, at init.** `||grad prior|| / ||grad pg|| =
   0.32` was measured on a warm-started rollout with a *miscalibrated* critic.
   Once the critic calibrates the policy gradient changes scale and that ratio
   moves. It is not tracked during training; `prior_nll` is logged but the
   ratio is not.
5. **No transfer/eval against anything external.** There is no scripted
   opponent, no CPU dm_soccer transfer eval, and no held-out scoring env. The
   only numbers are self-play, and self-play cannot rank a policy against
   itself — `fitness` is identically zero by construction.
6. **The alpha weights of the Eq. 5 mixture are uniform and fixed.** The paper's
   Eq. 6 makes them state-dependent. Not implemented.
7. **`--spawn uniform` and `--ball-jitter` are plumbed through but unexercised.**
   The run uses the mirror kickoff.
8. **Resume was never exercised.** `--resume` loads `checkpoint.pt` through
   `ppo.load_checkpoint`, whose optimizer state was saved from an Adam built
   over only the *trainable* parameters. That should be self-consistent here
   (the freeze happens before the optimizer is constructed, unlike the kick
   trainer's ordering hazard), but it has not been run.
9. **Long-run stability is unknown.** The gate's short run and the smoke cover
   minutes, not hours. `std` was drifting upward under the entropy bonus during
   the smoke, which is why the launch pins `--ent-ceil` and anneals `ent_coef`;
   whether that is enough is not established.
10. **The `upright` reading is not understood.** It sits near 0.6 rather than
    near 1.0, i.e. the ants spend a lot of the match tilted or down. Whether
    that is four ants knocking each other over, or the shoot gait degrading off
    its training distribution, has not been separated. Worth a look before
    anyone calls the play "watchable". **Partly answered after the run
    finished -- see 6.3.** It is not contact; it is falls that are never
    recovered from.

## 6. The finished run, evaluated from `final.pt`

The run completed its budget: **2,000,027,648 env steps, 15,259 iterations,
646.8 min wall, 0 obs-diverged, 0 sim-diverged, 0 non-finite gradients**, with
`final.pt` / `latest.pt` / `checkpoint.pt` written at exit.

The in-loop `videos/step_*.mp4` are a monitor, not an evaluation: one world
(world 0), 15 s, no aggregate behind them. `warp_port/eval_soccer2v2.py` is the
evaluation -- it plays complete matches over many worlds, aggregates FIRST, and
only then picks what to film, by rank inside that population. 4.2's method note
is the reason it is built that way round.

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -m rower_soccer.warp_port.eval_soccer2v2 \
    --ckpt runs_v2/soccer2v2_1f_base/final.pt \
    --out  runs_v2/soccer2v2_1f_base/final_eval.mp4 \
    --worlds 64 --matches 4 --also-stochastic \
    --json-out runs_v2/soccer2v2_1f_base/final_eval.json
```

64 worlds x 4 matches = **256 complete 45 s matches** per pass, ~170 s of
rollout, **0.36 GB of GPU** (`nvidia-smi --query-compute-apps`, measured live
next to the D3 run), 0 diverged. The env spec is read from the run's own
`config.json` rather than re-typed, so the eval env is the trained env.

### 6.1 The numbers

Two conventions, because they answer different questions. The trainer's video
path is **deterministic** (mean action) and that is what is filmed; training's
own `match` metrics were logged under the **sampled** policy.

| | deterministic | sampled | training log, last iter |
|---|---|---|---|
| goals per match | **1.48** | **1.81** | 1.83 |
| home / away | 0.80 / 0.68 | 0.84 / 0.97 | 0.90 / 0.93 |
| 0 goals / exactly 1 / 2+ | 28% / 28% / 45% | 23% / 22% / 54% | 21% / 25% / 54% |
| home win / draw / away win | 32% / 44% / 24% | 29% / 36% / 35% | -- |
| throw-ins per match | 2.47 | 2.50 | 2.46 |
| ball distance from centre spot | 8.46 m | 8.85 m | 8.91 m |
| ball path length per match | 84.7 m | 94.1 m | -- |
| nearest ant to ball, time-mean | 3.21 m | 3.19 m | -- |
| mean uprightness | 0.78 | 0.77 | 0.69 |

The sampled column reproduces the training log to within its own run-to-run
spread, which is the check that the eval is measuring the same task the trainer
was optimising. Three independent 256-match passes gave 1.57 / 1.56 / 1.48
(deterministic) and 1.90 / 1.79 / 1.81 (sampled): **the aggregates are stable
to about +/-0.06, but individual worlds are not reproducible run to run** --
mujoco_warp's GPU contact solve is not bitwise deterministic and a football
match is chaotic, so "which world was the median" changes between passes even
at a fixed seed. That is why nothing here is reported per world.

Matches never end early (`terminate_on_goal=False`, as in `match.py`), so
"endings" is the score distribution above: **28% of matches finish goalless**
and the mean is a mix of those and matches where someone got hold of the ball,
exactly as the smoke predicted. 18% of matches have a ball path under 30 m,
i.e. nearly a fifth of matches are effectively dead.

### 6.2 What the video actually shows

`runs_v2/soccer2v2_1f_base/final_eval.mp4` -- 45 s, 1624x1224, four panels, one
per QUARTILE of the 256 matches ranked by ball path length (ranks 12 / 37 / 62 /
87 %), each with a ball-tracking close-up inset so posture is visible; the
top-down alone cannot tell a standing ant from a fallen one. The panels finish
0-0, 1-0, 2-0 and 2-1, so six goals are scored inside the clip.

Honestly, frame by frame:

* the kickoff is the correct mirror formation with the ball on the centre spot;
* within ~4 s all four converge on the ball and contest it in a scrum. There is
  still **no passing, no spacing and no defending** -- self-play has made the
  four copies faster and more effective at reaching and striking the ball, and
  has not made them a team. Positionally this is the warm start with more
  urgency;
* the ball is genuinely driven around the pitch -- 84.7 m of ball travel per
  45 s match, where the warm start managed 6.8 m per 10 s (~31 m per match, on
  a different rollout, so treat that as an order-of-magnitude comparison and
  not a matched one). The four filmed matches finish 0-0, 1-0, 2-0 and 2-1 and
  each is filmed end to end, so six goals go in on screen; I did not sit and
  time each one, that is read off the panels' own final scores;
* creatures repeatedly run **into the goal frame and the net** and get stuck
  there;
* and in the bottom quartile the match simply **dies**: by ~14 s all four ants
  are flat, and the panel is frozen -- same positions, ball unmoved -- from
  there to the final whistle.

### 6.3 The `upright` question (5.10), partly answered

Two measurements over the same 256 matches, both new:

| | deterministic | sampled |
|---|---|---|
| uprightness, first 5 s of the match | **0.96** | 0.95 |
| uprightness, last 5 s | **0.70** | 0.68 |
| falls (up->down crossings) per match, all 4 players | 3.45 | 3.76 |
| recoveries (down->up) per match | 2.18 | 2.43 |
| **recoveries / falls** | **0.63** | 0.65 |
| players down at the final whistle | **32%** | 33% |
| uprightness while within 1.5 m of another creature | 0.768 | 0.773 |
| uprightness while further than 1.5 m from anyone | 0.783 | 0.766 |
| share of time within 1.5 m of another creature | 26% | 25% |

Read together: the ants start the match upright (0.96) and the population
degrades monotonically to 0.70 by the end, because **about a third of falls are
never recovered from** and a third of all players are lying down when the
whistle goes. Proximity to another creature makes **no difference** -- 0.768
crowded against 0.783 alone, a gap smaller than the difference between the two
action conventions -- so the "four ants knocking each other over" hypothesis is
not supported. What the number is measuring is a gait that falls on its own and
cannot stand back up; the dead bottom-quartile match is that failure taken to
its limit.

Caveats, plainly: the crowding split is a **correlation**, not a controlled
experiment (players are close precisely when they are all chasing the same
ball), and it does not establish *why* the gait falls -- "off shoot's training
distribution" remains the hypothesis, untested. The obvious next measurement is
the same checkpoint's fall rate in the shoot drill it came from; nobody has run
it. Nothing here changes 5.5: there is still no eval against any external
opponent.


## 7. Handing over

* Adding the KL-to-BC anchor when stage 5 lands: it is one more term in
  `SelfPlayPPO.update`, structurally identical to the prior term already there
  (a BC policy `pi_BC(z | football_obs)`, a weight, an anneal). The baseline to
  compare it against is this run.
* `runs_v2/` is gitignored, so checkpoints and logs live only on the box (and
  in GCS if `--gcs-bucket` is passed, which the baseline does not).

---

## 8. The pitch boundary: the ball bounces, the throw-in is gone (2026-08-29)

*Changed: `warp_port/scene.py`, `warp_port/soccer2v2_env.py`,
`warp_port/train_soccer2v2_warp.py`, `tests/test_soccer2v2.py` (12/12 -> 19/19).*

### 8.1 The rule that changed, and why

Until now the env implemented dm_soccer's **throw-in**: when the ball's centre
left the `field` box it was teleported to `xy * u`, `u ~ U[0.7, 0.9]`, with its
**velocity zeroed** (`_throw_in`, `DM_THROW_IN_SHRINK`). Measured over 256
matches of the 4.28B checkpoint, that fired **2.46 times per 45 s match**, and
on video it reads as the ball hitting an invisible wall and dying.

The DeepMind 2021 football paper specifies the opposite:

> "To emulate the football rules, the players can travel outside of the
> boundaries of the pitch (but cannot travel outside of the gradient-coloured
> physical hoardings), whereas the ball 'bounces off' of the pitch boundary.
> This simplification removes the need for a throw-in mechanism, and leaves the
> physics simulation to determine the range of strategies that players can
> execute (including deliberately bouncing the ball off the pitch boundary)."

**This is not a reinterpretation of dm_control — it is dm_control's other
branch.** `dm_control.locomotion.soccer.pitch.Pitch` takes `field_box=True`
(surfaced as `soccer.load(..., enable_field_box=True)`), and
`_fieldbox_pos_size`'s own docstring says:

> "Walls are placed around the field so that the ball cannot travel beyond
> `field` but walkers can walk outside of the `field` but not the surrounding
> pitch. Holes are left in the fieldbox at the goal positions to enable
> scoring."

It is also how dm_control **disables** the throw-in: `Pitch.register_ball` calls
`self._field.register_entities(ball)` only in the *non*-field-box branch, so with
a field box the out-of-play detector never sees the ball at all. So the change
here is a transcription of an upstream option, not an invention.

### 8.2 Where the bounce surface goes, and why not at the hoardings

There are now **two** boundaries, and the whole point is that they are at
**different radii**:

| surface | at (pitch_scale 0.3125) | collides with |
|---|---|---|
| field box (the *pitch boundary*) | \|x\| = 13.333, \|y\| = 9.583 | the **ball only** |
| `wall_nx/px/ny/py` (the *hoardings*) | \|x\| = 15.0, \|y\| = 11.25 | **everything** |

The 1.67 m strip between them is the ground players may run on and the ball may
not enter — which is exactly the paper's "players can travel outside of the
boundaries of the pitch (but cannot travel outside of the ... hoardings)". Put
the bounce at the hoardings instead and that strip vanishes, taking the rule
with it.

The field box is at `field_half`, i.e. dm_control's own `field` detector — the
line the throw-in used to fire on. Two independent reasons, beyond it being
upstream's choice:

1. **`field_half_x` IS the goal line.** The goal posts stand on |x| = 13.333.
   With the boundary on that plane, a ball crossing the goal line either goes
   through the mouth (a goal) or bounces off the rest of the line. That is the
   football rule, not an approximation of it. A boundary at 15.0 would let the
   ball roll 1.67 m *past* the goal line beside the post and come back, which no
   football does.
2. **It leaves the hoardings free to do their own job.** They already existed and
   already stop everything; they are now the players' limit and nothing else.

**One honest mismatch, recorded rather than papered over:** dm_control's pitch
*texture* has its white border at the edge of the ground plane, i.e. at
±15.0/±11.25, so the drawn touchline is ~1.67 m outside the line the ball
bounces off. That is an upstream texture property that predates this change (the
same is true in dm_control's own renders), and the physics was NOT moved to
match a texture. To stop the bounce reading as another invisible wall on video,
`_add_field_box` also draws a knee-high **site** strip along the real boundary —
sites carry no contype at all, so it cannot affect a single number.

### 8.3 Does it actually bounce? (measured, and it did not at first)

Restitution in MuJoCo is `solref = (timeconst, dampratio)`; `dampratio = 1.0` is
critically damped, i.e. no bounce. **The existing hoardings were measured first,
and they are not a bounce**: ball fired at 8 m/s returns at 0.71 m/s,

> **restitution 0.089** — the wall absorbs 99.2% of the kinetic energy, then the
> ball dies against it (final velocity 0.000).

So "just delete the throw-in and let the existing walls do it" would have
swapped a teleport for a ball that stops dead at the line. The field box needed
a restitution of its own, which required `geom_priority = 2`: the **ball** has
`priority = 1`, so *its* solref governs every contact it is in, and without
outranking it the boundary cannot be tuned without also changing how the ball
meets the ground and the creatures. mujoco_warp honours `geom_priority`
identically to MuJoCo (`collision_core._contact_params`).

The value was then swept, not guessed — ball fired at the boundary at
8/15/22/30/40/50 m/s on Warp:

| timeconst | dampratio range | restitution over 8..50 m/s | verdict |
|---|---|---|---|
| 0.005 | 0.45 .. 0.17 | 0.06 .. 76 | pumps almost everywhere |
| 0.01 | 0.45 .. 0.20 | 0.20 .. 0.71 | 0.17 **pumps** (e = 1.12) |
| 0.02 | 0.20 .. 0.10 | 0.46 .. 0.81 | 0.07 **pumps** (e = 4.8) |
| 0.03 | 0.20 .. 0.07 | 0.44 .. 0.96 | 0.05 **pumps** (e = 1.9) |

"Pump" means restitution **> 1**: the contact returns more energy than it
received, and the ball is flung clean past the hoardings (measured to x = 17.1
against a 13.33 boundary). That is the same energy-injection failure that NaN'd
`dribble_paper_v5/v6` through the ball's own solref. In a 24 h run it is fatal,
so the value is chosen for **margin**, not for the springiest bounce:

**`FIELD_BOX_SOLREF = (0.03, 0.15)`** — restitution **0.50 .. 0.64, essentially
flat from 8 to 50 m/s**, with the pump cliff measured at dampratio < 0.07 (a
2.1x margin) and a timeconst 12x the 0.0025 timestep rather than the 2x floor
that pumped.

Note the direction is the **opposite** of the ball's own solref, where *softer*
(0.02) was the unstable choice. It differs because the counter-body differs: a
22 kg creature pushed 20 cm into the ball separates violently; an immovable
static wall pushes back longer and more gently. The sweep above is on the
contact actually in question, not extrapolated from that one.

Measured through the gate's own probe (impact speed -> rebound speed, at the
wall rather than divided by launch speed):

| backend | +x | -x | +y | -y |
|---|---|---|---|---|
| CPU MuJoCo | 7.15 -> 3.09, e = 0.433 | same | 7.15 -> 2.89, e = 0.405 | same |

### 8.4 Players, and the goal that nearly broke

Players pass through the field box and are stopped by the hoardings, by
dm_control's **contact filter**, transcribed bit for bit: the boxes carry only
`_FIELD_BOX_CONTACT_BIT` (128), and the ball gains bit 7 on top of its normal
contype. Creature/ground/wall/goal geoms keep contype 1 / conaffinity 1, so
`1 & 128 == 0` both ways. The gate checks this on all **52 creature geoms**
rather than by driving an ant at a wall, because a policy that cannot reach the
strip would make a behavioural test vacuously pass.

**The thing that nearly broke, and did break once.** The goal mouth is a hole in
the boundary — `|y| < goal_half_width`, `z < goal_height` — built by
`_fieldbox_pos_size` from two corner boxes and a lintel above the crossbar. My
first transcription used `goal_geometry`'s **full** height where dm_control's
`goal_size[2]` is a **half**-height, which put the lintel's underside at z = 3.33
instead of z = 1.67 and left a goal-height-tall slot above the crossbar that a
lofted ball flew straight out through. It was found by probing the mouth at a
range of heights, not by reading the code again. A ball that escapes is worse
than out of play: it is **trapped** in the 1.67 m strip for the rest of the
match. For the same reason `FIELD_BOX_HALF_HEIGHT` is dm_control's absolute
20 m and is deliberately **not** scaled by `pitch_scale` — scaled it is 6.25 m,
and the documented 20-30 m/s ball ejections reach 20-45 m.

### 8.5 Stalling: the throw-in's other job, measured

The throw-in also stopped the ball dying in a corner. Same 4.28B policy, 256
matches (64 worlds x 4), deterministic actions, before vs after:

| metric | BEFORE (throw-in) | AFTER (bounce) |
|---|---|---|
| goals / match | 1.902 ± 0.100 | 1.547 ± 0.088 |
| throw-ins / match | 2.465 ± 0.117 | 0.000 ± 0.000 |
| ball escapes / match | 0.000 ± 0.000 | 0.000 ± 0.000 |
| ball path / match (m) | 96.6 ± 3.5 | 91.6 ± 3.3 |
| time within 1 m of the boundary | 0.097 ± 0.010 | 0.254 ± 0.016 |
| time in a corner (within 3 m of both lines) | 0.060 ± 0.010 | 0.128 ± 0.013 |
| time with ball speed < 0.5 m/s | 0.729 ± 0.009 | 0.692 ± 0.010 |
| longest unbroken stall (s) | 16.70 ± 0.88 | 16.89 ± 0.83 |

*256 matches each (64 worlds x 4), 45 s, deterministic actions, the SAME 4,276,224,000-step policy in both. ± is the standard error over matches.*

Reading it:

* **The throw-in is gone and nothing escaped.** 2.465 -> 0.000 throw-ins per
  match, 0 escapes in 256 matches.
* **The ball lives at the boundary now, as it must.** Time within 1 m of the
  line 0.097 -> **0.254** (2.6x) and time in a corner 0.060 -> **0.128** (2.1x).
  That is the mechanical consequence of deleting a rule whose whole job was to
  teleport the ball back toward the centre 2.5 times a match.
* **It does not, however, die there.** This was the real worry, and the answer
  is no: time with the ball essentially stationary went **down**, 0.729 ->
  0.692, and the longest unbroken stall is unchanged inside its error bar
  (16.70 +- 0.88 s vs 16.89 +- 0.83 s, difference 0.19 +- 1.20). The ball spends
  twice as long near the wall and is not any deader for it -- a bounce keeps it
  moving where a throw-in used to stop it dead (`_throw_in` **zeroed the
  velocity**). Caveat: the 0.69-0.73 stalled fraction is mostly this policy
  failing to reach the ball at all, not corners specifically, so this is
  evidence that corners did not get *worse*, not that stalling is solved.
* **The expected performance dip is real and measured:** goals/match
  **1.902 +- 0.100 -> 1.547 +- 0.088**, a drop of **0.356 +- 0.133** (~2.7
  standard errors, -19%). Ball path is down 5.0 +- 4.8 m, i.e. not
  distinguishable from no change.

The dip is what it should be. The observation layout, action space and reward
are byte-identical, so nothing the policy reads has moved; what changed is the
ball's dynamics on ~2.5 events per match, and the policy has never seen a
boundary that returns the ball at half speed instead of re-spotting it at rest.
Some of the drop is also mechanical rather than a skill loss: a throw-in pulled
the ball 10-30% back toward the centre, which is a free advance up the pitch
that the bounce does not give. **An early dip is not a failure, and this number
is the baseline the resumed run has to beat.** For comparison, the throw-in
run's own training curve sat at 1.93 goals/match over its last block of
iterations, which agrees with the 1.902 measured here.

### 8.6 The boundary is VISIBLE in the render

The clip's model comes from `probe_soccer2v2.Soccer2v2Renderer`, which calls
`build_soccer_scene` with the same arguments, so it draws the same world the
physics uses. Rendering the top-down camera with and without the field box and
differencing the frames: **16,805 pixels change, and 90% of them lie on the
boundary lines** (|x| in 13.3-14.3, |y| in 9.3-10.5). So the surface the ball
bounces off is drawn, and the bounce should not read as another invisible wall.

### 8.6a What the gate proves (19/19, was 12/12)

`tests/test_soccer2v2.py` gained five checks. `t_out_of_play_throw_in` is gone.

* **the ball BOUNCES** (cpu + warp) — fired at each of the four walls, the
  normal component reverses and the ball keeps a measured fraction of its speed,
  asserted inside `[0.25, 0.95]`. The band's lower edge is above the old
  hoardings' 0.089 and its upper edge excludes an energy pump.
* **players cross the boundary, the hoardings stop them** — the contact filter,
  on every creature geom, plus the ball must still meet ground and posts.
* **goals still register** — 6 shots that must score (centre, two angled, two
  that pass within a ball's width of a post, one lofted) and **one wide shot
  that must not**, so "it always scores" cannot pass either; then
  `detected_goal` and `score` on a live step.
* **no throw-ins, no escapes, the ball stays in** — `throw_ins` stays 0 and
  `ball_escapes` stays 0 over a rollout in which the ball is **re-launched at
  14 m/s in a random direction every 20 steps** (960 launches on the warp run).
  Random ant torques barely move the ball, so without the launches this would
  have proved the boundary holds against a ball that never reaches it.
* **NEGATIVE CONTROL** — the same env with the field box's contact bits cleared
  (i.e. the pre-change pitch) must FAIL the bounce check, and does: e = 0.037,
  ball reaches x = 14.857 past the 13.333 line.

`throw_ins` is deliberately **kept** as a field and still logged. If someone
reinstates the throw-in it goes from 0 to non-zero and says so; had the field
been deleted, the trainer's metric would have silently vanished instead.

### 8.7 Checkpoint archiving

Before this there was **no step-stamped checkpoint history at all**:
`checkpoint.pt`, `latest.pt` and `final.pt` are each overwritten every
`--ckpt-secs` (900 s), so a 4.28B-step run had exactly one archived copy in the
world (a manual `cp` at 2B) — no way to bisect a regression and one bad write
from losing the run.

* `--archive-steps` (default 1e9) additionally writes
  `runs_v2/<run>/archive/checkpoint_step_<N>.pt`, never overwritten. It carries
  the **optimiser state**, so an archive is resumable, not merely evaluable.
* The bucket is on the **step counter**, not the wall clock, so the cadence is
  reproducible across restarts, and it is seeded from the resumed step count so
  a resume does not immediately re-archive what it just loaded.
* An archive is also written at **clean exit**, which is the state that was most
  reliably lost before (`final.pt` is overwritten by the next run of the same
  name and has no optimiser state).

### 8.8 What I did NOT test

* **No trained policy has been trained against this boundary yet** at the time
  of writing. Everything above is the 4.28B throw-in policy evaluated on the new
  rule, plus the gate. The recovery claim in 8.5 is a prediction, not a result.
* **The `(0.03, 0.15)` solref is validated by a sweep and the gate, not by a long
  run.** The pump cliff was measured at dampratio < 0.07 with the ball fired
  head-on at a wall; it was NOT measured for a ball squeezed between a creature
  and the boundary, which is the geometry most likely to find an instability.
  `_sanitize` and the escape guard both cover that case if it happens, and
  `ball_escapes` is in `match_stats` so it would show.
* **No comparison against dm_control's own `enable_field_box` run.** The geometry
  and the contact filter are transcribed and unit-checked against dm_control's
  `Pitch` at this scale, but no side-by-side rollout was done.
* The `--video-rank ball_path` and first-video-at-iteration-0 changes are
  behavioural changes to the trainer that the gate does not cover.
