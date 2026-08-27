# D1 unit 1f — 2v2 self-play RL fine-tune (PIPELINE_V2 stage 6, no BC anchor)

*Written 2026-08-27. Builds on unit 1e (`soccer2v2_env.py`, `tests/test_soccer2v2.py`
12/12) and stage 3 (the four distilled drill priors in `runs_v2/_priors/`).*

Files:

| file | what it is |
|---|---|
| `rower_soccer/warp_port/train_soccer2v2_warp.py` | the trainer |
| `rower_soccer/warp_port/gate_soccer2v2_train.py` | its gate, 23/23 |
| `runs/soccer2v2_1f/warmstart_sheet.png`, `.mp4` | the policy before training |

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

512 worlds was chosen: it is the throughput knee and costs **~1.4 GB of our
own**, far under the ~12 GB budget. Torch's own peak allocation is 0.51 GB; the
rest is mujoco_warp's Data and the CUDA graph.

`--ent-ceil -0.3` (std ≤ 0.74) pins exploration at the level `shoot` converged
to rather than letting the entropy bonus inflate it: `DRILL_V4_NOTES`' follow_v5
collapse was exactly a converged policy drowning in its own exploration noise
once the fixed entropy bonus became the loudest term left. `--ent-anneal-steps`
removes that pressure on the same schedule as the shaping.

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
    anyone calls the play "watchable".

## 6. Handing over

* Adding the KL-to-BC anchor when stage 5 lands: it is one more term in
  `SelfPlayPPO.update`, structurally identical to the prior term already there
  (a BC policy `pi_BC(z | football_obs)`, a weight, an anneal). The baseline to
  compare it against is this run.
* `runs_v2/` is gitignored, so checkpoints and logs live only on the box (and
  in GCS if `--gcs-bucket` is passed, which the baseline does not).
