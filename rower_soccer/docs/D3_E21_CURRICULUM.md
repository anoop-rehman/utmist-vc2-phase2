# D3 M3 E2.1 — the curriculum ablation: was D2's competence the curriculum or the budget?

*2026-09-02. Extends [`D3_E2_RTG.md`](D3_E2_RTG.md), whose null result and whose
instrument this reuses unchanged. Every number below names the command that
produced it; anything not measured is in "Not tested" at the end.*

## The question

E2 found that at 5.0M steps neither the Transform2Act GNN nor a matched MLP
learns run-to-goal against the scripted opponent — **goal rate 0.00 in all
arms** — and, worse, that the return ranking *is* the fall-rate ranking:
`r(fall rate, return) = +0.989`, `r(forward progress, return) = +0.019`. A fall
ends the episode before the opponent's certain goal at step 491, dodges the
−1000, and banks +758 to +984.

D2 solved a locomotion task on the same body and the same CompetEvo reward.
Two candidate explanations, and E2 could not separate them:

1. **budget** — D2 trained far longer than E2's 5.0M;
2. **the curriculum** — D2's trainer does not optimise the env's reward at all.
   `CoEvoPPO.collect` mixes it, `r = alpha*dense + (1−alpha)*parse`, with alpha
   annealing 1 → 0. While alpha is near 1 the ±1000 is weighted near zero, so
   **the fall-dodge pays almost nothing early on**, and E2's flat reward makes
   falling attractive from step 0.

E2.1 runs the MLP arm — the stronger of E2's two baselines — at **20.0M steps**
in **three** conditions that differ in exactly one argument. The third
condition exists because of §0b: D2's curriculum never annealed away, so a
faithful port of the *nominal* schedule tests something D2 never experienced.

---

## 0. What reading D2's own trainer changed about the question

*This section is forensics on the premise, done before the runs, because the
premise as stated to me was not quite what the code does.*

### 0a. The curriculum is real and is exactly as described — in the code

`rower_soccer/competevo_port/ppo.py:211-234`, ported from
`competevo/runner/multi_agent_runner.py:150-167`:

```python
def alpha(self):
    if not self.curriculum_steps:
        return None
    return max(1.0 - self.total_steps / (self.A * self.curriculum_steps), 0.0)
...
    if alpha is not None:
        rew = alpha * info["dense"] + (1.0 - alpha) * info["parse"]
```

with `dense = forward − 0.5·Σa² − contact + 1.0` and `parse = ±1000 or 0`
(`run_to_goal_env.py:213-228`). Upstream's own config
(`/workspace/competevo/config/run-to-goal-ants-v0.yaml`) is
`use_exploration_curriculum: True`, `termination_epoch: 200`,
`max_epoch_num: 1000`, `min_batch_size: 50000` — **the anneal occupies the
first 20% of their run**: 10M agent-steps of 50M, then 40M at pure sparse.

### 0b. But D2's run never got past alpha = 0.846

The 98.3% figure comes from `runs/competevo_port/idle_ant_s42`
(`D2_MORPHOLOGY_COMPETENCE.md`, "Update, same day"), whose stored `args` are in
`log.json`. Two facts read out of them:

* `train_team_selfplay.py` has **no `--curriculum-steps` flag at all** and never
  passes `curriculum_steps` to `CoEvoPPO`, so the run took
  `dev_ppo.DEV_CURRICULUM_STEPS = 1000 × 50,000 = 50M` — the **dev/evo**
  config's number, not run-to-goal's 10M.
* Each learner accumulates `T × n_ego × L = 100 × 128 × 2 = 25,600` steps per
  iteration (`selfplay.py:550`), so after 600 iterations
  `lr.total_steps = 15,360,000` and, with `lr.A = L = 2`,

  ```
  alpha_end = max(1 − 15.36e6 / (2 × 50e6), 0) = 0.846
  ```

  (cross-checked against the log's own `steps` column, 51,200 per iteration
  summed over the two learners, 30.72M total.)

**So D2's alpha went 1.000 → 0.846 and never lower.** Its trainer optimised
`0.85·dense + 0.15·parse` at the very end and essentially pure dense throughout.
And because D2 trained with `--idle-opponent`, `parse` was **0 on every step
except its own goals** — the −1000 never fired at all.

**That does not weaken the hypothesis; it sharpens it.** D2 did not merely
*start* on the dense reward, it never meaningfully left it. If the curriculum is
the mechanism, then what mattered is the dense-dominant regime, and a
faithful port that anneals all the way to 0 is testing something D2 never
experienced in its second half.

### 0c. D2's budget is 3.1x E2's, not 15x

E2 §1 quotes "roughly 77M environment steps (600 × 256 × ~500)". The trained
budget is smaller: 600 iterations × 100 rollout × 256 worlds = **15.36M world
steps**; of the 61.44M agent-transitions simulated, half are opponent lanes
thrown away, leaving **15.36M agent-steps per learner** (30.72M summed over the
two). Against E2's 5.0M that is **3.07x, not 15x**.

**E2.1's 20.0M per arm is therefore 1.30x D2's per-learner budget, not a
quarter of it.** The budget arm of the question is answered more strongly than
the 20M figure suggested when it was chosen.

*Correction to `D3_E2_RTG.md` §1 and §9 carried by this experiment: "roughly
77M" should read "15.36M agent-steps per learner (30.72M over both)".*

---

## 1. What was built, and what gates it

Two edits, and nothing else.

| file | change |
|---|---|
| `design_opt/envs/run_to_goal.py` | `info` now carries `dense` and `parse`. **What the env returns as reward is unchanged** — `reward = dense + parse` is character-for-character the expression that was there. |
| `rower_soccer/t2a_port/train_e11_mlp.py` | `--curriculum-steps N`. Default **0 = off**, so E1.1 and E2 are byte-for-byte the runs they were. |

```python
def alpha(self, epoch):
    cs = self.args.curriculum_steps
    if not cs:
        return None
    return max((cs - epoch * self.batch) / cs, 0.0)
```

Three implementation decisions, each stated because each could have been made
differently:

1. **`(cs − done)/cs`, not `1 − done/cs`.** Algebraically identical, not
   identical in float64: the second form rounds twice and lands 1.11e-16 off
   CompetEvo's `(termination_epoch − epoch)/termination_epoch` at most epochs.
   The gate caught it at exact equality and **the code was changed, not the
   gate**. The chosen form is bit-identical to theirs over 400 epochs.
2. **The schedule is in agent-steps, not iterations** — `competevo_port/ppo.py`'s
   own reasoning, so a change of batch size cannot silently rescale it. This
   arm's `min_batch_size` is 50,000, which *is* their epoch, so the two forms
   agree epoch for epoch.
3. **The curriculum touches the PPO buffer and nothing else.** The episode
   return this trainer logs, and every number `e2_eval.evaluate` produces, stay
   the **raw env return** in both conditions, so a curriculum arm is measured on
   exactly the instrument a flat arm is.

**Not "improved".** At alpha = 0 CompetEvo's curriculum reward is the sparse
term **alone** — no forward term, no survive bonus, no control cost. That is
their rule, and it is ported rather than fixed, because a fixed version would
not answer whether *their* curriculum is what D2 had.

### The gate

```
cd /workspace/Transform2Act && source env-gpu.sh
.venv-gpu/bin/python /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/gate_e21.py
```

`runs/d3_e21_curriculum/logs/gate_e21.log` — **28 checks, 0 failed**, five
phases, each with at least one negative control.

| phase | what it establishes | headline |
|---|---|---|
| 1 env terms | the split is exact and the reward is unchanged | over 2,000 random-action steps `dense + parse == reward` to **0.000e+00**; `dense == forward − ctrl_cost − contact + survive` to **0.000e+00**; `parse ∈ {0, ±1000}`; 4 sparse events seen. NEG: on a sparse step `dense` alone is 1000.0 off the reward |
| 2 alpha | it is CompetEvo's schedule, and both E2.1 settings are what they claim | `alpha(e) == max((200−e)/200, 0)` for e = 0..399 at max error **0.000e+00**; 1.0 at the start, exactly 0 at `curriculum_steps`, pinned after; `cur` gives 1.0 / 0.5 / 0.0 at epochs 0 / 40 / 80; **`d2rep` gives 1.000000 → 0.846400, matching D2's own endpoint to 3.9e-10, with the sparse weight capped at 15.36% so the fall-dodge is worth at most +153.6**. NEG: `curriculum_steps = 0` returns `None`, not 1.0 |
| 3 the buffer | only the buffer changes | one seed → **bit-identical trajectory and bit-identical episode returns** at alpha ∈ {off, 1, 0.5, 0}; `buffer(a=1) + buffer(a=0) == buffer(flat)` to **0.000e+00**; a=0.5 is the exact half-and-half. **THE MECHANISM: at alpha=1 not one ±1000 reaches the buffer**, where the flat buffer has 3. NEG: at alpha=0 the buffer is the sparse term alone |
| 4 the fall-dodge | the premium that is removed, measured | `flat objective − alpha=1 objective == the sparse term`, exactly, per episode, and it is a whole number of ±1000s. On the **idle** policy: stopping early is worth **+998.7 to +1010.9** per episode under the flat reward and **0.00 to 10.91** under alpha=1 — the two distributions do not overlap, a factor of **422** on the means |
| 5 the flat arm | the control is E2's arm | default `alpha_now` is `None`; `ep_rets` are the raw env return in both conditions. NEG: past the schedule's end the buffer is *not* the env reward |

**`gate_e2.py` was re-run in full after the env edit: 41 passed, 0 failed**
(`logs/gate_e2_regression.log`), so E2's scene, opponent, frozen body, reward,
termination, observation and E1.1 regression are all untouched by this work.

### Two things the gate found that were not in the plan

* **The dense reward does not reward survival at initialisation.** `ctrl_cost`
  is `0.5·Σa²` and the MLP initialises at `log_std = 0`, so a fresh policy pays
  **~4.0 per step** against a survive bonus of 1.0; its flat reward is
  **−4.97/step**. The dense reward's first gradient is "quieten down", and it
  rewards survival only once the actions are small. The first draft of gate
  phase 4 asserted the opposite and failed; the *claim* was wrong, not the code.
* **Even at alpha = 1 the fall-dodge is not exactly zero.** Contact shoves from
  the opponent make the dense return dip locally, so stopping early is worth up
  to **+10.9** in one of five idle episodes. Against **+1010.9** under the flat
  reward. The gate is therefore a non-overlap test rather than a "== 0" test —
  stated because "the curriculum removes the incentive entirely" would be a
  false sentence.

---

## 2. Three conditions, and why the third one had to be added

The ablation was launched two-way and became three-way once §0b was read. The
reason, stated plainly because it is a design error caught mid-flight rather
than a plan:

**`curriculum_steps = 4M` on a 20M run means alpha anneals 1.0 → 0.0 over the
first 80 epochs and then sits at 0.0 — the SPARSE TERM ALONE — for the
remaining 16M steps.** But D2 sat at alpha ≈ 0.85 for its entire run, and with
an idle opponent its −1000 never fired at all. So the anneal arm spends 80% of
its training in exactly the regime under suspicion. If the ±1000 against a
*certain* opposing goal at step 491 is the real blocker, that arm fails for the
same reason the control does, and the ablation cannot separate "the curriculum
helps" from "the sparse regime is unlearnable here". A third arm that holds
alpha high throughout is what localises the cause.

### 2a. `cur` — CompetEvo's nominal schedule, scaled: 4,000,000

Three candidate anchors, all read out of source rather than assumed:

| anchor | value | as a fraction of a 20M run |
|---|---|---|
| CompetEvo's own run-to-goal config | 10M (200 epochs × 50k) of a **50M** run | 20% of theirs |
| D2's realised setting | 50M over a 15.36M run — alpha ends at **0.846** | the anneal never completes |
| **chosen for `cur`** | **4M (80 epochs × 50k) of a 20M run** | **20%, CompetEvo's own ratio** |

Copying the absolute 10M would put the crossover at the halfway mark of a
20M-step run — a schedule twice as slow, in run-relative terms, as CompetEvo
ever used. Copying D2's 50M would end at alpha = 0.6 and make the arm a
"mostly dense" test rather than a curriculum test. Preserving CompetEvo's own
**20% ratio** keeps the thing being ported the same thing.

**Alpha trajectory**: `1.000` at epoch 0 → `0.000` at epoch 80, then **0.000
for epochs 80-399**. 80% of this arm is *not* dense-shaped.

### 2b. `d2rep` — D2's REALISED condition: 130,208,333

D2's alpha was not a designed schedule; it was the arithmetic consequence of an
unset default (§0b). What it produced was a **linear ramp cut short at 15.36%
of itself**: `alpha = 1 − total_steps/(A·cs)` with `total_steps = 15.36M` and
`A·cs = 100M`, so `1.000 → 0.8464`, monotone, never plateauing.

`curriculum_steps = 130,208,333` is the value that makes a 400 × 50,000 = 20M
run complete **that same 0.1536 fraction**:

```
20,000,000 / 130,208,333 = 0.15360000    ->  alpha(400) = 0.84640000
D2's own                                      alpha_end  = 0.84640000   (differ by 3.9e-10)
```

**Why this mechanism and not an alpha floor.** An explicit `--alpha-floor` was
the other option and was rejected for two reasons. First, it is **new code**,
and E2.1's whole gate rests on the claim that the only thing that changes is
the contents of the PPO buffer; reusing the already-gated `--curriculum-steps`
argument adds nothing to audit. Second, a floor produces a **ramp-then-plateau**,
which is a shape D2 never had — D2's alpha decayed linearly for its whole run.
Matching the trajectory rather than only the endpoint is what makes this a
replication.

**Alpha trajectory**: `1.0000` at epoch 0, `0.9693` at 80, `0.9232` at 200,
`0.8464` at 400 — monotone, linear, never below 0.8464. The sparse term's
weight therefore runs `0.000 → 0.1536`, so **the fall-dodge is worth at most
+153.6 in this arm against +1000 in the control** (gated).

**What this arm is NOT.** It is not D2. D2 trained against an **idle** opponent
that never scores, so its `parse` was 0 on every step except its own goals;
here the opponent scores with certainty at step 491, so `parse` is −1000 on the
last step of almost every episode and enters the objective at up to 15.4%
weight. That difference is the point: the arm asks whether D2's *reward regime*
transfers to a scripted, non-idle opponent, not whether D2's run reproduces.

## 3. The three conditions

```
runs/d3_e21_curriculum/launch.sh cur_s1   | cur_s2      # anneal to 0
runs/d3_e21_curriculum/launch.sh flat_s1  | flat_s2     # the control
runs/d3_e21_curriculum/launch.sh d2rep_s1 | d2rep_s2    # alpha held high
```

They differ in **one argument**. Everything else — cfg, seed, batch (50,000),
minibatch (2,048), 10 optim epochs, lr 3e-4/3e-4, hdims 64,64, log_std 0,
10 sampler threads, eval every 5 epochs on 10 episodes, video every 40 — is
identical between conditions **and identical to E2's `mlp_s{1,2}` arm**.

| arm | argument | alpha over epochs 0 → 399 | what it asks |
|---|---|---|---|
| **flat** (control) | `--curriculum-steps 0` | — (raw env reward) | does 4x the budget alone fix E2? |
| **cur** | `--curriculum-steps 4000000` | 1.000 → 0.000 by epoch 80, then 0 | does early dense shaping alone suffice? |
| **d2rep** | `--curriculum-steps 130208333` | 1.000 → 0.846, monotone | does D2's realised regime transfer to a *scripted, non-idle* opponent? |

The control is a **genuine re-run**, not a comparison against E2's stored
numbers. (E2's matched MLP is the same code at 100 epochs instead of 400, so
the flat arm is also a 4x-budget replication of E2 in its own right.)

**2 seeds per condition**, seeds 1 and 2 — the same two E2 used — so all six
runs pair up seed-for-seed. **The third condition was added ~15 minutes after
the first four launched and runs concurrently**; the four were not restarted.
Measured headroom at the moment of that decision: load 11.5 of 48 cores, 78%
CPU idle with four arms live, so six arms at 10 sampler threads fit without
starving each other or the D1 tenant. A third *seed* was not run; a third
*condition* was judged worth more than a third seed, and n = 2 stays the
central statistical limitation (see "Not tested").

**Budget**: `--max-epoch 400` × `min_batch_size` 50,000 = **20.0M env steps per
arm**, 4x E2's and 1.30x D2's per-learner 15.36M (§0c).

**6 arms x 20.0M = 120M env steps in total.**

**Two book-keeping facts, recorded rather than hidden.**

1. **The headroom estimate that justified running six arms concurrently was
   optimistic.** It was taken as "load 11.5 of 48, 78% CPU idle" with four arms
   live — but that snapshot caught several arms in their single-threaded update
   phase. With six arms the sampler contention is real: `cur_s1`'s `T_sample`
   went 31 s → 41 s → 83 s as the two `d2rep` arms came online, and the
   per-epoch cost settled near **75-85 s** against the ~45 s four arms were
   running at. Six arms therefore took **~9 h** rather than the ~5.5 h two
   conditions would have. Running `d2rep` afterwards instead would have been
   ~11 h in total, so concurrency was still the faster route to all six — but
   the estimate that made the call was not a good one and is recorded as such.
2. **`inline_video`'s output directory is hardcoded to
   `runs/d3_e2_rtg/renders/`**, so E2.1's clips land in E2's render directory.
   Filenames carry the tag (`rtg_mlp_s1_cur_e0039_bmw.mp4` against E2's
   `rtg_mlp_s1_e0090_bmw.mp4`), so **nothing of E2's was overwritten** —
   checked. The path was left alone rather than changed, because editing a file
   E2's reproducibility depends on, for no gain, is not worth the risk.
3. **Disk, and whose it is.** `/workspace` went from 6.1 GB free at launch to
   13 GB during the run, and **none of that was E2.1**. A concurrent
   housekeeping pass (not this experiment) removed 596 stale local `wandb/run-*`
   directories (1.26 GB, each verified present in the cloud first), deleted the
   disposable `rtg_gnn_smoke` (450 MB), and archived 36 intermediate GNN
   checkpoints from `rtg_gnn_s{1,2}` and `ant_e11_gnn_s{1,2}` to
   `gs://vc2-2026-checkpoints/_t2a_archive/` (5.40 GB, byte-verified before
   deletion), **keeping `epoch_0100.p` and `best.p` for every run** — checked
   here directly, so E2's post-hoc at epoch 100 remains reproducible.
   **E2.1's own footprint is ~135 MB**: six results directories of ~11 MB each
   at 41 checkpoints x 419 KB, plus ~15 MB of clips. Because two agents were
   writing and pruning the same filesystem, **no disk delta in this document is
   evidence about E2.1's own usage** — the footprint above was measured on
   E2.1's own paths, not inferred from `df`.

   *One gotcha worth carrying forward, found by that pass and recorded here
   because it nearly cost the live D1 run: wandb writes into files inside a
   `run-*` directory without updating the DIRECTORY's mtime, so a
   `find -maxdepth 1 -mmin` filter marks a live run's directory as stale.*

---

## 5. THE RESULT: the curriculum, in D2's realised form, solves the task — in 4.0M steps, less than E2 spent

```
runs/d3_e21_curriculum/collect.sh          # e2_posthoc.py per arm, then e21_analyse.py
```

**One instrument, 20 episodes per arm, identical episode seeds, the same frozen
13-body / 8-motor ant, the same scripted opponent, the same raw env reward for
every measurement.** Body freezing re-verified per arm under each arm's OWN
trained policy: **134 mjModel arrays identical on all six trained arms.**

### 5a. The headline table — mean-action, 20.0M steps

| arm | mean-action R | sd | **goal** | lost | fell | ep len | **furthest forward** | **of the 5.0 m** | net dx | speed | action std | stochastic R |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **d2rep** s1 | +1479.8 | 405.4 | **0.95** | 0.05 | **0.00** | 274.6 | **5.00 m** | **100.0%** | +5.00 | +1.371 | 0.088 | +1414.5 |
| **d2rep** s2 | +1599.2 | 114.6 | **1.00** | 0.00 | **0.00** | 293.3 | **5.00 m** | **100.1%** | +5.00 | +1.334 | 0.086 | +1565.8 |
| **flat** s2 | +719.1 | 523.4 | 0.35 | 0.00 | 0.65 | 158.2 | 4.35 m | 86.9% | +4.18 | +1.969 | 0.173 | +761.4 |
| **flat** s1 | +463.9 | 436.4 | 0.15 | 0.00 | 0.85 | 170.2 | 3.44 m | 68.6% | +3.13 | +1.344 | 0.172 | +429.5 |
| **cur** s2 | +49.4 | 480.4 | 0.15 | 0.00 | 0.85 | 194.1 | 2.05 m | 41.0% | +1.40 | +0.538 | 0.321 | −91.9 |
| **cur** s1 | −128.4 | 570.3 | 0.05 | 0.15 | 0.80 | 252.7 | 2.33 m | 46.5% | +1.44 | +0.598 | 0.328 | −253.5 |
| *idle, zero torque* | −523.7 | 307.0 | 0.00 | 0.85 | 0.15 | 465.2 | 0.08 m | 1.5% | −2.08 | −0.306 | 0 | −523.7 |

Condition means (mean-action goal rate): **d2rep 0.975, flat 0.25, cur 0.10,
idle 0.00.** The ordering is identical on both protocols and on forward
progress, so nothing here depends on the mean-action/stochastic choice that
nearly inverted E1.1.

**`d2rep` solves the task.** 39 of 40 evaluation episodes reach the goal line,
**not one episode of 40 ends in a fall**, mean furthest-forward is the full
5.00 m, and `net/path` is 0.988-0.990 — it runs essentially straight at
1.33-1.37 m/s against a requirement of 0.68. For scale, D2's own ant managed
1.114 m/s in its front slot against an **idle** opponent
(`D2_MORPHOLOGY_COMPETENCE.md`); this is faster, against an opponent that
scores.

### 5b. The budget was never the blocker — d2rep had already solved it at 4.0M steps

Every arm re-scored on the **same** mean-action instrument at epoch 79
(4.0M steps), which is also the exact epoch `cur`'s alpha reaches 0:

| arm | | R | goal | fell | furthest forward |
|---|---|---|---|---|---|
| d2rep s1 | **@4.0M** | +1542.8 | **0.95** | 0.05 | **4.98 m** |
| d2rep s2 | **@4.0M** | +977.1 | **0.65** | 0.00 | **4.51 m** |
| flat s1 | @4.0M | −515.1 | 0.00 | 0.30 | 0.19 m |
| flat s2 | @4.0M | −562.8 | 0.00 | 0.25 | 0.20 m |
| cur s1 | @4.0M | −696.7 | 0.00 | 0.05 | 0.14 m |
| cur s2 | @4.0M | −641.2 | 0.00 | 0.15 | 0.12 m |

**`d2rep` reaches a 0.95 goal rate at 4.0M environment steps — 80% of E2's own
5.0M budget, and 26% of D2's 15.36M.** At that same 4.0M, the flat control and
the annealing curriculum are both at goal 0.00 and under 0.20 m of forward
progress, which is E2's null result reproduced.

So the answer to "curriculum or budget" is not a split decision. **The reward
mix is the mechanism, and it wins at a budget smaller than the one E2 already
spent failing.** A 77M-step run on the flat reward would have been ~19 h spent
learning nothing that 4.0M on the right reward mix does not show.

### 5c. But the budget is not nothing either — the flat control at 20M is far from E2 at 5M

The control was re-run rather than quoted, and at 4x E2's budget the flat
reward alone does move:

| | goal | fell | furthest forward | action std |
|---|---|---|---|---|
| E2 matched MLP s1/s2 @5.0M (stored) | 0.00 / 0.00 | 0.70 / 0.60 | 0.46 / 0.60 m | 0.633 / 0.624 |
| **E2.1 flat s1/s2 @5.0M** (`posthoc/e2_budget/`) | 0.00 / 0.00 | 0.50 / 0.55 | 0.25 / 0.25 m | 0.624 / 0.618 |
| **E2.1 flat s1/s2 @20.0M** | 0.15 / 0.35 | 0.85 / 0.65 | **3.44 / 4.35 m** | 0.172 / 0.173 |

**The control reproduces E2 at matched budget** — goal 0.00 in both seeds,
majority-fall, sub-metre forward progress, action std 0.62 against E2's
0.62-0.63. It is **not a bitwise replication**: E2 rendered video every 10
epochs and E2.1 every 40, and `e2_eval.roll` reseeds the global RNG per
episode, so the parent process's random stream diverges. That is my
explanation for the residual gap (fall 0.50-0.55 vs 0.60-0.70, forward
0.25 m vs 0.46-0.60 m) and **it is an inference I did not test** — confirming
it would need a 100-epoch re-run at `--video-every 10`. The conditions
compared against each other are unaffected: all six E2.1 arms share one video
cadence and differ only in `--curriculum-steps`.

At 20M the flat arm reaches 0.15-0.35 goal and 3.4-4.4 m. So **4x the budget on
the flat reward is a large improvement over E2 and still does not solve the
task**, while the right reward mix solves it in a fifth of that.

### 5d. `cur` is WORSE than no curriculum at all, and the reason is arithmetic

This is the counterintuitive number and it is not smoothed over: **annealing to
alpha = 0 is worse than never having a curriculum** — 0.10 goal against 0.25,
2.19 m against 3.89 m, and the highest fall rates in the experiment (0.80-0.85).

Two mechanisms, both derivable from the reward constants rather than guessed.

**(i) There is a critical alpha of 0.739, and `cur` falls below it at epoch 21.**
Measured on the idle floor's own 17 lost episodes, the dense return a full
491-step episode can bank is **+352.4**. The fall-dodge is worth
`(1 − alpha) × 1000`. Setting them equal:

```
(1 - a) * 1000 = a * 352.4     ->     a_crit = 0.739
```

Below alpha = 0.739, **ending the episode early outweighs everything a
full-length episode can possibly bank**, so falling is the optimum. Then:

| arm | alpha(0) | alpha(79) | alpha(399) | crosses 0.739 at |
|---|---|---|---|---|
| **d2rep** | 1.000 | 0.970 | 0.847 | **never** |
| cur | 1.000 | 0.013 | 0.000 | **epoch 21 (1.05M steps)** |
| flat | — | — | — | permanently below (sparse at full weight) |

`cur` crosses the threshold at 1.05M steps — long before any arm in this
experiment learns to locomote (`d2rep` takes off between epochs 44 and 84).
**D2's accidental schedule never crosses it.** That is the whole difference
between the two curriculum arms, and D2's setting was safe by accident, not by
design.

**(ii) At alpha = 0 there is no locomotion gradient at all.** CompetEvo's
curriculum reward at alpha = 0 is the sparse term **alone** — no forward term,
no survive bonus, no control cost (§1). A policy that cannot yet reach the goal
sees `0` on every step except a `−1000` it can dodge by falling. So for epochs
80-399 — **80% of the run** — `cur` has no gradient pointing toward the goal,
while `flat` always keeps the dense forward term at full weight. That is why
`cur` ends up below the control rather than merely equal to it.

Both mechanisms point the same way and the data separates them: at epoch 79
`cur` has the *lowest* fall rate of any arm (0.05-0.15, better than flat's
0.25-0.30) — the high-alpha phase did suppress falling exactly as predicted —
and by epoch 399 it has the *highest* (0.80-0.85). It learned not to fall, then
unlearned it once the objective stopped paying for staying up.

### 5e. The correlation structure inverts — return becomes a measure of competence

E2's central finding was `r(fall rate, return) = +0.989` and
`r(forward progress, return) = +0.019`: return measured falling, not running.
Recomputed here over the seven arms of the headline table:

| | E2 (7 arms, 5.0M) | **E2.1 (7 arms, 20.0M)** |
|---|---|---|
| r(fall rate, return) | **+0.989** | **−0.517** |
| r(forward progress, return) | **+0.019** | **+0.947** |

**The sign flips on both.** Return now tracks how far the agent gets
(r = +0.947) and is *negatively* associated with falling. Per condition, at the
episode level (2 seeds x 20 episodes pooled):

| condition | n | r(fell, R) | r(fwd, R) | mean fwd | fall rate | goal |
|---|---|---|---|---|---|---|
| d2rep | 40 | *undefined* | −0.051 | 5.00 m | 0.00 | 0.97 |
| flat | 40 | **−0.979** | +0.679 | 3.89 m | 0.75 | 0.25 |
| cur | 40 | −0.179 | +0.680 | 2.19 m | 0.82 | 0.10 |
| *idle* | 20 | **+0.985** | −0.120 | 0.08 m | 0.15 | 0.00 |

Two of these need stating honestly rather than quoting:

* **`d2rep`'s two correlations are degenerate, not informative.** `r(fell, R)`
  is undefined because the fall rate is exactly 0 — zero variance — and
  `r(fwd, R) = −0.051` is a ceiling effect: every episode is at 5.00 m, so
  there is no forward-progress variance left to correlate with. That is what
  solving the task looks like in this statistic, and it is why the goal column
  and not the correlation is `d2rep`'s headline.
* **The idle floor still shows E2's structure exactly** (`+0.985`). That is the
  control that matters: the instrument did not change, the *policies* did. E2's
  finding was true of E2's policies and remains true of an untrained one.

### 5f. What this says about the question as asked

> **Curriculum or budget? The reward mix, decisively — but only D2's realised
> one, and CompetEvo's nominal one is actively harmful.**

* **Both keep falling** was the predicted outcome for flat and for `cur`, and
  it happened for both.
* **The 15x budget scale-up was not needed and would not have been enough.**
  `d2rep` solved the task at 4.0M; `flat` at 20M — four times E2's budget —
  still reaches the goal in only 25% of episodes.
* **The mechanism proposed in the brief is confirmed with one correction.** It
  is not "early dense shaping lets the agent get moving before the sparse term
  fades in". `cur` does exactly that and fails. It is "**the sparse ±1000 must
  stay below (1 − 0.739) x 1000 = 261 points of weight for the whole run**",
  which D2's unset default happened to guarantee and CompetEvo's nominal
  200-epoch schedule does not.

---

## 6. Cost

| | |
|---|---|
| six MLP arms | ~83 s/epoch with all six live (45 s with four), 400 epochs, **~9 h wall clock, all six concurrent** |
| CPU | 10 sampler threads per arm, 60 processes; peak load ~38 of 48 cores |
| **GPU** | **none. Every arm ran `CUDA_VISIBLE_DEVICES=` — E2.1 is CPU-only**, which is what made 6 x 20M affordable. D1 held the card for the whole training window and was stopped cleanly by the user afterwards at 17.6B steps; E2.1 never touched it |
| disk | **~135 MB** — six results directories at ~11 MB (41 checkpoints x 419 KB) plus ~15 MB of clips. Measured on E2.1's own paths, not from `df` (§3 note 3) |
| post-hoc | 9 jobs in parallel, ~12 min; then 6 more at epoch 79, ~10 min |
| logging | six wandb runs, **metrics and video inline in one run each**, verified through the API: `metric rows=80, video/best_median_worst rows=10, video-in-summary=True, last epoch=399` on all six. No `_media` split, no run deleted |

## 7. Not tested / not claimed

* **n = 2 seeds per condition.** Six training runs cannot characterise seed
  variance. `flat`'s two seeds differ by 0.15 vs 0.35 goal and 3.44 vs 4.35 m —
  a factor of two on the headline metric — which is a direct warning against
  reading any single condition mean too hard. `d2rep`'s 0.95/1.00 is the only
  result here robust to that concern, because both seeds saturate.
* **The flat control is not a bitwise replication of E2** (§5c). It reproduces
  E2's qualitative result at matched budget; the residual gap is *probably* the
  video cadence perturbing the parent RNG stream, and **that explanation is an
  inference I did not test**.
* **`a_crit = 0.739` is computed from one measured constant** — the +352.4
  dense return of a full episode, measured on the idle floor. A policy that
  moves forward banks more than +352.4, so the true threshold falls as the
  agent improves; 0.739 is the threshold *for a policy that cannot yet move*,
  which is the regime that matters for whether it ever starts. It is a
  first-order argument, not a proof, and it was derived after seeing the
  result, not before.
* **Only three alpha schedules were run.** Nothing between `cur`'s 4M and
  `d2rep`'s 130M was tried, so the experiment does not locate where between
  them the transition happens, nor test the obvious follow-up — an alpha floor
  at ~0.85 that anneals no further.
* **Only the MLP arm.** The GNN was not re-run under any curriculum. E2's
  architecture question is still unanswered, though it is now *answerable*:
  `d2rep` is a reward regime on which this task is solvable, so a GNN-vs-MLP
  comparison run there would finally be comparing two controllers that can both
  do the task.
* **The fall-dodge was not removed from the task.** E2 §6's hazard is still
  present in the env exactly as CompetEvo defines it; `d2rep` avoids it by
  never weighting it heavily, not by the rule being fixed. E3 still has to
  decide about the rule itself.
* **This is not a reproduction of D2.** D2 trained against an **idle** opponent
  in mujoco_warp on a 2v2 team task; `d2rep` runs 1v1 against a scripted
  opponent that scores, in mujoco-py with CompetEvo's PGS/1000. It replicates
  D2's *alpha trajectory*, nothing else.
* **No hyperparameter was swept.** lr 3e-4, 64x64 tanh, log_std 0, batch
  50,000 — all inherited from E2 unchanged.
* **Nothing here says anything about morphology.** The body was frozen and
  verified frozen (134 arrays identical) under all six trained policies.
