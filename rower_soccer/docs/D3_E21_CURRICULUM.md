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
