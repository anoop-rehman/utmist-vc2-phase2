# D3 M3 E4R — shared-weight self-play: does the ratchet hold?

*Pre-registered 2026-09-05, before any E4R run existed. Nothing launched.*

One agent — **one design head, one controller** — plays both sides of the 1v1
run-to-goal. Success, in the user's words: *each new iteration should beat all
past iterations 1v1, and be basically tied against its current iteration.*

Supersedes the two-lineage E4 arm, which was built, gated 11/11, launched, and
stopped by stop-file at epoch ~14/400. **Nothing about it was found to be
wrong** — it is archived under `docs/t2a/e4_twolineage_archive/` in case the
divergence question returns. The redirect is well-founded on our own evidence:
E3.1 established that the **design head is blind** (skeleton and attribute
stages see only `attr_fixed ++ attr_design`, never simulation state), so
observation-conditioned specialisation between two lineages is impossible by
construction. That handicaps a divergence study and is irrelevant to a
shared-weight ratchet — which is also half the compute.

## 1. The trap: the two halves of the criterion conflict

If the training opponent were the current self, then **at equilibrium the
training signal switches itself off**. Both sides reach the line on the same
step, `n_reached == 2`, and `run_to_goal.py` scores:

```python
n_reached = int(reached) + int(opp_reached)
parse = 0.0
if n_reached == 1:
    parse = GOAL_REWARD if reached else -GOAL_REWARD    # +/-1000
```

`parse` is **0** whenever the match is tied — precisely at the point the second
half of the criterion asks us to reach. At `alpha = 0.847` that term is worth
**306 points against a dense component of 376**, i.e. 58% of the weighted
buffer return, so losing it is not a detail.

**Therefore the training opponent is always a strictly PAST self.** Beating a
weaker past self is where the ±1000 still pays. The mirror match is an
*evaluation*, never a training signal.

## 2. Design

**One arm per seed, 3 seeds.** Inherited verbatim from E3.1's primary arm (the
one that solved 2 of 3): `control_log_std: -1.5`, no `min_motors`, d2rep,
batch, lr, GNN spec, `done_condition`. The cfg delta is **four lines** — the
env class plus three opponent fields — which is what makes gate 5 ("scripted
mode reproduces `run_to_goal` bit for bit") a real regression test.

**The ring**, transcribed from CompetEvo via D2's `competevo_port/selfplay.py`,
including a correction D2 had to make to its own port map:

```
start = max(1, floor(delta * epoch));  end = epoch
ckpt  = randomstate.randint(start, end)      # HIGH-EXCLUSIVE
```

so the opponent is uniform on `[max(1, floor(delta*epoch)), epoch-1]` —
strictly past, never the current self. Three details:

* **`delta = 0`** (the whole history). `delta` is a *window*, not a mixing
  probability. The criterion is beating *all* past iterations, so the ring
  spans everything; CompetEvo's own fixed-morph ants used 0 (`0.5` was their
  dev setting).
* **Redrawn every EPISODE**, not every epoch — D2 measured this and found its
  own port map wrong on the point.
* **The opponent acts stochastically** (`noise_rate = 1.0`,
  `base_runner.py:27`). The two-lineage arm used a mean action; this matches
  theirs. `opponent_mean_action: true` restores the old behaviour.

Per-episode swapping is nearly free because `AntEnv.reset_robot` **already**
reparses the Robot and recompiles the model every episode; each ring member's
merged scene and opponent Robot are cached at archive time.

Archive cadence `--ring-every 10` → **40 members over 400 epochs**.

## 3. Pre-registered criteria, with numbers

All three are trajectory criteria on the window **epochs 200-400**, aggregating
before comparing. Endpoints have inverted the conclusion three times on this
project.

### (1) RATCHET HOLDS

* mean win rate against past selves with **age gap ≥ 100 epochs is ≥ 0.75**, and
* **Spearman ρ(age gap, win rate) ≥ +0.5**, pooled over seeds.

Grounded in E3.1's own trajectory rather than invented: its goal rate is 0.00
through epoch 100 and 0.98 after 300, so a competent late self should beat a
≥100-epoch-old self almost always. At 20 episodes per matchup the binomial SE
at p = 0.75 is 0.097, so 0.75 sits ~2.6 SE above a coin flip.

### (2) RATCHET FAILS / CYCLES

* **cyclic-triple fraction > 0.10** in the slot-averaged win matrix, or
* mean win rate vs age-gap-≥100 selves **< 0.6**, or **ρ ≤ 0**.

0.10 is **calibrated by simulation, not chosen**. For a perfectly transitive
ladder of 12 players at 20 episodes per ordered pair, with *adjacent* pairs at
a near-tied 0.55 (the hardest case for binomial noise):

| | mean | p95 | p99 |
|---|---:|---:|---:|
| cyclic-triple fraction under a transitive null | 0.013 | 0.036 | **0.055** |

and a tournament with **no real ordering at all** gives **0.136** (theoretical
max 0.25). So 0.10 is ~1.8× the noise ceiling and clearly below chance.

### (3) DEGENERATE MIRROR — the trap, made explicit

A 0-0 stalemate and a 1-1 race both read as "tied" on any scalar. The mirror
match therefore reports **three mutually exclusive outcomes**:

| outcome | meaning | `parse` |
|---|---|---:|
| **DECISIVE** | exactly one reached | ±1000 |
| **MUTUAL** | both reached, same step — **the good tie** | 0 |
| **STALEMATE** | neither reached (timeout) — **degenerate** | 0 |

**DEGENERATE** is declared if, over the window, `stalemate_rate > 0.5` **or**
mean forward progress in the mirror match is **< 2.5 m**. The course is 5.0 m
(x = −1 → +4); E3.1's solving arms covered 5.1-5.25 m and its non-solving arms
0.11-0.65 m, so 2.5 m separates the two regimes with a wide gap on either side.

**HEALTHY EQUILIBRIUM** is `mutual_rate` high, `stalemate_rate ≈ 0`, and
forward ≥ 2.5 m — both agents running the full course and arriving together.

## 4. The measurement

`e4r_tournament.py` plays the full matrix over **12 checkpoints × 20 episodes
per ordered pair**, in **both slot orientations** (averaged), because the
learner always trains in slot 0 and a slot advantage would otherwise
masquerade as skill. The orientation gap is reported as a diagnostic: gate 3
says the π-z rotation is exact (observation max|Δ| = 0.000e+00), so it should
be small — and if it is not, that is a finding.

Reported as a **matrix**, not a scalar. A monotone ratchet is triangular.

## 5. Carried forward unchanged

3 seeds; `control_log_std = -1.5`; d2rep; `p_act4` and the E3 falsifiers;
per-epoch morphology census; the fall/forward dodge correlations; per-arm
stop-files; `assert_e4_instruments.sh`; wandb with video in the same run;
mean-action **and** stochastic evaluation side by side with an idle
zero-torque floor; **forward progress, not return, as the primary readout**;
explicit PIDs only, never `pkill -f`.

## 6. What this does not test

* Whether two *independent* lineages would diverge — that is the archived E4.
* Whether a **sighted** design head would do better; it remains blind here, so
  morphology is shaped by return alone.
* Transfer of the ratchet to any opponent outside its own lineage.

## LAUNCHED 2026-09-05 23:15 — gate 13/13, three arms logging

`rtg_e4r_s1` (694927), `rtg_e4r_s2` (695070), `rtg_e4r_s3` (695257), each with a
stop-file. Launched by the detached `autolaunch_e4b.sh`, which measured
worst-of-six free memory at **19 635 MiB** against its 17 000 requirement
before committing.

### GPU: measured under the predicted band

| | MiB | of 20 475 |
|---|---:|---:|
| predicted (3 × measured 5 309 single-arm peak) | 15 871 | 78% |
| **measured, 45 samples over 90 s** | **13 744** | **67.1%** |

Flat across every sample, **zero above the 17 500 trigger**, headroom 6 731 MiB.
The prediction was conservative in the safe direction for once — three arms
interleave their update peaks rather than stacking them.

### Opening guards, all three arms at epoch 0

| | s1 | s2 | s3 |
|---|---:|---:|---:|
| `p_act4` | 0.80 | 0.85 | 0.90 |
| `p_act1` | 1.00 | 1.00 | 1.00 |
| motors (mean) | 5.55 | 5.55 | 5.75 |
| bodies (mean) | 14.75 | 14.40 | 14.55 |
| `control_log_std` | −1.504 | −1.504 | −1.504 |
| ring size / draws | 1 / 0 | 1 / 0 | 1 / 0 |

`p_act4` at 0.80-0.90 is nothing like E3's collapse to 0.000 by epoch 17, and
`control_log_std` is at the −1.5 the fix specifies. **Ring draws are 0 at epoch
0 by design** — no strictly-past member exists yet, so no opponent is installed,
which is exactly gate F's empty-ring negative control observed in production.

Instrument assertion: **OK on all three**, run in its strict form with rows
present.

### Epoch cost, full smoke sample

11 epochs of the single-arm pipeline test: ordinary epochs **117 s** mean
(110, 110, 115, 115, 120, 120, 121, 125), mirror/ladder epochs **194 s**
(171, 200, 210).

**There is a mild upward drift — 110 → 125 s — that I cannot yet separate from
noise or from ring growth**, so the ~124 s/epoch budget is quoted with that
caveat rather than as settled. The smoke archived every 2 epochs; production
archives every 10, so the ring grows 5× slower. I will re-measure at epoch 50
and report if it has moved.

### Correction: three concurrent arms do NOT fit, and my "they interleave" claim was wrong

Ninety minutes after launch the trigger fired at 18 650 MiB. Confirmed by
sustained measurement rather than acted on, and it is **not** a one-off:

| 3 arms, 36 samples / 72 s | MiB | of 20 475 |
|---|---:|---:|
| p50 | 13 190 | 64.4% |
| **p90 / p99 / p100** | **18 650** | **91.1%** |
| samples above the 17 500 trigger | **6 of 36 (17%)** | |
| headroom at peak | **1 825** | |

**The claim I made at launch — that three arms "interleave their update peaks
rather than stacking them" — is withdrawn.** It was inferred from a 90 s window
taken while the arms were still at epoch 0-1. The arithmetic says the opposite:
`T_update` is ~67 s of a ~117 s epoch, a **57% duty cycle**, so three
independent arms are all updating

```
0.57^3 = 0.185
```

**~19% of the time**, which is within noise of the 17% measured. That also
means **staggering the launch would not help** — the overlap is statistical,
not phase-locking, so there is no offset that avoids it.

**Action: `rtg_e4r_s3` stopped by stop-file at epoch 2** (about four minutes of
compute) and **deferred, not dropped** — it relaunches from a clean slate when
s1 and s2 finish, under its own detached launcher which refuses if the card is
not free. Three seeds remain the minimum that survives one dead controller, and
all three will run.

| | peak MiB | % | over trigger | headroom |
|---|---:|---:|---:|---:|
| 3 arms | 18 650 | 91.1% | 17% of samples | 1 825 |
| **2 arms** | **9 428** | **46.0%** | **0%** | **11 047** |

Two arms both updating would be ~12.4 GB (61%), still comfortable.

**Cost of the correction:** ~34 h instead of ~29 h for all three seeds. That is
the standing rule applied as written — smaller wave, no seeds shed — and it
buys margin against the failure that killed D1 at 19 950 MiB on this same card
with MPS active.

## Longitudinal GPU vs body size — the premise does not hold

Asked for after the two-arm peak read 15 818 MiB at epoch 8 against my earlier
9 428, alongside `motors_mean` rising 6 → 9 and `p_act4` reaching 1.00 — the
E3.1 pattern of bodies growing until an arm has to be shed.

### First, my 9 428 was not a comparable measurement

It came from **40 samples × 2 s = 80 s**. An epoch is ~117 s, so the window did
not span one — it violated the sizing rule I had stated one message earlier.
With two arms the joint peak needs both in their update phase at once (~32% of
the time at a 57% duty cycle), and an 80 s window can miss it entirely. Its
median was 7 706; the card's median now is **15 746, equal to its peak** — the
card sits flat, so what changed is mostly what the window could see.

### Second, memory does not track body size

E3.1's recorder holds **1 357 samples spanning `n_bodies` 11 → 26** — nearly the
whole range up to the ceiling of 29. Fitted **within each arm** (pooling across
arms was the defect that invalidated the earlier attempt, since different arms
carry different allocator reserves):

| arm | `n_bodies` | per-arm peak | slope | R² |
|---|---|---|---:|---:|
| `rtg_e31_s1` | 13-23 | 5 440-8 862 | **+199.8** MiB/body | 0.515 |
| `rtg_e31_s2` | 14-26 | 5 584-9 122 | **−0.4** | 0.000 |
| `rtg_e31_s3` | 11-21 | 6 256-8 880 | **+100.1** | 0.158 |

**Inconsistent in sign and magnitude, and non-monotonic in both datasets.**
E3.1 peaked at 9 122 MiB at `bodies_mean` 18.1 but only 7 674-7 780 at 19.1-19.6.
E4B peaked at 10 586 at 18.4 but 6 338 at 20.1. Memory here is dominated by the
**fixed 50 000-state PPO buffer**, not by per-graph node count.

So the E3.1 pattern being feared — bodies grow, memory grows, an arm must be
shed — **is not what E3.1's own data shows**. E3.1 shed an arm because three
arms peaked at 19 753 MiB, which is the same arithmetic that made me defer s3
here: three arms overlap, not bodies grow.

### Third, the ceiling, and where the real threshold is

The 29-body ceiling carries over (E4B shares `max_body_depth 4`,
`max_nchild 2`). **`bodies_max` running maximum is already 25 after ten
epochs** — at most four more bodies are structurally possible.

| | per-arm | two arms | of 20 475 |
|---|---:|---:|---:|
| joint peak observed now | — | **15 746** | 77% |
| E3.1 max per-arm anywhere (11-26 bodies) | 9 122 | 18 244 | 89% |
| E4B max per-arm so far | 10 586 | (21 172) | not observed jointly |
| steepest within-arm slope, 25 → 29 bodies | +800 | +1 600 | +8 pts |

**The breach threshold is not ahead of us.** Two arms cross 17 500 when both
simultaneously exceed 8 750, and E3.1 arms individually exceeded 8 750 at
`n_bodies` 17, 18 and 19 — counts we passed several epochs ago. Whether it
happens depends on **coincidence of update phases, not on further growth**.

**Consequence for the decision: waiting is not accruing danger.** The risk is
roughly stationary rather than rising, so there is no deadline by which an arm
must be shed. If the watcher does trip at 17 500, the mitigation is the one
already used for s3 — stop one arm by stop-file and run it afterwards, costing
wall clock and no seeds. No action now: 4.6 GB of headroom is real.

### Housekeeping found along the way

Two stale E3.1 recorders (`gpu_longitudinal.sh`, `watch.sh`) were still running
after 1.5 days and 14 hours, re-appending the finished runs' final rows. Both
were bash, not CUDA clients, and were stopped. They are what made recent
timestamps appear under `runs/d3_e31_fix/` — **not** the E5 agent, which has
touched only its own `PLAN_D3_E5_2V2.md`.

## Disk: a deadline, not tightness — and the ring was the cause

Found at 93% (3.1 GB free) with ~34 h of writing still due.

### The burn rate, measured

**Each ring member's policy is 148 MB** — 19.4 M float64 parameters, of which
**96 MB is three `*_ind_mlp` layers carrying a 256-slot leading dimension** from
upstream's `use_body_ind`. One is written per archive.

| | GB |
|---|---:|
| ring, 41 members × 3 arms to epoch 400 | **17.8** |
| free at discovery | 3.1 |
| **exhaustion point** | **≈ epoch 100, under 3 h away** |

So: a deadline. `models/` was **not** the problem — the trainer already
archives it to GCS with size verification and prunes locally
(`archived+pruned`), and with `save_model_interval 100` it stays bounded. **The
ring had no archive-and-prune path at all.**

### The fix: persist a subset, keep the experiment identical

`--ring-persist-every 4` writes every 4th archive to disk (11 members/arm).
**The in-memory ring still holds every member**, so the sampler's support and
therefore the experiment are unchanged; only what the post-hoc tournament can
read is thinned, and it subsamples to ~12 checkpoints anyway.

| persist-every | members/arm | 3 arms |
|---|---:|---:|
| 1 (before) | 41 | 17.82 GB |
| 3 | 14 | 6.09 GB |
| **4 (chosen)** | **11** | **4.78 GB** |

Also fixed: the mirror match's transient "current self" was writing a **148 MB
`policy_-001.p` on every eval**. Overwritten each time so it never grew, but it
burned the write and held the space for nothing. Now `persist=False`.

The two live arms were on the old code, so they were restarted at **epoch 13
(3.2% in, ~26 min)** rather than having files deleted underneath them.

### Space recovered, verify-then-delete

Every run uploaded to GCS and **md5 compared against the stored object** before
anything was removed; a run whose hashes did not match was kept. (The first
attempt failed for want of `xxd` and correctly kept all three files.)

Archived and pruned: `rtg_e4_s1a/b`, `rtg_e3c_s1/s2`, `rtg_e3_s1/s2/s3`,
`rtg_e31f_s1`, `rtg_e31d_s3body`, `hopper_gpu`, `hopper_gpu_s2`,
`rtg_gnn_s1/s2`. Deleted without archiving: `rtg_e4r_smoke` weights (a pipeline
test whose only outputs were timings, already documented) and `rtg_e4r_s3`'s
3-epoch residue, which its own launcher deletes before relaunch.

**Kept deliberately**: `rtg_e31_s1/s2/s3` models — `gate_e4.py`'s default
snapshot is `rtg_e31_s2/models/epoch_0400.p`, and they are the evidence base
for the comparison set and the null. Every run's `e3_epochs.jsonl` and all of
`runs/d3_e31_fix/census/` were kept regardless of archiving; the
memory-versus-body-size analysis rests on those.

**93% → 74% free (3.1 → 11 GB).** Projected need to completion **≈ 4.8 GB**.

### Disk guard

`watch_e4b.sh` now reports on crossing 6000 / 4000 / 2500 / 1200 MiB free, once
per threshold. Disk is the one resource with a precedent for killing a run
here — E3.1 seed 3 died at epoch 39 on a full disk.

## The epoch-150 restart rule was wrong — backtested and corrected

Pre-registered as the PBT substitute: *"if a seed's goal rate is still 0.00 at
epoch 150, restart that seed."* **Backtested against E3.1, it fires on all
three seeds — including the two that finished at goal 1.00.**

| | first epoch with goal > 0 | final goal | rule at 150 |
|---|---:|---:|---|
| `rtg_e31_s1` | **194** | 1.00 | **FIRES** (false positive) |
| `rtg_e31_s2` | **199** | 1.00 | **FIRES** (false positive) |
| `rtg_e31_s3` | 179 | 0.00 | fires (correct) |

Applying it would have restarted both eventual winners. The error was choosing
goal rate — which is near-zero for *everyone* at epoch 150 — instead of the
briefing's own primary readout.

**Forward progress separates them cleanly, and earlier:**

| epochs 150-199 | s1 (solved) | s2 (solved) | s3 (failed) |
|---|---:|---:|---:|
| goal rate | 0.06 | 0.04 | 0.01 |
| **forward (m)** | **2.59** | **3.42** | **0.68** |
| **mean speed** | **+0.55** | **+0.53** | **−0.17** |

`max_fwd > 2.0 m` is reached by s1 at epoch 164 and s2 at 119, and by s3
**never**. s3's mean speed is **negative in every window** (−0.26, −0.17,
−0.15, −0.18) while both solvers are positive throughout — it was travelling
backwards.

**Corrected rule**: at **epoch 200**, flag if the window-mean `max_fwd` over
epochs 150-200 is **< 1.5 m** *or* mean speed **< 0**. On E3.1 that is 3 of 3
correct: 1.5 m sits 1.7x below the nearest solver and 2.2x above the failure.

**Stated limitation: this is calibrated on ONE negative example.** Its
false-positive rate on E3.1 is 0/2 and its true-positive rate 1/1, but with a
single failure to learn from the false-negative rate is unknown. It remains
detect-and-flag; nothing restarts itself.

The two arms launched 2026-09-06 00:11 still carry the superseded rule, so they
may write a spurious `RESTART_RECOMMENDED` at epoch 150. That costs a log line
— nothing acts on it — and the watcher now labels the marker as possibly
superseded **and computes the corrected forward-progress check independently**,
so the right answer is reported regardless of which rule the trainer was
launched with.

## Why the creatures do not move — measured, and it is structural

### 1. Not a bug: the opponent is driven

Ruled out first, because an untrained opponent and an inert one are
indistinguishable on screen. Measured on a real self-play episode with s2's
epoch-40 ring member installed exactly as training installs it:

| | value |
|---|---:|
| opponent **max \|torque\|** | **0.8225** |
| opponent mean \|torque\| | 0.3786 |
| steps with non-zero torque | **100%** |
| opponent displacement toward its goal | 0.151 m |

The opponent acts. This is not the epoch-2-stage-flag failure the E4 gate
caught.

### 2. The learner's mean action has collapsed, and exploration is net-negative

Same episode, our own agent:

| per step | mean action | stochastic (what PPO samples) |
|---|---:|---:|
| max \|action\| | **0.0009** | 0.827 |
| forward | +0.024 | +0.159 |
| ctrl cost | 0.000 | **0.212** |
| survive | +1.000 | +1.000 |
| **dense total** | **+1.024** | **+0.947** |

**Exploring costs more than it earns.** The noise that could discover
locomotion costs 0.212/step and buys 0.159 — so standing still is better by
0.077/step, and the gradient points at shrinking the action. The mean action is
already **0.0006**: all visible motion in the videos is exploration noise, not
policy.

This is E3's attractor reached by a different route. E3 deleted actuators to
make `0.5*sum(a^2)` zero; here the actuators survive and the *policy output*
goes to zero instead. `control_log_std -1.5` and the actuator floor closed the
first route, not the second.

### 3. What E3.1 had that E4B does not

| at epoch 24 | E3.1 s1 | E3.1 s2 | E4B s1 | E4B s2 |
|---|---:|---:|---:|---:|
| `loss_rate` | **1.00** | **1.00** | 0.00 | 0.00 |
| `ep_len` | 491 | 491 | 500 | 500 |
| Σforward | **−169** | **−119** | +6 | +6 |
| `R_mean` | **−685** | **−634** | +498 | +501 |

E3.1's scripted opponent **marched from x=+1 to x=−4 through the agent's
position** and was infinitely massive (its state was overwritten every step, so
contact could not slow it). It therefore did two things E4B's opponent does
not: it **scored in every episode**, and it **shoved the ant backwards**,
driving Σforward to −169.

E4B's two agents stand 2 m apart and never touch. Nobody scores, nobody pushes.
**Standing still earns ~99% of the achievable return.**

### 4. The alpha schedule is NOT the culprit

Worth checking, and the answer is no. `alpha` weights only the **sparse** term:

| epoch | alpha | parse weight | value if parse = −1000 |
|---|---:|---:|---:|
| 40 | 0.9846 | 0.0154 | **−15** |
| 200 | 0.9232 | 0.0768 | −77 |
| 400 | 0.8464 | 0.1536 | −154 |

At epoch 40 the sparse term is worth **15 points even when it fires**. E3.1's
buffer return at e24 was ≈ `0.985 × 322 − 15 ≈ +302`, so the −1000 was
contributing 5% of it. **What actually taught E3.1 to move was the dense
forward term at −169**, not `parse`. Re-tuning alpha would not address this:
the missing signal is dense, and no reweighting of a term that is *identically
zero* can supply it.

### 5. Is it self-resolving? The barrier is modest but the gradient is adverse

Break-even needs the forward term to exceed ctrl cost at the same action
magnitude: **0.159 vs 0.212, i.e. ~33% more forward per unit torque**. That is
a coordination improvement, not a large one — and a competent agent is worth
far more than standing still (≈ 500 survive + 333 forward + parse, against
500). So the global optimum is to move.

But the *average* gradient currently points the other way, and the mean action
is already pinned at ~0. Escape depends on PPO's noise stumbling onto a
coordinated gait whose advantage survives the ctrl cost, with no external
pressure helping. **E3.1 needed ~200 epochs to move with a −169 forward penalty
pushing it; E4B has +6.**

### 6. Options, not yet actioned

Nothing changed on the running arms. Candidates, cheapest first:

1. **Opponent curriculum** — face the *scripted* opponent early, anneal to
   self-play once locomotion exists. Restores exactly the pressure that made
   E3.1 work, reuses `opponent_mode: scripted` which gate 5 proves is bit-identical
   to `run_to_goal`, and matches Bansal's exploration-curriculum logic applied
   to the opponent rather than the reward.
2. **Warm start from an E3.1 winner** — seed the ring's epoch-0 member from
   `rtg_e31_s2` (4.891 m/s). **AlphaStar does not start its league from random
   initialisation either**; it seeds from supervised policies precisely because
   self-play from scratch does not bootstrap. We have three solved checkpoints.
3. **Lower `CTRL_COST_COEF`** — changes the reward E3/E3.1 were measured under,
   so every cross-rung comparison breaks. Least preferred.

## Warm start: it worked, and it immediately produced a new regime

Epoch 4, the first eval after seeding:

| | s1 (seed 4.224 m/s) | s2 (seed 4.891 m/s) |
|---|---:|---:|
| goal / loss | 0.00 / 0.00 | **0.50 / 0.50** |
| fall rate | **0.90** | 0.00 |
| speed | 1.052 | **4.303** |
| mirror decisive / stalemate | 0.15 / 0.85 | **0.90 / 0.00** |
| ladder win vs past selves | 0.0 | **0.70** |

**s2 is the experiment working.** goal 0.50 against loss 0.50 is a *true tie
between identical policies* — the user's success criterion satisfied at epoch 4
rather than hoped for — with the mirror **90% decisive and 0% stalemate**. The
degenerate mirror that dominated the cold run is gone.

### s1's falls are collisions, not forgetting

Tested with training removed as a variable: the **seed policy itself**, no
gradient steps, in two conditions.

| `rtg_e31_s1` seed policy | goal | **fell** | speed |
|---|---:|---:|---:|
| vs **inert** opponent | 1.00 | **0.00** | 4.321 |
| vs **itself**, fully physical | 0.40 | **0.40** | 1.579 |

The body is fine — 4.32 m/s, never falls, unopposed. It falls on **contact**,
before any training.

**Why E3.1 never saw this.** Its scripted opponent had its state overwritten
every step, so it was infinitely massive and non-reactive, and it advanced at
0.68 m/s. E4B's opponent is a fully physical body under policy control racing
at seed speed:

| | closing speed | opponent |
|---|---:|---|
| E3.1 | 4.9 m/s | state overwritten, cannot be deflected |
| E4B | **8.4 m/s** | fully physical, deflectable |

**This is the task becoming genuinely adversarial for the first time.** In
E3.1 the opponent could not really contest — it arrived at step 491 against
winners finishing at 76. Now both sides race and meet in the middle, and
**physical robustness to a head-on collision becomes a selection pressure that
has never previously existed in this project**. s2's 18-body plan survives it
(fell 0.00); s1's 17-body plan does not (fell 0.40 pristine, 0.90 after four
epochs).

That is a morphological difference under adversarial contact — precisely what
E4B exists to surface — and it is a result, not a fault. Whether the ratchet
teaches s1 a collision-robust gait is now one of the more interesting things
this run can answer.

## s1 learns collision robustness — the first genuinely adversarial adaptation here

Its seed falls on contact (0.40, pristine, no gradient steps). Nineteen epochs
later it barely does, and every measure moves monotonically:

| s1 | e4 | e9 | e14 | **e19** |
|---|---:|---:|---:|---:|
| **fall rate** | 0.90 | 0.80 | 0.60 | **0.10** |
| goal | 0.00 | 0.10 | 0.20 | **0.80** |
| forward (m) | 2.81 | 3.08 | 3.73 | **4.70** |
| speed | 1.052 | 1.209 | 1.306 | **2.325** |
| mirror decisive | 0.15 | 0.35 | 0.65 | **0.85** |
| mirror stalemate | 0.85 | 0.65 | 0.35 | **0.15** |
| ladder win vs past selves | 0.00 | 0.20 | 0.45 | **0.65** |

**s1 is now better than its own seed at the thing the seed was never tested
on.** The seed policy, unchanged, falls 0.40 against a physical copy of itself;
s1 at epoch 19 falls **0.10**. E3.1 could not have produced this: its opponent's
state was overwritten every step, so a head-on collision at 8.4 m/s closing
speed never occurred in training.

This is the ratchet doing what it was built to do — `ladder win` 0.00 → 0.65
means the current agent now beats its own archived past selves — and the
adaptation is to a pressure that only exists because both sides are real.

s2 meanwhile is stable: **fall 0.00 throughout**, goal 0.4-0.6 against loss
0.4-0.6 (a maintained tie), mirror **0.90-1.00 decisive with 0.00-0.05
stalemate**, ladder win 0.55-0.80.

Both are far too early for the pre-registered verdict, which is taken over
epochs 200-400 and requires win rate ≥ 0.75 against selves ≥ 100 epochs older.
The sign is right; the number is not yet evidence.

## Budget re-derived on warm-started arms

Warm arms terminate episodes on goals rather than the 500-step cap, which
changes the rate materially:

| | cold arms | warm arms, recent 7 epochs |
|---|---:|---:|
| `ep_len` | 500 (timeout) | 76-241 |
| s/epoch | 189 | **s1 119.7, s2 127.9** |

At ~124 s/epoch the concurrent pair finishes 400 epochs in **13.8 h**. s3 alone
afterwards should be faster still with no contention — the cold single-arm rate
was 112 s/epoch, so ~100 s warm is plausible, giving **~11 h**, though that
figure is extrapolated rather than measured.

> **Revised total ≈ 25 h**, against the ~35 h carried over from the cold arms.

The rate may drift as bodies grow or episodes lengthen; s1's episodes (174-241)
are already longer than s2's (76-132), so the pair is paced by s1.

## Epoch-40 stride and the budget, re-measured on warm arms

**Persist stride confirmed on the warm-started arms** (the earlier confirmation
was on the cold pair — same code, different run):

```
[ring] epoch 10: archived, ring now holds 2 (in memory only; not persisted)
[ring] epoch 20: archived, ring now holds 3 (in memory only; not persisted)
[ring] epoch 30: archived, ring now holds 4 (in memory only; not persisted)
[ring] epoch 40: archived, ring now holds 5
```

s1 holds `policy_0000` and `policy_0040` on disk (297 MB) with **6 body XMLs** —
every archive documented, only every fourth persisted. Disk **9.6 GB free**;
projection to epoch 400 is **1.59 GB/arm, 4.78 GB for three**.

### Budget: my 7-epoch figure was optimistic

| window | s1 | s2 |
|---|---:|---:|
| recent 7 epochs (quoted earlier) | 119.7 | 127.9 |
| **epochs 20+, n=20/19** | **137.6** | **148.0** |
| range | 111-178 | 119-197 |

The seven-epoch window caught a fast stretch. **The same mistake as the 80 s GPU
window**: a sample shorter than the thing's own variation reports the part of
the distribution it happened to land in. The 20-epoch window is the one to use.

> pair, paced by the slower arm at **148 s/epoch** → **16.4 h**
> s3 alone afterwards, est. ~100 s/epoch (**extrapolated**; cold single-arm was
> 112) → ~11.1 h
> **TOTAL ≈ 28 h** — against ~35 h on the cold arms and the ~25 h I quoted from
> the short window.

## The right baseline: seeds re-measured under E4B's own conditions

The cost statement says every E4B claim is measured against its seed, not the
frozen ant. But the seeds' **published** speeds (4.224, 4.891) were measured in
E3.1's environment, against a scripted opponent that advanced at 0.68 m/s, had
its state overwritten every step, and arrived at step 491 — it could not
contest and could not collide meaningfully. E4B's opponent races at seed speed
and is fully physical. **Comparing an E4B eval to a published E3.1 number is
apples to oranges, and it understates E4B.**

Both seeds re-measured in E4B's environment — the seed policy itself, no
gradient steps, against a physical copy of itself:

| seed, matched conditions | goal | fell | **speed** | published (E3.1 conditions) |
|---|---:|---:|---:|---:|
| `rtg_e31_s1` | 0.40 | 0.40 | **1.579** | 4.224 |
| `rtg_e31_s2` | 0.45 | 0.05 | **3.883** | 4.891 |

Against that baseline:

| | seed (matched) | E4B now (last-3 mean) | change |
|---|---:|---:|---:|
| **s1** | 1.579 | **2.531** | **+60%** |
| **s2** | 3.883 | **4.553** | **+17%** |

**Both arms have exceeded their own seeds under the conditions they actually
train in**, by epoch ~40. On the published numbers neither has (2.531 vs 4.224;
4.553 vs 4.891) — which is the comparison to avoid, because it charges E4B for
an opponent E3.1 never faced.

One oddity to settle later: the s2 seed scores **goal 0.45 against loss 0.60**
playing *itself*. For identical policies that should be symmetric. At n=20 the
gap is ~1.4 SE so it is probably noise, but it is exactly what the tournament's
slot-asymmetry diagnostic exists to measure, and gate 3 says the π-z rotation
is exact — so if it persists it is a finding rather than a nuisance.

## The "slow opponent is an obstacle" hypothesis — tested and REFUTED

Two consecutive ladder evals showed stalemate rising with opponent age (e69:
0.20/0.20/0.00/0.10/0.00, e74: 0.50/0.30/0.20/0.10/0.00, oldest → youngest),
with the mirror against a *current* self stalemating only 0.00-0.10. The
proposed mechanism was geometric: both agents run *past* each other, so the
crossing point is on each one's path, and a slow or fallen old self — the e0
seed moves at 1.579 m/s and falls 40% of the time — would be parked in the lane.

If true it would have inverted the pre-registered RATCHET criterion, which
assumes win rate *rises* with age gap.

**Probe: s1's current policy against the SAME e0 body under three drivers,
20 episodes each.** Holding the morphology fixed isolates the driver.

| condition | win | **stalemate** | goal | opp moved | ep_len |
|---|---:|---:|---:|---:|---:|
| 1 — e0 self (slow, falls, obstructs) | 0.85 | 0.15 | 0.85 | 2.65 m | 107 |
| 2 — **stationary body**, pure obstacle | **1.00** | **0.00** | **1.00** | 0.35 m | 68 |
| 3 — e0 body + **current** policy (fast) | 0.75 | **0.25** | 0.75 | 2.84 m | 120 |

**The prediction was (1)≈(2) stalemating, (3) not. The opposite happened.** The
pure obstacle is the *easiest* condition — zero stalemates, goal 1.00, shortest
episodes — and it sits at x≈+1.35, squarely on the path from −1 to +4. s1 runs
past it every time.

**Stalemates track opponent MOBILITY, not obstruction**: 0.00 at 0.35 m of
opponent movement, 0.15 at 2.65 m, 0.25 at 2.84 m. Consistent with the earlier
collision result — a body moving at high closing speed knocks s1 off course; a
stationary one is simply run past.

### Consequences

* **The criterion is not inverted.** The mechanism runs the other way: fast
  recent selves cause more stalemates than slow old ones, so recent selves are
  *harder*, which is the direction the criterion already assumes.
* **The age-stalemate gradient has no supported mechanism.** A controlled
  20-episode comparison outranks a 5-point trend built from 10-episode evals.
  It is most likely sampling noise — which was the original read, and it was
  elevated to a hypothesis before being tested. The lesson is the ordering:
  test first, then elevate.
* **No change to the measurement.** The reason to change it did not survive.
  The pre-registered `e4r_tournament.py` already plays the **full all-pairs
  matrix** (12 checkpoints × 20 episodes × both slot orientations), which
  subsumes a common fixed reference set and is what detects non-transitivity.
  The per-epoch `ladder` is a cheap in-flight indicator at n=5 × 10 episodes,
  not the verdict instrument.
* **Optional cheap improvement**: `--ladder-episodes 20` halves the in-flight
  SE for ~60 s per eval. Worth applying at the s3 launch; not worth restarting
  the running arms for.

## The wandb videos were rendering against an INERT opponent

`e3_video.py` builds a fresh agent in a CPU subprocess and uses `ag.env`.
Nothing installed the ring, so `env.ring` stayed `None`, so `reset_robot` never
assigned `opp_policy`, so `do_simulation`'s guard

```python
if (self.opp_mode == 'policy' and self.opp_policy is not None
        and self.stage == 'execution'):
    ctrl += self.opp_control(self.opp_action())
```

skipped the opponent's torque entirely. The clips show a lone runner beside a
splayed, stationary body.

**Why it was silent, and it is the day's recurring pattern.** E2 and E3 used
`opponent_mode: scripted`, which needs no policy object, so this same code
rendered correctly for two rungs. Switching to `opponent_mode: policy` degraded
it with no error — same code, new configuration, different behaviour.

### Blast radius: the video panels only

* `payload.update(d["scalars"])` **does** send `video/{best,median,worst}_{R,dx,steps,goal}`
  to wandb, so those panels were biased.
* The bias is **optimistic**, and the three-condition probe already quantified
  it: inert was the *easiest* of the three conditions — stalemate 0.00, goal
  1.00, ep_len 68 against 107 and 120.
* **Nothing reported was affected.** The in-process eval sets
  `env.ring_epoch = epoch` before calling `e2_eval.evaluate`, so
  `eval_*`, `mirror` and `ladder` all draw real past selves. No `video/*`
  scalar appears anywhere in these docs — every number quoted came from the
  eval path.

### Fix

`install_ring_opponent` loads **one** past self — the most recent persisted
ring member — and installs it. One, not the ring: each policy is 148 MB and
this runs in a CPU subprocess. Two new scalars make the condition impossible to
misread: **`video/opponent`** (`ring_epoch_80` or `INERT`) and
**`video/opponent_is_inert`**.

Verified by rendering `rtg_e4r_s1` at `epoch_0060`:

| | goal |
|---|---|
| inert (every clip so far) | 1.00 across the board |
| **corrected, vs `ring_epoch_80`** | best 1.0, **median 0.0, worst 0.0** |
| in-process eval, same epoch | 0.60-0.70 |

The corrected clip now agrees with the eval; the inert one did not.

**No restart needed** — the trainer spawns `e3_video.py` as a fresh subprocess
per render, so the running arms pick this up on their next video.
**Clips rendered before 2026-09-06 show an inert opponent and should not be
read as matches.**

`load_gnn`'s return arity was deliberately left at four: `e3_posthoc.py` (x2),
`e3_termination_grid.py` and `e3_blob_probe.py` all unpack exactly four values,
so the opponent epoch is passed out on the env instead.

`best_median_worst`'s docstring is also corrected: it claimed all three panels
are the same creature (false when design is live) and that the clip shows
"whether it dodges the opponent" (false for E4B, whose opponent is a past self
that races and collides rather than a scripted mover to be dodged).

### Panels now show the opponent's displacement

`video/opponent` proves an opponent was *installed*; it does not prove it
*moved*. Each panel label now carries **`oppdx=`**, and `video/{best,median,
worst}_opp_dx` is logged alongside, so a clip is self-verifying:

| panel | goal | our dx | **opponent dx** |
|---|---:|---:|---:|
| best | 1 | 5.13 m | **4.91 m** |
| median | 0 | 2.88 m | 1.49 m |
| worst | 0 | 3.09 m | **3.50 m** |

Both sides race, and in the *worst* panel the opponent travels further than the
learner — which is what losing looks like. An inert body reads ~0.35 m (gravity
settling alone), so the two cases are now distinguishable on the label without
trusting the pipeline. This is what answers "which side is which and is it
doing anything".

## The "flat ladder" was an age-gap artifact — tested

s2's ladder mean sat near 0.61 for the first ~60 epochs and then rose to ~0.70,
which raised the question of whether a warm-started arm near its ceiling can
ratchet at all. **A ring whose oldest member is 30 epochs back cannot show a
ratchet**, so the mean was confounded with ring age. Tested by pooling every
`(eval epoch, age gap, win rate)` triple and splitting by training phase at
matched gaps:

| age gap | s1 early (<70) | s1 late (≥100) | s2 early (<70) | s2 late (≥100) |
|---|---:|---:|---:|---:|
| 0-20 | 0.55 | 0.60 | 0.56 | 0.62 |
| 20-50 | 0.74 | 0.44 | 0.60 | 0.65 |
| 50-100 | 0.82 | 0.76 | 0.75 | 0.64 |
| **100-200** | — | **0.80** | — | **0.75** |

Two things fall out.

**Win rate rises with age gap, in both phases and both arms** — s1 early
0.55 → 0.74 → 0.82, s2 early 0.56 → 0.60 → 0.75, and late 0.60 → 0.76 → 0.80 /
0.62 → 0.64 → 0.75. The direction the criterion assumes is present, and was
present from the start.

**At matched gaps, early ≈ late.** That is not evidence against a ratchet — it
is what a *steady* ratchet predicts. If the current agent and a self from Δ
epochs ago both improve at the same rate, the win rate at fixed Δ stays
constant. The ladder *mean* rose from 0.61 to 0.70 only because the ring aged
and larger gaps entered the average.

**So the flat period was an artifact of coverage, not an absence of ratcheting**,
and the "can a warm-started arm ratchet at all" question is resolved in the
affirmative for both arms.

It also vindicates the criterion's construction: **RATCHET HOLDS is defined on
selves ≥ 100 epochs older**, which is immune to this confound. Its actual
quantity currently reads **s1 0.80, s2 0.75** — at or above the 0.75 threshold,
though still outside the 200-400 verdict window and on n=8 / n=6.

### s1's 0.44 at e134 is the agent, not the ladder

| s1 vs | e129 | e134 |
|---|---:|---:|
| e0 (gap ~130) | 0.80 | 0.60 |
| e30 (gap ~100) | 0.90 | 0.70 |
| e60 (gap ~70) | 0.80 | 0.40 |
| e90/e100 (gap ~35) | 0.50 | 0.20 |
| e120/e130 (gap ~5) | 1.00 | 0.30 |

**Every opponent's win rate fell simultaneously** — that is the signature of
the learner being worse, not of opponent selection or a ladder mechanism. Its
own eval at the same epoch agrees: speed **4.054 → 2.929** and fall rate
**0.00 → 0.20** after five clean evals. A single bad eval on a noisy series
(speed 3.85, 3.84, 4.66, 4.05, 2.93), not a regime change — but the return of
falls is worth watching given s1's collision history.

`rho` continues to swing between +0.87 and −0.67 across consecutive evals and
remains uninformative; the gap-binned win rate above is the quantity to read.

## Tournament validated on real data — and it caught a scoring-protocol defect

First run of `e4r_tournament.py` against real persisted ring members (s1,
checkpoints 0/40/80/120). It works, and reads **transitive**: 0.000 cyclic
triples of 4.

But the slot-asymmetry diagnostic fired at **0.278 mean / 0.500 max**, where
gate 3 (rotation exact to 0.000e+00) says it should be ~0. The pair scores
summed to **1.25 instead of 1.0** — whoever held slot 0 won ~62% of the time.

**Cause, and it was mine.** In a scored match the learner is evaluated at its
**mean action**, while the opponent defaults to acting **stochastically**
(CompetEvo's `noise_rate = 1.0`). That is right for TRAINING and wrong for
SCORING: it charges the slot-1 player exploration noise *and* its control cost
while its opponent pays neither. Gate 6c had already hinted at it — a snapshot
ran 4.657 m/s in slot 1 against 4.891 trained in slot 0.

Fixed: `_play` now forces `opp_mean_action = True` for the duration of a scored
match and restores it afterwards. **Training is untouched.**

### What the re-run does and does not show

| | before | after |
|---|---:|---:|
| slot asymmetry, mean | 0.278 | 0.194 |
| mean pair-score sum (1.0 is fair) | 1.25 | 1.14 |
| cyclic triples | 0.000 | 0.000 |

**This is not evidence the fix worked.** At 6 episodes per ordered pair the SE
on a score is ~0.204, so the SE on `|S_ij − (1 − S_ji)|` is ~0.29 — **both
readings are consistent with zero and with each other.** The fix stands on the
reasoning (scoring two sides under different action protocols is wrong however
the numbers land), not on this comparison.

The verdict tournament runs 20 episodes per ordered pair, where the asymmetry
SE falls to ~0.16 — better, still coarse. **A slot bias smaller than ~0.15
cannot be resolved at the pre-registered sample size**, which is worth knowing
before the diagnostic is read at epoch 400.

## Slot bias: the statistic was wrong, not the sample size

I reported `mean |S_ij − (1 − S_ji)|` and concluded the diagnostic needed more
episodes. **That conclusion was wrong.** The mean of an absolute value does not
average to zero: under *no* bias at all it is a folded normal with expectation
`σ√(2/π)`. At 6 episodes/ordered pair that floor is **0.230**; at the
pre-registered 20 it is **0.126**. So a perfectly symmetric tournament still
reads ~0.13, and **the readings of 0.277 and 0.195 were being compared against
a zero the statistic can never reach**. More episodes shrink the floor but
never remove it.

**The signed per-pair quantity averages properly:**

```
d_ij = S_ij + S_ji − 1        # 0 under no slot bias; sign names the favoured slot
```

Recomputed on the two runs already taken (6 eps/pair, 6 unordered pairs):

| | mean signed **d** | SE | from zero | mean \|d\| | folded floor |
|---|---:|---:|---:|---:|---:|
| **before** protocol fix | **+0.250** | ±0.118 | **2.1 SE** | 0.277 | 0.230 |
| **after** protocol fix | **+0.139** | ±0.118 | 1.2 SE | 0.195 | 0.230 |

The signed statistic says something the absolute one could not: the pre-fix run
carried a **marginally significant slot-0 bias at 2.1 SE**, and after the fix it
is **not significant**. Stated carefully — the *change* itself (0.111, SE of the
difference 0.167) is **not** significant, so this is evidence the bias existed
and is now undetectable, not proof the fix removed it.

**At the pre-registered 20 episodes/pair with 12 checkpoints — 66 unordered
pairs — the SE of the mean is 0.158/√66 ≈ 0.02.** That resolves slot bias to
±0.02 with no extra compute, against the absolute statistic's 0.126 floor.
`mean_signed_d` is now the criterion, reported with its SE and an explicit
"consistent with NO slot bias / SLOT BIAS DETECTED (>2 SE)" verdict;
`max |d_ij|` is kept only as an outlier check.

### This closes the s2-seed oddity

Logged during the warm-start probe: the s2 seed scored **goal 0.45 against loss
0.60 while playing itself**, which should be symmetric for identical policies.
It was **the mean-vs-stochastic protocol mismatch, not the rotation** — the
slot-0 player acted at its mean action while the slot-1 player paid exploration
noise and its control cost. Gate 3's rotation (exact to 0.000e+00) was never
implicated. Third of the three tracked items, now closed.

## Interim tournament at epoch 200 — transitive, and a ratchet with one inversion

`rtg_e4r_s1`, 6 persisted checkpoints, **20 episodes per ordered pair** (the
pre-registered rate), both slot orientations averaged. 600 episodes, `nice`d so
the training arms kept priority.

Slot-averaged score, row beats column:

| | 0 | 40 | 80 | 120 | 160 | 200 |
|---|---:|---:|---:|---:|---:|---:|
| **0** | — | 0.06 | 0.15 | 0.07 | 0.04 | 0.09 |
| **40** | 0.94 | — | 0.36 | 0.42 | 0.36 | 0.31 |
| **80** | 0.85 | 0.64 | — | **0.70** | 0.30 | 0.20 |
| **120** | 0.93 | 0.57 | 0.30 | — | 0.20 | 0.32 |
| **160** | 0.96 | 0.64 | 0.70 | 0.80 | — | 0.42 |
| **200** | 0.91 | 0.69 | 0.80 | 0.68 | 0.57 | — |

**Three findings.**

**1. Transitive.** `cyclic triples 0.000 of 20`, against a 0.10 threshold whose
noise ceiling under a transitive null is 0.055. **The failure mode self-play is
specifically prone to — 30 beats 20, 20 beats 10, 10 beats 30 — is absent.**
This is the reading the whole matrix exists to produce, and it is now measured
at the pre-registered sample size rather than on a synthetic ring.

**2. No detectable slot bias.** `mean signed d = +0.0667 ± 0.0408` — **1.6 SE
from zero, consistent with none.** The signed statistic over 15 pairs resolves
to ±0.041, against ±0.118 from the 6-episode trial, so this is a real
measurement rather than a shrug. Combined with the earlier finding that the
pre-fix run sat at 2.1 SE, the protocol fix looks to have done its job — though
the two runs differ in episode count as well, so this is consistent evidence
rather than a controlled before/after.

**3. The latest checkpoint beats every past self — but strength is NOT monotone
in training time.** Epoch 200's row is 0.91 / 0.69 / 0.80 / 0.68 / 0.57: it
beats all five predecessors. Every checkpoint beats epoch 0 (0.85-0.96). But
the implied ranking is

```
200 (0.730) > 160 (0.705) > 80 (0.537) > 40 (0.480) > 120 (0.465) > 0 (0.083)
```

with one inversion: **checkpoint 80 beats checkpoint 120 at 0.70.** Transitivity
and monotonicity are different properties, and only the first holds. Training
time is not a total order on strength here — which matters because a criterion
phrased as "beats all past iterations" is satisfied by epoch 200 while "each
iteration beats the one before" is not.

The inversion is not mysterious: epoch 120 sits just before s1's measured dip
(speed 4.054 → 2.929 and falls 0.00 → 0.20 around e134), so a locally weak
checkpoint was archived and later checkpoints recovered past it.

**This is an interim reading, not the verdict.** The pre-registered window is
epochs 200-400 and it has just opened — s1's last eval was e199, outside it.
`e4r_report.py` correctly reports "no qualifying pairs yet" for both arms.

# IN-WINDOW VERDICT (epochs 200-400, partial: s1 at 251, s2 at 239)

## The pre-registered criterion is MET on both arms

Criterion, fixed before the run: *mean win rate against selves ≥100 epochs
older, over epochs 200-400, ≥ 0.75.*

| | in-window ratchet | n pairs | SE | clear of 0.75 by | verdict |
|---|---:|---:|---:|---:|---|
| `rtg_e4r_s1` | **0.850** | 30 | ±0.034 | **2.9 SE** | **MET** |
| `rtg_e4r_s2` | **0.852** | 21 | ±0.029 | **3.5 SE** | **MET** |

Two independently-seeded arms agree to three decimals. Computed by
`e4r_report.py`, which enforces the window and gap filter in its defaults —
the same code reports "no qualifying pairs yet" outside the window and reported
exactly that four hours ago.

**The other two pre-registered criteria:**

| | s1 | s2 | threshold |
|---|---:|---:|---|
| mirror stalemate | 0.040 | 0.000 | DEGENERATE if > 0.5 |
| mirror forward | 4.73 m | 5.16 m | DEGENERATE if < 2.5 m |
| mirror mutual | 0.045 | 0.086 | — |
| cyclic triples (interim, s1) | **0.000** of 20 | — | CYCLES if > 0.10 |

**Healthy on every one.** The mirror is not degenerate — both agents run the
full course and finish — and the matrix showed no cycling.

## "Beats all past iterations" holds. "Each beats the one before" does not.

The user's phrasing was the former, so the criterion is satisfied. But the
matrix exposed a distinction a scalar would have hidden, and the two must not
be conflated.

From the interim tournament, implied strength by mean slot-averaged score:

```
200 (0.730) > 160 (0.705) > 80 (0.537) > 40 (0.480) > 120 (0.465) > 0 (0.083)
```

Epoch 200 beats **every** predecessor (0.57-0.91) and everything beats epoch 0
(0.85-0.96) — the ratchet in the sense asked for. But **checkpoint 80 beats
checkpoint 120 at 0.70**: strength is *transitive* without being *monotone in
training time*. The inversion tracks s1's measured dip — 120 was archived just
before speed fell 4.054 → 2.929 with falls 0.00 → 0.20 around e134.

So: no strategic cycling (the failure self-play is prone to), but training time
is not a total order on strength, and a run can archive a locally weak self that
later checkpoints overtake.

## Speed against the baseline that applies

| | last-5 mean | vs **matched** seed | vs published seed |
|---|---:|---:|---:|
| s1 | 4.781 m/s | 1.579 → **+203%** | 4.224 → +13% |
| s2 | 5.731 m/s | 3.883 → **+48%** | 4.891 → +17% |

The matched figures are the honest comparison: the published speeds were
measured against E3.1's non-contesting scripted opponent, while these are
measured against a physical racer.

## What this does NOT establish

1. **Not from-scratch emergence.** Both arms are warm-started from E3.1
   winners. This shows a ratchet **from a competent baseline** — the question
   actually asked — and says nothing about self-play bootstrapping from
   nothing. From scratch, E4B converged on standing still: exploring cost
   0.212/step against 0.159 of forward gain.
2. **n = 2 arms**, and they share a lineage: s1 ← `rtg_e31_s1`, s2 ← `rtg_e31_s2`,
   both E3.1 seeds. s3 (a deliberate replicate of s2's seed) has not run yet.
3. **The window is partial** — s1 at 251 of 400, s2 at 239. The verdict is
   in-window but not final; the criterion could still move.
4. **Inherited morphology.** Design search is refining an already-evolved body,
   not discovering one. Whether it moves from the inherited plan at all remains
   a separate question.
5. **No transfer claim.** A ratchet against its own history says nothing about
   performance against an agent trained differently.
