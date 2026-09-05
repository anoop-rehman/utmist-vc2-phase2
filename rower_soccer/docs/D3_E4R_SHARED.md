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
