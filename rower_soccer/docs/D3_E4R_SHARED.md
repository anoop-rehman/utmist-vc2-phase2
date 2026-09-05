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
