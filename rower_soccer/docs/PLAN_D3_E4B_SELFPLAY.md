# D3 E4B — Shared-weight self-play: plan and literature review

*Written 2026-09-05. **Nothing has been launched.** This is a plan for review.*

---

## 0. WARM START — what this changes about the experiment

**Added 2026-09-06, before the seeded arms ran.** E4B from scratch converged on
standing still. Measured: exploring costs **0.212/step** in control against
**0.159/step** of forward gain, so the gradient shrinks the action and the mean
action collapsed to **0.0006** — all visible motion was exploration noise. E3.1
escaped that only because its scripted opponent scored in every episode and
**shoved the ant backwards** (Σforward **−169** against E4B's **+6**). Two
immobile past selves supply neither pressure, so standing still earns ~99% of
the achievable return.

Both the learner **and** the ring's epoch-0 member are therefore initialised
from a solved E3.1 checkpoint. Seeding only the ring would hand a random
learner an unbeatable opponent — Bansal's documented failure mode verbatim.
AlphaStar seeds its league from supervised policies for the same reason.

**Three things this costs, stated here rather than in a footnote:**

1. **This no longer tests from-scratch emergence.** It tests whether self-play
   **ratchets from a competent baseline**. That is the question actually asked
   — "each new iteration should beat all past iterations" — but it is *not* the
   question the original E4 framing implied, and the two must not be conflated
   later.
2. **The morphology is inherited, not evolved by E4B.** The design search now
   *refines an already-evolved body*. Whether it moves from the inherited plan
   at all is itself a result worth reporting, and if it does not, E4B has
   measured control improvement only.
3. **The baseline moves.** E3.1's bodies are now the *starting point*, so
   "beats the frozen ant by 2.8-3.3x" is no longer the relevant comparison.
   **The baseline for every E4B claim is its own seed checkpoint** — 4.891 m/s
   for s2/s3, 4.224 m/s for s1. A ratchet must beat *that*, not the ant.

### Which checkpoints, and why not three identical ones

Only **two** design-ON competent checkpoints exist. `rtg_e31d_s3body` reached
goal 1.00 at 3.006 m/s but ran with **`force_identity_design: true`**, so its
design head was never trained — it cannot seed a design-ON run.

| arm | seed | speed | body |
|---|---|---:|---|
| `rtg_e4r_s1` | `rtg_e31_s1/models/epoch_0400.p` | 4.224 m/s | 17b/8m, topo `2b3b3b54a170` |
| `rtg_e4r_s2` | `rtg_e31_s2/models/epoch_0400.p` | 4.891 m/s | 18b/6m, topo `50271e7f5d26` |
| `rtg_e4r_s3` | `rtg_e31_s2/models/epoch_0400.p` | 4.891 m/s | *replicate of s2* |

Two distinct morphologies plus one deliberate replicate. The replicate is not
waste: **s2 vs s3 differ only in RNG, while s1 vs s2 differ in RNG *and*
starting body**, so the pair decomposes seed variation into its two sources.
Three identical seeds would have given only the first; three distinct ones were
not available.

`epoch_0400.p` is used rather than `best.p` (epoch 371) because it has a
directly measured eval — goal 1.00, 4.891 m/s, 5.58 m — and is the checkpoint
whose numbers are already published. Its own best eval was 5.062 m/s at epoch
344, for which no checkpoint file exists.

### Verified before running, two-sided

A failed load and a fresh init both print "loaded" and both stand still, so the
gate measures a cold agent beside a warm one:

| | warm | cold |
|---|---:|---:|
| policy hash vs checkpoint | **identical** | differs |
| mean \|action\| | **0.4675** | 0.000482 |
| speed | **4.974 m/s** (2% off published) | 0.014 m/s |
| goal rate | **1.00** | 0.00 |

W4 is the negative control: the cold arm fails the speed test, so passing it is
evidence rather than a formality.

---

## 1. What was asked for

> Both creatures in the 1v1 share one body and one brain, optimised by
> self-play. Each new iteration should **beat all past iterations 1v1**, and be
> **basically tied against its current iteration**.

One agent — one design head, one controller — plays both sides of the
run-to-goal match. Success is a *ratchet*: today's agent beats every version of
itself that came before, and fights its own current version to a standstill.

That is the whole specification, and it turns out to be almost exactly the
criterion the foundational paper in this area wrote down in 2018. More on that
in §2.

---

## 2. What the literature says, and where we already agree by accident

### 2.1 Bansal et al. 2018 — the direct ancestor

[*Emergent Complexity via Multi-Agent Competition*](https://arxiv.org/abs/1710.03748)
is competitive self-play on MuJoCo bodies — the closest published setting to
ours, and the direct ancestor of CompetEvo, whose code we ported.

Their opponent-sampling rule: draw the opponent uniformly from
**`Uniform(δ·v, v)`**, where `v` is the current iteration and `δ ∈ [0,1]`.

* `δ = 1.0` → only the latest opponent
* `δ = 0.0` → uniform over the **entire history**

They tested `δ ∈ {1.0, 0.8, 0.5, 0.0}`. The result that matters for us:

> **`δ = 0.0` (the whole history) had the highest win rate for Ant.**
> `δ = 0.5` was best for Humanoid.

**Our task is an ant.** So the published evidence points directly at
whole-history sampling for our morphology.

Their justification for not simply using the latest opponent:

> *"training agents against the most recent opponent leads to imbalance in
> training where one agent becomes more skilled than the other agent early in
> training and the other agent is unable to recover."*

and their solution:

> *"we found that training against random old versions of the opponent to work
> much better [...] the policy at any time should be able to defeat random
> older versions of itself, thus ensuring continual learning."*

**That last sentence is the user's success criterion, stated as a design goal
eight years ago.** The request and the standard solution converged
independently. That is a good sign, and it means we are not inventing a
scheme — we are adopting a validated one.

### 2.2 AlphaZero — the cheapest option, and what assuming it would cost

AlphaZero trains against its own current self, with no archive at all. It works
because Go and chess are **largely transitive**: if A beats B and B beats C,
then A almost always beats C, so "get better" is a well-defined direction and
the latest agent is genuinely the strongest.

Using latest-only here would be assuming run-to-goal is transitive too. We have
no evidence for that, and §2.4 explains why symmetric physical contests are
exactly where the assumption tends to fail. It is the cheapest option and we
are explicitly *not* taking it — but we will **measure** whether the assumption
would have held (§4), so the question gets answered rather than assumed.

### 2.3 AlphaStar — PFSP and why a league exists at all

[Vinyals et al. 2019](https://www.nature.com/articles/s41586-019-1724-z) uses
**Prioritised Fictitious Self-Play**: opponents are sampled from the archive
with probability weighted by how well they do against the current agent, via
`f(x) = (1−x)^p` on the win rate `x` — so opponents that *beat* you are drawn
more often. Variants like `f(x) = x(1−x)` prefer evenly matched opponents.

The important structural point is *why* the league has main agents, **main
exploiters** and **league exploiters** at all: naive self-play in StarCraft
**cycles**. The exploiters exist to find and punish the systematic weaknesses
that a plain ratchet quietly accumulates. The league is an engineering response
to non-transitivity, which is the same thing our matrix in §4 is built to
detect.

### 2.4 Balduzzi et al. 2019 — the formal version of the concern

[*Open-ended Learning in Symmetric Zero-sum Games*](https://arxiv.org/abs/1901.08106)
states the problem cleanly:

> *"If a game is approximately transitive, self-play generates sequences of
> agents of increasing strength, but nontransitive games such as
> rock-paper-scissors can exhibit strategic cycles, making it unclear what the
> learning objective should be."*

In a non-transitive game **"better" is not well defined**, and a scalar score
cannot tell you whether you are climbing or going in a circle. This is the
formal justification for reporting a **matrix**, not a number.

### 2.5 CompetEvo and Transform2Act — read from the code, not the abstract

From our own port (`rower_soccer/competevo_port/selfplay.py`, transcribing
`runner/multi_evo_agent_runner.py:190-225`):

```
start = max(1, floor(delta * epoch));  end = epoch
ckpt  = randomstate.randint(start, end)      # HIGH-EXCLUSIVE
```

so CompetEvo implements **Bansal's rule exactly**, with the same `δ` semantics
(a *window*, not a mixing probability). Their fixed-morph ants used `δ = 0`;
`δ = 0.5` was their dev setting. Two details D2 verified against the code, one
of which corrected D2's own port map:

* the checkpoint is redrawn **every episode**, not once per worker-batch;
* the opponent acts **stochastically** (`noise_rate = 1.0`,
  `base_runner.py:27`), not at its mean action.

**Transform2Act itself has no self-play at all** — it is single-agent
morphology-plus-control learning. Everything competitive here comes from the
CompetEvo side. And one property of Transform2Act's own architecture shapes
this plan (see §7): its **design head is blind** — the skeleton and attribute
stages read only `attr_fixed ++ attr_design` and never see simulation state.
We measured this: moving the opponent 4 m changes the design head's input by
**0.000e+00**.

---

## 3. The opponent-sampling scheme, and the tradeoff

**Chosen: δ-uniform over the whole history (`δ = 0`), redrawn every episode.**

| scheme | what it buys | what it costs | verdict |
|---|---|---|---|
| **Latest-only** (AlphaZero) | cheapest; no archive | assumes transitivity; Bansal measured it causing runaway imbalance | rejected |
| **δ-uniform, δ=0** | Bansal's best setting **for Ant**; matches CompetEvo; directly serves "beat *all* past selves" | some compute spent on opponents that are far too weak to be informative | **chosen** |
| **δ-uniform, δ=0.5** | concentrates on recent, stronger opponents | drops exactly the old opponents the user's criterion asks about | rejected |
| **PFSP** (AlphaStar) | focuses on opponents that beat us — sample-efficient | needs a maintained win-rate estimate per opponent; more machinery; tuned for a league we do not have | not now; see below |
| **Full league** | strongest known defence against cycling | several agent types, many more arms; far beyond our compute | out of scope |

**The tradeoff, named plainly:** `δ = 0` spends real compute playing opponents
so weak they teach us little — late in training most of the archive is easy.
PFSP is the standard fix. We are not taking it because it adds machinery whose
benefit we cannot yet verify, and because **the user's criterion is explicitly
about the whole history**, which `δ = 0` samples natively.

**If the ratchet holds but learning is slow, PFSP is the first upgrade** — it
is a change to one sampling function (`OpponentRing.sample_epoch`) and nothing
else, and the win-rate matrix we already produce is exactly the input PFSP
needs.

### Why the opponent is never the *current* self

This is the design point that decides whether the whole thing works.

The two halves of the criterion **conflict at equilibrium**. If the training
opponent were the current self, then once they are evenly matched both reach
the line on the same step, and `run_to_goal.py` scores:

```python
n_reached = int(reached) + int(opp_reached)
parse = 0.0
if n_reached == 1:
    parse = GOAL_REWARD if reached else -GOAL_REWARD    # ±1000
```

`parse` is **zero whenever the match is tied**. The competitive part of the
reward **switches itself off exactly at the point the criterion asks us to
reach**. At our curriculum's `alpha = 0.847` that term is worth **306 points
against a dense component of 376** — 58% of the weighted training return, so
this is not a rounding error.

So: the training opponent is always a **strictly past** self, where winning
still pays. The mirror match is an **evaluation only**, never a gradient.

---

## 4. The measurement

### 4.1 A win-rate matrix, not a score

Every archived checkpoint plays every other. A healthy ratchet is a
**triangular** matrix. A cycle — 30 beats 20, 20 beats 10, 10 beats 30 — is
invisible to any scalar and obvious in a matrix.

* **12 checkpoints × 20 episodes per ordered pair.**
* **Both slot orientations, averaged.** The learner always trains in slot 0, so
  a slot advantage would otherwise masquerade as skill. The orientation gap is
  reported as a diagnostic: our rotation gate says the two slots are exactly
  equivalent (observation max|Δ| = **0.000e+00**), so it should be ~0, and if
  it is not, that is a finding.

### 4.2 The mirror match — and the trap that would fake success

**This is the most likely way to get a false success, so it gets its own
guard.** A 0-0 stalemate and a 1-1 photo finish both read as "tied" on any
scalar, and `run_to_goal` scores `parse = 0` for both. One is the goal; the
other is two creatures standing still.

The mirror match therefore reports **three mutually exclusive outcomes plus
distance travelled**:

| outcome | meaning | `parse` |
|---|---|---:|
| **DECISIVE** | exactly one reached the line | ±1000 |
| **MUTUAL** | both reached, same step — **the good tie** | 0 |
| **STALEMATE** | neither reached; timeout — **degenerate** | 0 |

**This guard has already fired.** In a 12-epoch pipeline test, at epoch 3 the
untrained agent scored **stalemate 1.00, forward 0.14 m** — correctly reported
as degenerate. A naive draw-rate metric would have reported "draw rate 1.00"
and looked like a perfectly matched equilibrium at epoch 3.

---

## 5. Pre-registered outcomes

Fixed before launch. All are **trajectory** criteria over epochs **200-400**,
aggregated before comparing — on this project an endpoint has inverted the
conclusion three separate times.

### Outcome 1 — RATCHET HOLDS
* mean win rate against selves **≥100 epochs older is ≥ 0.75**, **and**
* **Spearman ρ(age gap, win rate) ≥ +0.5**, pooled across seeds.

Grounded in our own data rather than invented: E3.1's goal rate is 0.00 through
epoch 100 and 0.98 after 300, so a competent late agent should beat a
100-epoch-old self nearly always. At 20 episodes the binomial SE at p = 0.75 is
0.097, so 0.75 sits ~2.6 SE above a coin flip.

### Outcome 2 — RATCHET FAILS or CYCLES
* **cyclic-triple fraction > 0.10**, **or**
* mean win rate vs ≥100-epoch-older selves **< 0.6**, **or** ρ ≤ 0.

**0.10 is calibrated by simulation, not chosen.** For a *perfectly transitive*
ladder of 12 players at 20 episodes per pair, with adjacent pairs at a
near-tied 0.55 — the hardest case for noise — the cyclic-triple fraction comes
out at:

| | mean | p95 | p99 |
|---|---:|---:|---:|
| transitive null | 0.013 | 0.036 | **0.055** |

and a tournament with **no real ordering at all** gives **0.136**. So 0.10 is
~1.8× the noise ceiling and comfortably below chance.

### Outcome 3 — DEGENERATE MIRROR
* **stalemate rate > 0.5**, **or**
* mean forward progress in the mirror match **< 2.5 m**.

The course is 5.0 m (x = −1 → +4). E3.1's solving arms covered 5.1-5.25 m; its
non-solving arms covered 0.11-0.65 m. 2.5 m separates those regimes with a wide
gap either side.

**HEALTHY EQUILIBRIUM** = high mutual rate, stalemate ≈ 0, forward ≥ 2.5 m:
both creatures run the full course and arrive together.

---

## 6. PBT: **recommended against**, with the arithmetic

PBT ([Jaderberg et al. 2017](https://arxiv.org/abs/1711.09846)) is usually sold
as hyperparameter search, which is not our problem. Its **exploit** step is,
though: periodically replace the worst population members with perturbed copies
of the best. That is a direct answer to something we actually measured — a
**~1-in-3 controller failure rate**. E3.1's seed 3 scored goal 0.00, and the
frozen-body diagnostic later proved its *body* was fine (goal 1.00 at 3.006 m/s
with a fresh controller). It was a bad controller draw, and PBT's exploit step
is exactly the mechanism that would have culled it.

**It still does not fit, for two independent reasons.**

**(a) The population would be far below the size PBT needs.** The paper's own
population sizes: **DeepMind Lab 40, Atari 80, StarCraft II 30, translation 32,
GAN 45.** Its exploit step is truncation selection — *"if the current agent is
in the bottom 20% of the population, we sample another agent uniformly from the
top 20%"*. And its own ablation:

> *"if the population size is too small (10 or below) we tend to encounter
> higher variance and can suffer from poorer results ... a population size of
> between 20 and 40 is sufficient to see strong and consistent improvements."*

At a population of 3, "bottom 20%" and "top 20%" are each **0.6 of a member**.
The step degenerates into "copy the best onto the worst" decided by a
20-episode evaluation — and with a 1-in-3 failure rate and that much evaluation
noise, it is roughly a coin flip whether you cull a genuine failure or a
perfectly good run having a bad week. **At this size the exploit step amplifies
noise rather than removing it.**

**(b) The compute is short by 5-7×.** A viable population of 20, at our
measured **112 s/epoch** and 400 epochs, is **249 hours of arm-time**. The
cgroup grants **10.2 CPUs** and the card **20 GB**; we fit about **3 arms at
once** (~5 GB each). That is ~83 h wall clock at best — against ~14 h for the
plan in §7 — and the GPU alone would need ~100 GB to hold 20 arms.

**(c) It would collide with the seed count, not complement it.** We have room
for ~3 concurrent arms. Spending them on PBT members leaves **one seed**.
Seeds and population members are not interchangeable: seeds are *independent*
estimates, PBT members are *coupled* by copying. We would lose the ability to
distinguish "the method works" from "one lucky run" — and 3 seeds is already
the minimum that survives one dead controller.

**The cheaper thing that addresses the same problem.** The failure PBT would
have caught is detectable directly: E3.1's s3 was stuck at goal 0.00 by epoch
~140, while its eventual solvers were already climbing. So we pre-register an
**early-restart rule** instead: *if a seed's goal rate is still 0.00 at epoch
150, restart that seed with a new controller init and keep the run.* That
recovers a dead draw for ~1/3 of a run's cost, keeps the seeds independent, and
needs no population at all.

---

## 7. Budget — measured, not estimated

Measured on a real single-arm ring run (12 epochs, `--num-threads 10`, the
production configuration), timing each epoch's arrival:

| | s/epoch |
|---|---:|
| trainer's own `T_sample + T_update + T_eval` (n=6) | **112.4** |
| wall clock, ordinary epoch (n=4) | **112** |
| wall clock, epoch with mirror + ladder eval | 171 |

The two independent clocks agree to within 0.4 s, and the ordinary-epoch
figure is stable across every epoch measured (110, 110, 115, 115 s).

The mirror/ladder eval added **59 s** at test size (18 episodes). At production
size (20 mirror + 5×10 ladder = 70 episodes) that is ≈ 230 s, so running it
**every 20 epochs** amortises to ~12 s/epoch.

> **Production single arm: ≈ 124 s/epoch → 13.8 h for 400 epochs.**

For 3 seeds, the honest range — the concurrency factor is **measured for 2
arms** (170-181 s each vs 110 alone, i.e. 1.6×) and **extrapolated for 3**:

| plan | wall clock | GPU | confidence |
|---|---:|---|---|
| 3 seeds **sequential** | ~41 h | ~5 GB | measured |
| 2 concurrent, then 1 | ~34 h | ~11 GB (measured) | measured |
| **3 concurrent** | **~29 h** | ~16.5 GB (estimated) | extrapolated |

**Recommendation: start 3 concurrent, measure the rate over the first 10
epochs, and fall back to 2+1 if the GPU peak approaches the 17 500 MiB
trigger.** The previous budget I gave was built on an unmeasured baseline and
was wrong by ~50%; the 124 s/epoch above is measured, and the only extrapolated
number in the table is flagged as such.

For comparison, the archived two-lineage E4 design cost **~60 h** for the same
3 seeds. This is **less than half**, because it runs one arm per seed instead
of a pair.

---

## 8. What this does not test

* **Whether two independent lineages would diverge.** That is the archived E4
  experiment. It is a different question and this plan cannot answer it.
* **Whether a *sighted* design head would do better.** Transform2Act's design
  head never sees simulation state, so the body is shaped only by returns, not
  by anything about the opponent. Everything here is measured under that
  constraint.
* **Transfer beyond its own lineage.** A ratchet against its own history says
  nothing about performance against an agent trained differently — the very
  gap AlphaStar's exploiters exist to expose.
* **Whether the game is transitive in general.** We measure transitivity *over
  the checkpoints this run happens to produce*, which is a path through
  strategy space, not the whole space.

---

## Sources

- Bansal et al. 2018, *Emergent Complexity via Multi-Agent Competition* — https://arxiv.org/abs/1710.03748
- Jaderberg et al. 2017, *Population Based Training of Neural Networks* — https://arxiv.org/abs/1711.09846
- Vinyals et al. 2019, *Grandmaster level in StarCraft II* (AlphaStar, PFSP + league) — https://www.nature.com/articles/s41586-019-1724-z
- Balduzzi et al. 2019, *Open-ended Learning in Symmetric Zero-sum Games* — https://arxiv.org/abs/1901.08106
- Silver et al. 2018, *AlphaZero* — https://www.science.org/doi/10.1126/science.aar6404
- CompetEvo / Transform2Act: read from the ported code in this repo —
  `rower_soccer/competevo_port/selfplay.py`,
  `design_opt/models/transform2act_policy.py`
