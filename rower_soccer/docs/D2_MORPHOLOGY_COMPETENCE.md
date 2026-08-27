# D2 — can these morphologies play the game at all?

*2026-08-27. Every number is from `competence_eval.py` at seed 42, 32 worlds,
1,600 steps, ~100 completed episodes per run, against an IDLE opponent. The
commands are given so each can be re-run.*

> **Revision note.** An earlier version of this document explained the result
> with mean-speed arithmetic and used that to reinterpret `DESIGN_2V2.md`'s
> back-agent finding. **Both of those were wrong and have been removed.** What
> replaced them is a per-episode displacement measurement, which supports a
> narrower claim. The retraction is written out in the last section rather than
> quietly edited away.

## Why this exists

The 2h sweep trained 27 runs and ranked teams by win rate. The ranking was not
usable: every cell containing a bug or a spider ended 82–99% of its episodes in
a timeout, so the "winner" was whichever side lost more slowly. Ranking teams
presupposes the creatures in them can reach the goal, and nothing had
established that.

Two errors sit behind that sweep, both mine:

1. **It never trained a mixed team.** `--creatures` takes agents in the port's
   order `(A1, B1, A2, B2)` and `team_lanes` is `[[0, 2], [1, 3]]`, so
   `ant,bug,ant,bug` is team A = (ant, ant) against team B = (bug, bug). I
   passed front,back,front,back. All 27 runs are homogeneous-vs-homogeneous.
2. **The morphology ranking I first computed was an artifact** — spider 16.2% >
   ant 10.7% > bug 1.6%, driven entirely by a degenerate spider-vs-spider cell
   whose three seeds disagreed 0 / 0 / 88.

## The measurement

Against an **idle** opponent — one receiving exactly zero torque — reaching the
goal is locomotion plus navigation and nothing else.

```
PYTHONPATH=. .venv/bin/python -m rower_soccer.competevo_port.competence_eval \
    --runs "runs/competevo_port/t2h_*_s42" --opponent idle --worlds 32 --steps 1600
```

The idle side is a free negative control: an unactuated body must measure ~0
m/s, and it does (−0.02 to −0.05 across all nine runs).

## Result: no. Neither bug nor spider scores, even unopposed

| morphology | goal rate | speed (front / back) | upright |
|---|---|---|---|
| ant | **33.0%** | +0.554 / +0.962 m/s | 98.4% |
| bug | **0.0%** | +0.244 / +0.586 m/s | 97.5% |
| spider | **0.0%** | +0.128 / +0.007 m/s | 81.3% |

The two failures are **not the same failure**:

* **The bug is not falling over.** It is upright 97.5% of the time — as much as
  the ant — and it locomotes. It is slow.
* **The spider barely moves**, and its back slot is motionless: +0.007 m/s.

## What actually settles it: per-episode displacement

Mean speed cannot decide reachability, and the ant is the proof — see the
retraction below. The decisive measurement is how far each episode *actually*
got. Distance required is fixed by the scene: the front agent spawns at ∓1.0
with its goal line at ±4.0 (**5.0 m**) and the back agent spawns at ∓4.0
(**8.0 m**), inside a 500-step × 0.015 s = **7.5 s** limit.

| morphology | slot | needs | median | p90 | max | ever arrives? |
|---|---|---|---|---|---|---|
| ant | front | 5.0 | 3.39 | 4.98 | **5.03** | yes |
| ant | back | 8.0 | 6.05 | 7.97 | **8.04** | yes |
| bug | front | 5.0 | 1.77 | 2.47 | 3.09 | **never** |
| bug | back | 8.0 | 4.26 | 5.53 | 7.03 | **never** |
| spider | front | 5.0 | 0.83 | 1.01 | 1.17 | **never** |
| spider | back | 8.0 | 0.04 | 0.14 | 0.16 | **never** |

The ant's maxima land exactly on the requirement (5.03 against 5.0, 8.04
against 8.0) because the episode ends on arrival — that equality is the
signature of reaching, not a coincidence.

**So the answer is reachability, on the evidence of displacement rather than
speed.** Over roughly a hundred episodes each:

* the **bug never arrives in either slot**, but it is *marginal*: its best back
  episode covers 7.03 m of the 8.0 required, 88% of the way;
* the **spider is nowhere near** — 23% of the front distance at best, and 2% of
  the back.

That difference matters for what to do next. A bug that gets 88% of the way is
a candidate for a longer clock, a shorter distance, or more training. A spider
whose back slot moves 4 cm per episode is not.

## What follows

**Re-running the mixed-team sweep with the ordering corrected would produce
nine flavours of timeout**, because two of the three morphologies never arrive
in any composition. Fix reachability first.

Three candidates, increasing in how much they change the task:

1. **Train for speed with `--idle-opponent`**, so nothing interferes, and see
   whether the gait improves past the requirement. This answers "could they
   learn", which nothing above does — every number here comes from a
   200-iteration policy trained against a live opponent.
2. **Shorten `back_x`**, already one of `DESIGN_2V2.md` §9's options.
3. **Raise the time limit**, which makes every slot reachable at current gaits
   but lengthens every episode.

`--idle-opponent` is gated 5/5 (`gate_idle_opponent.py`); the default self-play
path is bit-identical with the flag off (`gate_team_selfplay` 13/13,
`tests/test_selfplay` 11/11). Runs for all three morphologies are in flight at
`runs/competevo_port/idle_{ant,bug,spider}_s42`.

## A within-morphology result worth its own line

The **same ant body** scores 71.9% against idle bugs and **0.0% against idle
spiders**. Both opponents are unactuated at eval; the difference is entirely in
what each policy learnt during training. Ant upright also drops from 96.7% to
68.5% in the spider-trained run.

The reading that fits: **spiders knock their opponents down early enough that
the ant never achieves a first goal, and so never bootstraps.** That is a
hypothesis with two pieces of support, not a demonstrated mechanism. It is
testable by training an ant against idle spiders.

## Not tested

* Whether longer training produces faster gaits — (1) above is running.
* Whether a shorter `back_x` or a longer limit fixes the bug. Both unrun.
* Whether the bug's slowness is its morphology or its `SCALE_MAX`/gear
  settings. Nothing here distinguishes a body that cannot go faster from a
  policy that never learnt to.
* Seeds 43 and 44. Everything above is seed 42.
* The displacement table is one run per morphology (the homogeneous cells).

## Corrections made while producing this

Recorded because each would have been believed, and two of them were:

1. **The ending histogram read `info["end_goal"]`**, which does not exist. It
   printed all zeros — indistinguishable from "no episodes ended".
2. **Speed counted episode resets as motion.** On reset the body teleports to
   spawn, contributing a field-length of negative progress, and resets land
   most often on the teams that score most — so the measure ran *backwards*: a
   67.9%-scoring ant read 0.084 m/s and a 0%-scoring one read 0.362.
3. **The control step is 0.015 s, not the drills' 0.025**, understating every
   speed by 1.67×. It imports `CONTROL_DT` from `scene.py` now.
4. **The spider-vs-spider discrepancy was not stochastic-vs-mean actions**, as
   I guessed. `t2h_spsp_s42` is 100% wipeout on 36–72 step episodes through
   iter 184 and 100% timeout on 500-step episodes by iter 194. Averaging the
   last 8 evals reported 83.4% wipeout — a state the policy occupied at neither
   end of the window.
5. **RETRACTED: "the mechanism is arithmetic, not tactics."** I multiplied mean
   speed by the time limit, compared it to the distance, and concluded the bug
   and spider were provably unable to arrive. The control refutes it: the ant
   scores 33.0% while *both* its slot means sit below the implied thresholds
   (0.554 against 0.667, 0.962 against 1.067). A mean over all episodes
   understates what an agent covers in the episodes where it commits, so
   mean-speed × time-limit is not a reachability test. The conclusion happened
   to survive; the argument did not, and it was replaced with displacement.
6. **RETRACTED: "the back agent's task is unreachable at achievable speed."**
   Built on the same bad arithmetic, and used to reinterpret `DESIGN_2V2.md`
   §11's spectator finding as a geometry problem rather than a coordination
   one. The ant's back slot covers the **full 8.04 m** and does arrive, so the
   back agent is not physically incapable and §11's reading stands untouched by
   anything measured here.

---

# Update, same day — the training runs answered "could they learn"

Three runs, 600 iterations each at 256 worlds, homogeneous per-slot, trained
with `--idle-opponent` so nothing interferes. Scored afterwards with
`competence_eval` in **both** conditions, because the answer differs sharply
between them.

| | unopposed goal | opposed goal | unopposed speed (front/back) | opposed speed |
|---|---|---|---|---|
| ant | **98.3%** | 1.0% | +1.114 / +3.255 | +0.199 |
| bug | **94.4%** | 1.0% | +1.671 / +2.271 | +0.209 |
| spider | **0.0%** | 0.0% | +0.463 / +0.300 | +0.102 |

## 1. The bug's morphology was never the limitation

From **0.0% to 94.4%**, with speed rising from 0.244/0.586 to 1.671/2.271 m/s —
comfortably past both the 0.667 front and 1.067 back requirements. The 2h
sweep's zero was a fact about the training conditions, not about the body. Any
plan that discards the bug as unusable is discarding a morphology that reaches
the goal in 94% of episodes once it is given a task it can get reward on.

## 2. The spider's probably is

Three times the training and no interference, and it still never arrives: best
episode 4.68 m of the 5.0 needed. What it *did* learn is instructive — upright
went from 81.3% to **100%**, and speed roughly tripled from a near-standstill.
It solved stability and not locomotion. The clip
(`runs/d2_sweep_clips/idle_trained_spider.mp4`) shows exactly that: splayed,
level, and barely translating across 37 seconds.

This is the strongest evidence so far that the spider needs a change to the
body or its `SCALE_MAX`/gear settings rather than more compute. It is still not
proof — one seed, one setting.

## 3. Unopposed training does not transfer, for any morphology

Every one of these policies collapses when an opponent appears: **98.3% → 1.0%**
for the ant, 94.4% → 1.0% for the bug, with speed dropping ~11x in all three
cases including the spider. The uniformity across morphologies is what makes
this a statement about the training condition rather than about any one body.

So `--idle-opponent` is a **diagnostic and not a curriculum**. It answers "can
this body do the locomotion", which is what it was built for, and it does not
produce a policy worth warm-starting self-play from. The flag's help text
already said results from it are not comparable to a self-play run; this is the
number behind that warning.

`render_sweep` drives both sides and therefore reports the *opposed* figures
(ant 2.1%, bug 0.0%, spider 0.0%) for the same checkpoints. That is not a
contradiction of the 94-98% above, it is the other column of this table — but
it is exactly the sort of pair that gets quoted out of context, so both are
recorded together here.

## What this changes

* **Do not drop the bug.** Its body clears the requirement with room to spare.
* **The spider needs a body fix, not more iterations**, on the evidence
  available.
* **Reachability is no longer the open question for the bug**; transfer from
  unopposed to opposed play is. That is a different and more interesting
  problem, and it is what the 2h sweep was actually up against.

### Not tested, added here

* One seed per morphology, 600 iterations, 256 worlds. The 2h sweep used 512.
* Whether a *mixed* opposed/unopposed curriculum transfers where pure
  unopposed training does not.
* Whether the spider improves with a raised `SCALE_MAX` or different gearing.
* The ant's front slot reads "never arrives" (max 4.60 m) in the unopposed
  eval. This is very likely truncation rather than failure — `win_rule` is
  `team_first`, so the back agent crossing at 8 m ends the episode and cuts the
  front agent's displacement short. Stated as the likely reading, unverified.
