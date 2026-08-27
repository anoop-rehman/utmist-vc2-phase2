# D2 — can these morphologies play the game at all?

*2026-08-27. Every number here is from `competence_eval.py` at seed 42, 32
worlds, 1,600 steps, or from arithmetic over measured constants. Commands are
given so each can be re-run.*

## Why this exists

The 2h sweep trained 27 runs and ranked teams by win rate. The ranking was not
usable, and not because of noise: every cell containing a bug or a spider ended
82–99% of its episodes in a timeout, so the "winner" was whichever side lost
more slowly. Ranking teams presupposes that the creatures in them can reach the
goal, and nothing had established that.

Two separate errors sit behind that sweep, both mine, and both recorded here so
the next reader does not re-derive them:

1. **The sweep never trained a mixed team.** `--creatures` takes agents in the
   port's order `(A1, B1, A2, B2)` and `team_lanes` is `[[0, 2], [1, 3]]`, so
   `ant,bug,ant,bug` is team A = (ant, ant) against team B = (bug, bug). I
   passed front,back,front,back. All 27 runs are homogeneous-vs-homogeneous.
2. **The morphology ranking I first computed was an artifact.** It read spider
   16.2% > ant 10.7% > bug 1.6%, driven entirely by a degenerate
   spider-vs-spider cell whose three seeds disagreed 0 / 0 / 88. Reading the
   *distribution of episode endings* rather than the headline rate is what
   caught it.

## The measurement

Two questions, deliberately separated, because a team can fail to score in two
unrelated ways:

* **Can it move?** Mean forward speed toward its own goal, and the fraction of
  steps spent upright.
* **Can it score?** Goal rate against an **idle** opponent — one receiving
  exactly zero torque. With nobody to interfere, reaching the line is
  locomotion plus navigation and nothing else.

```
PYTHONPATH=. .venv/bin/python -m rower_soccer.competevo_port.competence_eval \
    --runs "runs/competevo_port/t2h_*_s42" --opponent idle --worlds 32 --steps 1600
```

The idle side is a free negative control: an unactuated body must measure ~0
m/s, and it does (−0.02 to −0.05 across all nine runs).

## Result: no. Neither bug nor spider can score, even unopposed

Side A, the driven side, against an idle opponent:

| morphology | goal rate | forward speed | upright |
|---|---|---|---|
| ant | **34.8%** | **+0.788 m/s** | 87.9% |
| bug | 0.7% | +0.364 m/s | 98.9% |
| spider | **0.0%** | +0.096 m/s | 64.0% |

The two failures are **not the same failure**:

* **The bug is not falling over.** It is upright 98.9% of the time — more than
  the ant — and it locomotes. It is simply slow.
* **The spider barely moves and spends a third of its time down.** 0.096 m/s at
  64% upright is a locomotion failure in the ordinary sense.

## The mechanism is arithmetic, not tactics

Measured from the compiled scene:

| | value |
|---|---|
| time limit | 500 steps × 0.015 s = **7.5 s** |
| front agent spawn → own goal line | −1.0 → +4.0 = **5.0 m** |
| back agent spawn → own goal line | −4.0 → +4.0 = **8.0 m** |

So the speed a creature must sustain to score at all:

| slot | required | ant 0.788 | bug 0.364 | spider 0.096 |
|---|---|---|---|---|
| front | **0.667 m/s** | reaches | cannot | cannot |
| back | **1.067 m/s** | **cannot** | cannot | cannot |

That single table accounts for every zero in the sweep. The bug and the spider
do not fail to score because they cannot navigate or cannot coordinate; at the
gaits they actually reached, **the goal line is outside the distance they can
cover before the clock stops.**

### This also explains a result the design doc attributes to coordination

`DESIGN_2V2.md` §11 records the back agent as a spectator — 0.0% of crossings
after 80 epochs of native training — re-opens section 9's decision 1 about
roles and interference, and concludes "the task survives at four bodies and
8 m, and the back agent is decorative under a first-crossing rule."

The back agent's line is **8 m away and needs 1.067 m/s.** The best gait
anything in this sweep reached is 0.788 m/s. The back agent is not decorative
because the reward structure fails to pay it, and not because roles were learnt
badly — **its task is unreachable at achievable speed.** The doc's own figure
is consistent with this: agent 3 moves from +4.0 to +1.47, which is 2.53 m of
progress against the 8 m it needs.

This does not settle the design question, but it changes what the question is.
`back_x` is not a preference among three options; it is the difference between
a task the second player can complete and one it cannot.

## What follows

The prerequisite the sweep skipped is now measured, and it fails. **Re-running
the mixed-team sweep with the ordering corrected would produce nine flavours of
timeout**, because two of the three morphologies cannot score in any
composition. Fix reachability first.

Three candidates, in increasing order of how much they change the task:

1. **Shorten `back_x`** so the back agent's distance is inside the same budget
   as the front's. Cheapest, and it is already one of section 9's options.
2. **Raise the time limit**, which makes every slot reachable at current gaits
   but lengthens every episode.
3. **Train for speed first** — bug and spider on run-to-goal with
   `--idle-opponent`, so nothing interferes, and see whether the gait improves
   past the threshold. This is the one that answers "could they learn", which
   the numbers above deliberately do not.

`--idle-opponent` was built for (3) and is gated 5/5
(`gate_idle_opponent.py`); the default self-play path is bit-identical with the
flag off (`gate_team_selfplay` 13/13, `tests/test_selfplay` 11/11).

## A within-morphology result worth its own line

The **same ant body** scores 71.9% against idle bugs and **0.0% against idle
spiders**. Both opponents are unactuated at eval; the difference is entirely in
what each policy learnt during training. Ant upright also drops from 96.7% to
68.5% in the spider-trained run.

The reading that fits: **spiders knock their opponents down early enough that
the ant never achieves a first goal, and so never bootstraps.** That is a
hypothesis with two pieces of support (the upright drop, and the total absence
of learning), not a demonstrated mechanism. It is testable by training an ant
against idle spiders.

## Not tested

* Whether longer training produces faster gaits. Every speed here is from a
  200-iteration run, and the threshold is a moving target if gaits improve.
* Whether a shorter `back_x` or a longer time limit actually fixes it. Both are
  arithmetic predictions, unrun.
* **Front and back speeds are pooled.** The 0.788 m/s ant figure averages both
  slots, so it is an upper bound on the back agent's own speed and a lower
  bound on the front's. Separating them would sharpen the table above and is
  one flag's worth of work.
* Whether the bug's slowness is its morphology or its `SCALE_MAX`/gear
  settings. Nothing here distinguishes a body that cannot go faster from a
  policy that never learnt to.
* Seeds 43 and 44. Everything above is seed 42.

## Corrections made while producing this

Recorded because each would have been believed:

* **The ending histogram read `info["end_goal"]`**, which does not exist. It
  printed all zeros — indistinguishable from "no episodes ended". It reads
  `env.last_end` now.
* **Speed counted episode resets as motion.** On reset the body teleports to
  spawn, contributing a field-length of negative progress, and resets land most
  often on the teams that score most — so the measure ran *backwards*: a
  67.9%-scoring ant read 0.084 m/s while a 0%-scoring one read 0.362.
* **The control step is 0.015 s, not the drills' 0.025.** Hardcoding the drill
  value understated every speed by 1.67×. It imports `CONTROL_DT` from
  `scene.py` now, so the diagnostic and the reward cannot drift apart.
* **The spider-vs-spider discrepancy was not stochastic-vs-mean actions**, as I
  guessed. `t2h_spsp_s42` is 100% wipeout on 36–72 step episodes through iter
  184 and 100% timeout on 500-step episodes by iter 194. Averaging the last 8
  evals reported 83.4% wipeout — a state the policy occupied at neither end of
  the window. Aggregate before comparing rates, but check the window holds one
  regime first.
