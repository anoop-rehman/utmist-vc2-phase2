# D3 M3 — from a reproduced Transform2Act to 2v2 soccer with evolved bodies

*Written 2026-08-29, after M2 was met (`port_s2_fixed`, final-20 `exec_R_eps`
10,240 against the reference's 7,482 / 10,594). Everything below is either a
measured fact with the command that produced it, an explicit proposal, or an
explicit open question.*

## The goal, stated once

2v2 soccer where the **body plan is part of what training optimises**, teams may
be heterogeneous, and the controller generalises across morphologies. M2 proved
the machinery reproduces on the paper's own task. M3 is the walk from there to
our task.

---

## 0. Three facts established before planning

These change the shape of the first experiments, so they come first.

### 0a. The CompetEvo observation does NOT depend on the opponent's morphology

The natural worry — "if the opponent evolves a limb, does the observation
vector change width and break everything?" — does not apply. Measured in
`competevo_port/dev_env.py:306` and `team_scene.py:226`:

```
obs = [ stage flag (1) | scale vector (20) | own qpos (15) | own qvel (14)
      | 2 x (n_agents - 1) other-root xy ]
```

`other_qpos_xy` is the **root xy only** — two numbers per other agent,
regardless of that agent's joint count. An agent never reads another creature's
joint state. So an evolving opponent is structurally fine, and the fallback
plans (opponent exposes only its centre of mass; or a scripted opponent) are
**not needed**. They stay in reserve for other reasons, not this one.

*This removes a blocker from experiments E3-E4 that we had budgeted for.*

### 0b. The 2D locomotion task is PLANAR, so "start from the ant" is not a drop-in

`assets/mujoco_envs/hopper.xml`:

```xml
<joint name="rootx" type="slide" axis="1 0 0"/>
<joint name="rootz" type="slide" axis="0 0 1"/>
<joint name="0_joint" type="hinge" axis="0 1 0"/>
```

The root translates in **x and z and rotates about y** — motion confined to the
xz plane. There is no y translation and no yaw. A quadruped whose legs radiate
in the xy plane has nowhere to put them. "Run the same 2D locomotion task
starting from the ant" therefore requires inventing a 2D ant, which is a
different creature from the one D1 and D2 use, and would answer a question
about that invented creature.

**Transform2Act already ships a 3D ant task**: `design_opt/envs/ant.py` +
`design_opt/cfg/ant.yml` + `assets/mujoco_envs/ant.xml`. That is the natural
home for "start from a quadruped and evolve it".

### 0c. Their ant is NOT our ant

| | bodies | joints | motors |
|---|---|---|---|
| Transform2Act `ant.xml` | 5 | 5 | 4 |
| CompetEvo `dev_ant_body.xml` (D1/D2's ant) | 13 | 10 | 8 |

Theirs is a torso plus four single-segment limbs, one actuator each — 5 joints,
being a free root plus four hinges (`nq 11 / nv 10`), measured by
`gate_their_ant.py`; this table said 6 when first written. Ours is
the DeepMind-style ant: four legs of two segments, hip and ankle per leg. They
are different creatures with different capabilities, and any result on one does
not transfer to the other by assumption.

So "start from the ant" splits into two different experiments, and **the one on
the critical path to soccer is the one that uses OUR ant**, because the soccer
creature has to be the creature D1 and D2 already train.

---

## 1. The experiment ladder

Ordered so each rung fails cheaply and independently. Every rung is a *decision
point*, not just a run: each has a stated question and a stated
what-would-falsify-it.

### E0 — Does the skeleton stage still explore on a body that is already good?

**Cheapest and most informative first.** On the reference's own completed
hopper run, by epoch 100 **199 of 200 sampled designs share one topology** and
`exec_R_eps` then climbs from 1,376 to 6,836 over the remaining 900 epochs
(`D3_HANDOFF.md`). Nine tenths of a Transform2Act run is attribute tuning and
control, not body plan.

That was measured starting from a 3-segment line — a body with obvious room to
improve. Starting from a *competent* body (an ant) may make the skeleton stage
converge even faster, in which case the headline M3 comparison
("design freedom on vs off") would mostly be measuring **attribute tuning**, and
we would learn that after spending the compute rather than before.

* **Run**: their `ant.yml`, 3 seeds, ~100 epochs each. Their ant, their task,
  zero porting.
* **Measure**: `topology_census.py` every 10 epochs — distinct topologies among
  200 sampled designs, and whether the mean-action design changes at all.
* **Also measure**: cross-seed design divergence. Do three seeds converge on the
  same body, or three different ones? This is the variability question, and it
  is answerable at 100 epochs.
* **Falsifies**: if all three seeds pin one topology by epoch ~20 and never move,
  M3 should be framed as attribute-and-control search, and the "evolve
  genuinely different bodies for different roles" ambition needs a different
  mechanism (longer skeleton annealing, an exploration bonus, or population
  diversity pressure).

#### E0 RESULT, 2026-08-29 — run, and it did NOT falsify. See `D3_E0_ANT.md`.

Three seeds, 100 epochs, on THEIR ant on their CPU reference (our GPU port has
no ant path — four hopper-only hardcodings, listed in `gate_their_ant.py`).

* **The skeleton stage keeps exploring.** At epoch 20 all three seeds sample
  187/188/187 distinct topologies out of 200; at epoch 100 they are at 34/20/27
  with the most common topology holding 20/41/26%. The hopper is at 3 distinct
  and 89% by epoch 100. The mean-action design changes at 9, 7 and 9 of the 11
  censuses and had still not settled at epoch 100.
* **But seeds do not give different bodies.** The three final bodies are
  Jaccard 0.76-0.82 apart and 0.39-0.50 SMD apart — LESS divergent than the
  hopper's two seeds (0.88, 0.70) — while their gaits differ enormously
  (`exec_R_eps` 800 / 256 / 331; 37.0 / 5.9 / 15.6 m travelled; 47% / 4% / 25%
  airborne). Same body, different gait. The "different bodies for different
  roles" ambition needs role pressure or a diversity term, not more seeds.
* **Their ant's design box is far better behaved than their hopper's**: zero
  parameters at any bound across 1,745 sampled capsules and actuators, against
  40-44% of hopper capsules at the minimum radius; floor penetration 0.09-0.11 m
  against 0.27 m; net/path 0.93-0.99.
* **Correction to `D3_HANDOFF.md` carried by this experiment**: the "199 of 200
  designs share one topology by epoch 100" figure is an epoch-1000 number. At
  epoch 100 the measured value is 178/200 (89%) on `hopper_gpu`.

Open question §3.2 ("how much of E0 to run?") is answered: 100 epochs answers
the convergence and variability questions and does not converge a body.

### E1 — Our ant inside Transform2Act, on a locomotion task

The porting rung. Transform2Act's `Robot` parses a specific XML dialect
(integer body names, `<n>_joint` naming, `attr_design` fields it can perturb).
Our `dev_ant_body.xml` is a different dialect.

* **Build**: a converter from `dev_ant_body.xml` to their `Robot` format, plus a
  gate that the resulting body is the same creature — mass, geometry, joint
  ranges and actuator gears equal to the CompetEvo ant to tolerance, and one
  rendered clip looked at. **DONE 2026-08-29 —
  [`D3_M3_E1_ANT_CONVERTER.md`](D3_M3_E1_ANT_CONVERTER.md).**
* **Run**: their ant task with our ant as the initial design. **DONE 2026-08-30 on TWO seeds x 100 epochs — [`D3_E1_ANT.md`](D3_E1_ANT.md).** Originally 3 seeds, ~100
  epochs.
* **Question**: does it evolve, and into what? Same variability measurement as
  E0, now on the creature that actually matters.
* **Note**: this is on the critical path regardless of E0's outcome — every rung
  from E3 onward needs our ant in their representation.

#### E1 RESULT, 2026-08-30 — run on TWO seeds. See [`D3_E1_ANT.md`](D3_E1_ANT.md).

**Two seeds x 100 epochs, not three.** Seed 3 was stopped at epoch 62 by user
decision to free the GPU for E1.1 and is excluded from every number; an earlier
seed-3 attempt aborted at 39 epochs on a full disk and is quarantined. **n=2 is
the central limitation: two points cannot establish a spread**, so the contrasts
with E0's three seeds below are suggestive only.

* **The skeleton stage explores here too, and converges more slowly.** Epoch 20:
  190 and 187 distinct topologies of 200, indistinguishable from their ant's
  187/188/187. Epoch 100: **63 and 101 distinct, most-common share 5.5-7.0%**,
  against their ant's 34/20/27 at 20-41% and the hopper's 3 at 89%. On
  concentration our ant at epoch 100 sits where their ant was at epoch 40-50.
  The mean-action design changed at 10 of 11 censuses on both seeds.
* **But the reachable space is much smaller, and this was established before
  running.** `min_body_depth: 1` (not `max_nchild`) is what forbids a fifth leg,
  and it binds their ant identically. The real constraint is that
  `add_child_to_body` clones the parent and our depth-1 leg stubs are
  **jointless**: our ant can gain **at most 4 actuators** (8→12) against their
  24 (4→28), 12 of its 16 possible additions are passive dead weight, and it can
  erode to a **0-motor** blob theirs cannot reach.
* **`done_condition.max_ang: 60` was flagged as possibly ending a rolling
  quadruped early. It does — for THEIR ant, not ours.** Untrained episodes:
  ours mean 509 steps with 43% running the full 1000; theirs mean **26** with
  97% ending on tilt.
* **The evolved creature stops being a quadruped.** Settled torso height halves
  (0.561 → 0.270 on both seeds), mass nearly doubles, and four knee-up walking
  legs become long splayed limbs that are **airborne 71-76%** of the episode.
  `exec_R_eps` final-10 3192 and 2721; 33.3 m in 222 steps and 119.5 m in 1000;
  net/path 0.999-1.000. Their ant *gained* a stance; ours *lost* one.
* **The design box is no longer pristine.** 7.2% of one seed's capsules at the
  MAXIMUM radius and 7.5% of the other's gears at the minimum, where their ant
  had 0.0% everywhere. Still far from the hopper's 44%/34%. Floor penetration is
  *better* than their ant's (0.049-0.053 m vs 0.110).
* **Cross-seed:** Jaccard 0.75, SMD 0.58 for the one pair — marginally more
  divergent than any of E0's three pairs, on one measurement.

**The consequence for M3**: on a task whose reward is `dx/dt` with no energy
term, morphology search converts the soccer creature into a straight-line
bounder with its belly near the floor. Before morphology search is pointed at
soccer, **the task needs changing, not the design space**.

**GPU budget correction, learned expensively**: E0's "3 concurrent = 19.2 GB"
does NOT transfer. **Two of ours peaked at 19.95 GB of 20.475** and OOM-ed the
live D1 run off the card. One E1-class run at a time alongside D1.

### E1.1 — Is the GNN controller as good as plain PPO? (added 2026-08-29, user)

E1 runs design+control. **E1.1 nulls the skeleton and attribute stages** so only
the execution stage does anything, on the **DeepMind ant**, and asks the
question E0 could not: how good is Transform2Act's *controller* on its own,
measured against ordinary PPO on the same body?

**The comparison must be run in-house, not against published Ant numbers.** The
two reward functions are different objectives, measured from source:

| | forward | control cost | contact | survive |
|---|---|---|---|---|
| gym `Ant-v2/3/4` (`gym/envs/mujoco/ant.py:14-19`) | `dx/dt` | `0.5 * sum(a^2)` | `0.5e-3 * sum(clip(cfrc)^2)` | **+1.0 / step** |
| Transform2Act ant (`design_opt/envs/ant.py:153-165`) | `dx/dt` | `1e-4 * mean(a^2)` | none | **0.0** (cfg default) |

Three incompatibilities, any one of which breaks a naive comparison:

1. **The survive bonus.** Gym pays +1.0 every step; over a 1,000-step episode
   that is +1,000 of the ~5,000-6,000 a published PPO Ant run reports. Roughly a
   fifth of the headline number is standing still.
2. **Control cost differs by ~40,000x.** `0.5 * sum` over 8 actuators against
   `1e-4 * mean` is a factor of 8 from sum-vs-mean and 5,000 from the
   coefficient.
3. Gym charges a contact cost; Transform2Act does not.

So "the GNN reached X, published PPO reaches Y" would compare two different
objectives and mean nothing.

**Design.** Run **PPO with an ordinary MLP policy inside the Transform2Act ant
env itself**, morphology frozen exactly as for the GNN arm, same reward, same
episode structure, same step budget. Then the only difference between arms is
the policy architecture. Published Ant numbers are useful as a *sanity check on
the environment* -- if our PPO-in-their-env lands nowhere near the literature
after adjusting for the reward difference, the env is the suspect -- but they
are not the comparison.

**SETTLED 2026-08-29 (user).** The design stages are **run but forced to an
identity action**, not skipped. Skipping would change episode length and strip
the stage flag's meaning from the observation, altering the task as well as the
policy; forcing identity holds everything constant except the one thing under
test. Gate that the body is genuinely unchanged from the first step to the last
-- an "identity" transform that quietly perturbs a length would make this a
comparison of two different bodies.

**SETTLED 2026-08-29 (user).** The baseline is plain-MLP PPO run **inside the
Transform2Act ant env**, same reward, same episode structure, same budget, with
only the policy architecture differing. Published Ant numbers are a sanity
check on the environment, never the baseline.

**What would falsify.** If the GNN materially underperforms a plain MLP on the
same body, same reward and same budget, then every design+control result rests
on a weaker controller than the task allows, and that is a bigger problem than
any morphology finding.

#### E1.1 RESULT, 2026-08-30 — IT FIRED. See [`D3_E1_ANT.md`](D3_E1_ANT.md) sections 13-17.

**The GNN controller loses to a plain MLP on the same frozen body, same reward,
same 5.0M-step budget.** One instrument, mean-action, 20 episodes per arm, body
freezing verified array-by-array under each arm's own trained policy:

| arm | seed means | ratio |
|---|---|---|
| GNN (Transform2Act, design heads nulled) | 2622, 2430 -> **2526** | — |
| MLP PPO, batching matched to the GNN | 3091, 2855 -> **2973** | **1.18x the GNN** |
| MLP PPO, published PPO-MuJoCo batching | 1180, 1016 -> 1098 | 0.43x the GNN |

Seed ranges do not overlap; episode-level Welch t = 4.70.

**Three qualifications that must travel with that**: (1) it depends entirely on
the baseline being well-configured -- against *published* PPO batching the GNN
wins by 2.1-2.6x, and only Transform2Act's own 50,000/2048 batching makes the
MLP win, so running both batchings is the only reason the answer is right;
(2) n=2 seeds per arm; (3) the MLP acts on the 8 actuators directly while the
GNN emits one scalar per node over 13 and discards 5, and the GNN carries design
heads taking gradients from discarded actions -- **whether SKIPPING the design
stages rather than forcing identity would close the gap is untested.**

**Consequence for the ladder**: E1's `exec_R_eps` (3346, 2704) came from this
controller. `exec_R_eps` comparisons across rungs should NOT be read as
morphology quality, and part of what the skeleton stage appears to gain may be
compensation for controller weakness. E2 is now partly answered and partly
sharpened: the GNN *can* learn this task well (105 m per episode, net/path
0.998) but is not the best controller available for it.

**A measurement trap worth carrying forward**: Transform2Act's `exec_R_eps` is a
separate **mean-action evaluation** pass (`transform2act_agent.py:214`), not a
training return. Comparing it against a trainer that logs stochastic training
returns flatters the GNN by ~1.3x. The first draft of this comparison made that
error.

### E2 — GNN control, frozen morphology, 1v1 run-to-goal vs a SCRIPTED opponent

**Spec settled by the user 2026-08-30.** Frozen morphology, GNN control,
**1v1 run-to-goal**, **both agents our DeepMind ant**, opponent **scripted**
(not learned, not self-play). Design stages run but forced to identity, exactly
as E1.1 settled.

This is the first rung where the Transform2Act machinery meets OUR task rather
than the paper's locomotion task, so it is the real integration step toward
soccer.

* **Falsifies**: if the GNN cannot learn this task on a fixed body, nothing
  above it is interpretable and the problem is the controller, not the
  evolution.

**The comparison has to be built, not borrowed.** D2's run-to-goal numbers are
from *self-play* with CompetEvo's own MLP. A GNN-vs-scripted result is not
comparable to an MLP-vs-self-play one -- different opponents make different
tasks. So E2 needs its own **matched MLP arm on the identical setup**: same
scripted opponent, same body, same reward, same budget, only the policy
architecture differing. E1.1 is the precedent and the reason: there, the answer
*flipped* depending on whether the MLP baseline used published or matched
batching.

**Prior from E1.1**: on a frozen body the GNN came in **18% below** a
well-configured MLP (2,526 vs 2,973). So the expectation is that the GNN is
workable but behind. What E2 adds is whether that gap holds on a task with an
opponent and a goal line rather than open-field locomotion.

**The scripted opponent must be specified, not improvised.** Write it down --
what it does, how fast, whether it reacts -- because it is now part of the task
definition and every later rung inherits it.

#### E2 BUILD DONE, 2026-08-30. See [`D3_E2_RTG.md`](D3_E2_RTG.md).

**The scripted opponent, which E3-E5 inherit**: our ant, all 8 motors at zero
torque, its ENTIRE state overwritten after every control step to
`x = +1.0 - 0.68*dt*k`, `y = 0`, `z = 0.5347`, yaw 180 deg, root velocity
`(-0.68, 0, 0)`, hinges at the stance it settles into at zero torque (hips 0,
ankles 51.87 deg). **Rigid, non-reactive, constant-speed** -- its trajectory is
a function of the step index alone, bit-identical in every episode of every
seed of every arm, and no contact can slow it, push it or knock it over.

**0.68 m/s is not a free choice**: 5.0 m (x=-1 to the goal at x=+4) inside
500 control steps x 0.015 s = 7.5 s is the 0.667 m/s the task's own clock
already demands, and 0.68 is that advanced 2% so that running out of time is
realised as a **loss to a visible opponent** rather than as a silent
truncation. It crosses x=-4 at control step **491 of 500**. Beating the
opponent and beating the clock are therefore the same requirement, which makes
E2's goal rate directly comparable to D2's.

**Beatable, and not trivially**: D2 measured the same ant under the same reward
over the same 5.0 m in 7.5 s at **98.3% goal, 1.114 m/s** after unopposed
training, and **33.0% at 0.554 m/s** after the 2h sweep -- 0.68 m/s sits inside
the band this body's policies span. A zero-torque agent scores 0% and is
bulldozed backwards to x = -3.22.

**Gated**: `gate_e2.py`, 41 checks, 0 failed, 7 phases each with a negative
control -- including the body-freeze gate the rung requires (134 mjModel arrays
identical across 20 episodes of destructive random design actions; 96 change
without the flag) and a check that E1/E1.1's arms are untouched by the two
shared-file edits.

The `exec_R_eps` trap is now designed out of the artefacts as well as the
analysis: the two arms' wandb keys are named `..._MEANACTION_eval` and
`..._STOCHASTIC`, and the comparable curve is `e2/eval_*`, produced by one
shared instrument (`e2_eval.evaluate`) that both trainers and the post-hoc
table call.

#### E2 RESULT, 2026-08-30 — the task is NOT learned at this budget, by either
architecture. See [`D3_E2_RTG.md`](D3_E2_RTG.md) sections 6-7.

One instrument, mean-action, 20 episodes per arm, body frozen under each arm's
own trained policy (134 mjModel arrays identical on all four trained arms).

| arm | mean-action R | **goal** | fell | furthest forward | of the 5.0 m |
|---|---|---|---|---|---|
| GNN s1 / s2 | −655 / −537 | **0.00** | 0.05 / 0.15 | 0.22 / 0.30 m | 4.4% / 6.0% |
| MLP matched s1 / s2 | −195 / −204 | **0.00** | 0.70 / 0.60 | 0.46 / 0.60 m | 9.1% / 11.9% |
| idle, zero torque | −524 | **0.00** | 0.15 | 0.08 m | 1.5% |

* **Goal rate is 0.00 for both arms across 40 episodes each.** The best single
  episode covered **1.95 m of the 5.00 m** required. E2's intended question --
  does E1.1's 18% GNN deficit persist on a task with an opponent and a goal
  line -- **cannot be answered on goal rate**, because neither arm can do the
  task at 5.0M steps. Pre-registered: D2 needed ~**77M** steps for 98.3% on the
  same body, reward, distance and clock; E2 spent 5.0M to stay matched to E1.1.
* **The GNN is statistically indistinguishable from ZERO TORQUE**: −595.9
  against the idle control's −523.7, episode-level Welch t = **0.87**.
* **The MLP leads on return by 3.0x (Welch t = 5.03) and the mechanism is
  FALLING OVER.** A fall ends the episode before the opponent's certain goal at
  step 491 and so never pays the −1000, which is worth **+750 to +900** inside
  every arm; conditional on the ending the GNN actually scores HIGHER than the
  MLP (+280/+251 against +74/+100). The whole gap is the mixture -- the MLP
  falls in 60-70% of episodes, the GNN in 5-15%. **"The MLP controller is
  better on this task" would be the wrong sentence**, and the return column
  alone would have produced it.

* **The published-batching MLP settles it.** Run as E1.1's precedent demands
  (batch 2048 / minibatch 64 / lr anneal, 5.0M steps), it has the **best return
  of any arm** (+32.0 seed mean; seed 1 at **+174.9 ± 4.9**) and is the **most
  degenerate policy in E2** -- it falls over in **every one of 20 episodes**,
  action std collapsed to 0.039, furthest forward 0.14 m of 5.00. Rank all seven
  rows by return and the ordering IS the fall-rate ordering:
  **Pearson r(fall rate, return) = +0.989**, while
  **r(forward progress, return) = +0.019**.

**So E2's real finding is about the TASK, not either architecture**: on
CompetEvo run-to-goal against a scripted opponent that always scores, at 5.0M
steps, **episode return is not a measure of competence** -- any controller
comparison that ranks by return here ranks by exploitation of the fall-dodge.

**The consequence for E3, which inherits this opponent**: the scripted
opponent's goal is *certain*, so "fall before step 491" is a reliable local
optimum worth ~+826 (measured on the idle control). CompetEvo's own rule set
creates it -- a fall ends the episode and `goal_rewards` pays nobody -- but a
learned or idle opponent makes it contingent, and a scripted one that always
scores makes it reliable. **E3 must decide deliberately whether to keep it.**
The second consequence is budget: on this reward, 5.0M steps is not enough for
this body to locomote, and no architecture comparison on this task means much
until it is.

### E2.1 — the curriculum ablation: was D2's competence the curriculum or the budget?

E2 left exactly one question dividing two explanations for "D2 could and E2
could not": D2's much larger budget, or the fact that **D2's trainer does not
optimise the env reward at all** (`CoEvoPPO.collect` mixes
`alpha*dense + (1-alpha)*parse`, so early training barely feels the −1000 that
makes falling attractive). E2.1 runs E2's matched MLP arm at **20.0M steps**
(4x E2's) in two conditions differing in one argument — CompetEvo's curriculum
ported in, versus E2's flat reward re-run under identical conditions — 2 seeds
each, measured on E2's own instrument with **forward progress as the headline
rather than return**.

#### E2.1 RESULT, 2026-09-03 — the reward mix, decisively. See [`D3_E21_CURRICULUM.md`](D3_E21_CURRICULUM.md).

Three conditions x 2 seeds x 20.0M steps, MLP arm, one mean-action instrument,
20 episodes each. **Body frozen on all six (134 mjModel arrays identical).**

| condition | alpha 0 -> 399 | **goal** | fell | **furthest forward** | speed |
|---|---|---|---|---|---|
| **d2rep** (D2's realised regime) | 1.000 -> 0.847 | **0.95 / 1.00** | **0.00 / 0.00** | **5.00 / 5.00 m** | 1.37 / 1.33 m/s |
| **flat** (control, E2's reward) | -- | 0.15 / 0.35 | 0.85 / 0.65 | 3.44 / 4.35 m | 1.34 / 1.97 m/s |
| **cur** (CompetEvo's nominal anneal) | 1.000 -> 0.000 by ep 80 | 0.05 / 0.15 | 0.80 / 0.85 | 2.33 / 2.05 m | 0.60 / 0.54 m/s |
| *idle, zero torque* | -- | 0.00 | 0.15 | 0.08 m | -0.31 m/s |

* **The task IS solvable on this body, and the blocker was the reward mix, not
  the budget.** `d2rep` reaches **0.95 goal at 4.0M steps** -- 80% of E2's own
  budget and 26% of D2's -- and 39 of 40 episodes at 20M, with **not one fall
  in 40**. The planned 15x scale-up was unnecessary.
* **Budget alone is not enough.** The flat control at 4x E2's budget reaches
  only 0.25 goal; at E2's own 5.0M it reproduces E2's null (goal 0.00).
* **CompetEvo's NOMINAL curriculum is worse than no curriculum** (0.10 vs 0.25
  goal). Two reasons, both from the reward constants: below a **critical alpha
  of 0.739** the fall-dodge outweighs everything a full episode can bank
  (+352.4 measured), and `cur` crosses it at epoch 21 while `d2rep` never does;
  and at alpha = 0 the objective is the sparse term ALONE, so 80% of `cur`'s
  run has no locomotion gradient at all.
* **E2's correlation structure inverts.** Over the same seven-arm statistic,
  `r(fall rate, return)` goes **+0.989 -> -0.517** and
  `r(forward progress, return)` **+0.019 -> +0.947**. Return becomes a measure
  of competence. The idle floor still shows E2's structure (+0.985), so it is
  the policies that changed, not the instrument.

**Consequence for E3**: run-to-goal against this scripted opponent is a
solvable task once the sparse term is held under ~26% weight for the whole run.
The fall-dodge hazard is NOT fixed -- `d2rep` avoids it rather than removing
it -- so E3 still has to decide about the termination rule. And E2's
architecture question is now answerable: `d2rep` is a regime where a
GNN-vs-MLP comparison would compare two controllers that can both do the task.

### E3 — Evolving ant vs a FIXED opponent, run-to-goal

The first adversarial rung, deliberately without self-play so there is only one
moving part.

* **Opponent**: a frozen CompetEvo run-to-goal checkpoint. Per §0a its
  morphology never enters our agent's observation, and ours never enters its
  — so a frozen opponent is safe even as our agent's body changes.
* **Fallback if the frozen policy proves brittle**: a scripted opponent (run
  straight at the goal). Simpler, fully specified, and removes any question of
  the opponent being off its training distribution.
* **Question**: can Transform2Act's design+control loop win an adversarial task?
* **Measure**: win rate against the fixed opponent, and whether the evolved body
  differs from E1's (same creature, different pressure — a role effect).

### E4 — Self-play, both sides evolving, run-to-goal 1v1

* Both agents run skeleton -> attribute -> execution, both learn, opponent
  sampled from the checkpoint ring as in D2's `CoEvoPPO`.
* **The interesting question**: does co-evolution produce an arms race in
  morphology, or do both sides converge on the same body?
* **Known risk**: D2 measured that a shared design head cannot condition on role
  or opponent at all until `--role-in-design` was added (SMD 0.110 -> 0.833).
  The equivalent question here is what Transform2Act's design head can see. It
  must be checked before this rung, not after.

### E5 — 2v2 run-to-goal

Teams of two, still the run-to-goal task. D2 has all of this except the
Transform2Act policy: team lanes, opponent ring over whole past teams, team
credit, per-slot nets for heterogeneous compositions.

* **The specific thing to watch**: D2 found the back agent is a spectator under
  a first-crossing rule. With evolvable bodies, does the back agent evolve a
  *different* body suited to interference rather than racing? That is the first
  rung where "heterogeneous teams" means something learned rather than
  configured.

### E6 — 2v2 soccer

The target. Everything above is in service of arriving here with a controller
that generalises across bodies and a design head that has been shown to
specialise by role.

---

## 2. What runs in parallel, and what the machine can take

Measured, not estimated:

| | value |
|---|---|
| GPU | 1 x RTX 4000 Ada, **20 GB** |
| CPU | **48 cores**, load average ~0.7 idle |
| RAM | **251 GB** (46 GB in use) |
| D1 self-play @ 512 worlds | 1.14 GB own / 2.20 GB process, **~95% SM** |
| D3 t2a training arm | 3.4-4.6 GB, design stages on **CPU** |

**Memory is not the constraint; SM time is.** Peak observed all session was
~9.3 GB of 20, with four training clients up. What contends is compute: D1 alone
sustains ~54k fps and dropped to ~50k with three other GPU clients running, so
roughly an 8% tax at that level of sharing.

Practical budget:

* **D1 continuously + 2-3 D3 arms** is comfortable. VRAM is nowhere near the
  limit and the design stages are CPU work against 48 mostly-idle cores.
* Beyond ~4 concurrent GPU clients the SM tax stops being negligible and every
  run slows together; better to queue than to over-subscribe.
* **NVIDIA MPS is active. Never kill a CUDA process** — under MPS it can corrupt
  the live survivors, which has destroyed two runs on this project. Every
  trainer now takes `--stop-file`; use it.

So E0, E1 and E2 can run **simultaneously with D1**, and their results read
together.

---

## 3. Open questions for the user

1. **Which ant for E0?** Their 5-body / 4-motor ant is runnable today with zero
   porting; our 13-body / 8-motor ant needs the E1 converter first. Recommend
   running E0 on theirs immediately for the variability answer, and treating E1
   as the real experiment.
2. **How much of E0 to run?** 100 epochs answers the topology-convergence and
   seed-variability questions. It does not produce a converged body.
3. **E3's opponent**: frozen learned checkpoint (more realistic) or scripted
   (fully specified, no distribution-shift confound)? Recommend running the
   scripted one first precisely because it cannot be blamed.

---

## 5. 2026-08-29 — E1's prerequisite is built and gated

`rower_soccer/t2a_port/competevo_to_t2a.py` puts `dev_ant_body.xml` into
Transform2Act's `Robot` representation;
`rower_soccer/t2a_port/gate_competevo_ant.py` gates it in four phases plus nine
negative controls. Full write-up, with every number and every tolerance, in
[`D3_M3_E1_ANT_CONVERTER.md`](D3_M3_E1_ANT_CONVERTER.md). The headline:

* **The conversion is not lossy on the creature.** 95 compiled `mjModel` arrays,
  compared against the model D1/D2 *actually compile*
  (`scene.dev_run_to_goal_xml`) rather than against the asset file: largest
  residual exactly `0.000e+00`. 500 steps of physics from an identical state
  with identical recorded actions in an identical arena: `max|dqpos|` exactly
  `0.000e+00`. Section 4's open question — "whether their `Robot` XML dialect
  can express our ant's two-segment legs without loss" — is **answered: it can**,
  because a chain of bodies with one capsule each is precisely the bone model
  `Robot` implements. `dev_ant_body.xml`'s body names already match `reindex()`
  exactly, so the converter mostly gives the fragment a `<worldbody>`.
* **`Robot` mutates it**: add a limb, remove a limb, change a length — each
  recompiles and steps, and `AntEnv` runs skeleton -> attribute -> execution on
  it. Rendered and looked at: same creature as D1/D2's, standing on its feet,
  nothing through the floor.
* **It found a real bug in `khrylib/robot/xml_robot.py`.** `Body.get_params`
  pads one zero for a jointless body and `Body.set_params` did not consume it.
  No Transform2Act robot has a jointless body; ours has four (the leg stubs), so
  one attribute transform silently reset each stub capsule to radius 0.065 from
  0.08. Fixed with the missing three lines; a strict no-op for hopper, swimmer,
  gap and their ant.
* **And two in our own `t2a_port/xml_global_to_local.py`**: `legacy_inertial`
  was silently dropped for a local-coordinate input, and both legacy passes
  assumed density 1000 where `ant.xml` and our ant use 5.0 — a 200x mass error
  had anyone pointed `two_stage_pipeline` at an ant. Hopper is unaffected, so no
  existing number moves.

Three caveats E1's reading has to carry, none of them the converter's doing:

1. E1 trains in **their** stack, where MuJoCo 2.1 counts a capsule's caps as ¾ of
   a sphere: legs **3.5% lighter** than D1/D2's ant (0.8787 vs 0.9109 kg).
2. With mass and inertia corrected to 1e-14 the two engines agree to `1.2e-14`
   through 17 contact-free steps and part at the **first floor contact**. That is
   the 2.1-vs-3.12 contact solver and no XML can fix it.
3. Transform2Act's floor inherits `margin="0.01"`; CompetEvo's is 0. **Our ant's
   feet touch down 1 cm earlier on their floor** — visible in the render as a
   settled torso 0.010 higher.

**Not done, deliberately**: no training run. The smoke shows the agent builds on
a 13-body graph, a PPO iteration completes, and all three heads take finite
non-zero gradients. E1's 3 seeds x ~100 epochs has not been started.

## 6. Not tested / not assumed

* Whether Transform2Act's design head can see role or opponent at all. Its
  hopper task is single-agent, so the question does not arise there and has not
  been asked. **E4 depends on the answer.**
* ~~Whether their `Robot` XML dialect can express our ant's two-segment legs
  without loss.~~ **Answered 2026-08-29: it can, exactly** (section 5). What is
  still open is the *engine*: E1 trains under mujoco-py 2.1, whose capsules are
  3.5% lighter and whose contact solver is not 3.12's.
* ~~Whether the GNN controller matches the MLP on a fixed body (E2).~~
  **Partly answered 2026-08-30, and the answer is that the question was
  the wrong shape**: on run-to-goal at 5.0M steps neither controller
  learns the task (goal rate 0.00 both), the GNN is indistinguishable
  from zero torque, and the MLP's 3.0x return lead is entirely the rate
  at which it exploits a degenerate fall. E1.1 answered the locomotion
  version of this on Transform2Act's own task; E2 could not answer the
  run-to-goal version because the task was not learned.
* Any claim that E0's result on *their* ant transfers to *our* ant. Per §0c they
  are different creatures.
