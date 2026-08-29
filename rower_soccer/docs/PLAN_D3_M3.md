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
| Transform2Act `ant.xml` | 5 | 6 | 4 |
| CompetEvo `dev_ant_body.xml` (D1/D2's ant) | 13 | 10 | 8 |

Theirs is a torso plus four single-segment limbs, one actuator each. Ours is
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

### E1 — Our ant inside Transform2Act, on a locomotion task

The porting rung. Transform2Act's `Robot` parses a specific XML dialect
(integer body names, `<n>_joint` naming, `attr_design` fields it can perturb).
Our `dev_ant_body.xml` is a different dialect.

* **Build**: a converter from `dev_ant_body.xml` to their `Robot` format, plus a
  gate that the resulting body is the same creature — mass, geometry, joint
  ranges and actuator gears equal to the CompetEvo ant to tolerance, and one
  rendered clip looked at. **DONE 2026-08-29 —
  [`D3_M3_E1_ANT_CONVERTER.md`](D3_M3_E1_ANT_CONVERTER.md).**
* **Run**: their ant task with our ant as the initial design, 3 seeds, ~100
  epochs.
* **Question**: does it evolve, and into what? Same variability measurement as
  E0, now on the creature that actually matters.
* **Note**: this is on the critical path regardless of E0's outcome — every rung
  from E3 onward needs our ant in their representation.

### E2 — GNN control only, morphology frozen, on run-to-goal

The sanity rung, and the user's own suggestion. We know the CompetEvo ant can
learn run-to-goal with CompetEvo's own MLP policy. Can Transform2Act's **GNN**
controller learn the same task on the same fixed body?

* **Run**: run-to-goal, our ant, skeleton and attribute stages disabled (or
  emitting no-ops), execution stage only.
* **Baseline**: D2's own run-to-goal numbers on the same body.
* **Falsifies**: if the GNN cannot match the MLP on a fixed body, nothing above
  it is interpretable, and the problem is the controller, not the evolution.
* **Cheap**: no design search, so it converges faster than a full run.

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
* Whether the GNN controller matches the MLP on a fixed body (E2).
* Any claim that E0's result on *their* ant transfers to *our* ant. Per §0c they
  are different creatures.
