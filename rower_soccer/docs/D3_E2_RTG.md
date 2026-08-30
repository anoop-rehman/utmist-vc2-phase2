# D3 M3 E2 — Transform2Act's GNN controller on 1v1 run-to-goal vs a scripted opponent

*2026-08-30. The experiment is `PLAN_D3_M3.md` section 1, rung E2, with the
spec the user settled that morning. Every number below names the command that
produced it; anything not measured is in "Not tested" at the end.*

**Creature**: OUR ant on BOTH sides —
`rower_soccer/competevo_port/assets/dev_ant_body.xml`, the DeepMind ant D1 and
D2 train (13 bodies, 9 joints = 1 free + 8 hinge, 8 motors), converted to
Transform2Act's `Robot` dialect and gated in
[`D3_M3_E1_ANT_CONVERTER.md`](D3_M3_E1_ANT_CONVERTER.md).
**Task**: CompetEvo's `run-to-goal-ants-v0`, not Transform2Act's locomotion
task. **Morphology frozen** — the design stages run and are forced to the
identity action, exactly as E1.1 settled. **Opponent scripted** — not learned,
not self-play.

This is the first rung where the Transform2Act machinery meets OUR task, so
most of the work is the integration and the gates on it.

---

## 1. THE SCRIPTED OPPONENT — the exact definition every later rung inherits

Implemented in `design_opt/envs/run_to_goal.py` (`set_opponent`), gated in
`gate_e2.py` phase 2.

**Body.** Our ant, identical to ours: same 13 bodies, same 9 joints, same 8
motors, same masses, same geometry. It is a name-prefixed (`opp_`) clone of
our ant in the same MJCF, spawned at CompetEvo's registered pose for agent 1 —
`(+1, 0, 0.75)`, yawed 180° to face −x — while ours takes agent 0's,
`(−1, 0, 0.75)` facing +x.

**Control.** All eight of its motors are commanded **0** for the whole episode.
Its actuators exist in the model and are never written.

**Motion — prescribed, open-loop, non-reactive.** At execution step `k`
(`k = 0` at the first control step), *after* that step's physics, the
opponent's ENTIRE state is overwritten:

| | |
|---|---|
| root position | `x = +1.0 − v·Δt·k`,  `y = 0`,  `z = 0.5347` |
| root orientation | yaw 180°, no roll, no pitch (`quat = (0,0,0,1)`) |
| root velocity | `(−v, 0, 0)`, zero angular velocity |
| 8 hinge angles | `q*` = hips 0°, ankles 51.8746° |
| 8 hinge velocities | 0 |

with **`v = 0.68 m/s`** and `Δt = timestep(0.003) × frame_skip(5) = 0.015 s`,
so it advances exactly **1.02 cm per control step**.

`z = 0.5347` and `q*` are not magic numbers: they are the stance this same ant
settles into under gravity at zero torque in this engine, measured once at env
construction (`RunToGoalEnv._settle_opponent`, 3.0 s of simulation from the
asset pose, no noise) and reported by
`rtg_scene.py --report-settle`. They are re-measured on every construction, so
they cannot drift from the asset.

**Therefore the opponent is a rigid, non-reactive, constant-speed obstacle in
the shape of our ant.** It does not walk — it glides in its standing stance. It
does not steer, does not observe our agent, and **no contact can slow it, push
it or knock it over**. Its trajectory is a function of `k` alone and is
therefore bit-identical in every episode of every seed of every arm.

**Two consequences, stated rather than discovered later:**

1. **It is effectively infinitely massive.** Within a control step our agent's
   contacts do push it; the push is then discarded by the snap. A head-on
   collision therefore pushes OUR agent backwards and cannot be won. Measured
   (`gate_e2.py` phase 2): a zero-torque agent is bulldozed from x = −1.00 to
   **x = −3.22**; an agent thrashing with uniform random torques, to −3.85.
   Avoiding the collision is free — the reward has no `y` term and the arena
   has no side walls — so lateral avoidance is a real and cheap tactic, and
   part of what the task asks for.
2. **It cannot fall, and CompetEvo's shared fall rule is therefore applied to
   OUR agent only.** Theirs ends the episode if *either* ant drops below
   z = 0.28. Ours never does (it is held at 0.5347), so the rule would be
   vacuous for it — but inheriting it unexamined would end every episode at
   step 0 if the stance were ever lowered.

### Why a kinematic opponent, and why 0.68 m/s

**Why kinematic.** A scripted opponent that *runs* would need a hand-written
torque controller for a quadruped; no simple one makes an ant locomote, and a
learned one is excluded by E2's spec. A prescribed trajectory is the only
constant-speed opponent that is *fully* specified — zero gains, zero tuning,
zero seed dependence — which is what "every later rung inherits it" requires.

**Why 0.68 m/s.** The task's own clock already demands a speed: 5.0 m (x = −1
to the goal at x = +4) inside 500 control steps × 0.015 s = 7.5 s, i.e.
**0.667 m/s**. The opponent is that clock made physical. It is advanced by 2%
so that running out of time is realised as a **loss to a visible opponent**
rather than as a silent truncation: at 0.68 m/s it crosses x = −4 at control
step **491 of 500** (gated, exact). Beating the opponent and beating the clock
are then the same requirement, which is what makes E2's goal rate directly
comparable to D2's on the same body, reward, distance and clock.

### Is it beatable? Yes on the evidence, and not trivially

**Requirement**: 5.0 m in ≤ 490 control steps = **0.68 m/s** mean speed.

* **Achievable.** `D2_MORPHOLOGY_COMPETENCE.md` measured the *same* ant, the
  *same* CompetEvo reward and the *same* 5.0 m in 7.5 s: after 600 iterations
  at 256 worlds against an idle opponent, **98.3% goal rate at +1.114 m/s** —
  1.64× the requirement.
* **Not trivial.** The same document's 2h-sweep policy (200 iterations, live
  opponent) reached the goal **33.0%** of the time at 0.554 m/s, *below* the
  requirement. So 0.68 m/s sits inside the band this body's policies actually
  span: weak policies lose, good ones win.
* **Not won by standing still.** A zero-torque agent scores 0%, is bulldozed
  backwards, and loses every episode at step 491 (gate phase 2).

**The caveat that must travel with this, stated before the runs, not after**:
D2 reached 98.3% with roughly **77M environment steps** (600 iterations ×
256 worlds × ~500 steps). E2's budget is **5.0M steps per arm**, matching
E1.1 — about **15× less**. A low goal rate at 5.0M steps would therefore be a
statement about the budget, not about either architecture. The
architecture *comparison* is unaffected: both arms get exactly the same
budget, the same opponent and the same reward. D2's engine is mujoco_warp with
Newton/100; E2's is mujoco-py 2.1 with CompetEvo's own PGS/1000, so the
transfer of D2's speed figure is strong evidence, not proof.

---

## 2. What was built

| file | what it is |
|---|---|
| `rower_soccer/t2a_port/rtg_scene.py` | emits `assets/mujoco_envs/rtg_ant.xml` — our ant + the opponent + goal lines + CompetEvo's `<option>` block |
| `design_opt/envs/run_to_goal.py` | `RunToGoalEnv(AntEnv)` — the scripted opponent, CompetEvo's reward, CompetEvo's termination, the opponent-relative observation |
| `design_opt/envs/__init__.py` | registers `run_to_goal` |
| `design_opt/envs/ant.py` | one line: `frame_skip` from `env_specs`, default 4 (E0/E1/E1.1 unchanged) |
| `design_opt/cfg/rtg_{gnn,mlp}_s{1,2}.yml` | the four arms — byte-identical except name and seed |
| `rower_soccer/t2a_port/train_e2_gnn.py` | GNN trainer: their loop + stop file + inline wandb + inline evaluation + inline video |
| `rower_soccer/t2a_port/train_e11_mlp.py` | E1.1's MLP baseline, generalised to `env_dict[cfg.env_name]`, plus the same inline wandb/eval/video |
| `rower_soccer/t2a_port/e2_eval.py` | **the one instrument** — rollout, both protocols, goal/loss/fall rates, and the best/median/worst clip |
| `rower_soccer/t2a_port/e2_video.py` | renders a clip from a checkpoint, as a subprocess |
| `rower_soccer/t2a_port/e2_posthoc.py` | the headline table: body-freeze under the trained policy + both protocols |
| `rower_soccer/t2a_port/e2_wandb.py` | inline wandb from `.venv-gpu` |
| `rower_soccer/t2a_port/gate_e2.py` | the gate, 7 phases |

### The one structural idea

`Robot.load_from_xml` (`khrylib/robot/xml_robot.py:511`) parses
`tree.getroot().find('worldbody').find('body')` — **the first body element and
nothing else**. So the opponent is written as a *sibling* of our ant in the
same MJCF: it is present to the physics, to collisions and to the renderer, and
completely invisible to Transform2Act's design machinery. It can never be
mutated, indexed, actuated or observed as a node by the skeleton or attribute
stages. That is why the opponent is XML rather than a second `Robot`, and it is
what makes E3-E5 (an evolving agent against a fixed opponent) a change of
policy rather than a change of representation.

### What is CompetEvo's, verbatim

Taken from `rower_soccer/competevo_port/run_to_goal_env.py` and `scene.py`,
themselves gated against CompetEvo's own code in
`competevo_port/tests/test_parity.py`:

* the registration — agent 0 at `(−1,0,0.75)` facing +x, agent 1 at
  `(+1,0,0.75)` yawed 180°; goal lines at x = ±4 as real colliding cylinders;
* `<option integrator="RK4" timestep="0.003" solver="PGS" iterations="1000"/>`
  — their world options **exactly**. mujoco-py 2.1 implements PGS, so unlike
  the mujoco_warp port nothing had to be swapped;
* the dev merger's collision trick: agent 0's geoms `contype 1 / conaffinity 0`,
  agent 1's `contype 0 / conaffinity 1` — neither ant self-collides, the two DO
  collide with each other;
* the dense reward `forward − 0.5·Σa² − contact + 1.0`, `forward` on the torso
  **subtree COM** over `dt`, the control cost on the RAW (unclamped) action;
* `contact_cost = 0`, because their `cfrc_ext` is never populated and the term
  is a constant zero in every CompetEvo run (re-measured here: `max|cfrc_ext|`
  is exactly `0.000e+00`);
* the sparse reward **±1000** iff exactly one agent crosses its goal line;
* termination on a fall (root z < 0.28), on a goal, or on a non-finite state;
  truncation at 500 control steps.

### What is deliberately NOT CompetEvo's, and why

1. **The fall rule applies to our agent only** (§1, consequence 2).
2. **Transform2Act's `done_condition.max_ang` / `min_height` / `max_height` are
   not set.** CompetEvo's run-to-goal has no tilt condition, and two
   termination rules for one task would be untraceable. E1/E1.1 keep
   `max_ang: 60` because they run Transform2Act's OWN task; E2 runs CompetEvo's.
3. **The observation.** CompetEvo's is a flat 31-vector
   `[own qpos (15) | own qvel (14) | opponent root x,y (2)]` in the world
   frame. Transform2Act's is a per-body matrix that deliberately excludes the
   root's x,y so a policy transfers across designs and positions; injecting
   absolute world x,y would break that. E2 instead appends **three** columns,
   broadcast identically to every node row:
   `(opp_com_x − own_com_x, opp_com_y − own_com_y, 4.0 − own_com_x)` — the
   opponent's position relative to ours, and the distance still to run. That is
   CompetEvo's information content in a translation-invariant frame. The node
   row goes from E1's 22 columns to **25**. Both arms are fed exactly these
   columns, so the comparison is unaffected.

---

## 3. The gate

```
cd /workspace/Transform2Act && source env-gpu.sh
.venv-gpu/bin/python /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/gate_e2.py
```

`runs/d3_e2_rtg/logs/gate_e2.log` — **41 checks, 0 failed**, in seven phases,
each with at least one negative control because a gate that cannot fail is not
evidence.

| phase | what it establishes | headline |
|---|---|---|
| 1 scene | the merged scene is our creature twice over | 27 bodies, 16 motors; `Robot` sees exactly our 13 and none of the opponent's; our mass/inertia/ipos/geom-size/gear vs the gated single-ant asset, max delta **0.000e+00**; CompetEvo's spawn poses, goal lines and `<option>` block; the two collision masks |
| 2 opponent | the script is the script | opponent root x follows `1 − v·Δt·k` with max error **0.000e+00**; **bit-identical trajectory** under a passive and a thrashing agent (non-reactivity); COM = root to 7.6e-9; crosses x = −4 at step **491**; stance is a rest state. NEG: `opponent_speed 1.0` moves the crossing to step **334** |
| 3 frozen body | morphology really is frozen | 20 episodes of **destructive random design actions** (every body told to add or remove, full-range attribute kicks): all **134 mjModel arrays identical**, XML byte-identical, 13 bodies throughout, stage sequence exactly 5 skeleton + 1 attribute every episode. NEG: without `force_identity_design` the same actions change **96 arrays** and give body counts 17/17/15/12/18 |
| 4 reward | it is CompetEvo's, term by term | over 200 steps of random actions, `reward == forward − 0.5·Σa² + 1.0 (±1000)` to max error **4.4e-16**; `max|cfrc_ext| = 0` |
| 5 termination | the three end conditions | a standing agent with a stationary opponent truncates at exactly 500; crossing x = +4 ends the episode with r = +1001; root z < 0.28 ends it as a fall |
| 6 observation | the three appended columns | equal `(opp_dx, opp_dy, goal_dx)` **exactly**, identical on every node row, first 13 sim columns are E1's untouched, row width 25 = 4 + 16 + 5 |
| 7 E1.1 regression | E2's edits to shared files are no-ops | E1.1's env still `AntEnv`, frame_skip 4, dt 0.04, sim_obs_dim 13, node row 22 |

**Phase 3 is the gate the user asked for**: the body is unchanged from the
first step to the last. It is run twice — here with destructive *random*
actions before training, and again in `e2_posthoc.py` with **each arm's own
trained policy** after training, because a gate that only ever saw random
actions could miss a policy that learned some other path.

---

## 4. The comparison, and how it is measured

**The MLP arm is built, not borrowed.** D2's run-to-goal numbers come from
self-play with CompetEvo's own MLP; a GNN-vs-scripted result is not comparable
to an MLP-vs-self-play one, because a different opponent is a different task.
So E2 runs its own matched MLP arm on the identical setup: same scripted
opponent, same body, same reward, same episode structure, same 5.0M-step
budget, same batch (50,000) and minibatch (2,048), same gamma/GAE-lambda/clip
read from the same cfg. The four cfgs are byte-identical except for their name
and seed. Only the policy architecture differs.

**E1.1 is the precedent and the warning**: there the verdict *flipped* with the
MLP's batching — the GNN won 2.1-2.6× against published PPO-MuJoCo batching and
lost 1.18× against Transform2Act's own. The matched batching is therefore the
baseline here, as E1.1 established.

**Three differences between the arms that are NOT "architecture" in the narrow
sense**, carried over from E1.1 and stated so they are not mistaken for it:

1. **The MLP normalises its observations and the GNN does not.**
   `transform2act_agent.py:109` sets `running_state = None` — Transform2Act
   applies no observation normalisation at all — while the MLP arm carries the
   Welford normaliser the published PPO recipe uses. That is each
   architecture's own published configuration, not a handicap chosen here. It
   matters slightly more in E2 than in E1.1 because the three appended columns
   are distances: they span about ±8 m, which is inside the ±10 the
   `clip_qvel: true` velocity columns already span, so they are not out of
   family with what the GNN already sees unnormalised — but the GNN sees them
   raw and the MLP sees them standardised.
2. **The action spaces differ.** The MLP writes the 8 actuators directly; the
   GNN emits one scalar per node over 13 nodes and discards 5.
3. **The GNN carries skeleton and attribute heads that take gradients from
   actions the env throws away**, which is inherent to "run but forced to
   identity". Whether SKIPPING the design stages would close any gap is
   untested here, exactly as in E1.1.

**Protocol** (`e2_eval.evaluate`, called from all three places):

* **mean-action is the headline**, stochastic reported beside it, and each
  arm's **learned action std** recorded, because the two protocols disagreed in
  opposite directions in E1.1;
* Transform2Act's `exec_R_eps` is a separate mean-action *evaluation* pass, not
  a training return, and is never compared against the MLP trainer's stochastic
  `exec_R_eps`. **No number in the results table comes from a training log**;
* one instrument for every arm, 20 episodes each, identical episode seeds;
* the task's own success metric — **goal rate, loss rate, fall rate** — beside
  return, because a return alone does not say whether the thing reaches the
  goal.

---

## 5. Logging

Metrics and video go to **one wandb run per arm**, logged inline from the
training process in the **same `wandb.log` call**, with `epoch` declared as the
step metric and no explicit `step=`. E0/E1 needed separate `_media` runs
because they backfilled metrics post-hoc and then rendered video in a second
pass, so the video landed behind the run's current step and wandb dropped it.
E2 has no such gap: both trainers log as each epoch finishes.

Two environment facts that had to be solved for this (`e2_wandb.py`):
`/workspace/Transform2Act/.venv-gpu` has mujoco-py and **no wandb**, so wandb
is installed beside it at `/workspace/t2a_pylibs` and put first on `sys.path`;
that pulls in protobuf ≥ 4, which breaks `tensorboardX`'s protobuf-3 generated
code, so `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is forced first.

Video is rendered in a **subprocess** off a saved checkpoint — the GNN trainer
holds a CUDA context and forks sampler workers every epoch, and an offscreen GL
context in that process is a way to lose a run — and the mp4 comes back and is
logged into the trainer's own run at that epoch.

**With morphology frozen all three panels are the same creature**, so the clip
shows **gait and tactics** — how it starts, whether it dodges the scripted
opponent, whether it crosses the line — and **not** design variation, which is
what the same clip meant in E0 and E1.

**Verified through the wandb API, not by an exit code**
(`rower_soccer/t2a_port/e2_wandb_verify.py`, run from `/workspace` because the
repo's own `wandb/` artefact directory shadows the package):

```
[OK] d3_e2_mlp_s1  state=finished  metric rows=2  video/best_median_worst rows=1
     video-in-summary=True  last epoch=9
```

— metrics and the video in **one run's history**, from one training process,
with no `_media` split. That check is run again on every arm at the end.

**One book-keeping fact, recorded rather than hidden.** The six arms were
launched once, run 3-36 epochs, and stopped cleanly by stop-file so that two
wandb keys could be renamed: both trainers were logging a key called
`exec_R_eps`, and it means a **mean-action evaluation** in the GNN arm and a
**stochastic training return** in the MLP arm — precisely the confusion that
nearly produced a wrong answer in E1.1. They are now
`e2/exec_R_eps_MEANACTION_eval` and `e2/train_R_eps_STOCHASTIC`, and the
comparable curve is `e2/eval_*` from the shared instrument. The aborted
attempt's logs are quarantined in
`runs/d3_e2_rtg/logs/aborted_keyrename/`; **no wandb run was deleted** — the
same run ids were resumed, so each arm's history carries ~3-36 rows from the
aborted attempt before the real run's epoch 0. Nothing in the results below
comes from a wandb series.

---

## 6. A hazard in the task, found before the results and measured

**A fall is worth ~+826 more than a loss, so "fall before step 491" is a local
optimum.** Measured on the idle negative control's own 20 episodes
(`runs/d3_e2_rtg/posthoc/idle.json`):

| ending | n | mean return | mean length |
|---|---|---|---|
| our agent falls | 3 | **+178.2** | 319 |
| the opponent scores | 17 | **−647.6** | 491 |

An episode that ends on a fall never reaches step 491, so it never pays the
−1000; it keeps the +1.0/step survive bonus it has already banked and stops.
An episode that survives to 491 pays the −1000 in full.

**This is CompetEvo's own rule set, not something E2 introduced**: their
`_get_done` ends the episode on a fall and `goal_rewards` pays nobody in that
case, so "fall rather than lose" is available in their task too. What the
*scripted* opponent changes is that its goal is **certain** rather than
contingent — it scores at step 491 in every episode unless our agent has
already crossed — which sharpens a contingent incentive into a reliable one.
D2 never saw it because D2 trained against an **idle** opponent that never
scores, so the −1000 never fired; that is part of why D2's 98.3% is not a
number E2 can inherit.

It is stated here rather than fixed, because changing the termination rule
after the arms were launched would make E2 uninterpretable. **Every later rung
inherits it along with the opponent**, and E3 should decide deliberately
whether to keep it — the two obvious alternatives are to pay the loser its
−1000 on a fall as well (removing the dodge) or to drop the sparse term when an
episode ends on a fall for both sides (keeping CompetEvo's rule and accepting
the incentive).

---

## 7. THE RESULT: neither controller learns the task, and the return difference is a mixture effect

```
./runs/d3_e2_rtg/collect.sh          # e2_posthoc.py per arm, then e2_compare.py
```

**One instrument, 20 episodes per arm, identical episode seeds, the same frozen
13-body / 8-motor ant, the same scripted opponent, the same reward, the same
5.0M-step budget.** Body freezing re-verified per arm under each arm's OWN
trained policy: **134 mjModel arrays identical on all four trained arms.**

| arm | mean-action R | sd | **goal** | lost | fell | ep len | furthest forward | of the 5.0 m | action std | stochastic R |
|---|---|---|---|---|---|---|---|---|---|---|
| **GNN** s1 | −655.1 | 225.0 | **0.00** | 0.95 | 0.05 | 490.9 | 0.22 m | 4.4% | 0.551 | −972.7 |
| **GNN** s2 | −536.8 | 345.0 | **0.00** | 0.85 | 0.15 | 479.6 | 0.30 m | 6.0% | 0.559 | −1116.1 |
| **MLP matched** s1 | −194.5 | 423.3 | **0.00** | 0.30 | 0.70 | 240.8 | 0.46 m | 9.1% | 0.633 | −597.7 |
| **MLP matched** s2 | −203.9 | 392.1 | **0.00** | 0.40 | 0.60 | 334.4 | 0.60 m | 11.9% | 0.624 | −514.0 |
| *idle, zero torque* | −523.7 | 307.0 | **0.00** | 0.85 | 0.15 | 465.2 | 0.08 m | 1.5% | 0 | −523.7 |

Seed means (mean-action): **GNN −595.9, MLP-matched −199.2**, idle −523.7.

### 7a. The headline is the goal column, and it is zero everywhere

**Neither architecture reaches the goal in a single one of 40 evaluation
episodes.** The best episode any arm produced went **1.95 m of the 5.00 m
required** (MLP s2); the GNN's best was 1.29 m. Averaged, the furthest either
ever got from its own start line is 0.22-0.60 m. At 5.0M steps, on CompetEvo's
reward, on this body, **neither controller learns to run**.

That was pre-registered as the budget risk in §1 and it fired: D2 needed roughly
**77M** environment steps for 98.3% on the same body, reward, distance and
clock, and E2 spent **5.0M** per arm to stay matched to E1.1. This is a
statement about the budget, not about either architecture — but it means E2's
intended question ("does E1.1's 18% GNN deficit persist on a task with an
opponent and a goal line?") **cannot be answered on goal rate**, because
neither arm can do the task.

### 7b. The GNN is statistically indistinguishable from doing nothing

GNN −595.9 against the zero-torque control's −523.7: **episode-level
Welch t = 0.87** (n 20 vs 40). After 5.0M steps the Transform2Act controller has
not separated itself from a policy that emits no torque at all. It does move
slightly further forward than idle (0.22-0.30 m against 0.08 m), and it does so
while paying control cost, which is why its return is if anything *worse*.

### 7c. The MLP "wins" by 3.0x, and the mechanism is falling over

MLP-matched −199.2 against GNN −595.9: **+396.7 per episode, Welch t = 5.03**
(n 40 vs 40), seed ranges non-overlapping. Quoted alone that reads like a
repeat of E1.1's verdict. **It is not, and the decomposition says why.**
Splitting every arm's episodes by how they ended:

| arm | fell: n | fell: mean R | fell: len | lost: n | lost: mean R | lost: len |
|---|---|---|---|---|---|---|
| GNN s1 | 1 | **+279.6** | 488 | 19 | −704.3 | 491 |
| GNN s2 | 3 | **+250.6** | 415 | 17 | −675.7 | 491 |
| MLP s1 | 14 | **+74.0** | 134 | 6 | −821.1 | 491 |
| MLP s2 | 12 | **+99.5** | 230 | 8 | −658.9 | 491 |
| idle | 3 | **+178.2** | 319 | 17 | −647.6 | 491 |

Within **every** arm, ending on a fall is worth roughly **+750 to +900** over
surviving to the opponent's goal, because a fall stops the episode before step
491 and so never pays the −1000 (§6). And **conditional on the ending, the GNN
scores HIGHER than the MLP on falls** (+280/+251 against +74/+100 — it survives
longer before falling, so it banks more of the +1.0/step). The entire 3.0x
return gap is therefore the **mixture**: the MLP falls in 60-70% of episodes and
the GNN in 5-15%.

**So the correct sentence is not "the MLP controller is better on this task".
It is: neither controller solves the task, and the MLP found the degenerate
optimum the task's own termination rule offers while the GNN did not.** Reading
the return column alone would have produced the wrong sentence, which is the
same failure mode E1.1 recorded and the reason the success metric was required
beside the return.

### 7d. The two protocols disagree, in opposite directions per arm

Mean-action is the headline; the stochastic column moves each arm differently
because each learns its own noise (GNN 0.551/0.559, MLP 0.633/0.624):

* the **GNN gets much worse** stochastically (−595.9 → −1044.4), because sampled
  actions do not buy it the early fall and it still pays the noise;
* the **MLP gets worse too but less** (−199.2 → −555.9), and its fall rate goes
  *up* (0.60-0.70 → 0.75-0.80);
* **idle is identical by construction** (no distribution), which is the check
  that the two protocols are the same code path.

The MLP still leads on both protocols, so the ordering is protocol-independent
here — but the 3.0x becomes 1.9x, and neither number means what it looks like
without §7c.

