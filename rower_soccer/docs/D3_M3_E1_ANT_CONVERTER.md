# D3 M3 E1 — the CompetEvo ant inside Transform2Act's `Robot`

*2026-08-29. Deliverable: a gated converter, not a training run. Every number
below is from a command in this document; where something is untested it says
so.*

The reason this exists is `PLAN_D3_M3.md` §0c: Transform2Act's ant is a torso
with four single-segment limbs (5 bodies, 6 joints, 4 motors); the ant D1 and
D2 train is the DeepMind/gym ant (13 bodies, 9 joints, 8 motors, four legs of
three links). Every rung of M3 from E3 onward needs **our** ant in **their**
representation, because the soccer creature has to be the creature D1 and D2
already train.

## What was built

| file | what it is |
|---|---|
| `rower_soccer/t2a_port/competevo_to_t2a.py` | the converter |
| `rower_soccer/t2a_port/gate_competevo_ant.py` | the gate, phases A–D + negative controls |
| `rower_soccer/t2a_port/render_e1_ant.py` | the render |
| `rower_soccer/t2a_port/smoke_e1_train.py` | "does training start and take gradients" |
| `Transform2Act/assets/mujoco_envs/ant_competevo.xml` | the converted asset (generated) |
| `Transform2Act/design_opt/cfg/ant_competevo.yml` | their ant task, our ant |

Three changes were needed **outside** the converter. All three are in
`/workspace/Transform2Act` and all three are described below; two of them are
bugs in code this project already depends on.

## 1. What the converter does — and the finding that makes it small

`dev_ant_body.xml` already satisfies almost every structural invariant
`khrylib/robot/xml_robot.py` imposes, by accident of CompetEvo's own naming:

* Bodies are already `0`, `k`, `1k`, `11k`, which is **exactly** what
  `Body.reindex()` generates. `sync_node()` renames nothing.
* Joints are already `<body>_joint`; motors are already named for their joint.
* At most one hinge and exactly one capsule per body (root: one sphere).
* Every joint sits at its body's origin — `Joint.__init__` *asserts* this.
* Each capsule's far end coincides with its single child's origin. This is the
  one that mattered: `Robot` **redraws every capsule** from `bone_start` to
  `bone_end = mean(children's origins)` on the first attribute transform, so a
  capsule that did not already end there would be silently moved. It does.

So the conversion is four things, none of them a restructuring:

1. **Give it a `<worldbody>`.** `dev_ant_body.xml` is a fragment — its
   `<worldbody>` is *commented out*, because CompetEvo's scene builder splices
   the `<body>` into a merged arena. `Robot.load_from_xml` does
   `getroot().find('worldbody').find('body')` and would crash. The shell
   (floor, light, skybox, `track` camera, assets) is taken verbatim from
   `Transform2Act/assets/mujoco_envs/ant.xml`, because the *task* is theirs.
2. **`conaffinity="0"` on the geom default.** The asset says `1`, but that
   value is never compiled: `scene.py`'s `_dev_agent_default_xml` overrides it
   per agent with `conaffinity=i, contype=1-i`, so agent 0 — the creature D1/D2
   simulate — has `contype=1, conaffinity=0`: self-collision **off**, floor
   collision on. That is also what Transform2Act's ant uses. Emitting the
   asset's literal `1` would have given our ant self-colliding legs that
   D1/D2's ant does not have. The gate compares against
   `scene.dev_run_to_goal_xml()` rather than against the asset file, which is
   what caught this; it is negative control N5.
3. **Rewrite the root's placement.** Root `pos`/`euler` are placeholders —
   `_dev_ant_body_xml(agent_id, pos, euler)` overwrites both with the
   registered init pose. They are not part of the creature, so they are a CLI
   knob, defaulting to `0 0 0.75` / `0 0 0`.
4. **Canonicalise names anyway.** Steps 1–3 are all this asset needs, but the
   renamer is generic (BFS, T2A's `reindex` rule, joints and motors carried
   along) so the converter is not a bet on CompetEvo's names happening to
   match. `--require-name-noop` asserts it is a no-op here, and it is.

Motor **order** is left exactly as the source has it. It is not load-bearing for
Transform2Act (`action_to_control` looks actuators up by name, and
`add_child_to_body` appends new motors at the end anyway), but it means
`data.ctrl` indices coincide between the two models so the physics gate replays
one recorded action array on both with no permutation.

**Kept in LOCAL coordinates, deliberately.** `xml_robot.py` supports both and
reads `compiler/@coordinate`, but every XML Transform2Act ships is `global`, and
MuJoCo removed global coordinates in 2.3.3 — which is why
`t2a_port/xml_global_to_local.py` exists for the batched port. Our source is
already local and `Robot` never rewrites the attribute, so **every design
descended from this ant compiles under modern MuJoCo directly**, with no
conversion step and no `assert_no_rotation` landmine from the root's `euler`.

### Is the conversion lossy?

**No, on the creature.** Phase A compares 95 compiled model arrays and finds
zero difference (largest residual exactly `0.000e+00`). Their `Robot` dialect
expresses two-segment legs without compromise, because a chain of bodies with
one capsule each is precisely the bone model `Robot` implements.

**Yes, in three respects that are *not* the converter's doing** and that E1's
reading has to carry:

* **Engine.** E1 trains in their stack (mujoco-py 2.1 / mujoco210), where a
  capsule's caps count as ¾ of a sphere. The converted ant weighs
  **0.878710 kg there and 0.910880 kg under our mujoco 3.12** — legs 3.5%
  lighter than the ant D1/D2 train. Correctable (phase D reproduces their mass
  and inertia to 1e-14 with `xml_global_to_local.legacy_capsule_inertial`), but
  not corrected *inside* their stack, because their stack is the one that is
  wrong.
* **Contact solver.** With mass and inertia matched to 1e-14 the two engines
  agree to `1.155e-14` for all 17 contact-free steps and separate the instant
  the first foot touches the floor (step 18: `1.155e-14 → 2.297e-03` in one
  step). No XML can fix that.
* **Floor margin.** Transform2Act's `ant.xml` gives its floor no `margin`, so
  it inherits `margin="0.01"` from the file's `<default><geom>`; CompetEvo's
  floor is outside every class and gets MuJoCo's `0`. **Our ant's feet touch
  down 1 cm earlier on their floor.** Visible in the render: settled torso
  height 0.551 (CompetEvo floor) vs 0.561 (theirs), lowest geom z 0.0097 vs
  0.0197 — a difference of exactly 0.01. Left as theirs, because E1's arena is
  theirs and E0 runs on the same floor; `--floor-margin` is not implemented,
  and if E3+ needs the CompetEvo contact regime this is the knob to add.

**Two Transform2Act body-plan features our ant does not use, stated because
they bound what E1 can find:** their `add_body_condition.max_nchild: 2` means
the torso (4 children) can never gain a limb, and `max_body_depth: 4` caps legs
at three links — so the skeleton stage can add a 4th link to a leg or a 2nd
child to a stub or a hip, and can remove leaves, but cannot grow a 5th leg from
the torso. Not tuned; inherited from `ant.yml` unchanged on purpose.

## 2. The gate

```
PYTHONPATH=. .venv/bin/python -m rower_soccer.t2a_port.gate_competevo_ant --ours
cd /workspace/Transform2Act && source env-gpu.sh && \
  .venv-gpu/bin/python .../t2a_port/gate_competevo_ant.py --theirs
PYTHONPATH=. .venv/bin/python -m rower_soccer.t2a_port.gate_competevo_ant --cross
```

### A — same creature (our venv, tol 1e-12): PASSED

Compared against **the model D1 and D2 actually compile**
(`scene.dev_run_to_goal_xml(n_agents=1)`), not against the asset file. The field
list is **derived from `mujoco.introspect`**, not hand-written: every `mjModel`
array whose leading extent is `nbody`/`ngeom`/`njnt`/`nv`/`nq`/`nu`/`nactuator`/
`nout`, paired by name (bodies, joints, actuators) or through the owning body
(geoms — Transform2Act's `Geom.sync_node` *deletes* geom names, so they cannot
be paired by name) or through the owning joint (dofs, qpos slots).
`two_stage_pipeline.differing_fields` is the precedent: it found 21 arrays
differing between two designs of one topology, including `body_iquat` and
`geom_sameframe`, so a hand list would miss some.

```
[PASS] A: all 95 indexed arrays equal    largest residual 0.000e+00
       skipped 44 arrays by rule (int-typed, name ends in adr/id/num:
       indices into other tables, plus geom_group/rgba/matid — colour and
       render layer, since the scene tints agents by team)
[PASS] A: total robot mass       0.910880083 kg over 13 bodies
                                 (per-body 0.039158–0.327249)
[PASS] A: topology               13 bodies, 9 joints (1 free + 8 hinge), 8 motors
[PASS] A: actuator gears         all 8 at 150
[PASS] A: hinge ranges           4 hips [-30,30] deg, 4 ankles [30,70] deg
[PASS] A: capsule radii          all 12 at 0.08
[PASS] A: capsule half-lengths   [0.141421, 0.282843]
[PASS] A: torso sphere radius    0.25
[PASS] A: nexclude/npair/neq/ntendon/nsite/nsensor/nmocap/nflex/nuserdata equal
```

The count checks are there because a contact exclusion or an equality
constraint in one model and not the other would never show up in a per-element
comparison.

### B — same physics (our venv, tol 1e-9): PASSED, and it is **exactly zero**

500 `mj_step`s, identical `qpos`/`qvel`, identical recorded pseudo-random
actions, **identical arena**.

```
[PASS] B: all 16 geoms (arena included) identical
[PASS] B: trajectories agree for the whole rollout   max|dqpos| 0.000e+00 over 500 steps
[PASS] B: same contact set at every step            500/500 steps
[PASS] B: the ant actually moved                    root moved 0.243 m, joints swept 0.959 rad
```

Making the arena identical is not a formality and it is where the floor-margin
finding came from. Run each robot on *its own* floor and the two rollouts are
bit-identical for 67 steps, then one foot-floor contact appears in one and not
the other and they separate to 0.109 in qpos. The gate therefore transplants the
CompetEvo arena into the converted model **and writes every geom attribute the
converted file's `<default>` would otherwise inject explicitly, read off the
reference's own compiled model**, so the arena is the CompetEvo arena and not a
hybrid. `B: all 16 geoms identical` checks that it worked rather than assuming
it.

### C — `Robot` can mutate it (their venv, mujoco-py 2.1): PASSED

```
[PASS] C: converted XML compiles and steps in their stack
          nbody 13 nq 15 nv 14 nu 8, mass 0.878710174
[PASS] C: Robot round-trip leaves every model array unchanged   134 arrays
[PASS] C: identity attribute transform is a no-op on the compiled model
[PASS] C: identity attribute transform through the TRAINING path (pad_zeros) is a no-op
[PASS] C: attribute transform changes a length, and the COMPILED model shows it
          bone_offset |.| 0.565685 -> 0.608822 (+0.043136);
          compiled capsule length +0.043137
[PASS] C: ...and no other geom changed size
[PASS] C: skeleton ADD compiles and steps      bodies 13->14, nu 8->9, nq 15->16
[PASS] C: the added limb is actuated
[PASS] C: skeleton REMOVE compiles and steps   bodies 13->12, nu 8->7
[PASS] C: AntEnv runs skeleton -> attribute -> execution on our ant
          5 skel + 1 attr + 100 exec steps, R=21.421, final bodies 16
```

### D — cross-engine (both venvs): PASSED, and it is the honest caveat

```
[PASS] D: mass differs by exactly the known MuJoCo 2.1 capsule-cap bug
          theirs 0.878710 kg, ours 0.910880 kg, ratio 0.96468
[PASS] D: the legacy-inertial closed form reproduces their mass exactly
          max|dmass| 1.697e-14 kg over 14 bodies
[PASS] D: ...and their inertia exactly              max|dI| 4.826e-15
[PASS] D: same first-contact step in both engines   step 18
[PASS] D: contact-free flight agrees to machine precision (steps 0-17)
          max|dqpos| 1.155e-14 corrected, 3.411e-04 uncorrected
```

The inertia agreement is an **independent confirmation of
`legacy_capsule_fit.py`**, whose closed form was fitted on a synthetic capsule
grid and had never been checked on a different robot.

### N — negative controls: all rejected

Five model corruptions that the phase-A comparison must reject, and four
structural ones the *validator* must reject before anything compiles:

```
[PASS] N: rejects one motor gear 150 -> 151
[PASS] N: rejects one capsule radius 0.08 -> 0.0801
[PASS] N: rejects one ankle axis sign flipped
[PASS] N: rejects one hinge range 30 70 -> 30 71
[PASS] N: rejects the asset's literal conaffinity=1  <-- the bug this gate caught
[PASS] N: validator rejects a capsule that does not end at its child's origin
[PASS] N: validator rejects a joint moved off its body origin
[PASS] N: validator rejects a second hinge on one body
[PASS] N: validator rejects a rotated non-root body frame
```

The gear control **failed on first run**, and that is the most useful thing the
gate did. `actuator_gear` is `('nout', 6)` in MuJoCo 3.x, `actuator_gainprm` is
`('nactuator', …)`, and only `actuator_ctrlrange` is `('nu', 2)` — so keying the
field sweep on `nu` compared 2 of the 31 actuator arrays and silently skipped
the gears. This is the same class of mistake as the hand-maintained field list
`two_stage_pipeline` was written to avoid, one level up: deriving the field
list from MuJoCo is not enough if the *dimension names* are guessed.

## 3. Three changes outside the converter

### 3a. `khrylib/robot/xml_robot.py` — a real bug, latent until our ant

`Body.get_params(pad_zeros=True)` appends **one zero for a body with no
joints**. `Body.set_params(pad_zeros=True)` did not consume it. Every field
after the pad was therefore read one slot early, so for a jointless body
`size` was read from the pad and `ext_start` from `size`.

No robot Transform2Act ships has a jointless body — hopper/swimmer/gap have a
3-joint root and a hinged child; their ant has a free root and four hinged
children — so the asymmetry never fired. **Our ant has four**: the leg stubs
between torso and hip. Without the fix, one attribute transform silently reset
each stub capsule to radius **0.065** (from 0.08) and `ext_start` **0.143**
(from 0), i.e. deformed the robot on the first design step of every episode.

The fix is the missing three lines, symmetric with `get_params`:

```python
if pad_zeros and len(self.joints) == 0:
    params = params[1:]
```

It is a strict no-op for every robot they ship (verified: none has a jointless
body, and `add_child_to_body` clones joints so none can appear). Gate check
`C: identity attribute transform through the TRAINING path (pad_zeros) is a
no-op` fails without it and passes with it. Note that `Robot.set_params` — the
path `xml_robot.py`'s own `__main__` demo uses — pads nothing and was always
fine, which is why the bug survived their own smoke test.

### 3b. `rower_soccer/t2a_port/xml_global_to_local.py` — two bugs

* `convert(..., legacy_inertial=True)` **returned early for a non-global
  input**, silently dropping the correction. Harmless while every Transform2Act
  asset was global; not harmless now that `ant_competevo.xml` is local by
  construction. The coordinate shift is now conditional and the legacy pass runs
  either way.
* Both legacy passes assumed MuJoCo's default density of **1000**. That is right
  for `hopper.xml`, which sets no density — and wrong by a factor of 200 for
  `ant.xml` and for our converted ant, whose `<default><geom>` says
  `density="5.0"`. `_default_density()` now reads it, and raises rather than
  guessing if a `<default class=…>` sets its own. Measured before the fix:
  `max|dmass|` 12.92 kg against their 0.879 kg robot.

  **This means `two_stage_pipeline.compile_design` and `xml_to_fields` would
  have produced 200x-wrong masses had anyone pointed them at an ant.** They
  have only ever been run on hopper, so no existing number is affected — but
  this is worth knowing before E1's designs go anywhere near the batched port.

### 3c. `design_opt/envs/ant.py` — one default-preserving line

```python
self.model_xml_file = self.env_specs.get(
    'model_xml_file', 'assets/mujoco_envs/ant.xml')
```

so a cfg can point the same task at another body. `ant.yml` sets no
`env_specs`, so E0 on their ant is unaffected.

`design_opt/cfg/ant_competevo.yml` is `ant.yml` with only `env_specs` added:
the converted asset, and `init_height: false`. Their ant is reset to z=0.4,
which is right for a torso of radius 0.25 with flat single-segment limbs; ours
is the gym ant and both gym and CompetEvo start it at 0.75, so at 0.4 it begins
interpenetrating the floor. Everything else — `robot_param_scale`,
`done_condition.max_ang: 60`, batch size, annealing — is theirs, unchanged, and
**untuned for a quadruped**; changing the optimiser at the same time as the
creature would make E1's answer uninterpretable.

## 4. What the render shows

`MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m rower_soccer.t2a_port.render_e1_ant`
→ `/tmp/claude-0/-root/453bc0de-a27f-4894-ad03-7d048158ee36/scratchpad/e1_ant_render.png`
(1440x2000, 5 rows x 3 views; the two XMLs it renders are written beside it as
`e1_render_src.xml` and `e1_render_mutated.xml`).

Rows 1 and 2 are the CompetEvo ant as D1/D2 compile it and the converted ant, at
t=0, from top-down / three-quarter / low side. **They are the same creature.**
Top-down: a torso sphere with four legs radiating on the 45-degree diagonals,
each visibly a chain of three capsules with the seams where the links meet, same
lengths and same radii in both. The only differences are the floor texture
(their green `grid_new` vs CompetEvo's white `MatPlane`) and the red goal rod at
x=-4 that only exists in the CompetEvo scene.

Rows 4 and 5 are the same two after 2 s of settling under gravity with zero
control. Both stand on their feet with the ankles bent knee-up, foot-down, as
the gym ant does; nothing is through the floor and nothing is inside-out.
Settled torso z 0.551 vs 0.561 and lowest geom z 0.0097 vs 0.0197 — a difference
of **exactly the 0.01 floor margin** described above, which is a satisfying
visual confirmation of a number that came out of the physics gate.

Row 3 is a mutated design — one added limb via `Robot.add_child_to_body`, then
every bone offset lengthened in x through the same `pad_zeros` path training
uses. It renders as a five-limbed ant with longer, splayed legs. It compiles and
steps. Worth having looked at, because it is what training will actually
produce, and it is the row that would have shown the jointless-body bug of §3a
as visibly shrunken stubs had it still been there.

## 5. Reproducing it, and where the Transform2Act-side files are versioned

`/workspace/Transform2Act` is a vendored reference checkout, so the four files
this task put there are mirrored into this repo the way `abs_displacement.patch`
and `hopper_gpu_s2.yml` already were:

| in this repo | goes to |
|---|---|
| `rower_soccer/docs/t2a/ant_competevo.yml` | `design_opt/cfg/ant_competevo.yml` |
| `rower_soccer/docs/t2a/ant_competevo.xml` | `assets/mujoco_envs/ant_competevo.xml` (or regenerate, below) |
| `rower_soccer/docs/t2a/e1_transform2act_side.patch` | `git apply` — the two code changes of §3a and §3c |
| `rower_soccer/docs/t2a/e1_ant_render.png` | the render of §4, versioned so it can be looked at without a GPU |

```bash
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
cd /workspace/utmist-vc2-phase2

# regenerate the asset (deterministic; --require-name-noop asserts the source is
# already in T2A's naming, which is the finding of section 1)
PYTHONPATH=. .venv/bin/python -m rower_soccer.t2a_port.competevo_to_t2a \
    --out /workspace/Transform2Act/assets/mujoco_envs/ant_competevo.xml \
    --require-name-noop

# gate: phases A, B and the negative controls
PYTHONPATH=. .venv/bin/python -m rower_soccer.t2a_port.gate_competevo_ant --ours
# gate: phase C, and record their engine's trajectory for D
cd /workspace/Transform2Act && source env-gpu.sh && \
  .venv-gpu/bin/python \
  /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/gate_competevo_ant.py --theirs
# gate: phase D, cross-engine replay
cd /workspace/utmist-vc2-phase2 && \
  PYTHONPATH=. .venv/bin/python -m rower_soccer.t2a_port.gate_competevo_ant --cross

# render
MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m rower_soccer.t2a_port.render_e1_ant

# smoke (2 epochs at min_batch_size 4000; ~5 min on a GPU shared with E0)
cd /workspace/Transform2Act && source env-gpu.sh && \
  .venv-gpu/bin/python \
  /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/smoke_e1_train.py
```

E1's real run, when it is time (**not started**):

```bash
cd /workspace/Transform2Act && source env-gpu.sh
.venv-gpu/bin/python design_opt/train.py --cfg ant_competevo --num_threads 20
```

## 6. Not tested / not claimed

* **No training run.** The smoke is 2 epochs at `min_batch_size` 4000 against
  E1's real 50,000 x 1,000, and it passes:

  ```
  [PASS] env built on our ant   13 bodies, attr_design_dim 5, sim_obs_dim 13,
                                8 actuators, xml assets/mujoco_envs/ant_competevo.xml
  [PASS] skeleton  head receives a finite, non-zero gradient   17 tensors, ||g|| 3.8181e-03
  [PASS] attribute head receives a finite, non-zero gradient   18 tensors, ||g|| 5.2652e-02
  [PASS] control   head receives a finite, non-zero gradient   18 tensors, ||g|| 4.2085e-02
  [PASS] skeleton / attribute / control head parameters actually moved
         max|dw| 1.050e-03 / 1.047e-03 / 1.057e-03
  ```

  The three head-parameter sets are disjoint (`Transform2ActPolicy` prefixes
  every parameter with `skel_`/`attr_`/`control_`), so "this head got a
  gradient" is a real statement — a design head silently receiving none would
  train a fixed body while looking exactly like a working run.

  **The two reward numbers it printed (`exec_R_eps` 2.05 then -0.09) mean
  nothing** and are recorded here only so nobody later mistakes them for a
  result. The smoke says nothing about whether the ant learns to walk, what it
  evolves into, or how E1 compares to E0. E1's real run is 3 seeds x ~100 epochs
  and has not been started.
* **Whether the converted ant is a *good* starting design for their task** is
  exactly E1's question and is not answered here.
* **`done_condition.max_ang: 60` on a quadruped** is inherited unchecked. A
  torso sphere with a free joint can roll; if E1 episodes terminate absurdly
  early this is the first thing to look at.
* **The floor-margin difference is not corrected**, only measured. E1 trains on
  their floor.
* **The 2.1-vs-3.12 contact-solver difference is not correctable** and is not
  quantified beyond "they part at the first contact". Whether it matters for
  transferring an E1-trained controller back into D1/D2's stack is an open
  question that E2 will run into first.
* **Only `dev_ant_body.xml` has been converted.** The validator is generic and
  `dev_spider_body.xml` / `dev_bug_body.xml` are untried.
* **Nothing about 2v2, self-play or soccer** is touched here; this is E1's
  prerequisite only.
