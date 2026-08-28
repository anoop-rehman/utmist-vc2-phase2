# Direction 3 (Transform2Act) — state of play and how to pick it up

*Written 2026-08-14, at the point where D3 pauses pending more GPU. Everything
here is either a file path, a measured number with the command that produced it,
or an explicit statement that something was not tested.*

## Where it stands

| unit | state |
|---|---|
| 3a — clone, install, smoke | **done**, `docs/repro/TRANSFORM2ACT_M1_REPRO_NOTES.md` |
| 3b — port map | **done**, `docs/repro/TRANSFORM2ACT_PORT_MAP.md` |
| 3c — GNN playground | **done**, `rower_soccer/t2a_port/gnn_playground.py` |
| 3d — GPU port | **done** (2026-08-27). All six steps built and gated. |
| 3e — paper-number validation | **running**, `runs/t2a_port/port_s1` |
| 3f — design+control on our drills | not started |
| M1 at paper scale (hopper) | **done, MET** — see the M1 notes, "M1 IS MET" |
| M1 at paper scale (ant) | abandoned; the pod it ran on is gone |

*The rest of this file is chronological. The sections dated 2026-08-27 at the
bottom supersede the 2026-08-14 text above wherever they disagree — in
particular decision 4's "~3.8x" and the "step 1 of 6" state.*

***Read the LAST section first if you are picking up 3e.** "Update 2026-08-27
(second)" settles that fp32 is not why the port under-trains and shows that the
gap is episode LENGTH rather than reward rate; both still hold. It then names
`agent_specs.batch_design` as the leading cause, and **"Update 2026-08-28" ran
that experiment and refutes it** — the fix is correct, matches their code, and
makes training worse. Read 2026-08-28 for what the lead is now (a
sampled-rollout discrepancy that predates the first gradient step) and for
three cheap diagnostics that have not been spent.*

## The two training runs

**`hopper_gpu` — complete, 1000/1000 epochs.** `exec_R_eps` by 100-epoch block:

| epochs | 0-99 | 100-199 | 200-299 | 300-399 | 400-499 | 500-599 | 600-699 | 700-799 | 800-899 | 900-999 |
|---|---|---|---|---|---|---|---|---|---|---|
| `exec_R_eps` | 858 | 1,529 | 2,500 | 3,366 | 4,382 | 4,758 | 5,091 | 5,166 | 5,638 | 6,317 |

Final-20 mean **6,836**, max 7,452. Monotone throughout.

**This does NOT establish that M1 is met, and the distinction matters.**
`TRANSFORM2ACT_M1_REPRO_NOTES.md:115` records the paper's 2D locomotion as
converging to "~4000+", and that figure is OURS, not the paper's — an attempt to
verify it on 2026-08-12 failed (the arXiv abstract carries no numbers and the PDF
exceeds the fetch limit). So what is established is a clean, monotone,
fully-completed 1000-epoch run that lands well above a number we wrote down.
**Anyone claiming the reproduction gate is met should read the 2D Locomotion
result out of the paper first.** That is the single cheapest open item in D3.

Checkpoints: `/workspace/Transform2Act/results/hopper_gpu/models/` —
`epoch_{100..1000}.p` plus `best.p`, 157 MB each, **1.7 GB total**. Worth pruning
to `epoch_1000.p` + `best.p` before any disk pressure.

**`ant_gpu` — running.** Epoch 88 of 1000, `exec_R_eps` 150-161 and climbing,
~32 min/epoch at `--num_threads 32`, ETA ~2d17h. Launched 2026-08-14 after
hopper freed the box. It is CPU-bound on its 32 samplers, not GPU-bound (the card
sits near 6%), so **if a kick or CompetEvo run needs cores, drop this to 16
threads rather than queueing behind it.**

## What 3d step 1 delivered

`rower_soccer/t2a_port/dense_policy.py` — their policy on dense `[G, N, F]`
tensors with one shared adjacency per topology group.
`rower_soccer/t2a_port/gate_dense_policy.py` gates it, **8/8**:

* their `epoch_0400` checkpoint loads with `strict=True` (65 tensors, 0 missing,
  0 unexpected);
* **0.00e+00** max action difference against their policy on real observations
  at all three stages, 66 states;
* **1.44e-15** for a G=30 batch against the same graphs one at a time;
* two negative controls that fail on demand (edge-dropping, `body_index` roll),
  both 60/60.

Run it with:

    cd /workspace/Transform2Act && source env-gpu.sh
    .venv-gpu/bin/python /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/gate_dense_policy.py

## The decisions already made, with the measurement behind each

These are settled; do not re-litigate them without new evidence.

1. **Group worlds by topology; do NOT compile a superset and mask.** A batch
   contains 21 distinct topologies from an untrained policy and 2 once trained
   (`topology_census.py`, 200 designs per checkpoint). Grouping is a handful of
   compiles, and it is exact — the masking approach's failure mode (a deactivated
   body that still carries mass or contact geometry) is the class of bug this
   project has shipped twice.
2. **Keep `IndexLinear`'s Python loop.** Real batches carry 7-12 distinct
   `body_index` values against a `max_index` of 256, and at that count the loop
   beats a batched gather by ~6x (3.3 ms vs 19.5 ms at 50k nodes).
3. **Replace the `cumsum`-and-difference log-prob reduction with `index_add`,
   and drop to fp32.** Measured: fp64 cumsum-diff 1.7e-10 error, fp32 cumsum-diff
   1.3e-1 (0.18% of a typical log-prob, which PPO then exponentiates into the
   clipped ratio), fp32 `index_add` 2.0e-5. float64 is load-bearing only for
   their choice of reduction. In the dense form a per-graph sum is `.sum(1)`, so
   this comes free.
4. **Plan against ~3.8x, not the raw env-step ratio.** 74% of hopper's wall-clock
   is rollout and 26% is the PPO update, so a physics-only port is Amdahl-capped
   there. D2 made the opposite mistake (19x raw, 3x real).

## Remaining steps of 3d, in order

2. Dense masked GraphConv into the training path (verified free in 3c).
3. Batched execution env with topology grouping — the real work. Gate it by
   building morphology M inside a group and separately as its own compiled
   model, then asserting the TRAJECTORIES agree over several hundred steps from a
   shared initial state. Not the observation: the trajectory.
4. `index_add` + fp32, validated against their fp64 run.
5. Two-stage design pipeline (design stages on CPU, execution batched).
6. 3e: their Table 1 numbers.

## An observation about the method, not the port

By epoch 100 the skeleton stage has **stopped exploring** — 199 of 200 sampled
designs share one topology, and every mean-action design does — while
`exec_R_eps` climbs from 1,376 to 6,836 over the remaining 900 epochs. The back
nine tenths of the run is attribute tuning and control, not body plan.

That matters for 3f, which was motivated by wanting a machine that finds
genuinely different bodies for different roles. On this task at these settings,
the skeleton search converges early and stays put. One task, one seed — an
observation, not a claim about the method, but worth designing around rather
than discovering later.

## Open items, cheapest first

1. **Read the paper's 2D Locomotion number** and settle whether M1 is met. Costs
   minutes; currently gates a claim we would otherwise be making on our own
   arithmetic.
2. Prune `results/hopper_gpu/models/` from 1.7 GB.
3. Decide whether `ant_gpu` runs to 1000 epochs or is cut short — it holds 32
   cores for ~2.7 more days and its result is a second M1 data point, not a
   dependency of 3d.
4. 3d step 3, the batched execution env, which is the actual remaining work.

## Not tested

Nothing in `dense_policy.py` has been trained — it is gated for forward-pass
equivalence only. There is no batched Transform2Act env, no training loop, and no
measurement of what the port's throughput would actually be; the 3.8x above is an
Amdahl ceiling computed from hopper's phase timings, not an achieved number.

---

## Update 2026-08-24 — M1, and step 3's two blockers cleared

### M1 is NOT met, and the bar we were checking against was our own

`~4000+` was never in the paper. The paper has **no results table and no numeric
return anywhere** — two tables, both hyperparameters, everything else a curve.
Read off Figure 3, 2D Locomotion converges to **~9,000**. Our completed
1,000-epoch run reached 6,836, about 76% of it.

A candidate excuse was examined and destroyed. The paper's equation 17 pays
`|p^x_{t+1} − p^x_t| / δt + 1` and the released code has no `abs`, so the code
might have been implementing a harder task than the figure. Adding the `abs`
(opt-in flag, default off) and training a seed:

| at epoch 50 | signed (released) | `\|Δx\|` (eq. 17) |
|---|---|---|
| reward | 1,321 | 3,324 |
| **net displacement** | **2.57 m** | **0.60 m** |
| net / path | **0.999** | **0.032** |

The `|Δx|` agent vibrates in place — 18.6 m of motion for 0.6 m of progress —
while scoring 2.5x the reward. The paper reports and renders plausible
locomoting agents, so the `|·|` is a write-up error and Figure 3 came from the
shipped code. **~9,000 is real and the 24% gap is ours.**

Note how nearly this went wrong: the reward curves showed `|Δx|` learning
2.5-6.5x faster, which reads as evidence until you notice it is a tautology.
`displacement_probe.py` exists because reward is not comparable across reward
forms and displacement is.

**Seed variance is small here**, unlike D2: two signed seeds tracked each other
within a few percent for 200 epochs (211/206 at epoch 10, 494/520 at 40,
1,314/1,357 at 50). So 6,836 is not a bad draw.

### Step 3 was blocked twice, and both are now cleared

Before writing a batched execution env, the assumption underneath it was
checked: they simulate with **mujoco-py 2.1 / mujoco210**, the port with the
**modern bindings under mujoco_warp**.

1. **Their XML does not compile in modern MuJoCo at all** — `xml_robot.py`
   emits `<compiler coordinate="global">` and global coordinates were removed
   in 2.3.3. `t2a_port/xml_global_to_local.py` converts; the conversion is a
   pure translation only because their generator never emits `quat`/`euler`,
   and `assert_no_rotation` enforces that rather than assuming it.

2. **MuJoCo 2.1 computed capsule mass and inertia differently, and that is what
   they trained against.** Recovered exactly by sweeping their compiler over a
   49-point (r, h) grid:

   ```
   mass  = rho*pi * (2 r^2 h + r^3)                      # cylinder + 3/4 sphere
   I_ax  = rho*pi * (r^4 h + r^5 / 2)
   I_tr  = rho*pi * (r^4 h + 2/3 r^2 h^3 + r^5 / 3 + r^3 h^2)
   ```

   Verified against their compiler at 4.2e-16 (mass) and 1.8e-14 (inertia).
   Their bodies are 1.3-3.2% lighter than the same XML compiled today.

`convert(legacy_inertial=True)` emits explicit `<inertial>` from these, and the
**whole model then round-trips to machine precision** — mass 4.3e-11, inertia
4.4e-13, all geometry at 1e-16. So the batched env can compute their physics
in closed form on the GPU, with no per-world compiler call; at ~50,000 worlds
per epoch that call was the thing that would have made step 3 impossible.

**Trap worth knowing:** their compiler line says `inertiafromgeom="true"`, which
makes MuJoCo ignore an explicit `<inertial>` **silently**. The elements were
emitted, correct, and discarded, and every field still read as mismatched. Set
`inertiafromgeom="auto"`.

### What step 3 still has to decide

The model matches exactly; the **engine** does not. Trajectories separate at
1e-3 by step 11 and 1e-2 by step 72 over 300 recorded-action steps — RK4 plus
contacts across two MuJoCo versions, not a model error, and not closable by
fixing the model.

So a batched port cannot bit-match their episodes, and the question is what it
is for:

* **A faithful reproduction** whose numbers are comparable to their curves —
  needs the legacy inertial (now available) and has to accept that individual
  trajectories diverge while the *distribution* should not. Worth checking by
  training a short run in the port and comparing the learning curve, not the
  trajectory.
* **A correct-physics environment**, using modern MuJoCo's fixed capsules. Valid
  and arguably better, but its numbers are not comparable to theirs and M1 would
  have to be restated against a fresh baseline.

Recommend the first, because M1 is the point of D3 and it is a comparison.

### Files added

| file | what |
|---|---|
| `t2a_port/xml_global_to_local.py` | global→local, plus the legacy capsule formulas |
| `t2a_port/physics_bridge_gate.py` | two-venv model + trajectory comparison |
| `t2a_port/legacy_capsule_fit.py` | the sweep that recovered the formula |
| `t2a_port/displacement_probe.py` | net vs path displacement; settled the `\|Δx\|` question |

### Watch this: the two new seeds are running well ahead of the run that scored 6,836

`exec_R_eps` at matched epochs, against the completed run's recorded 100-epoch
block means:

| epoch | old run (block mean) | seed 1 | seed 2 |
|---|---|---|---|
| 200 | 2,500 | 3,375 | 2,382 |
| 300 | 3,366 | **5,405** | **6,098** |
| ~330 | — | **5,640** | **6,637** |

The completed run's *final* figure was 6,836 after **1,000** epochs. Seed 2 is at
6,637 by epoch **339**, and the two new seeds track each other, so this is not
one lucky draw.

Epochs are epochs here — `min_batch_size` is 50,000 either way — so this is not
a wall-clock artefact. If the new seeds plateau anywhere near ~9,000, **M1 flips
from "not met" to "met, and the 6,836 run was simply a bad one"**, which would
retract the headline finding above. Do not restate M1 either way until they
finish (~18-22 h from 2026-08-25 01:00 UTC).

**The leading hypothesis is `--num_threads`, and if true it is a finding in its
own right.** The old run used 32 threads; these use 24 and 16.
`sample_worker` loops until *its share* of `min_batch_size` is collected and
then finishes the episode in progress, so thread count changes the fraction of
each batch that is a truncated tail: at 32 threads each worker collects 1,562
steps against episodes up to 1,000 long, at 16 it collects 3,125. **`num_threads`
is not only a speed knob in this codebase — it changes the data distribution.**

Cheap to test once the seeds land: one more seed at `--num_threads 32`,
everything else identical. If it reproduces ~6,800 while 16 and 24 reach higher,
the mechanism is confirmed and every previous Transform2Act number here needs
its thread count recorded beside it.

#### Correction: the lead is real but narrowing, and "M1 may flip" was overstated

The comparison above quoted SINGLE epochs of the new seeds against the old run's
100-epoch BLOCK MEANS. That is not like-for-like: per-epoch variance here is
enormous (seed 2 goes 6,098 at epoch 300, 2,958 at 400, 6,562 at 437, 3,852 at
445; over epochs 300+ its sd is 1,202 and its minimum is 860). Comparing a noisy
point against a smoothed one is how the earlier claim got its size.

Block means, like for like:

| block | seed 1 | seed 2 | old run |
|---|---|---|---|
| 0-99 | 902 | 930 | 858 |
| 100-199 | 2,330 | 2,413 | 1,529 |
| 200-299 | 4,403 | 4,701 | 2,500 |
| 300-399 | 5,177 | 5,695 | 3,366 |
| 400-499 | 5,625 | 5,837 | 4,382 |

**The lead survives the correction** — the new seeds are ahead in every block —
**but it is shrinking**: 1.5x, 1.8x, 1.6x, 1.3x. And the shapes differ. The old
run was still climbing steeply at 400-499 (4,382, on its way to 6,836 by epoch
1,000); the new seeds are flattening (5,177 -> 5,625).

So the honest projection is **not** "these are heading for ~9,300". It is "these
converge sooner, to somewhere plausibly in the 6,000-7,000 region" — which would
land near the old run's 6,836 and leave **M1 still not met**. The `num_threads`
hypothesis, if it holds, would then explain a difference in *convergence speed*
rather than in *final performance*, which is a much less interesting result and
does not rescue the gap.

Do not restate M1 until epoch 1,000. But the earlier note's framing — that a
reversal was likely — was built on a bad comparison and should not be relied on.

#### Correction 2: the truncated-tail mechanism above is FALSE, and M1 is met

Both of the above are superseded. See `repro/TRANSFORM2ACT_M1_REPRO_NOTES.md`,
"M1 IS MET -- with the table recomputed".

**M1 is met.** Seed 1 (24 threads) finishes its 900-939 block at 8,625 and seed
2 (16 threads) its 900-999 block at 10,210, against the paper's ~9,300 with a
band of roughly 7,700-10,300. The "converging to 6,000-7,000" projection above
was wrong; so was the projection it corrected. **Do not project these curves in
either direction. Run them out.**

**The truncated-tail mechanism is refuted.** `while logger.num_steps <
min_batch_size` is the *outer* loop in `sample_worker`, and `num_steps` moves
only in `LoggerRL.end_episode`, so a worker cannot notice it has passed its
budget until the episode it is in has ended. It **overshoots and never
truncates** -- verified by counting `mask == 0` entries against `num_episodes`
in a real batch: equal at every thread count tried. There are no truncated tails
at any thread count, so thread count cannot set their fraction.

What `num_threads` really varies is (1) realized batch size, as a *non-monotone*
sawtooth that does not track the performance ordering, (2) the effective RNG
stream, via `seed_worker` -- so it is a de facto `--seed` -- and (3) the number
of eval episodes each logged point averages over. Details and measurements in
the M1 notes.

**This inverts the port's problem.** Their sampler emits *only complete
episodes*; a fixed-`T` batched sampler truncates *every* world at the rollout
boundary. The port has partial episodes their code never has, and needs a
bootstrap `V(s_T)` at every cut that their GAE never performs.

##### M2 acceptance criterion, settled

1. **Time-limit semantics: follow Transform2Act, not CompetEvo.** `hopper.py`
   computes `done = not (... and (self.control_nsteps < max_nsteps))`, so the
   1,000-step time limit sets `done = True` and `mask = 0`, and their GAE
   bootstraps **zero** at the limit. This is the *opposite* of
   `competevo_port/ppo.py:187` ("mask = 1 on truncation, so GAE bootstraps
   across the boundary"). **Do not carry the CompetEvo convention into the
   Transform2Act port.** Bootstrap only at true rollout-boundary cuts.
2. **Batch target: what their sampler delivers, not the nominal 50,000.** At the
   operating point that is ~57,000-64,000 agent-steps per PPO iteration, i.e.
   ~280-315 gradient steps at minibatch 2,048 x 10 epochs.
3. **Sampler shape: reset all worlds together, roll `T = k * max_ep_len` with
   per-world auto-reset**, so the batch is exactly `N * k` complete episodes and
   there are zero rollout-boundary truncations -- the same *kind* of learning
   signal as theirs. With `max_ep_len = 1006` (1,000 exec + 5 skeleton + 1
   attribute), `T = 1024` and `N = 56` gives 57,344: the 16-thread batch.
4. **Eval: record how many complete episodes back each logged point.** Theirs is
   `N` at convergence. Evaluating over 256 worlds gives a visibly smoother and
   differently-biased curve than their reference; match it deliberately or say
   so.
5. **STRICKEN from the criterion: "reproduce the fewer-threads-is-better
   effect."** There is no such mechanism to reproduce, and the observation it
   came from is confounded with seed.

---

## Update 2026-08-27 — 3d step 5 gated, 3e launched, and four corrections

*Written by the agent that ran 3d steps 5-6. Every number below names the
command that produced it; where something is a projection it says so. The
"Not tested" section at the end is not decoration.*

The four corrections, so they are not buried:

1. **`PORT_MAP` section 6's phase split is wrong.** Their PPO update is
   **65-70%** of wall-clock, not 26%, so the "~3.8x Amdahl ceiling for a
   physics-only port" describes a different port. Measured over three complete
   1,000-epoch logs.
2. **`xml_to_fields`'s premise is wrong arithmetic.** A batch holds ~57
   designs, not 50,000, and a compile is 4.5 ms. The pipeline compiles.
3. **Settled decision 5's sampler shape is not implementable as written**, and
   settled decision 4 implies a world count that cannot be a constant. Both
   adjusted, with the measurement, in `train_t2a.py`'s docstring.
4. **`hopper_gpu_t32` did not "plateau".** It fell into the alive-bonus
   local optimum -- a body that cannot fall -- which makes the seed spread on
   this task bimodal, not 42%.

### The gates that were supposed to already pass, re-run

All three, from scratch, before anything was built on them.

| gate | command | result |
|---|---|---|
| `gate_dense_policy.py` | their venv, `hopper_gpu` epoch 1000 | **was 8/8, now 18/18** |
| `gate_batched_exec.py --check --backend warp` | repo venv | **11/11** |
| `gate_batched_exec.py --check --backend cpu` | repo venv | **11/11** |

The dense-policy gate grew ten checks, because 3e needs four things step 1
never touched -- sampling, per-graph log-probs, the critic, and the batching
the PPO update depends on:

```
[PASS] skel_trans: per-graph log-prob matches their cumsum reduction   8.88e-16
[PASS] attr_trans: per-graph log-prob matches their cumsum reduction   7.11e-15
[PASS] execution:  per-graph log-prob matches their cumsum reduction   1.15e-14
[PASS] dense critic matches theirs            66 observations, 4.55e-13
[PASS] negative control: the critic reads node 0, not a pool           60/60
[PASS] DIFFERENT topologies of the same size batch together            2.22e-15
[PASS] padded graphs give the same per-graph log-prob                  3.33e-15
[PASS]   control: dropping the node mask changes the log-prob          66/66
[PASS] RunningNorm's statistics ignore padded rows
[PASS]   control: an unmasked update DOES corrupt them   mean moves 1.32e-01
```

The log-prob check samples an action rather than taking the mean, because a
log-prob evaluated only at the mode cannot see an error in the standard
deviation or in the quadratic term.

### 3d step 5: the two-stage pipeline

Three new files; `gate_two_stage.py` is **15/15 on both backends**.

| file | what |
|---|---|
| `t2a_port/design_stage.py` | their skeleton + attribute stages, CPU, **no MuJoCo at all** |
| `t2a_port/two_stage_pipeline.py` | topology grouping, per-world model fields, group envs |
| `t2a_port/gate_two_stage.py` | the gate, `--emit` in their venv then `--check` in ours |

#### The design stages need no MuJoCo, and that had to be measured

Their `apply_skel_action` calls `reload_sim_model` after every skeleton edit and
`get_sim_obs` then reads `self.data.qpos`, so the obvious reading is that a
design step needs a compiled model. Measured instead, in their venv over 20
sampled episodes of `hopper_gpu_s2` epoch 1000: the design-stage `sim_obs` is
the **same constant at every design step of every episode** --

    root row  [0, 1.25, 0, 0, 0]        every other row  [0, 0, 0, 0, 0]

-- because `reload_sim_model` leaves `data.qpos` at `qpos0` and the only
non-zero entry of `qpos0` is the root's `rootz` slide joint, which carries
`ref="1.25"` in `assets/mujoco_envs/hopper.xml` and no design parameter
touches. The 1.25 is **read from the exported XML on every episode**, and a
`ref` on any generated joint is an assertion, not an assumption
(`_assert_no_child_ref`, and the gate injects one to prove the assertion
fires).

Consequence: the design stages are pure Python over their own `Robot`, which
imports cleanly into the repo venv (numpy + lxml only). Cost **2.0 ms per
world** -- 0.53 construct, 1.17 for five skeleton steps, 0.34 for the attribute
step.

#### The design half is bit-exact against their env

`gate_two_stage.py`, replaying **their recorded actions** through our CPU design
stage over 100 of their episodes (60 untrained + 40 at epoch 1000), 600 design
steps:

```
[PASS] design: observation matches theirs at every design step   max |d| 0.00e+00
[PASS] design: the exported XML is byte-identical to theirs      100/100 episodes
[PASS] design: body order, edges and body_index match theirs     0 mismatches
[PASS] design: projected design parameters match theirs          max |d| 0.00e+00
```

with three negative controls that each bite (forcing "add a child" changes the
body 20/20; zeroing the attribute action changes the XML 20/20; an injected
joint `ref` is caught).

#### xml_to_fields' premise was wrong arithmetic, and the pipeline COMPILES

`xml_to_fields.py` computes per-world model fields in closed form on the
premise that "compiling 50,000 of them per epoch is the thing that would make
step 3 impossible". That premise does not survive contact:

* **50,000 is agent-steps, not designs.** A ~57,000-step batch at ~1,000 steps
  per episode holds **~57 designs**, one compile each.
* **A compile is 4.5 ms** and the global->local conversion 0.5 ms, measured over
  40 real designs. 57 worlds is 0.26 s of CPU per PPO iteration.
* **The closed-form surface is far bigger than it looks.** Asking MuJoCo which
  arrays actually differ between two designs of the *same topology* returns
  **21**: `actuator_acc0, actuator_gear, body_inertia, body_invweight0,
  body_ipos, body_iquat, body_mass, body_pos, body_subtreemass, bvh_aabb,
  cam_poscom0, dof_M0, dof_invweight0, dof_length, geom_aabb, geom_pos,
  geom_quat, geom_rbound, geom_sameframe, geom_size, light_poscom0`.

So `TopologyGroup` compiles every world and reads the fields off the compile.
`xml_to_fields.py` keeps its job as a gate on those closed forms and as the
fallback if a task ever really does need thousands of designs per batch.

**`differing_fields()` turns coverage into an assertion.** Every field that
differs must either be written per world or be on `WARP_INERT` with a reason
that is re-derived from the installed `mujoco_warp` -- and it earned its keep
immediately: `cam_poscom0` and `light_poscom0` were on the skip list as
"rendering only" and the check refused them, because warp does carry those
arrays. The rule is now "if warp can batch it, write it", with no judgement
calls; only `bvh_aabb`, `dof_M0`, `geom_sameframe`, `body_sameframe` (absent
from warp's Model) and `dof_length` (read only under `mjENBL_SLEEP`, asserted
off) are skipped.

#### The trajectory gate

The one that matters. World *i* inside a group of 8, against world *i* compiled
and rolled entirely on its own, from a shared per-world initial state with a
shared action tape, 300 steps:

| backend | result |
|---|---|
| fp64 CPU | **max abs qpos diff 0.000e+00** over 8 worlds x 300 steps |
| fp32 warp | **9.31e-10 at step 1** (tol 3e-6); envelope 1.9e-5 @10, 1.7e-1 @50, 2.0e+0 @300 |

The warp envelope is chaos, not a defect -- PORT_MAP section 14 already measured
identical worlds separating by ~1e1 over 400 fp32 steps -- so the fp32 gate
asserts one step and prints the rest. Four negative controls, each of which
must break the match, and each does at a step-1 difference five to six orders of
magnitude above the passing case:

| control | step-1 diff | worst |
|---|---|---|
| no per-world fields written | 2.97e-03 | 5.30 |
| per-world fields rolled by one | 4.32e-03 | 5.18 |
| `actuator_gear` left unwritten | 5.33e-04 | 2.39 |
| `body_mass`/`body_inertia`/`body_subtreemass` unwritten | 6.00e-04 | 5.08 |

Plus a control on the group key itself: forcing two different topologies into
one group is refused by `differing_fields()` on the shape mismatch.

#### Grouping: the key, and what it costs

The key is the **ordered** tuple of body names. Names encode the path from the
root, so the *set* of names determines the tree and the XML document order;
the *order* of `robot.bodies` is creation order, so two worlds can reach the
same tree by different skeleton actions and index their nodes differently.
Keying on the ordered tuple means a group shares one adjacency and one
`body_index` vector exactly, with no reordering step to get wrong. Measured
fragmentation against the unordered key:

| reference | designs | ordered key | name-set key |
|---|---|---|---|
| untrained | 60 | **17** groups (sizes 23, 7, 6, 4, 3, 3, 2, 2, 2, 1x8) | 16 |
| epoch 1000 | 40 | **2** groups (25, 15) | 1 |

One extra group untrained, one extra at convergence. Canonicalising node order
would collapse them and would be provably safe (the policy is permutation
equivariant), but it is new untested code and the cost is one group.

### 3e: the trainer, and two measurements that change how it should be planned

`t2a_port/train_t2a.py` is the port's PPO loop: their policy and critic (dense,
gated against theirs), their GAE, their clipped objective, their optimizers and
learning rates, with the design stages on the CPU and the execution stage on
topology-grouped batched physics.

#### Their wall-clock is NOT 74% rollout. It is 65-70% UPDATE.

`PORT_MAP.md` section 6 records "T_sample ~100 s (49%), T_update ~52 s (26%),
T_eval ~51 s (25%)" from a snapshot of the live `hopper_gpu` run, and concludes
"Amdahl's ceiling for a physics-only port is therefore ~3.8x". Recomputed over
**all 1,000 epochs** of each completed run, from
`results/<cfg>/log/log_train.txt`:

| run | block | T_sample | T_update | T_eval | total | update share |
|---|---|---|---|---|---|---|
| `hopper_gpu_s2` | 0-99 | 34.6 | 88.0 | 13.2 | 135.9 | 65% |
| | 500-599 | 28.5 | 92.3 | 11.0 | 131.8 | 70% |
| | 900-999 | 16.8 | 54.6 | 6.4 | 77.8 | 70% |
| `hopper_gpu` | 500-599 | 32.3 | 89.7 | 14.2 | 136.1 | 66% |
| `hopper_gpu_t32` | 100-199 | 20.0 | 50.4 | 9.1 | 79.5 | 63% |

**The update is where their time goes, and the 3.8x figure describes a port
that this one is not.** The port runs the update in fp32 on dense `[G, N, F]`
tensors; measured on a 2,048-graph minibatch of the real shapes:

| | per PPO gradient step | x280 (their 10 epochs x 28 minibatches) |
|---|---|---|
| this port, fp32 | **27.0 ms** | **7.6 s** |
| this port, fp64 | 108.1 ms | 30.3 s |
| theirs (float64, ragged, GPU) | -- | **55-92 s** |

so the update phase alone is **7-12x**. (Their own update is on the GPU; the
gain is dense-vs-ragged plus fp32, not CPU-vs-GPU.)

#### `nconmax`/`njmax` are PER WORLD, and getting that wrong cost 2.6x

`mujoco_warp.put_data`'s docstring says both are allocated *per world*.
`two_stage_pipeline.py` first passed `nconmax * n_worlds`, which asks for a
4.8 GB constraint Jacobian at 1,024 worlds (it OOMs) and, below that, silently
gives the solver arrays a thousand times larger than the problem. Corrected to
32/128 per world (measured peak `nacon` is ~5 per world), the batched execution
env, **policy in the loop**, on the shared card:

| worlds | before the fix | after | env only, after |
|---|---|---|---|
| 64 | 3,774 /s | **5,435 /s** | 7,049 /s |
| 256 | 7,332 /s | **18,863 /s** | 25,308 /s |
| 512 | 5,838 /s | **36,419 /s** | 48,462 /s |
| 1,024 | OOM | **67,512 /s** | 89,650 /s |
| 2,048 | -- | **123,210 /s** | 156,409 /s |

Their sampler, like for like, is ~3,000 steps/s. Note the shape: a batched step
costs 11.8 ms at 64 worlds and 16.6 ms at 2,048, i.e. it is **almost entirely
fixed launch overhead**. That single fact drives everything below.

#### The sampler shape had to be adjusted, and here is the measurement

Settled decision 5 says "reset all worlds together, roll `T = k * max_ep_len`
with per-world auto-reset, so the batch is exactly `N*k` complete episodes".
Two things make that not implementable as written:

1. **Auto-reset restarts the same BODY.** The design stages run before the
   rollout, so an auto-reset world begins a new execution episode on the
   morphology it already had, while their `env.reset()` calls `reset_robot()`
   and draws a new design every episode (`hopper.py:310, 318`).
2. **Episodes are not `max_ep_len` long.** 928 +/- 51 at convergence, but tens
   of steps early; a cut at fixed `T` truncates whatever episode each world is
   in, which is what decision 5 exists to avoid.

The trainer therefore samples in **generations**: design N worlds, roll until
every one is `done`, stop, repeat until the agent-step budget is met. That
gives what decision 5 was for -- only complete episodes, zero rollout-boundary
truncations, **no bootstrap anywhere** -- in every regime, and each episode
gets its own design as theirs does. It also overshoots and never truncates,
exactly as `sample_worker` does.

**And N cannot be a constant.** Decision 4 fixes the batch at ~57,000-64,000
agent-STEPS. Early in training a hopper survives ~30 steps, so their batch
holds **~1,700 episodes**, not 64. A fixed 64 worlds would have produced 1,848
steps per iteration instead of 57,344 -- a thirtieth of their gradient signal
-- which is what the first smoke run did. The trainer now sets
`N = clip(ceil(budget / mean_episode_length), 32, max_worlds)` from the
measured length, so it carries ~1,000-1,900 worlds early and ~62 at
convergence.

#### The cost model, and the two things it made me fix

Because a batched step costs ~12 ms whatever the world count, a generation
costs **(number of topology groups) x (longest-surviving episode in each
group)** BATCHED steps -- not agent-steps. Logged every epoch as
`batched_steps`. On a real epoch-0 batch that is 112 groups and 4,560 batched
steps for 57,598 useful agent-steps: an average of **12.6 agent-steps per GPU
launch**, on a card that does 2,048 for the same launch. One long-lived world
in a two-world group costs as many launches as a thousand-world group.

The same "many small launches" problem was, at first, much worse in the update.
Measured, on the same batch, before and after:

| | buckets | T_update (340 gradient steps) |
|---|---|---|
| minibatch bucketed by (stage, node count) | 29 | **130.2 s** |
| minibatch bucketed by stage, graphs PADDED | **3** | **30.7 s** |

Padding is the hazard `dense_policy.py`'s docstring named -- a zero row is not
a neutral sample -- so it is gated rather than assumed:
`gate_dense_policy.py` now checks that a padded block gives the same per-graph
log-prob as the unpadded graphs (3.33e-15 over 66 real observations), that
dropping the node mask changes it (66/66), that `RunningNorm`'s statistics
after a masked update on a padded block equal an update on the real rows alone,
and that an unmasked update **does** corrupt them (mean moves 1.32e-01). The
gate is now **18/18**.

#### What an epoch actually costs, measured

`runs/t2a_port/port_s1.log`, epoch 0, seed 1, `hopper_gpu_s2` config, against
their own epochs 0-99 block means from `results/hopper_gpu_s2/log/log_train.txt`:

| | T_sample | T_update | T_eval | epoch |
|---|---|---|---|---|
| theirs, 16 threads, epochs 0-99 | 34.6 | 88.0 | 13.2 | **135.9 s** |
| **this port, epoch 0** | **88.0** | **29.4** | **0.6** | **118.0 s** |

That batch is 57,522 agent-steps in 2,008 complete episodes -- their operating
point (settled decision 4) reproduced, not approximated.

Three epochs of that configuration ran before the memory fix below forced a
relaunch: **118.0, 119.3, 115.3 s**.

**About 1.15x, and the phases have swapped.** The update is **3.0x
faster**, eval is **21x faster** (their eval is 16 sampler processes; ours is
one group of 16 worlds, 48 batched steps -- the mean-action design collapses to
a single topology, as `topology_census.py` said it would). Sampling is **2.5x
slower**, for the reason above: 104 topologies from an untrained policy, each
rolled in its own under-filled batch, 4,384 batched steps for 57,522
agent-steps.

**This should improve a lot as the run trains, and that sentence is a
projection, not a measurement.** By epoch 100 the skeleton stage has converged
to 1-2 topologies, so a generation becomes ~2 groups x ~1,000 batched steps --
2,000 launches for ~62,000 agent-steps instead of 4,384 for 57,522 -- while
their own epoch cost stays roughly flat. **Nobody should quote an end-to-end
speedup for this port until an epoch in the converged regime has been timed.**
Eight epochs have been timed across three configurations, all untrained, all
within a few percent of each other (115-124 s); the launched run's epoch 0 is
122.0 s.

#### And a memory bug the launch found

The first launch of `port_s1` reached **5.1 GB of GPU**, over the 6 GB budget
for this work once the other jobs on the card were counted. Cause:
`build_groups` constructed every group before any of them was rolled, so all
**112** `mujoco_warp` `Data` objects were resident at once -- memory scaling
with the topology count rather than with anything useful. The run was stopped
with its stop file (not killed), and `iter_groups` now builds and drops one
group at a time; the GPU rolls them sequentially anyway, so nothing is lost.
`build_groups` stays for the gate, which genuinely wants several groups alive
at once.

**And after that fix it still climbed**, 0.4 -> 4.1 -> 5.3 GB over two epochs,
because `mujoco_warp` allocates a fresh `Data` per topology group and warp's
CUDA mempool caches every block it has ever handed out -- so the resident set
grows to the high-water mark over all group shapes ever seen, not to the
working set. `--mempool-mb` (default 512, the run uses 256) sets
`wp.set_mempool_release_threshold`, and the trainer drops the batch and calls
`torch.cuda.empty_cache()` at each epoch boundary. Measured on the launched
run: **~4.2 GB at the mid-epoch peak, back to ~0.8 GB between epochs**, and it
no longer ratchets. The peak is the number that matters to whoever else is on
the card; most of it is torch holding the epoch's stored observations,
actions and per-row adjacencies (`gpu_mib` in the JSON log reports torch's own
peak, 2.9 GB). If it needs to come down further, store one adjacency per
distinct graph with a per-row index instead of a per-row `[n, n]` block --
`adj` is the largest per-transition item and it is the same handful of matrices
repeated. Not done.

The rule this earns: **on a shared card, a growing resident set is a bug even
when nothing is leaking.** A caching allocator plus a per-group `Data` is
enough to take a 0.4 GB working set to 5.3 GB of held memory.



### Watching the creatures: one of the three seeds is stuck in a survival trap

Rendered with `t2a_port/render_checkpoint.py` (CPU, offscreen, read-only) and
measured with `t2a_port/displacement_probe.py`, 12 mean-action episodes each:

| | `hopper_gpu_s2` epoch 1000 | `hopper_gpu_t32` epoch 650 |
|---|---|---|
| bodies / actuators | 7 / 6 | **9 / 8** |
| `exec_R_eps` | 11,750 | 1,958 |
| episode length | 950.4 | **1,000.0 (never falls)** |
| **net displacement** | **86.40 m** | **7.66 m** |
| path length | 86.40 m | 11.56 m |
| net / path | **1.000** | 0.663 |
| net speed | **11.36 m/s** | 0.96 m/s |

**`hopper_gpu_s2` is a genuine runner.** net/path = 1.000 -- it never steps
backwards. On video it is a long two-capsule boom held at ~40 degrees with its
far end in the air, and a cluster of short limbs scissoring underneath at high
frequency: a lean-forward-and-skitter gait, not a hop. Not a physics exploit,
but not a hopper either.

**`hopper_gpu_t32` is stuck in the alive-bonus local optimum, and this is why
it plateaued at ~1,500 from epoch 100 to 670.** It evolved a nine-body tangle
that lies across the ground, survives all 1,000 steps of every episode, and
travels 7.7 m. Its return decomposes almost exactly as `1,000 x alive_bonus +
7.66 m / 0.008 s = 1,000 + 958 = 1,958`. **It has found a body that cannot
fall and stopped searching.** No metric in the training log says "sprawled and
not locomoting" -- `exec_R_eps` just sits flat -- and it took a render plus a
displacement probe to name it.

This matters for 3e's design. `TRANSFORM2ACT_M1_REPRO_NOTES.md` records seed
spread as 42% (7,482 vs 10,594 on the two finished seeds). Including t32 the
spread is not 42% but **6x**, and it is not a spread at all -- it is two
qualitatively different basins, run-and-score or sprawl-and-survive. A
distribution comparison against Figure 3 needs enough seeds to see both, and
`net displacement` should be logged beside `exec_R_eps` so the mode is visible
without rendering.

#### And the thing the video could not show: the limbs go through the floor

The tracking camera keeps the creature centred, which makes ground contact
almost impossible to judge by eye, so it was measured instead. Per step, the
lowest point of any non-floor capsule (centre minus half-length times |z-axis|
minus radius), one mean-action episode each, in THEIR venv on THEIR
checkpoints:

| run | steps | deepest point below the floor | mean depth | >2 cm | >10 cm |
|---|---|---|---|---|---|
| `hopper_gpu_s2` e1000 | 874 | **0.314 m** | 0.009 m | 12% of steps | 2% |
| `hopper_gpu_t32` e650 | 999 | **0.414 m** | 0.012 m | 26% | 1% |
| `hopper_gpu` e1000 | 999 | **0.236 m** | 0.010 m | 16% | 2% |

Against capsule radii of 0.03-0.08 m and segment lengths of 0.44-1.04 m, a peak
of 0.24-0.41 m is **half a limb underground**. `hopper_gpu_s2` is airborne
(no ground contact at all) on **85%** of its steps and averages 0.16 ground
contacts per step while travelling at 11 m/s.

Some penetration is by design -- their XML sets `solref=".02 1"` and
`solimp=".8 .8 .01"`, a deliberately soft contact with a 1 cm impedance width,
and the mean depth of ~1 cm is exactly that. The **peaks** are not: they are a
very light capsule (radius 0.03, density 1000) driven by a gear-400 actuator
through a contact that cannot push back hard enough.

**This is a property of the reference environment, in all three runs, including
the one that meets M1.** The port reproduces it rather than introducing it, and
nothing in the port can or should fix it. It is recorded here because D3's plan
says to watch for a body that exploits sim physics, and this is one -- it just
does not show up in the video, only in a contact probe. Before 3f puts this
design space on a soccer pitch, the contact model and the actuator bounds
should be revisited together.

**The optimizer presses on its bounds.** Across 40 sampled epoch-1000 designs
(240 actuators, 280 capsules): **32% of capsules sit at the minimum radius**
(0.03, the `lb`) and **18% of gears at the maximum** (400, the `ub`). With no
energy cost in the reward, the optimum is "as light and as strongly actuated as
the bounds allow", and the bounds -- not the physics -- are what stops it.
Worth knowing before 3f puts this design space on a soccer task.

### An operational rule, learned by breaking it

**Never wrap a CUDA process in `timeout`.** `timeout` forwards its own SIGTERM
to the child, so killing the wrapper kills the CUDA process too -- which is how
a smoke run got SIGTERMed on 2026-08-27 despite the standing "never kill a CUDA
process under MPS" rule. The four other MPS clients (one D1 soccer run, three
CompetEvo self-play runs) were checked immediately afterwards and were all
still running and still on the GPU; that is luck, not evidence that the rule is
soft.

`train_t2a.py` now takes `--stop-file`: `touch` the path and it saves and exits
at the next epoch boundary. Use that. Long runs are launched with `setsid
nohup` and **no** `timeout`.

### One bug found while the run was in flight, fixed for the NEXT seed

`self.len_est` -- the measured episode length that sizes the next generation's
world count -- was being written by **both** passes of `sample()`. The eval
pass runs mean actions and its episodes are a different length (21.2 against a
training 31.8, read off epoch 14 of `port_s1`), so the world count for each
training generation was being derived from the eval distribution.

It is benign in the direction it currently errs: a too-small `len_est` asks for
*more* worlds than needed, which costs an extra generation rather than
under-filling the batch, and `batch_steps` stayed on target throughout (57,523
at epoch 14, against the 57,344 decision 4 asks for). It would not stay benign
at convergence, where mean-action episodes run *longer* than sampled ones and
the error flips sign.

Fixed in `train_t2a.py` (`if lens and record:`). **`port_s1` is running the
unfixed heuristic** -- it was not restarted, because the batches it is
producing are correct and a restart costs more than the bug does. The next seed
launched gets the fix, and that is a difference between seeds that has to be
recorded rather than forgotten.

### Not tested

* **The port has not been trained to convergence and its learning curve has not
  been compared to theirs.** Everything above is gates and throughput. Nothing
  here says the port learns the same thing; that is exactly what 3e is for, and
  it is the claim that matters.
* **Nothing checks the PPO update against theirs numerically.** The policy, the
  critic and the per-graph log-probs are each gated to 1e-13 or better against
  their networks on their observations, and the GAE and the clipped objective
  are transcriptions of `khrylib/rl/core/common.py` and
  `khrylib/rl/agents/agent_ppo.py` -- but no test drives one optimiser step in
  both codebases and compares the resulting weights. That is the cheapest
  remaining gate and it has not been written.
* **The design stage cannot see a design that fails to compile.** Their
  `apply_skel_action` wraps `reload_sim_model` in a bare `except` and ends the
  episode; this port does not compile until after the design stages, so a
  mid-design XML that fails would be noticed one step late (or not at all if
  the final XML compiles). 100/100 designs compiled in the gate and no failure
  has ever been observed, but the rate has not been measured on a long run.
* **The fp32 log-prob has not been measured against fp64 on a real batch.**
  `gate_dense_policy.py` runs the dense policy in float64, so the 1e-14
  agreement it reports is an fp64 result. The dense reduction is `.sum(1)` over
  at most ten nodes rather than their cumsum-and-difference over 50,000, so the
  cancellation PORT_MAP section 5 measured cannot arise -- but "cannot arise"
  is an argument, not a number, and the trainer runs in fp32.
* **fp32 physics has not been validated over a full training run.**
  `gate_batched_exec.py` measures the fp32 reward error at 2.10e-04 and argues
  it is uncorrelated over an episode; that argument has not been checked
  against an actual fp32-vs-fp64 training comparison.
* **The node-ordering fragmentation is unfixed.** Canonicalising node order
  would merge 17 groups into 16 untrained and 2 into 1 at convergence, halving
  the sampler's cost in the converged regime. It is provably safe (the policy
  is permutation equivariant) and it is not implemented or gated.
* **`WarpBackend.step` synchronises the device on every step.** Removing that
  sync could hide most of the ~12 ms fixed cost per batched step, which is the
  single biggest lever left on sampling throughput. Not attempted: whether warp
  and torch share a stream here was not established, and guessing wrong is a
  silent race.
* **The floor-penetration finding has not been reproduced in the port.** It
  was measured in THEIR venv on THEIR checkpoints. The port's contact
  parameters are gated identical (`physics_bridge_gate.py`: `geom_solref`,
  `geom_solimp`, `geom_friction`, `geom_margin` all 0.0e+00 apart), so it
  should behave the same, but that has not been checked.
* **The port has been timed for exactly two epochs, both untrained.** Nothing
  here is a measurement of a converged epoch, and the converged regime is where
  the port is expected to win. Do not quote an end-to-end speedup yet.
* **Eval count.** `n_eval` is logged per settled decision 6, and the default is
  16 to match their 16-thread run. It has not been checked that 16 mean-action
  episodes of the port have the same spread as 16 of theirs.

### The decision 3e needs from whoever owns it

The port's leverage is in the update, and settled decision 4 is what caps the
rest: ~57,000-64,000 agent-steps per iteration is only ~62 episodes at
convergence, so the sampler runs 62 worlds on a card that costs the same at
2,048. Three ways forward, and they are not equivalent:

1. **Take what is there and run seeds.** Faithful to every settled decision.
   Epoch 0 measured at 123.6 s against their 135.9 s, and the gap should widen
   in the port's favour as topologies converge. This is what has been launched.
2. **Spend on the sampler.** The two levers are (a) removing `WarpBackend`'s
   per-step `wp.synchronize_device()`, which currently prevents any pipelining
   across the ~112 small group rollouts, and (b) canonicalising node order to
   merge order-variant groups (17 -> 16 untrained, 2 -> 1 converged). (a) needs
   a stream-ordering answer before it can be trusted; (b) needs a permutation
   gate. Together they are plausibly the difference between 1.1x and 2-3x
   early.
3. **Relax the batch size.** 512 or 1,024 worlds per iteration with
   proportionally fewer iterations to the same 50 M simulation steps -- Figure
   3's own x-axis. This is where the measured 67,000-123,000 steps/s lives and
   it would make a seed a matter of hours. It is a deliberate departure from
   settled decision 4 and it changes the algorithm (bigger batch, fewer
   gradient steps), so it must not be done quietly.

**Recommendation: (1) is running; (2) is the next unit of work; (3) only as a
labelled second experiment.** The point of 3e is a comparison, and (3) changes
the thing being compared.

### What is running, and how to stop it

```sh
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
cd /workspace/utmist-vc2-phase2
PYTHONPATH=. MUJOCO_GL=egl setsid nohup .venv/bin/python \
    -m rower_soccer.t2a_port.train_t2a --cfg hopper_gpu_s2 --run port_s1 \
    --outdir runs/t2a_port --seed 1 --eval-worlds 16 --max-worlds 1024 \
    --mempool-mb 256 --epochs 1000 --save-interval 100 \
    --stop-file /tmp/stop_t2a_port_s1 \
    > runs/t2a_port/port_s1.log 2>&1 &
```

```
log:      runs/t2a_port/port_s1.log    and runs/t2a_port/port_s1/log_train.txt
ckpts:    runs/t2a_port/port_s1/models/epoch_*.p  (every 100) + best.p, gitignored
stop:     touch /tmp/stop_t2a_port_s1  -- saves stopped.p and exits at the next
                                          epoch boundary. DO NOT kill it, and do
                                          NOT wrap it in `timeout`.
pace:     ~119 s/epoch measured over the first two (untrained) epochs, so under
          33 h if it never got faster -- and it should get faster as the
          skeleton stage converges to one or two topologies. n = 1 seed; 3e
          wants at least three, and hopper_gpu_t32 says one of them may land in
          the survival trap rather than the running basin.
```

Each log line is their format plus a JSON line carrying `batch_steps`,
`n_train_eps`, `n_eval`, `gens`, `groups`, `batched_steps`, `gen_fill`,
`buckets`, `contact_buf_peak` and `steps_per_s_sample`. `n_eval` is there
because settled decision 6 asks for it.



### Files added or changed

| file | what |
|---|---|
| `t2a_port/design_stage.py` | **new.** Their design stages, CPU, no MuJoCo |
| `t2a_port/two_stage_pipeline.py` | **new.** Topology grouping, per-world field writes, group envs |
| `t2a_port/gate_two_stage.py` | **new.** The step-5 gate, 15/15 on both backends |
| `t2a_port/train_t2a.py` | **new.** The port's PPO loop |
| `t2a_port/dense_policy.py` | sampling, per-graph log-probs, `RunningNorm.update`, `DenseTransform2ActValue` |
| `t2a_port/gate_dense_policy.py` | +6 checks (log-prob x3, critic, critic control, mixed-topology batching): 8/8 -> 14/14 |

### How to re-run everything

```sh
# gates 1 and 2 -- their venv
cd /workspace/Transform2Act && source env-gpu.sh
.venv-gpu/bin/python .../t2a_port/gate_dense_policy.py                  # 14/14

# gate 3 -- ours, both backends
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
cd /workspace/utmist-vc2-phase2
PYTHONPATH=. .venv/bin/python -m rower_soccer.t2a_port.gate_batched_exec --check --backend warp   # 11/11
PYTHONPATH=. .venv/bin/python -m rower_soccer.t2a_port.gate_batched_exec --check --backend cpu    # 11/11

# gate 4 -- step 5. The reference is emitted from THEIR venv first:
cd /workspace/Transform2Act && source env-gpu.sh
.venv-gpu/bin/python .../t2a_port/gate_two_stage.py --emit --checkpoint 1000 --episodes 40 --tag e1000
.venv-gpu/bin/python .../t2a_port/gate_two_stage.py --emit --checkpoint 0    --episodes 60 --tag e0
cd /workspace/utmist-vc2-phase2
PYTHONPATH=. .venv/bin/python -m rower_soccer.t2a_port.gate_two_stage --check --backend cpu    # 15/15
PYTHONPATH=. .venv/bin/python -m rower_soccer.t2a_port.gate_two_stage --check --backend warp   # 15/15
```

---

## Update 2026-08-27 (second) — fp32 is NOT why the port trains to 1/12, and what is

*Written by the agent asked to test the precision hypothesis. It was tested and
it is dead. A different cause was found, gated, and fixed; the fix has NOT been
trained. Every claim below names the command that produced it, and the "Not
tested" section at the end is the important part.*

The short version, in the order the evidence arrived:

1. `--fp64` **changes the policy/update dtype only.** `mujoco_warp` is
   **float32-only** and there is no switch.
2. `--fp64` **had never run** — it raised on the first forward pass. Fixed.
3. **Their converged policy scores 11,547 inside the port's fp32 pipeline**
   against 11,228-11,964 in their own log. The port's env, physics, design
   stage, reward and done condition are therefore not the problem, and neither
   is fp32.
4. The same policy in **fp64 CPU physics scores 11,547** — identical. fp32
   physics costs nothing measurable.
5. The 12x gap is **entirely episode LENGTH**, not reward rate. Their agents
   reach the 1,000-step time limit by epoch 50 and stay there; ours die at 103.
6. **`agent_specs.batch_design` was never ported.** It is set true in every
   hopper cfg they ship, and it makes their minibatches stage-pure. Without it
   the port takes **15x more Adam steps on the design towers per epoch**, each
   from a seventeenth of the data.

### 1. What `--fp64` actually changes: the torch half, and nothing else

`train_t2a.py:267` sets `self.dtype` from `--fp32/--fp64` and uses it for the
policy, the critic, the stored batch and the loss. It does **not** reach the
simulator. `batched_exec_env.py:151` sets the env's dtype from the BACKEND —
`self.dtype = self.backend.qpos.dtype` — and the backend chooses it:

* `WarpBackend` aliases `mujoco_warp`'s own buffers with `wp.to_torch`, so its
  dtype is whatever warp allocated;
* `CompeteCpuBackend` takes `dtype=torch.float64`.

**`mujoco_warp` is float32-only.** Its `Data`/`Model` arrays are declared
`array(..., float)`, which is `wp.float32` in a warp kernel, and `_src/io.py:426`
says so in as many words: *"C MuJoCo tolerance was chosen for float64
architecture, but we default to float32 on GPU"* — it then raises `opt.tolerance`
to 1e-6 to stop the solver chasing precision it does not have. Measured, not
inferred:

```sh
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
cd /workspace/utmist-vc2-phase2 && PYTHONPATH=. .venv/bin/python -c "
import mujoco, mujoco_warp as mjw, warp as wp
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><joint type=\"slide\"/>'
                                   '<geom type=\"sphere\" size=\".1\"/></body></worldbody></mujoco>')
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
wm, wd = mjw.put_model(m), mjw.put_data(m, d, nworld=4, nconmax=8, njmax=8)
print(wd.qpos.dtype, wd.qvel.dtype, wm.body_mass.dtype, wm.opt.timestep.dtype)"
# warp._src.types.float32 x4
```

So **`--fp64` buys fp64 in the torch half and the physics stays fp32 no matter
what.** An fp64 training arm can only ever test the policy/update precision;
there is no configuration of this port that runs fp64 physics on the GPU. That
is the reframing asked for, and it holds.

### 2. `--fp64` crashed on the first forward pass, and had never been run

The env hands the policy `obs` and `adj` in the BACKEND's dtype. Under
`--fp64` the policy is float64 and those arrive float32, and `nn.Linear`
refuses the mix:

```sh
cd /workspace/utmist-vc2-phase2 && PYTHONPATH=. .venv/bin/python -c "
import torch, yaml
from rower_soccer.t2a_port.dense_policy import DenseTransform2ActPolicy
cfg = yaml.safe_load(open('/workspace/Transform2Act/design_opt/cfg/hopper_gpu_s2.yml'))
p = DenseTransform2ActPolicy(cfg['policy_specs'], 4, 5, 3, 3, control_action_dim=1).to(torch.float64)
p.act('execution', torch.randn(2,4,12), torch.zeros(2,4,4), torch.zeros(2,4,dtype=torch.long))"
# RuntimeError: expected scalar type Double but found Float
```

The fp32 path only works because both halves happen to be float32. `--backend
cpu` is broken in the mirror image (float64 env, float32 policy) and always was.

**Fixed** in `train_t2a.py`'s `rollout`: `obs`, `adj`, `nobs` and `r` are cast
to the trainer's dtype at the sim boundary. `.to()` on a matching dtype is a
no-op, so **the fp32 path is unchanged** — `port_s1` (pid 992213) is unaffected
and was not restarted.

### 3. The port's environment is not the problem: THEIR policy scores THEIR number in it

This is the measurement that should have been made before any precision test,
and it is cheap. Their `hopper_gpu_s2` epoch-1000 policy loads into the dense
policy `strict=True` (65 tensors, 0 missing, 0 unexpected) and is rolled through
the **port's** design stage, XML conversion, topology grouping, fp32
`mujoco_warp` physics, reward and done condition:

| | episodes | mean length | `exec_R_eps` | reward/step |
|---|---|---|---|---|
| **their policy, port env, fp32 warp** (seed 5) | 16 | 929.3 | **11,403.9** | 12.27 |
| **their policy, port env, fp32 warp** (seed 23) | 32 | 938.8 | **11,547.4** | 12.30 |
| **their policy, port env, fp64 CPU MuJoCo** (seed 5) | 8 | 938.4 | **11,547.5** | 12.31 |
| their policy, THEIR env, from their log epoch 998/999 | 16 | ~919 | **11,963.9 / 11,228.2** | 12.31 / 12.22 |
| `gate_batched_exec --check --backend cpu` closed loop (fp64) | 20 | 926.1 +/- 55.2 | **11,352.3 +/- 882.0** | SMD -0.051 |
| `gate_batched_exec --check --backend warp` closed loop (fp32) | 20 | 902.1 +/- 56.2 | **10,986.1 +/- 930.6** | SMD -0.451 |
| their policy, THEIR env, same gate's reference | 20 | 928.4 +/- 51.5 | **11,397.6 +/- 848.2** | — |

```sh
cd /workspace/Transform2Act && source env-gpu.sh   # re-export their pickle; see end_probe.py's docstring
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
cd /workspace/utmist-vc2-phase2
PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -m rower_soccer.t2a_port.end_probe \
    --their-npz their_e1000.npz --worlds 32 --seed 23 --mean-action
PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -m rower_soccer.t2a_port.end_probe \
    --their-npz their_e1000.npz --worlds 8  --seed 5  --mean-action --backend cpu
```

Two things follow, and both are load-bearing:

* **The port's environment is faithful.** Reward/step agrees with theirs to
  three significant figures and the return sits inside their own epoch-to-epoch
  spread. Every earlier gate checked a component; this checks the whole
  pipeline against a number the port cannot fake.
* **fp32 physics costs at most a few percent.** The two probes above give
  11,547.5 (fp64 CPU) and 11,403.9-11,547.4 (fp32 warp) -- no separation at 8-32
  episodes. The paired 20-episode closed loop inside `gate_batched_exec` is
  more sensitive and does see one: **10,986 fp32 against 11,352 fp64, 3.2%**,
  moving the gate's standardised mean difference against their env from -0.05
  to -0.45. So fp32 physics is not free -- but 3% is not 1,150%, and the
  `PORT_MAP` section 13 trajectory divergence does not matter at the level of a
  return. (That closed loop was ALREADY in the gate at 11/11 on both backends
  and already said the port's env reproduces their score; what it had not been
  used for was to rule out the precision hypothesis, which it does.)

**This is what settles the precision question, not the training arm.** If fp32
were degrading the task, their policy could not score their number in it.

### 4. The gap is episode LENGTH, and the reward RATE was never the problem

`exec_R_eps` is a rate times a length, and the two move in opposite directions
here, so the aggregate hides the mechanism. Split out (`exec_R` is already in
both logs; length is their ratio):

| epoch | port R/step | port len | port `R_eps` | ref s1 R/step | ref s1 len | ref s1 `R_eps` | ref s2 R/step | ref s2 len | ref s2 `R_eps` |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 1.00 | 42.2 | 42.2 | 1.00 | 42.1 | 42.1 | 1.00 | 42.1 | 42.1 |
| 10 | 1.01 | 43.0 | 43.4 | 2.68 | 78.7 | 211.0 | 2.74 | 75.2 | 206.1 |
| 25 | 4.28 | **27.2** | 116.7 | 3.64 | 91.4 | 332.7 | 3.60 | 97.8 | 352.2 |
| 50 | 4.33 | 52.2 | 225.7 | 1.31 | **1002.7** | 1313.6 | 1.36 | **997.9** | 1357.1 |
| 100 | 3.58 | 85.8 | 306.9 | 1.82 | 1000.6 | 1821.1 | 1.53 | 1002.6 | 1534.0 |
| 156 | 3.83 | 94.4 | 362.1 | 2.31 | 998.7 | 2307.0 | 2.53 | 1001.0 | 2532.6 |
| 250 | 4.03 | 102.9 | 414.4 | 5.12 | 1000.0 | 5120.2 | 4.12 | 769.6 | 3170.8 |
| 391 | 4.19 | 107.7 | 451.6 | 5.61 | 1000.7 | 5613.8 | 5.18 | 962.1 | 4983.8 |
| 400 | 4.20 | 107.4 | 450.5 | 6.21 | 999.3 | 6205.6 | 5.61 | 527.2 | 2957.6 |

Read the shapes, not the endpoints. **Both curves start at 42 steps, which is
free-fall**: a hopper is dropped from `rootz = 1.25` and the episode ends when
the root passes 0.7, which takes `sqrt(2*0.55/9.81) = 0.335 s = 42` control
steps. The reference **buys survival first** — by epoch 50 its episodes hit the
1,000-step limit and its reward RATE has fallen to 1.31, i.e. almost pure alive
bonus — and spends the next 950 epochs converting that into speed. The port
does the opposite: it takes the reward rate to 4.3 by epoch 25 while its
episodes get SHORTER than the untrained ones, and then creeps back up 27 -> 107
over 375 epochs. **At epoch 50 the port's agent is faster per step than theirs
and scores a sixth as much.**

What the port's agent actually does, at epoch 400 (64 sampled episodes, 9
topologies):

```sh
PYTHONPATH=. MUJOCO_GL=egl .venv/bin/python -m rower_soccer.t2a_port.end_probe \
    --ckpt runs/t2a_port/port_s1/models/epoch_0400.p --worlds 64 --seed 7 --trace
```

**100% of episodes end `FELL`, none time out, none go non-finite**, at a mean
length of 103.3 with a spread of 98-112 across every design. The height trace
explains why it is so tight: it is one ballistic arc.

```
t=   0 h=1.2537   t=  30 h=1.8657   t=  60 h=1.9489   t=  90 h=1.3505
t=  10 h=1.5116   t=  40 h=1.9582   t=  70 h=1.8267   t= 100 h=0.9988
t=  20 h=1.7170   t=  50 h=1.9896   t=  80 h=1.6275   -> done at ~105
```

It launches, coasts to **1.9896** — the `max_height` cut-off is 2.0 — and dies
on landing. The policy has learned to ride the ceiling of the done condition
for one jump. Note that no metric in the training log says this; `exec_R_eps`
just climbs slowly, exactly as `hopper_gpu_t32`'s sprawl did.

### 5. The cause: `agent_specs.batch_design` was never ported

`hopper_gpu_s2.yml` opens with `agent_specs: {batch_design: true}`, and
`use_mini_batch` is `cfg.mini_batch_size (2048) < cfg.min_batch_size (50000)`,
so this branch runs on every one of their epochs
(`design_opt/agents/transform2act_agent.py:272-297`):

```python
perm_np = np.arange(num_state); np.random.shuffle(perm_np)          # shuffle
...
if self.cfg.agent_specs.get('batch_design', False):
    perm_design_np, perm_design = self.get_perm_batch_design(states)  # then SORT BY STAGE
```

`get_perm_batch_design` returns `inds[0] + inds[1] + inds[2]` — skeleton rows,
then attribute rows, then execution rows. Minibatches are consecutive slices of
that array, so **each of their minibatches is stage-PURE** except at the two
boundaries. The port sliced a plain `randperm`, so **every** port minibatch was
stage-mixed. Nothing in `t2a_port/` or `PORT_MAP.md` mentions `batch_design` or
`agent_specs` at all; it was not a rejected option, it was unseen.

It is not a wash. A batch holds `6 * n_episodes` design rows (5 skeleton + 1
attribute per episode) against `batch_steps` execution rows, and the trainer's
own logged `minibatches` confirms the composition exactly at every epoch:

| epoch | episodes | design rows | exec rows | minibatches (logged = computed) | minibatches touching a design tower — theirs | ours |
|---|---|---|---|---|---|---|
| 0 | 2,023 | 12,138 | 57,605 | 34 | **6** | 34 |
| 50 | 1,405 | 8,430 | 58,431 | 32 | **5** | 32 |
| 100 | 881 | 5,286 | 57,484 | 30 | **3** | 30 |
| 400 | 568 | 3,408 | 58,500 | 30 | **2** | 30 |

Each minibatch is one Adam step. So per optimisation epoch the design towers
take **2 steps in the reference and 30 in the port** at epoch 400 — 15x — and
the port's steps are computed from ~120 design rows instead of 2,048. Adam
divides a gradient by its own running RMS, so a tower that sees a seventeenth
of the data does **not** take a seventeenth of a step; it takes a full-sized
step from a seventeenth of the data, ten times more often. The attribute tower
is worse still: it is 1/6 of the design rows, so it fits inside a single
minibatch of theirs (10 Adam steps per epoch) against the port's 300.

That is a plausible mechanism for exactly what the port shows — a skeleton
distribution that never settles (**30-35 topology groups at epoch 400**, where
`topology_census.py` found 2 in their epoch-1000 batch) and a control policy
that never gets a stable body to master, so it learns a body-independent trick
instead. **It is a mechanism, not yet a demonstration.** See "Not tested".

**Fixed and gated.** `train_t2a.py` grows `stage_sorted_perm()` and reads
`agent_specs.batch_design` from the cfg, so the default now follows theirs;
`--no-batch-design` restores the old behaviour for an A/B.

```sh
PYTHONPATH=. .venv/bin/python -m rower_soccer.t2a_port.gate_batch_design   # 9/9
```

The gate builds the real epoch-400 composition, checks the permutation is a
permutation, that at most two minibatches straddle a boundary, that exactly two
carry a design row, that the stage profile equals a transcription of their
`get_perm_batch_design`, and that within a stage the order is still shuffled —
with three controls that must fail and do (an unsorted permutation puts design
rows in 30/30 minibatches; a constant sort key purifies nothing).

### 6. The fp64 training arm, matched to `port_s1`

Launched anyway, because "their policy scores their number" is an argument
about the env and someone could still ask about the update:

```sh
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
cd /workspace/utmist-vc2-phase2
PYTHONPATH=. MUJOCO_GL=egl setsid nohup .venv/bin/python \
    -m rower_soccer.t2a_port.train_t2a --cfg hopper_gpu_s2 --run port_s1_fp64 \
    --outdir runs/t2a_port --seed 1 --fp64 --eval-worlds 16 --max-worlds 1024 \
    --mempool-mb 256 --epochs 1000 --save-interval 100 \
    --stop-file /tmp/stop_t2a_port_s1_fp64 > runs/t2a_port/port_s1_fp64.log 2>&1 &
```

Identical to `port_s1` in every argument but `--fp64`. **Two differences from
`port_s1` that are not precision and must be recorded**: it carries the
`len_est` fix (`port_s1` does not), which changes the world count per
generation but not `batch_steps`; and it predates the `batch_design` fix, like
`port_s1`, so the A/B is clean for precision.


It ran **52 epochs** (0-51) and was ended with its stop file, not killed —
`stop file /tmp/stop_t2a_port_s1_fp64 present -- saving and exiting cleanly at
epoch 52`, leaving `runs/t2a_port/port_s1_fp64/models/stopped.p`. 52 epochs is
enough because **the port's pathology is fully formed by epoch 15**, and both
arms show it:

| epoch | fp32 `R_eps` | len | R/step | fp64 `R_eps` | len | R/step | ref s1 `R_eps` | len | ref s2 `R_eps` | len |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 42.2 | 42.2 | 1.00 | 42.1 | 42.1 | 1.00 | 42.1 | 42.1 | 42.1 | 42.1 |
| 5 | 42.3 | 42.2 | 1.00 | 42.7 | 42.6 | 1.00 | 93.5 | 66.3 | 207.0 | 131.0 |
| 10 | 43.4 | 43.0 | 1.01 | 43.6 | 43.0 | 1.01 | 211.0 | 78.7 | 206.1 | 75.2 |
| 15 | 74.7 | **21.9** | 3.40 | 72.4 | **21.0** | 3.45 | 275.6 | 86.4 | 265.5 | 82.2 |
| 20 | 93.3 | 23.9 | 3.91 | 88.2 | 22.4 | 3.94 | 335.2 | 90.4 | 311.0 | 92.0 |
| 25 | 116.7 | 27.2 | 4.28 | 108.5 | 24.9 | 4.35 | 332.7 | 91.4 | 352.2 | 97.8 |
| 30 | 206.3 | 48.4 | 4.27 | 144.1 | 31.8 | 4.54 | 370.7 | 99.1 | 420.7 | 116.9 |
| 35 | 217.3 | 50.2 | 4.33 | 222.8 | 50.7 | 4.40 | 463.7 | 138.8 | 465.5 | 132.6 |
| 40 | 224.0 | 51.0 | 4.39 | 223.3 | 50.8 | 4.40 | 493.7 | 150.1 | 519.8 | 195.4 |
| 45 | 228.2 | 52.5 | 4.35 | 224.2 | 50.9 | 4.41 | 1095.6 | 725.6 | 594.7 | 191.8 |
| 50 | 225.7 | 52.2 | 4.33 | 210.0 | 46.6 | 4.50 | 1313.6 | 1002.7 | 1357.1 | 997.9 |

Block means, like for like:

| block | fp32 `R_eps` (len) | fp64 `R_eps` (len) | ref s1 | ref s2 |
|---|---|---|---|---|
| 0-24 | **64.8** (32.4) | **62.7** (32.8) | 216.1 | 223.8 |
| 25-49 | **206.0** (47.3) | **195.5** (44.3) | 670.3 | 553.8 |

**fp64 is 3-5% BELOW fp32, and the reference's own two seeds differ by 21% in
the same block.** The difference between the arms is not distinguishable from
nothing; the difference between either arm and the reference is a factor of
3.3. Both arms make the *same wrong move at the same epoch*: between epoch 10
and 15 the reward RATE jumps from 1.01 to ~3.4 while episode LENGTH **halves**,
43 -> 21, and length then crawls back to ~50 while the rate sits at 4.4. The
reference does the opposite in the same window.

Two caveats on how controlled this is. The arms share `--seed 1` and start from
the same initialisation (`exec_R_eps` 42.2 against 42.1), but their RNG streams
are not identical once fp64 changes the numbers the samplers consume, so this
is a paired-start comparison, not a bitwise-controlled one — which is why the
block means matter more than epoch 30's 206.3 against 144.1. And the fp64 arm
carries the `len_est` fix that `port_s1` does not.

**Verdict: precision is not why the port trains to a twelfth of the reference,
and the fp64 arm was the weakest of the four pieces of evidence saying so.**

Also worth recording, because it is a design-diversity measurement and it is
the thing `batch_design` predicts: sampled (not mean) designs at 32 worlds,
same probe, same seed —

| policy | distinct topologies in 32 sampled designs | bodies |
|---|---|---|
| `port_s1` epoch 400 | **8** | 6-8 |
| their `hopper_gpu_s2` epoch 1000 | **2** | 7 exactly |

with both collapsing to 1 under mean actions. Their `topology_census.py` found
1 topology in 199 of 200 designs by **epoch 100**. The port's skeleton
distribution is still wide at epoch 400 — which is what a design tower taking
15x the Adam steps on a seventeenth of the data would look like, and is the
observable to watch in the next experiment.

#### The throughput and memory cost of fp64

| | `port_s1` (fp32) | `port_s1_fp64` |
|---|---|---|
| `T_update`, epoch 0-1 | **30.4 / 29.4 s** | **64.0 / 68.5 s** |
| `T_sample`, epoch 0-1 | 91.0 / 89.9 s | 75.1 / 82.7 s |
| torch peak (`gpu_mib`), epoch 0-1 | 2,944 / 3,167 MiB | **5,380 / 6,312 MiB** |
| process GPU, `nvidia-smi` mid-epoch | ~4.2 GB | **7.4-7.9 GB** |

The **update is 2.2x slower** and torch's peak is **1.9x** — consistent with
the isolated 27.0 ms -> 108.1 ms per gradient step already recorded above,
diluted by the parts of an epoch that are not fp64 matmul. **Sampling is not
slower**, and that is the tell: the physics is fp32 in both arms, so only the
policy forward inside the rollout changes. (The two `T_sample` columns are not
directly comparable anyway — they were measured under different contention on a
shared card.)

**Card budget, reported as asked**: `port_s1` ~0.8 GB between epochs and ~4.2 GB
at the mid-epoch peak; `port_s1_fp64` 7.4-7.9 GB at peak. With both plus the
probes the card reached **~9.1 GB of 20 GB**. The fp64 arm alone is at the
stated ~8 GB ceiling, which is another reason not to run fp64 arms casually.

### What is NOT tested

* **The `batch_design` fix has not been trained.** It is gated for what the
  permutation does, not for what it buys. The mechanism above (15x the Adam
  steps on the design towers, from a seventeenth of the rows) is an argument
  from the reference's source plus row counts; **no training run has shown that
  fixing it closes any part of the gap.** That is the next experiment and it is
  not optional before anyone believes this section's headline.
* **`port_s1` and `port_s1_fp64` both predate the fix**, so neither is evidence
  about it either way.
* **Nothing here rules out a second cause.** `batch_design` is the only
  unported piece found by reading their `update_params`/`update_policy` line by
  line against the port's; the rest (fixed log-probs recomputed under frozen
  norm statistics, `value_opt_niter = 1`, grad clip 40 on the policy only, no
  value clip, advantage normalised by std over the whole batch, `noise_rate`
  1.0 so every training transition is sampled, `end_reward` false,
  `running_state` None) all match. Matching by reading is not matching by
  measurement: **`gate_dense_policy.py` still does not drive one optimiser step
  in both codebases and compare the weights**, which the previous handoff
  already listed as the cheapest missing gate and which would have caught this
  one.
* **fp64 physics on the GPU is untestable here**, so "would fp64 physics train
  differently" is answered only indirectly — by their policy scoring the same
  in fp32 warp and fp64 CPU MuJoCo, which is a statement about evaluation, not
  about training.
* **A pre-training discrepancy in the TRAINING distribution is unexplained.**
  At epoch 0 the eval metric matches theirs to 0.3% (`exec_R_eps` 42.25 against
  42.12/42.10), but the sampled rollouts do not: `train_R` 1.00 and
  `train_R_eps` 28.53 here against **0.83 and 28.93** in BOTH their seeds, i.e.
  their sampled episodes are ~34.9 steps long with a mean `dx/dt` of -0.17 m/s
  and ours are 28.5 with 0.00. The design sampler is not the cause -- 200
  untrained port designs give **24 distinct topologies** against the 21 their
  `topology_census.py` reports for 200, with a similar size profile. Small, and
  it predates training, so it should be chased before any further training
  comparison is called clean.
* **`--backend cpu` in the trainer.** Now usable with `--fp64` (and gated only
  to the extent that `end_probe --backend cpu` ran); with `--fp32` it is still
  broken in the mirror image, because `CompeteCpuBackend` is float64 and the
  cast added here goes one way only. Nothing trains on the CPU backend, so this
  is a latent trap rather than a live bug.
* **Correction to settled decision 6.** "Theirs averages `exec_R_eps` over
  `num_threads` episodes" is true only at convergence. Their eval is
  `sample(cfg.eval_batch_size, mean_action=True)` with `eval_batch_size = 10000`
  **agent-steps** (`design_opt/utils/config.py:49`), split over the workers, so
  at ~950-step episodes it is 16 episodes but at epoch 10 (79-step episodes) it
  is ~128. The port's 16 is right at convergence and 8x too few early. It does
  not explain anything here -- the gaps at those epochs are 5x, far outside
  eval noise -- but the early part of the port's curve is noisier than theirs
  by construction.

### The single next experiment

**Train one seed with `batch_design` on, everything else matched to `port_s1`,
and compare episode LENGTH at matched epochs — not the return.**

```sh
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
cd /workspace/utmist-vc2-phase2
PYTHONPATH=. MUJOCO_GL=egl setsid nohup .venv/bin/python \
    -m rower_soccer.t2a_port.train_t2a --cfg hopper_gpu_s2 --run port_s2_bd \
    --outdir runs/t2a_port --seed 1 --eval-worlds 16 --max-worlds 1024 \
    --mempool-mb 256 --epochs 1000 --save-interval 100 \
    --stop-file /tmp/stop_t2a_port_s2_bd > runs/t2a_port/port_s2_bd.log 2>&1 &
```

The read-out is cheap and early: **the reference's episodes reach the 1,000-step
limit by epoch 50.** `port_s1`'s were 52.2 there and 27.2 at epoch 25. So by
epoch 50-75 -- about 1.5 hours -- this either shows episode length climbing
toward the limit or it does not, and no one has to wait 1,000 epochs to learn
which. Log `eval_len` beside `exec_R_eps` when reporting it; the return alone is
what hid this for 400 epochs.

If it does not move, the next candidates in order are (1) write the missing
one-optimiser-step gate against their codebase, which is the only thing that
would catch a second `batch_design`-shaped omission, and (2) two more seeds,
because `hopper_gpu_t32` shows this task's seed spread is bimodal rather than
tight. The sampler shape and the batch/world count are **not** on that list:
the port's batch at epoch 0 is 2,023 episodes against their ~1,760 for the same
50,000-step budget, its generations produce only complete episodes as theirs
do, and its untrained design distribution matches theirs.

### Files added or changed

| file | what |
|---|---|
| `t2a_port/train_t2a.py` | `stage_sorted_perm` + `agent_specs.batch_design`; fp64 dtype cast at the sim boundary |
| `t2a_port/gate_batch_design.py` | **new.** 9/9, three controls that must fail |
| `t2a_port/end_probe.py` | **new.** Termination census; runs THEIR checkpoint through the port |

## Update 2026-08-28 — `batch_design` was trained, and it does NOT close the gap

*Written by the agent asked to run the experiment the previous section named as
"the single next experiment". It was run. The answer is negative, and it is
negative in a way that changes the diagnosis: the previous section's headline
("`agent_specs.batch_design` is the leading cause of the 12x") is **not
supported** and should be read as refuted, not merely unconfirmed.*

The short version:

1. `gate_batch_design.py` re-run independently: **9/9**, three controls fail on
   demand. The permutation is right, and their `get_perm_batch_design` and the
   cfg were re-read from their source rather than taken from this file.
2. One seed was trained with `batch_design` **on**, matched to `port_s1`:
   `runs/t2a_port/port_s1_bd`, 101 epochs, ended with its stop file.
3. **It is worse than the arm with the bug**, on the decisive readout and on
   every other one. At epoch 100 the sampled (training) episode is **33.7
   steps with `batch_design` on** against **65.3 with it off** and **674-780 in
   the reference**. It is 28.5 steps at epoch 0, so in 100 epochs the
   `batch_design`-on arm bought **5 steps of survival**.
4. The port's *training* episodes are 6 steps shorter than theirs **at epoch 0,
   before any gradient step**, in every arm — 28.5-29.6 against 34.6-34.9 — and
   that gap never closes in any arm. That, not the minibatch permutation, is
   now the most specific unexplained thing on the table.
5. A side effect worth keeping: stage-pure minibatches make `T_update`
   **2.1x cheaper** (13.7 s against 28.1 s per epoch), because each minibatch
   then touches one `Bucket` instead of three.

### The gate, re-run and re-derived from their source

```sh
cd /workspace/utmist-vc2-phase2
PYTHONPATH=. .venv/bin/python -m rower_soccer.t2a_port.gate_batch_design
# 9/9 checks passed
```

Checked against their code rather than against the previous section's summary
of it:

* `design_opt/cfg/hopper_gpu_s2.yml:2-3` — `agent_specs: {batch_design: true}`,
  and `min_batch_size: 50000` / `mini_batch_size: 2048` so `use_mini_batch` is
  true. Confirmed by reading the file.
* `design_opt/agents/transform2act_agent.py:250-256` — `get_perm_batch_design`
  buckets by `x[2]` and returns `inds[0] + inds[1] + inds[2]`.
* `transform2act_agent.py:281-284` — it is applied AFTER the plain shuffle, so
  it is a stable sort of the shuffled order by stage.
* `transform2act_agent.py:287` — `optim_iter_num = int(math.floor(num_state /
  mini_batch_size))`, i.e. the tail is dropped, which is what the port's
  `batch.size // self.mini` also does.
* `design_opt/envs/hopper.py:197-198` — `if_use_transform_action` indexes
  `['skeleton_transform', 'attribute_transform', 'execution']`, so stage ranks
  are 0/1/2 and `train_t2a.py`'s `STAGE_RANK` matches.

So the mechanism described in the previous section is real: their design towers
take 2 Adam steps per optimisation epoch at epoch 400 and the port's took 30.
**The mechanism is real and its effect on training is the opposite of the one
predicted.**

### The run

```sh
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
cd /workspace/utmist-vc2-phase2
PYTHONPATH=. MUJOCO_GL=egl setsid nohup .venv/bin/python \
    -m rower_soccer.t2a_port.train_t2a --cfg hopper_gpu_s2 --run port_s1_bd \
    --outdir runs/t2a_port --seed 1 --eval-worlds 16 --max-worlds 1024 \
    --mempool-mb 256 --epochs 1000 --save-interval 100 \
    --stop-file /tmp/stop_t2a_port_s1_bd > runs/t2a_port/port_s1_bd.log 2>&1 &
```

Identical to `port_s1`'s argv except `--run`/`--stop-file`. `batch_design` is
not on the command line because the default now follows the cfg, so
`train_t2a.py` was given a startup line that writes the setting into the run's
own log — otherwise which arm a run is cannot be recovered afterwards:

```
runs/t2a_port/port_s1_bd/log_train.txt:1
run port_s1_bd  cfg hopper_gpu_s2  seed 1  batch_design True
  (cfg agent_specs.batch_design True, --batch-design None)  dtype torch.float32
```

It reached **epoch 101** in 2.76 h, was **ended with its stop file, not
killed**, and left `models/epoch_0100.p`, `best.p` and `stopped.p`.

Two differences from `port_s1` that are not `batch_design` and must be
recorded: `port_s1_bd` carries the `len_est` fix and the fp64 sim-boundary cast
(a no-op in fp32); `port_s1` predates both. **`port_s1_fp64` is the control
that covers this** — it is the same code lineage as `port_s1_bd` with
`batch_design` OFF, and it tracks `port_s1` to within a few percent for its 52
epochs. So the `len_est` fix is not what separates the arms.

### The readout: episode LENGTH on the TRAINING distribution

The previous section read episode length off the eval metric. **Do not do that
here.** `--eval-worlds 16` is 16 mean-action episodes on 16 sampled designs, and
on this arm it produced two spurious excursions — `eval_len` jumped 38 -> 116 at
epoch 71 and 25 -> 116 at epoch 75 and fell straight back — while the training
distribution (500-2,000 sampled episodes per epoch) did not move at all. The
training columns below are `train_R_eps / train_R`, both already in both logs.

| epoch | bd ON len | bd ON `R_eps` | bd off (`port_s1`) len | `R_eps` | bd off (`fp64`) len | `R_eps` | ref s1 len | `R_eps` | ref s2 len | `R_eps` |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 28.5 | 28.5 | 28.5 | 28.5 | 29.6 | 29.6 | **34.9** | 28.9 | **34.6** | 28.7 |
| 10 | 29.7 | 32.7 | 30.6 | 42.3 | 31.2 | 42.7 | 84.0 | 150.4 | 81.8 | 160.3 |
| 25 | 29.7 | 36.5 | 33.2 | 75.4 | 33.3 | 77.0 | 125.8 | 341.0 | 165.8 | 374.7 |
| 50 | 31.0 | 41.5 | 41.6 | 148.9 | 41.9 | 148.4 | 550.2 | 836.3 | 488.3 | 766.6 |
| 75 | 31.2 | 46.7 | 51.5 | 194.0 | — | — | 664.0 | 1009.2 | 743.7 | 1115.6 |
| 100 | **33.7** | **56.9** | **65.3** | **235.8** | — | — | 674.3 | 1220.5 | 780.2 | 1209.2 |

Block means, like for like:

| block | 0-24 | 25-49 | 50-74 | 75-100 |
|---|---|---|---|---|
| bd ON, train len | 29.6 | 30.1 | 30.9 | **32.6** |
| bd off (`port_s1`), train len | 31.0 | 37.0 | 46.9 | **58.9** |
| bd off (`fp64`), train len | 31.3 | 37.3 | 42.3 | — |
| ref s1 / s2, train len | 78.9 / 84.2 | 362.5 / 316.3 | 567.9 / 620.3 | 622.7 / 723.6 |
| bd ON, train `R_eps` | 33.2 | 38.6 | 43.3 | **51.1** |
| bd off (`port_s1`), train `R_eps` | 48.0 | 111.0 | 173.5 | **216.4** |
| ref s1 / s2, train `R_eps` | 172.2 / 183.8 | 623.8 / 573.9 | 884.0 / 950.7 | 1051.0 / 1155.0 |

And the eval metric the previous section used, for continuity — same verdict,
noisier:

| block | 0-24 | 25-49 | 50-74 | 75-100 |
|---|---|---|---|---|
| bd ON, `eval_len` / `exec_R_eps` | 42.0 / 42.0 | 42.6 / 47.8 | 53.0 / 80.5 | 34.7 / 76.1 |
| bd off (`port_s1`) | 32.4 / 64.8 | 47.2 / 206.0 | 53.9 / 226.0 | 73.7 / 274.1 |
| ref s1 | 76.5 / 216.1 | 362.7 / 670.3 | 937.9 / 1283.2 | 947.9 / 1454.7 |
| ref s2 | 85.4 / 223.8 | 220.9 / 553.8 | 999.6 / 1418.2 | 1000.1 / 1524.4 |

**The reference's two seeds differ from each other by 8-15% in these blocks.
The `batch_design`-on arm is 45% below the `batch_design`-off arm on training
length at 75-100 and 76% below it on training return. That is far outside the
seed spread, and it is in the wrong direction.**

Reproduce either table with the scripts used here (they are 40 lines of regex
over the two logs; nothing was hand-copied):

```sh
cd /workspace/utmist-vc2-phase2
# train_R_eps / train_R per epoch for all five arms
.venv/bin/python - <<'PY'
import re
def rd(p):
    o={}
    for l in open(p):
        m=re.match(r'^(\d+)\t.*train_R ([\d.eE+-]+)\ttrain_R_eps ([\d.eE+-]+)', l)
        if m: o[int(m.group(1))]=(float(m.group(3))/float(m.group(2)), float(m.group(3)))
    return o
for p in ['runs/t2a_port/port_s1_bd/log_train.txt','runs/t2a_port/port_s1/log_train.txt',
          '/workspace/Transform2Act/results_hopper_gpu.log','/workspace/Transform2Act/results_hopper_gpu_s2.log']:
    d=rd(p); print(p, {e:tuple(round(x,1) for x in d[e]) for e in (0,25,50,100) if e in d})
PY
```

### Verdict

**`batch_design` does not close the gap, does not partly close it, and makes
training strictly worse over 101 epochs.** The previous section's headline
should be treated as refuted.

What the run does establish, and it is not nothing:

* **Almost all of what the port was "learning" was coming from the
  over-stepped design towers.** Take the 15x extra Adam steps away and the arm
  goes nearly flat: training length 28.5 -> 33.7 over 100 epochs. The
  `batch_design`-off arms' gains — including the epoch-15 move where length
  *halves* to 21 and the reward rate jumps to 3.4 — are design-side, i.e. the
  port was finding bodies that fall forward, not a controller that hops.
* **So the port's EXECUTION tower is barely learning in any arm**, and that is
  a sharper statement of the defect than "the port trains to 1/12". The
  execution tower's Adam-step count is almost unchanged by this fix (28 of 30
  minibatches instead of 30 of 30), which is why the fix could not have helped
  it and why it did not.
* **`batch_design` should nevertheless stay on**, because it is what their code
  does and the port is supposed to be their code, and because it is 2.1x
  cheaper per update. It is now a matched-behaviour item, not a fix.

### The pre-training discrepancy is now the lead

The previous section listed this as a small loose end. This run promotes it,
because it is the only measured difference that is present in every arm,
present **before the first gradient step**, and pointed the same way as the
whole failure:

| | untrained sampled episode length | untrained `train_R` (= 1 + mean dx/dt) |
|---|---|---|
| ref seed 1 | **34.9** | 0.83 (mean dx/dt −0.17 m/s) |
| ref seed 2 | **34.6** | 0.83 |
| `port_s1` | 28.5 | 1.00 (mean dx/dt **0.00**) |
| `port_s1_fp64` | 29.6 | 1.00 |
| `port_s1_bd` | 28.5 | 1.00 |

Under **mean** actions the port and the reference agree to 0.3% (`exec_R_eps`
42.2 against 42.1). Under **sampled** actions they do not. PPO learns from the
sampled distribution, so a discrepancy that appears only there is exactly the
shape of a defect that would leave every mean-action gate green while training
fails.

What was checked and matches, so this is not those: `control_log_std: 0` and
`attr_log_std: -2.3` are read from the cfg into `nn.Parameter`s in
`dense_policy.py:236-239`, and `fix_control_std`/`fix_attr_std` are false in
both, i.e. learnable in both; `action_to_control` maps node column 0 to the
actuator index in both (`hopper.py:95-102` against
`batched_exec_env.py:275-282`); the reward and done conditions are the same
function (`hopper.py:156-179` against `batched_exec_env.terms`). What has
**never** been checked is a sampled rollout, end to end, from identical
weights.

### What is NOT tested

* **`port_s1_bd` was stopped at epoch 101, not run to 1,000.** The claim is
  "it is behind the `batch_design`-off arm and behind the reference at every
  epoch up to 101 and the separation is widening", not "it can never catch up".
  A late crossover after epoch 101 has not been excluded, only made
  unattractive to look for.
* **One seed.** `hopper_gpu_t32` showed this task's seed spread is bimodal.
  The separation here (76% on training return) is much larger than the
  reference's own two-seed spread (8-15%), which is why one seed is being
  treated as sufficient for a negative — but it is one seed.
* **The two eval excursions at epochs 71-75 are unexplained.** They are almost
  certainly the 16-episode mean-action eval landing on a lucky design sample,
  since the training distribution does not move across them, but nobody looked
  at the designs. `models/epoch_0100.p` is on disk if someone wants to.
* **The claim "the port's gains are design-side" is an inference**, from the
  fact that removing the design-tower over-stepping removes almost all of the
  learning. **No per-tower measurement was made.** Logging the parameter delta
  or gradient norm per tower per epoch would turn it into a measurement and
  costs one restart.
* **The `batch_design` implementation is gated for the permutation only.** It
  is not gated against their optimiser: `gate_dense_policy.py` still does not
  drive one PPO step in both codebases and compare weights. That gate is now
  two sections old as "the cheapest missing gate" and it is the thing that
  would have caught this one before 2.76 GPU-hours went into it.
* **Nothing here re-opens fp32, the environment, the sampler shape, or the
  batch size.** Those were settled in the previous section and this run did
  not touch them.

### The next experiment, cheapest first

1. **Compare a SAMPLED rollout from identical weights, both codebases.**
   Export one randomly-initialised policy, roll ~200 episodes with sampled (not
   mean) actions through their `hopper` env and through the port's, and compare
   the episode-length histogram and mean `dx/dt`. Minutes, no GPU-hours, no
   training. It targets the 28.5-against-34.9 discrepancy directly, and every
   existing gate is blind to it because they all drive mean actions.
   `end_probe.py` already loads their weights into the port and already has a
   `--mean-action` switch to turn off.
2. **Log the per-tower parameter delta and gradient norm per epoch**, then
   re-read 20 epochs of either arm. Turns "the port's gains are design-side"
   into a measurement and says outright whether the execution tower is learning.
3. **The one-optimiser-step cross-codebase gate.** Same batch into their
   `update_policy` and the port's `update`, compare per-tower weight deltas.
   Still the only thing that would catch another `batch_design`-shaped omission.

Two more seeds is **not** on this list yet. The separation being chased is a
factor of 20, and three cheap, non-training diagnostics are unspent.

### What is running

* **`port_s1` (pid 992213), epoch ~985 of 1000, still training WITH the bug.**
  It is the matched `batch_design`-off control for everything above and it is
  the only arm that has ever gone past epoch 101. **Not stopped — that is the
  owner's call.** Its continued value: it is ~15 epochs from finishing a
  complete 1,000-epoch curve, which is the reference-shaped artefact to compare
  against, and stopping it now buys back almost nothing. Its cost: ~0.8 GB
  idle, ~3.4 GB mid-epoch on a 20 GB card.
* `port_s1_bd` — **ended cleanly** at epoch 101 via `/tmp/stop_t2a_port_s1_bd`.
* `train_soccer2v2_warp` (D1, pid 3595703) — untouched, ~1.3 GB.

**Card**: `port_s1_bd` used ~0.9 GB between epochs and **4.5 GB at the
mid-epoch peak** (torch `gpu_mib` 3,416). With `port_s1` and the D1 run the
card peaked at ~9.3 GB of 20 GB. Nothing was killed; no CUDA process was
wrapped in `timeout`.

### Files added or changed

| file | what |
|---|---|
| `t2a_port/train_t2a.py` | one startup log line recording `run`/`cfg`/`seed`/`batch_design`/`dtype`, so an arm is identifiable from its own log |
| `runs/t2a_port/port_s1_bd/` | the 101-epoch `batch_design`-on arm, `epoch_0100.p` + `stopped.p` |
