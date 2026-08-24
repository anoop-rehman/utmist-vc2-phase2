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
| 3d — GPU port | **step 1 of 6 done** (dense policy, gated). Steps 2-6 open. |
| 3e — paper-number validation | not started |
| 3f — design+control on our drills | not started |
| M1 at paper scale (hopper) | **done, all 1000 epochs** |
| M1 at paper scale (ant) | **running**, epoch 88, ETA ~2d17h |

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
