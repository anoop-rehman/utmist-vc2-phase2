# Transform2Act port map (D3 unit 3b)

*Written 2026-08-12, from a read of `/workspace/Transform2Act` at the commit the
`hopper_gpu` run is training from. Companion to `COMPETEVO_PORT_MAP.md`.
Every claim here is either a file:line reference or a number measured by
`rower_soccer/t2a_port/gnn_playground.py` (unit 3c) — nothing is from the paper.*

The whole of `design_opt/` is **2,360 lines**. This is a small codebase with one
genuinely hard idea in it, and the port difficulty is concentrated in a single
place (topology changes mid-episode), not spread out.

## 1. What an episode is

Not an RL episode in the usual sense — it is a *design* followed by a *rollout*,
and the design steps go through the same policy and the same PPO buffer as the
control steps.

| stage | steps | physics? | action slice used | reward |
|---|---|---|---|---|
| `skeleton_transform` | `cfg.skel_transform_nsteps` (5 for hopper) | **no** | `a[:, -1]`, one categorical per node | 0 |
| `attribute_transform` | 1 | **no** | `a[:, control_dim:-1]`, continuous per node | 0 |
| `execution` | up to `done_condition.max_nsteps` (1000) | yes | `a[:, :control_dim]` | the locomotion reward |

`hopper.py:105-172`. The two design stages **do not step MuJoCo at all** — they
edit an XML, call `reload_sim_model`, and return an observation. That single
fact is what makes this portable: the expensive part of the episode is a
fixed-topology rollout, and the topology only changes at six known points.

Each skeleton step is a per-node choice from `{no-op, add child, remove}`
(`skel_num_action = 3 if cfg.enable_remove else 2`, `hopper.py:38`), gated by
`allow_add_body` / `allow_remove_body` (depth and child-count limits from
`cfg.add_body_condition`). The attribute step writes continuous parameters —
geom size, offsets, actuator gear — through `body.set_params`.

## 2. The observation is a graph, per node

`hopper._get_obs` (`hopper.py:258-274`) returns a **list**, not an array:

```
[ obs [N, attr_fixed + sim_obs + attr_design],   # per-node features
  edges [2, E],                                  # both directions, tree edges
  use_transform_action [1],                      # which of the 3 stages
  num_nodes [1],
  body_index [N] ]                               # only if use_body_ind
```

Three details that a port gets wrong silently:

* **Edges are the robot tree, not fully connected.** `robot.get_gnn_edges()` is
  the default; `get_graph_fc_edges` is only used when `obs_specs.fc_graph` is
  set, and `hopper_gpu.yml` does not set it. A port that assumes fully-connected
  will train fine and be a different algorithm.
* **`body_index = int(body.name, base=max_nchild+1)`** (`hopper.py:250-256`).
  Body names are numerals in base `max_nchild+1`, so the index encodes the
  body's *path from the root*. This is what lets per-body-type weights be shared
  across morphologies: "second child of the first child of the root" has the
  same parameters in every body plan that has such a node. `max_index=256` caps
  it, so deep or wide trees silently collide.
* **`design_cur_params` is the projected parameter**, not the action, when
  `use_projected_params` (default true, and set for hopper). The observation
  reports what the XML actually got after clipping to `lb`/`ub`, not what the
  policy asked for.

## 3. The policy: three towers, one per stage

`transform2act_policy.py`. Each stage has its own, independent:

```
RunningNorm -> [pre_mlp] -> GNNSimple -> [mlp] -> JSMLP(indexed by body_index)
```

and they share nothing but the observation layout. Input dims differ:

| tower | input | output per node |
|---|---|---|
| `skel_*` | `attr_fixed + attr_design` | `skel_action_dim` logits |
| `attr_*` | `attr_fixed + attr_design` | `attr_design_dim` Gaussian mean |
| `control_*` | `attr_fixed + sim_obs + attr_design` | `control_action_dim` Gaussian mean |

Only the control tower sees `sim_obs` — the design towers are blind to the
physical state, which makes sense (there is no meaningful state yet) and matters
for a port that would otherwise feed them the full vector.

`forward` splits a mixed batch by stage, runs only the towers that have rows,
and reassembles into one `[total_nodes, control_dim + attr_dim + 1]` action
tensor with the unused slices left at zero (`policy:207-224`). The env asserts
those zeros (`hopper.py:146`).

**`GNNSimple`** (`models/gnn.py`) is `torch_geometric.nn.GraphConv` layers with
optional residual and `cat_input`. For hopper: `hdims [64,64,64]`, `aggr: add`,
`bias: true`, relu, no residual, no cat_input.

**`JSMLP`** (`models/jsmlp.py`) is an MLP whose every layer is an `IndexLinear`:
a `[256, out, in]` weight bank selected per node by `body_index`. Its forward is
**a Python loop over `ind.unique()`** with one `addmm` per distinct index.

## 4. The critic

`transform2act_critic.py`: one tower (`RunningNorm -> GNN -> MLP [512,256] ->
Linear(…,1)`), and the value for a graph is **read off its first node**
(`critic.py:78-81`), not pooled. The one-hot stage flag is concatenated to the
observation (`design_flag_in_state: true`, `onehot_design_flag: true`), so the
critic knows which stage it is valuing — necessary, since design steps pay 0 and
execution steps do not.

## 5. The per-graph log-prob, and why the whole codebase is float64

`policy.get_log_prob` needs a per-graph sum of per-node log-probs. It does this
by `cumsum` over the entire batch, indexing at each graph boundary, and
**differencing consecutive boundaries** (`policy:238-241`, and again for each of
the three stages).

That is catastrophic cancellation by construction: on a 50,000-step batch the
running cumsum reaches ~1e5 while the wanted quantity is ~1e1. Measured in
`gnn_playground.py`:

| reduction | max abs error vs fp64 segment sum |
|---|---|
| fp64 cumsum-and-difference (theirs) | 1.7e-10 |
| **fp32 cumsum-and-difference** | **1.3e-1  (0.18% of a typical log-prob)** |
| fp32 segment sum (`index_add`) | 2.0e-5 |

**So float64 is load-bearing only for their choice of reduction.** Swap
`cumsum`-and-difference for `index_add` and fp32 is accurate to 2e-5 — five
orders of magnitude better than fp32 with their reduction, and good enough for a
PPO ratio. This is the single cheapest correctness-preserving change in the
port, and it halves memory and bandwidth for everything else.

A 0.18% error on a log-prob is not cosmetic: PPO exponentiates it into the
importance ratio and clips at 1±0.2, so it perturbs exactly the quantity the
clip is meant to bound.

## 6. Where the wall-clock actually goes

From the live `hopper_gpu` run (32 workers, this pod), per epoch:

| phase | seconds | share |
|---|---|---|
| `T_sample` | ~100 | 49% |
| `T_update` | ~52 | 26% |
| `T_eval` | ~51 | 25% |

`T_eval` is also rollouts (`agent.optimize_policy` calls `self.sample(...,
mean_action=True)`), so **74% of wall-clock is environment rollout** and 26% is
the PPO update. A batched-physics port attacks the 74%; the update is already on
the GPU and is not where the time is. Amdahl's ceiling for a physics-only port
is therefore ~3.8x, and that is the honest number to plan against — not the
raw env-step ratio, which is the mistake made in D2 (see
`COMPETEVO_PORT_MAP.md`; the measured end-to-end speedup there was ~3x against a
~19x raw ratio).

## 7. The one genuinely hard problem: topology changes mid-episode

Batched GPU physics compiles one model for all worlds. Transform2Act changes the
model six times per episode, and after the skeleton stage different worlds have
**different numbers of bodies**. Three approaches, ranked:

### A. Superset model with an activation mask (recommended)

Compile the maximal tree once — for hopper, `max_body_depth 4` and
`max_nchild 3` bound it at 1+3+9+27 = 40 bodies — and let each world activate a
subtree. Deactivation means: zero the actuator gear, freeze the joint, and
shrink the geom to epsilon *inside its parent* so it cannot collide or add
inertia.

Pros: one compiled model, fixed `[W, N, F]` tensors, the adjacency becomes a
per-world mask, and the whole ragged-batching problem disappears.

Cons, and they are the dangerous kind: a "disabled" body that still has mass,
contact geometry, or a live degree of freedom changes the physics of a body plan
that should not contain it. **This is precisely the failure mode this project
has shipped twice — an env that is numerically fine and physically wrong.** The
gate must be: build morphology M as a subset of the superset, build M as its own
compiled model, and assert the trajectories agree to machine epsilon over
several hundred steps from a shared initial state. Not the observation — the
trajectory.

### B. Group worlds by topology after the design stages

Skeleton actions are discrete and heavily constrained, so many worlds land on
the same tree. Compile one model per distinct topology per generation and run
each group as its own batch.

Pros: exact by construction, no masking risk. Cons: the number of distinct
topologies is unbounded in principle, compile cost lands inside the training
loop, and group sizes are ragged — the tail groups waste the GPU.

Worth measuring before dismissing: **how many distinct topologies does a
50,000-step batch actually contain?** If it is tens, B is simpler and safer than
A. That measurement is cheap (log `cur_xml_str` hashes for one epoch of the live
run) and should be taken before committing to A.

### C. Keep the design stages on CPU

They cost no physics. Run all six design steps in the existing single-threaded
path, then hand fixed topologies to batched GPU physics for the execution stage.

This is not really an alternative — it is a component of both A and B, and it is
free. The design stages are 6 steps against up to 1000 execution steps.

## 8. What we already have

The attribute-transform half of this problem is **solved in our own tree**.
`rower_soccer/competevo_port/design.py`'s `DesignWriter` writes per-world model
fields (geom sizes, masses, gears) into a batched backend for a fixed topology,
with `mj_setConst` for the derived quantities, and it has a parity gate. That is
exactly the attribute stage. The genuinely new work is topology.

## 9. Port order (proposed for 3d)

1. **`index_add` instead of cumsum-and-difference, in fp32.** Measured above,
   independent of everything else, and it can be validated against their fp64
   run directly.
2. **Dense masked GraphConv.** Verified in 3c to match PyG to **0.0 max error in
   fp32** and 9e-16 in fp64, so this is a free representation change.
3. **Measure the distinct-topology count** (section 7B) before choosing A or B.
4. **Execution stage on batched physics**, fixed topology, one body plan — the
   narrowest possible thing that can be gated against their trajectories.
5. **Topology strategy** per step 3, with the trajectory-equivalence gate.
6. **Paper-number validation (3e)** — their Table 1 final returns.

## 10. Things measured that turned out NOT to be problems

Recorded because they were on the suspect list and the port should not spend
time on them:

* **`IndexLinear`'s Python loop is not obviously the bottleneck.** Batching it
  into a gather + `baddbmm` is correct (max err 7.8e-5) but the speedup depends
  entirely on the regime, and one of them is a *slowdown*:

  | nodes | body types | loop | batched | speedup |
  |---|---|---|---|---|
  | 5,000 | 8 | 5.0 ms | 3.4 ms | 1.5x |
  | 5,000 | 40 | 16.9 ms | 2.0 ms | 8.4x |
  | 5,000 | 256 | 109.8 ms | 4.4 ms | **24.9x** |
  | 50,000 | 8 | 3.3 ms | 19.5 ms | **0.17x** |
  | 50,000 | 40 | 25.9 ms | 19.3 ms | 1.3x |
  | 50,000 | 256 | 139.3 ms | 22.7 ms | 6.1x |

  The loop costs one kernel per *distinct* index; the batched form pays a
  gathered `[n, out, in]` weight tensor (3.3 GB at 50k nodes and 128x128). At
  many nodes and few body types the loop wins outright. **Do not port this
  blind** — measure the actual distinct-`body_index` count in a real batch
  first, and if it is large, chunk the batched form rather than materialising
  the full gather.
* **Variable-size graph batching is not hard.** Their `batch_data`
  (`policy:113-126`) concatenates node features and offsets edge indices — the
  standard trick, and it already runs on GPU unchanged. Under superset-model
  batching (7A) it stops being needed at all.

## 11. Measured: the topology census settles both open decisions

Sections 7 and 10 both end in "measure this before choosing". `topology_census.py`
does, on the live run's own checkpoints. Only the design stages are run (they
involve no physics), 200 designs per configuration.

| policy | distinct topologies / 200 designs | distinct `body_index` values |
|---|---|---|
| untrained (worst case) | **21** | **12** |
| epoch 100 | 2 (199 of one) | 7 |
| epoch 200 | 2 (196 of one) | 8 |
| epoch 100/200, mean-action | 1 | 7 |

### Decision 1: group by topology (7B), not superset masking (7A)

A 50,000-step epoch contains roughly 50 designs at 1,000-step episodes, and even
a 1,000-world GPU batch samples from a distribution that is producing **21
distinct topologies at its most diverse and 2 once trained.** Compiling one model
per distinct topology per generation is therefore a handful of compiles, not
thousands.

That removes the whole reason to consider the superset-with-masking approach,
whose failure mode — a deactivated body that still carries mass or contact
geometry — is exactly the class of bug this project has shipped twice. **Take
the exact approach; it is also the cheap one.** Section 7A stands as a fallback
if a future task turns out to have a genuinely wide topology distribution, but
hopper does not, and nothing should be built for it speculatively.

### Decision 2: keep `IndexLinear`'s loop

Section 10 found the loop beats a batched gather when there are few distinct
indices and loses badly when there are many, and left the regime unmeasured.
It is **7-12 distinct indices**, against a `max_index` of 256. At 50,000 nodes
and 8 body types the sweep measured the loop at 3.3 ms against the batched form
at 19.5 ms — the batched version would be a **6x slowdown**. Do not port it.

### An observation about the method, not the port

By epoch 100, **199 of 200 sampled designs share one skeleton**, and every
mean-action design does. The skeleton stage has effectively stopped exploring
while `exec_R_eps` continues to climb from 1,376 (epoch 100) to 3,757 (epoch
404). Whatever is producing that improvement over the back three quarters of the
run, it is the attribute stage and the controller, not the body plan.

This matters for D3's stated motivation — wanting a machine that finds genuinely
different bodies for different roles. On this task, at these settings, the
skeleton search converges early and then stays put. Worth knowing before
designing 3f around the assumption that it keeps searching. It is one task and
one seed, so it is an observation, not a claim about the method.

## 12. Built and gated: the dense policy (3d step 1)

`rower_soccer/t2a_port/dense_policy.py` is their policy on dense `[G, N, F]`
tensors with one shared adjacency per group -- the representation section 11's
topology-grouping decision implies. Their epoch-400 checkpoint loads with
`strict=True` (65 tensors, 0 missing, 0 unexpected), and
`gate_dense_policy.py` compares it against theirs on real observations pulled
from their env: **0.00e+00 max abs difference at all three stages**, 66 states.

The gate corrected two things written above.

* **Section 2 warned that edge direction is easy to get wrong. It is not a
  hazard at all.** `robot.get_gnn_edges()` emits both directions of every tree
  edge, so the adjacency is symmetric and a port that transposed it would be
  numerically identical. This was found by writing a negative control ("the
  answer must change under transposition") that could never have failed, and
  noticing that it did not. The check now asserts the symmetry instead.
* **A negative control on a discrete head has to read the head, not the
  action.** The skeleton stage ends in an argmax, which absorbs any perturbation
  that leaves the winning logit winning: the edge-dropping control read 42/60
  while measuring how decisive the logits were, not whether the perturbation
  reached them. Reading the pre-argmax head output gives 60/60.

Deliberately not ported: the `cumsum`-and-difference reduction of section 5. In
the dense form each graph's nodes have their own axis, so a per-graph sum is
`.sum(1)` and the cancellation that forces float64 never arises.

## 13. Ran the physics bridge gate: the model ports EXACTLY, the trajectory does not

Section 9 step 4 is "execution stage on batched physics". Its unstated
precondition is that their MuJoCo and ours agree on the same morphology, and
`physics_bridge_gate.py` existed to check that but had never been run. It has
now, both halves: emit from `hopper_gpu_s2`'s epoch-1000 checkpoint under
mujoco-py 2.1.2.14 / mujoco210, check under `mujoco` 3.12.0.

### The model is exact -- but ONLY with `--legacy-inertial`

| | default | `--legacy-mass` | `--legacy-inertial` |
|---|---|---|---|
| `body_mass` | 1.582e-01 (**1.66% of range**) | 3.5e-10 | **4.9e-12** |
| `body_inertia` | 4.780e-02 (**5.18% of range**) | 3.2e-02 (3.47%) | **3.5e-13** |
| traj max over 1,000 steps | 6.31e-01 (at 300) | 6.03e-01 (at 300) | **3.55e-01** |

`body_pos`, `geom_pos`, `geom_size`, `geom_ipos` and `dof_damping` are exact
(<= 5.6e-17) in every case, so the global->local conversion itself is not in
question.

**`--legacy-mass` alone is not enough and is the trap.** It fixes the mass and
leaves the inertia 3.5% wrong, which reads as a pass on the field most people
would check. `--legacy-inertial` writes an explicit `<inertial>` carrying both
and subsumes it. **The batched port must emit legacy inertials.** Without them
it simulates a robot 1.7% off in mass and 5.2% off in inertia -- a different
robot, whose numbers are not comparable to their published curves, failing in
exactly the shape of "the port trains to a lower reward".

*(Reminder from `xml_global_to_local`: the explicit `<inertial>` is silently
ignored unless `inertiafromgeom="auto"` is also set.)*

### Every solver option already matches, and the remaining gap is not settable

Dumped from both stacks and compared: `timestep`, `integrator`, `solver`,
`iterations`, `tolerance`, `jacobian`, `cone`, `impratio`, `noslip_*`,
`density`, `viscosity`, `disableflags`, `enableflags`, `gravity`, and the
per-element `geom_solref`, `geom_solimp`, `geom_friction`, `geom_margin`,
`dof_armature`, `actuator_gear`, `actuator_ctrlrange` -- **all identical**, the
array fields to 0.0e+00.

One field differs: `mpr_iterations` 50 vs modern `ccd_iterations` 35, a changed
default. Setting ours to 50 leaves the trajectory **bit-identical** (max
3.550e-01 either way), because capsule-plane and capsule-capsule use analytic
primitives and never enter MPR. So there is no knob left to turn: the residual
is MuJoCo 2.1 vs 3.12 integrator/solver code.

### What the trajectory divergence actually looks like

Open-loop replay of a fixed pseudo-random control sequence, `legacy_inertial`
on, 1,000 steps. Per coordinate, against that coordinate's own range of motion
in the reference:

| coordinate | max abs diff | ref range | % of range |
|---|---|---|---|
| `1_joint` | 1.184e-01 | 7.03e-01 | **16.8%** |
| `11_joint` | 1.195e-01 | 9.30e-01 | 12.9% |
| `211_joint` | 3.550e-01 | 2.92e+00 | 12.2% |
| `31_joint` | 2.942e-01 | 3.62e+00 | 8.1% |
| `21_joint` | 3.393e-01 | 4.32e+00 | 7.9% |
| `rootx` | 6.458e-02 | 1.10e+00 | 5.9% |
| `rootz` | 1.732e-02 | 9.58e-01 | **1.8%** |

It crosses 1e-3 at step 10 and 1e-2 at step 68, then grows and **stays bounded**
-- it does not blow up. The divergence is concentrated in the *joints*, not the
root: the two builds put the hopper in slightly different poses while it stays
in roughly the same place.

Note this is the **worst case**. Open-loop replay has nothing correcting the
drift; a closed-loop policy pushes back on it. Measuring the closed-loop version
needs the observation pipeline, i.e. step 4 itself, so it is deferred to the
step-4 gate rather than guessed at here.

### Consequence for M2's validation

**The port cannot be gated on trajectories, and should not be.** Per-step
agreement was already ruled out in `D3_HANDOFF.md`; this puts a number on it and
shows it is not a bug to be fixed. So:

* **Gate the model, exactly.** Field-by-field against a compile, which is what
  `xml_to_fields.py --gate` already does and what this gate now does for the
  conversion. That part *is* reproducible to 1e-12 and any regression is a real
  defect.
* **Gate the port on distributions.** Learning curve and final-return
  distribution against their run, not episode traces. Combined with section 11's
  M1 finding that seed alone spreads final return ~30%, that means **several
  seeds, comparing distributions** -- a single matched run proves nothing in
  either direction.
* **Do not chase the residual.** Every settable option is already matched.
