# D3 M3 E5 — 2v2 run-to-goal with evolving morphologies: plan and inventory

*Written 2026-09-05. **Nothing has been launched.** No shared code has been
changed. This is a plan for review.*

*Machine state at the time of writing, measured (`nvidia-smi`, `ps -eo pid`,
`/sys/fs/cgroup/cpu/cpu.cfs_quota_us`): one E4R arm up — PID 667512 plus ten
`--num-threads 10` forks — running `--cfg rtg_e4r_smoke --max-epoch 12`, i.e.
the 12-epoch smoke and not the 400-epoch production wave; GPU 5,309 MiB of
20,475 in use; CPU quota **10.2 CPUs** (1020000/100000), load average 12.8.
`runs/d3_e4r_ring/` contains only an empty `renders/`. Reported, not acted on.*

---

## 0. Before anything else: the premise in the ladder is out of date

`PLAN_D3_M3.md` states E5's question as:

> D2 found the back agent is a **spectator** under a first-crossing rule. With
> evolvable bodies, does the back agent evolve a *different* body suited to
> interference rather than racing?

**The first sentence is a stale reading of D2 and it has to be corrected before
the rung is designed, because the correction changes what E5 is for.**

What D2 actually measured, in order (`DESIGN_2V2.md`):

| when | back pair's share of all crossings | source |
|---|---|---|
| transplanted 1v1 pair, no 2v2 training | **0.000-0.005** | §8 step 1, quoted at `role_metrics.py:24-27` |
| native 2v2 training, epochs 60-80 | **0.0%** | §11 "The back agent is still a spectator, and this is now a training result" |
| native 2v2 training, epoch 200, one seed | **35.8%** | §12 |
| **six independent runs, 3 seeds × 2 `goal_credit` arms** | **34.5 / 70.8 / 63.1 / 49.7 / 52.4 / 59.0 %, mean 54.9%** | §14 "What is now solid: the back agent is a real player" |

D2's own summary of §14: *"the second player is not decorative, the sparse team
reward does find a use for it, and neither the y-gated goal nor a shorter
`back_x` is needed."* §12 also records that §11d's "stable stalemate" reading
was withdrawn — *"a flat stretch in a self-play curve is not evidence of
convergence"* — which is the same error, one rung earlier, in the same
document.

**So "the back agent is a spectator" is a statement about a transplanted 1v1
policy and about the first 80 epochs of one run. It does not survive
replication.** Carrying it into E5 would have set the rung up to re-answer a
question D2 already closed.

### What is actually still open, and it is sharper

D2's §15 answers the *attribute-level* version of the morphology question and
closes it:

| front-vs-back genome, largest per-dimension SMD, 256 worlds | s42 | s43 | s44 | mean |
|---|---|---|---|---|
| role hidden (2f) | 0.112 | 0.101 | 0.118 | **0.110** |
| **role visible (`--role-in-design`, 2g)** | **0.873** | **0.716** | **0.909** | **0.833** |

7.6×, no overlap, and free (goal rate 80.2% vs 84.6%, indistinguishable at a
50-point within-arm range). And §15 states its own limit exactly:

> The specialisation is within a fixed skeleton — same four legs, same joints —
> so **Transform2Act is still the only route to *topological* difference.**

**That sentence is E5's actual mandate.** The open question is not "does the
back agent do anything" (answered: yes, 54.9% of crossings) and not "can bodies
differ by role" (answered: yes, at 0.833 SMD, once the design head can see the
role). It is:

> **When the design search can add and remove limbs, does role-conditioning
> produce two teammates with different *topologies* — and are those topologies
> functionally different, one better at racing and one better at
> interference — or does role-conditioning only rescale the same skeleton?**

D2 cannot ask that: its design space is a 20-dim scale vector applied to a
fixed compiled model (`competevo_port/design.py:1-9` — *"The topology never
changes, so ONE compiled model serves every world"*). Transform2Act can.

### And one measured fact says the question is hard, not easy

`DESIGN_2V2.md` §4, per-episode dense returns over 566 completed episodes:

| agent | mean dense return | sd(R_i) | corr(R_i, R_mate) |
|---|---:|---:|---:|
| 0 (A1, front) | 419.0 | 110.6 | 0.030 |
| 1 (B1, front) | 443.4 | 94.3 | 0.101 |
| **2 (A2, back)** | **484.0** | 137.2 | 0.030 |
| **3 (B2, back)** | **489.4** | 133.1 | 0.101 |

> *"The **back agent is not a freeloader in dense terms — it is the better paid
> one** (484/489 against 419/443). It travels further before the episode ends
> and is knocked over less. That inverts the naive lazy-agent story here: the
> danger is not that the back ant does nothing, it is that running forward
> **already pays it well**, so nothing in the dense reward pushes it toward a
> defensive or interfering role."*

**This is the single most important constraint on E5 and it is easy to miss.**
The dense term — `forward − 0.5Σa² − contact + 1.0`, weighted `alpha` against
the sparse ±1000 — pays the back agent *more* for racing than it pays the
front, because the back agent has 8 m of runway and gets knocked over less.
An interference-specialised body has to be found **against** the dense
gradient, on the strength of the sparse team term alone, and at D2's `alpha =
0.90` the sparse term is only ~24% of the dense one in magnitude.

Two consequences carried into §4:

* the rung's positive outcome is *a priori* less likely than the ladder's
  phrasing suggests, and the pre-registration must make "no specialisation"
  a clean, reportable answer rather than a disappointment;
* if E5 returns Outcome 3 (no specialisation) with a healthy task, **the first
  follow-up is the reward, not the architecture** — the dense term is paying
  for exactly the behaviour the question hopes to see displaced. That
  follow-up is named here so it is not invented after seeing the result.

---

## 1. Inventory: what exists, what is welded, what must be built

Read from the code, not from the docs. Every row was opened.

### 1.1 D2's team machinery — genuinely policy-agnostic

| piece | file:line | why it is portable |
|---|---|---|
| **N-agent contact bitmask** | `competevo_port/team_scene.py:20-31, 77-88` | `contype = 1<<i`, `conaffinity = ALL ^ (1<<i)`. Pure MJCF arithmetic. `tests/test_team2v2.py` asserts the colliding-pair *set* matches CompetEvo's at n=2 and that the naive formula is broken at n=4. **T2A's `rtg_scene.py:139-142` uses exactly the broken 2-agent trick** (`contype=1/conaffinity=0` and `0/1`), so this is a fix T2A needs and D2 already derived. |
| **Team spawn geometry** | `team_scene.py:57-69` `team_init_pose` | `pos = [(-1,0),(+1,0),(-4,0),(+4,0)]`, `euler = [0,180,0,180]`, agent order (A1,B1,A2,B2) chosen so *"a 2v2 scene truncated to its first two agents is the validated 1v1 scene"*. Data, not code. |
| **Role-symmetric other-ordering** | `team_scene.py:208-227, 246-254` | Each agent's others are ordered `[teammate, opp_near, opp_far]`, so obs slot 0 is always "my teammate". A concept, three lines. |
| **The team rules** | `team_env.py:41-60, 183-231` | `win_rule="team_first"`, `goal_credit ∈ {team, scorer, split}`, `down_rule ∈ {any, ignore, frozen, recover, team_down}`, wipe-out payout, the `alive` mask that stops a corpse earning +1/step. These are **rules**, and the rule text is what transfers. |
| **Opponent ring over whole past teams** | `selfplay.py:147-234` (`OpponentRing`), `train_team_selfplay.py:17-21` | `sample_epoch` is `start = max(1, floor(delta*epoch)); randint(start, epoch)` **high-exclusive**. Pure arithmetic. **T2A already has an equivalent** in `t2a_port/e4r_ring.py:107-124`; what transfers from D2 is the *keying decision*: one ring entry per **team**, one slot per **world**, so a world plays a whole past team, never a mix (`DESIGN_2V2.md` §6, backed by PSRO's joint-policy-correlation 34.2%/71.7% and Hanabi cross-play 23.97 → 2.52). |
| **Credit assignment** | `team_env.py:191-206` + `selfplay.py:534-540` | There is **no separate credit-assignment module**. "Team credit" is two facts: the env pays `parse = ±GOAL_REWARD` to both members of the scoring team, and GAE runs per `(world, lane)` on `reward = parse + dense` with each agent's own `dense`. `DESIGN_2V2.md` §4 calls this *"individual dense + shared sparse"* and grounds it in Kuba et al. (the `(n−1)` variance factor is 1 at n=2), MAPPO/IPPO over COMA, and Liu et al.'s 2v2 soccer. **This is 20 lines of reward arithmetic and one GAE axis, and it is the single most reusable thing in the inventory.** |
| **Role metrics** | `role_metrics.py` (crossing split, teammate CPD, topple attribution), `probe_2v2.py:412-505` (`probe_roles`) | Metric *definitions*, including the correction that a fixed absolute jitter over-weights the role one-hot and the CPD table must be std-scaled (`DESIGN_2V2.md` §11 "This table replaces a wrong one"). And the warning that decides every "who did the work" number: under `team_first`, `winner = mine & one_team` marks **both** members, so the per-agent split is 50/50 by construction — *"Every 'who does the work' number below is built on `reached`, never on `winner`"* (`role_metrics.py:14-19`). |

### 1.2 D2's team machinery — welded to D2's policy or D2's backend

| piece | file:line | what it is welded to |
|---|---|---|
| **`CoEvoPPO`, the team-lane trainer** | `selfplay.py:271-590` | Generalised from two agents to two **sides** in place (`:300-317`), `L = len(team_lanes[0])`. But it drives a **batched torch env over `self.n` worlds** with rectangular `[n, A, obs_dim]` / `[n, A, act_dim]` tensors (`collect`, `:497-560`) and fills `DevSelfPlayPPO` ring buffers. Transform2Act's observation is a **ragged per-graph python list** `[obs [N,F], edges [2,E], stage flag, num_nodes, body_ind]` pushed one sample at a time (`khrylib/rl/agents/agent.py:100-101`). There is no rectangular tensor to fill. **Not portable.** |
| **`TeamActorCritic`** | `team_policy.py:88-172` | D2's dense MLP actor-critic, `expand_obs` gathering against a registered permutation buffer, `role_in_design` widening `scale_mlp`. Entirely D2's architecture. |
| **`SlotTeamActorCritic` (2h "Option A")** | `slot_policy.py:87-270` | One `TeamActorCritic` per (side, slot), with per-slot column gathers, padded action layout, and a flat masked dispatch for the PPO update. Architecture-specific. |
| **`TeamRunToGoalDevEnv`** | `team_env.py:82-307` | Subclasses `RunToGoalDevEnv`, which is the batched-torch port with `CompeteWarpDevBackend` / `CompeteCpuDevBackend` and a **fixed compiled model** whose design is a per-world field write (`design.py:1-9`). It physically cannot change topology. |
| **`StackedDevActors` fast opponent path** | `dev_ppo.StackedDevActors`, disabled at `selfplay.py:369-393` | Already falls back to the per-slot path for both `role_in_design` and per-slot policies. Irrelevant to T2A. |

Two further facts from D2 that E5 inherits and should not re-derive:

* **Contact budget at four agents, measured**: max **12** contacts per world at
  4 agents against 7 at 2, so a 2v2 scene is nowhere near `nconmax`/`njmax`
  (`DESIGN_2V2.md` §10). T2A compiles with `mujoco_py` rather than
  `mujoco_warp`, so this is a reassurance about the physics, not about the
  solver limits, and D2's own §10 records that **four-agent solver fidelity was
  never gated** — there is no reference 4-agent scene of CompetEvo's to compare
  against.
* **Cross-play was never measured.** D2's §6 whole-team-sampling recommendation
  rests on PSRO's 34.2%/71.7% joint-policy-correlation loss and Hanabi's
  23.97 → 2.52, both from other domains. §10 lists "no cross-play measurement"
  as an open item, and it is directly relevant to E5's ring: E5 adopts
  whole-team sampling on the same borrowed evidence.

**Summary of 1.1/1.2: what D2 contributes to E5 is a rulebook and a set of
calibrated metrics, not a trainer.** The trainer, the env and the policy are
all the wrong stack.

### 1.3 The T2A side (E3.1 / E4B) — what exists

| piece | file:line | state |
|---|---|---|
| Design+control on an adversarial task | `rtg_e31_s{1,2,3}.yml`, `train_e3_gnn.py` | **Works.** goal 1.00 at 4.89 m/s (s2) and 3.72 m/s (s1) against the frozen ant's 1.50 m/s. 2 of 3 seeds; the failure was the controller, not the body (`D3_E31_FIX.md`). |
| `control_log_std = -1.5` | `rtg_e31_s1.yml:58`, `rtg_e4r_s1.yml:57` | **Verified in the cfg that governs the code path.** E3's `rtg_e3_s1.yml:72` still carries `0`. Any E5 cfg inherits −1.5 or it deletes its actuators, exactly as E3 did on 3 of 3 seeds. |
| Actuator floor `min_motors: 4` | only `rtg_e31f_s1.yml:112`; `ant.py:63-81` implements it, default 0 = off | Available, and **uninformative at n = 1** (`D3_E31_FIX.md`). E5 should not adopt it without a reason. |
| Policy-driven opponent with its own evolved body | `Transform2Act/design_opt/envs/run_to_goal_sp.py` | **Works.** `opp_*`-prefixed sibling in a merged scene, `opp_control` maps per-body actions onto `opp_*` motor slots by name (`:263-279`), and `do_simulation` (`:290-309`) adds the opponent's torque **downstream of the control-cost billing** — because *"dense control cost is precisely the term that deleted every actuator in E3."* |
| Opponent ring, T2A flavour | `t2a_port/e4r_ring.py:79-124` | Archives body XML + merged scene + live `Robot` + live policy + pickled state dict; samples `Uniform[floor(delta·epoch), epoch−1]`. |
| π-z slot rotation | `run_to_goal_sp.py:54-57, 150-204` | Slot 1 is slot 0 rotated π about z; the free joint's `qvel[3:6]` is body-local and therefore **unchanged**, measured not derived. |
| Ragged-graph batching | `transform2act_policy.py:114-127` | **Different topologies in one batch already work** — block-diagonal `edge_index` offsets, `num_nodes_cum` boundaries, per-row `body_ind` gather in `IndexLinear`. E3.1 ran 12- to 18-body variants in one buffer. |

### 1.4 What must be built for E5 — nothing here exists today

1. **A 4-body merged scene in T2A's MJCF dialect.** `rtg_scene.py` hardcodes two agents in five places: one `OPP_PREFIX` (`:59`), two-entry `INIT_POS`/`INIT_EULER` (`:61-62`), two goal geoms (`:67`), one appended opponent body (`:112-143`), and the binary collision mask (`:139-142`). The contract *"the first `<body>` under `<worldbody>` must be our ant's root"* (`:105-107`) must become "the first two are ours".
2. **Per-agent state addressing that does not assume the opponent is the tail.** `run_to_goal.py:110-124`: `nq = m.nq - qs` takes everything after the opponent's first joint. With three other bodies that silently returns a 3-agent slice. `run_to_goal_sp.py:178-185` already carries the warning: *"would be −3 in any future scene with a third body."*
3. **Task columns 3 → 7.** `run_to_goal.py:163-171` tiles `(opp_dx, opp_dy, goal_dx)` onto every node row. 2v2 needs `(mate_dx, mate_dy, opp_near_dx, opp_near_dy, opp_far_dx, opp_far_dy, goal_dx)` in D2's role-symmetric order. This changes `sim_obs_dim` (`ant.py:47`) → `state_dim` (`transform2act_agent.py:105`) → **every network input width, so every existing E3/E3.1/E4 checkpoint stops loading.**
4. **The team win rule inside the T2A env.** `run_to_goal.py:220-243` is `n_reached = int(reached) + int(opp_reached)`, `parse = ±1000` iff `== 1` — the exact rule `team_env.py:41-53` documents as not surviving four agents (two teammates crossing together is a win, not a draw).
5. **A two-slot rollout that puts both teammates' transitions in one PPO buffer against one shared policy.** *This is the largest single build and it does not exist in either stack.* `khrylib/rl/agents/agent.py:36-83` is a strictly single-agent loop: one `env.reset()`, one `select_action`, one `env.step(action)`, one `push_memory`. E4R's shared-weight design is one agent playing **both sides at different times**; E4's two-lineage design is two agents that **never share a gradient**. Neither is "one head, two role-tagged agents, both contributing to the same buffer in the same episode" — which is the only configuration in which a role channel carries information at all.
6. **A team-keyed ring.** `e4r_ring.py:88-101` archives one body, one `Robot`, one policy per member. A team member is two bodies + one shared policy + a merged scene containing both.
7. **The role channel** (see §3).

---

## 2. The integration problem, stated honestly

### 2.1 Where the two stacks meet

They meet at exactly one place and it is not where the inventory suggests.

D2's team machinery is not a library that can be handed a different policy. It
is a **batched GPU environment** (`RunToGoalDevEnv` over `self.n` worlds, one
compiled `MjModel` shared by every world, design applied as a per-world field
write) plus a PPO trainer whose every buffer is `[T, N, A, D]`. Transform2Act
is a **single-env, per-episode-recompile, ragged-graph** stack: `reset_robot`
reparses the `Robot` and calls `reload_sim_model`, then five skeleton steps and
one attribute step each export XML and recompile again — **7 `mujoco_py`
compiles per episode** (`ant.py:83-118`, `mujoco_env_gym.py:95-107`).

You cannot batch a topology that changes mid-episode into a shared compiled
model. That is not an engineering gap; it is the reason `design.py` exists and
the reason it says *"That cannot exist on a GPU."*

**So the meeting point is the rulebook, not the runtime.** E5 is
"reimplement D2's four rules — 4-agent contact mask, `team_first` win rule,
individual-dense-plus-shared-sparse credit, team-keyed ring — inside T2A's
`run_to_goal_sp.py` lineage", not "make T2A's policy a drop-in for
`CoEvoPPO`".

### 2.2 The hard part

Not the scene. Not the rules. **The hard part is the shared-policy two-slot
rollout, and the reason is the design stages.**

In T2A an episode is: reset → 5 skeleton steps → 1 attribute step → up to 500
execution steps. With two learners in one episode there are two independent
design phases and each one recompiles the *shared merged scene*. Options, and
each has a cost:

| option | what it means | cost |
|---|---|---|
| **Sequential design** | slot 0 runs its 6 design steps, then slot 1 runs its 6, then execution begins | 12 recompiles of a 4-body scene per episode; the design *stages* are not simultaneous, so the stage flag (`if_use_transform_action`, `hopper.py:197-198`) becomes per-agent and the policy's stage partition (`transform2act_policy.py:129-145`) must be told which agent each graph belongs to |
| **Simultaneous design** | both slots take design actions on the same step | one recompile per step instead of two, but `Robot` owns the whole tree (`swap_opponent`'s docstring, `run_to_goal_sp.py:322-327`) and two `Robot`s over one XML is a change to `khrylib` |
| **Design one, freeze one** | only the front slot evolves; the back slot uses the front's body | destroys the question |

**Recommendation: sequential design, two `Robot`s over two disjoint subtrees of
one merged XML.** It is the only option that needs no `khrylib` change and no
simultaneous-stage semantics, and the recompile arithmetic (§6) says it is
affordable.

The second-hardest part is bookkeeping that has already bitten this project
once: `run_to_goal_sp.py:236-241` records that the stage flag is **2** for
execution, not 0, and `transform2act_policy.py:115` unpacks **five** elements
unconditionally, so `obs_specs.use_body_ind: true` is mandatory and the tuple
layout cannot be extended by appending — `x_i[-3]` is the stage flag by
negative index.

---

## 3. Blockers, checked in the code

### 3a. The design head is blind — and for E5 this is not a handicap, it is the whole experiment

**Confirmed independently, in the file as it stands on disk.**
`Transform2Act/design_opt/models/transform2act_policy.py`, lines 170 (attribute
stage) and 194 (skeleton stage), byte-identical:

```python
obs = torch.cat((obs[:, :self.attr_fixed_dim], obs[:, -self.attr_design_dim:]), dim=-1)
```

What is dropped is the entire middle block `obs[:, attr_fixed_dim :
attr_fixed_dim + sim_obs_dim]` — root z, root quaternion, root linear and
angular velocity, every hinge qpos/qvel, **and** the task columns
`RunToGoalEnv.get_sim_obs` tiles onto every node (`run_to_goal.py:169-171`).
The asymmetry is declared in the constructor, `:23-25`:

```python
self.control_state_dim = attr_fixed_dim + sim_obs_dim + attr_design_dim
self.skel_state_dim    = attr_fixed_dim + attr_design_dim
self.attr_state_dim    = attr_fixed_dim + attr_design_dim
```

and the two `RunningNorm`s (`:33`, `:59`) are sized from it, so the width is
baked into buffer shapes, not only into a slice. On the run-to-goal ant the
design heads see **9 of 25 columns**. The critic (`transform2act_critic.py:57-68`)
is the odd one out: it never slices, so it is already sighted.

There is **no cfg flag** for a sighted design head anywhere. Searched both
trees: `design_obs` matches only the local variable in the four `_get_obs`
methods; `sighted` and `role_in_design` have zero hits in `/workspace/Transform2Act`
and zero in `rower_soccer/t2a_port/`. `D3_E4_PREREQ.md:152` recorded the plan
*"build the sighted head behind a cfg flag, leave it unused"* — **it was never
built.** The port's own dense re-implementation hard-codes the same slice
(`t2a_port/dense_policy.py:301-304`).

#### The consequence, which is stronger than "it would be harder"

Under a **shared** design head with no role input, both teammates start every
episode from the same base body. Therefore `attr_fixed` (the `depth` one-hot,
`ant.py:265-268`) and `attr_design` (`get_attr_design`, `:285-291`) are
**identical for the two slots**, and the design heads' input is identical.

The per-epoch morphology readout is the **mean-action** design
(`e4_selfplay.py:80-91` → `e3_morph.run_design_stages(env, policy, True)`),
which is deterministic. So:

> **On a blind shared design head, the front and back slots' mean-action bodies
> are bit-identical. Front-vs-back distance is exactly `0.000e+00`, by
> construction, not approximately zero.**

This is the same shape of fact as `D3_E4_PREREQ.md`'s measurement (moving the
opponent 4 m changes the design head's input by 0.000e+00) and it should be
**gated, not run**: a 20-minute gate that dumps both slots' mean-action bodies
from an untrained shared policy and asserts they are identical is worth more
than a 30-hour blind training arm, and it costs nothing.

**Therefore the central design decision of this rung is: E5 does not have a
blind arm as its null.** The blind case is analytically zero and provable. What
E5 runs is the **sighted** arm, and the null it is measured against is internal
and same-role (§4.2).

#### How to make it sighted — and the cheapest route is not the policy file

Verified by reading `ant.py:47-48` and `transform2act_policy.py:20-25`:

```python
self.sim_obs_dim    = self.get_sim_obs().shape[-1]        # ant.py:47
self.attr_fixed_dim = self.get_attr_fixed().shape[-1]     # ant.py:48
```

`attr_fixed_dim` is **measured from the getter at construction**, propagates
through `transform2act_agent.py:102-108`, and both `attr_state_dim` and
`skel_state_dim` are built from it. `get_attr_fixed` (`ant.py:261-283`) is a
per-body loop over opt-in pieces keyed off `cfg.obs_specs.attr`.

> **Adding `'role'` to `obs_specs.attr` and a four-line branch to
> `get_attr_fixed` makes the design heads role-sighted with *zero* edits to
> `transform2act_policy.py`.** The leading-`attr_fixed_dim` slice at `:170` and
> `:194` picks the new columns up for free.

That is D2's `--role-in-design` intervention transplanted, and it is smaller
here than it was there. Three caveats:

* it also reaches the **control head and the critic**, which consume the full
  row. D2 wanted that (`DESIGN_2V2.md` §5 recommendation items 1-2, grounded in
  VDN: *"parameter-shared agents cannot specialise unless given an
  identifier"*), so it is correctness, not a confound — but it must be stated,
  because it means E5's control policy is also role-conditioned and the
  behavioural and morphological effects are not separable within one arm;
* **every existing checkpoint stops loading.** `skel_norm`/`attr_norm` buffers
  and `*_gnn.in_fc.weight` all change width. E5 is a fresh baseline regardless,
  because §1.4 item 3 already changes `sim_obs_dim`;
* the role is constant across an agent's nodes and constant over the episode,
  so **it carries information only if both teammates' transitions land in the
  same buffer under the same head** — which is build item 5. The channel and
  the rollout are one decision, not two.

#### The alternative that would look cheaper and is not

Per-slot policies (D2's Option A, `slot_policy.py`) would give specialisation
without any role channel. D2 states the cost precisely (`slot_policy.py:16-24`):

> *"the 2g measurement does not transfer. There is no shared design head left
> to make role-visible, so 'does the design head seeing the role produce
> specialisation' is not a question you can ask of this architecture —
> specialisation is unconditionally available."*

Under per-slot heads a positive result is uninformative: two independent
searches produce different bodies whether or not roles exist. **E5 must use one
shared head with a role channel, or it cannot ask its question.** It also costs
double: two `Transform2ActAgent`s, two buffers, no shared samples.

### 3b. Can the machinery represent two different evolved bodies on one team?

Three separate questions, three different answers.

**(i) D2's per-slot machinery: yes for *configured* heterogeneity, no for
*evolved* heterogeneity.** `gate_hetero.py` gates ant/bug/spider on one team
with per-creature genome tables, ragged index tables and `cap_keep`/`body_keep`/
`act_keep` scatter guards, and `slot_policy.MixedPolicyObsEnv` expands the
padded scene observation so each slot gathers its own columns. That is real and
it works. But the heterogeneity is **chosen at scene-compile time** —
`--creatures ant,ant,spider,spider` (`train_team_selfplay.py:164-169`) — and
each creature's design space is a fixed-width scale vector over a fixed
skeleton. `design.py:1-9`: *"The topology never changes, so ONE compiled model
serves every world."* **D2 cannot discover a new topology, so it cannot
represent an evolved difference in topology.** `DESIGN_2V2.md` §15 says so in
its own words.

**(ii) T2A's policy: yes, and it is already the normal case.** `batch_data`
(`transform2act_policy.py:114-127`) concatenates ragged node blocks along dim 0
and offsets `edge_index` per graph — the standard block-diagonal trick — so a
batch is one big disconnected graph. `num_nodes_cum` recovers boundaries for
the per-graph log-prob cumsum-diff (`:249-251`) and for the critic's
first-node value read (`transform2act_critic.py:80`). `body_ind` concatenates
the same way and `IndexLinear` gathers per row (`jsmlp.py:17-25`). E3.1 ran
12- to 18-body variants in one buffer.

Two constraints, neither of them a topology limit: the `[N, action_dim]` width
must match across graphs (`:242` concatenates; column slices at `:234-238` are
absolute) and `attr_fixed_dim`/`sim_obs_dim`/`attr_design_dim` must match
(`:116` concatenates). **Two teammates of the same spec with different bodies
satisfy both.**

**(iii) T2A's *scene*: no, and this is the real blocker.** `rtg_scene.build`
takes one `opponent_src` and appends one body (`:112-143`); `_opp()` assumes the
opponent is the tail of `qpos`/`qvel` (`run_to_goal.py:110-124`); `opp_control`
knows one `opp_robot` and one prefix (`run_to_goal_sp.py:263-279`);
`e4r_ring.add` builds a 2-body scene (`e4r_ring.py:88`). **The policy can
represent two evolved teammates; the scene cannot hold them.** That is build
items 1, 2 and 6.

One piece of good news, checked: the π-z rotation generalises. `team_init_pose`
puts team A at euler 0 and team B at euler 180 with x-mirrored spawns, so team
B *is* team A rotated π about z — the front/back distinction is a difference in
spawn **x** (∓1 vs ∓4), not in frame. `qmul_zpi` (`run_to_goal_sp.py:54-57`)
therefore still maps a team-B agent into a team-A frame; only the x offset is
new. A four-pose scene does not need a new symmetry, only a new offset.

---

## 4. Proposed experiment design

### 4.1 Shape

* **Task**: 2v2 run-to-goal, 500 control steps, `frame_skip 5` (dt 0.015 s),
  goal at x = ±4, spawns `(∓1, 0)` front and `(∓4, 0)` back per
  `team_init_pose`.
* **Rules, taken from D2 and not re-derived**: `win_rule = team_first`,
  `goal_credit = team`, `down_rule = team_down`, individual dense + shared
  sparse. D2's §13/§14 ablation of `goal_credit` is explicitly **not**
  reproduced — its goal-rate comparison did not survive replication (50.7-point
  within-arm range) and its teammate-CPD effect is p ≈ 0.10 at n = 3.
  `team` is chosen because it is the only credit rule that makes
  blocking-and-walking-past rational for the blocker (`team_env.py:55-60`).
* **Learning**: **one shared Transform2Act policy** drives both slots of the
  learning team; both slots' transitions go into one buffer; the role one-hot
  enters via `obs_specs.attr: ['depth', 'role']`.
* **Opponent**: a whole past **team** drawn from the ring, `delta = 0.0`
  (Bansal's best setting for Ant, and E4B's choice), redrawn per episode. Never
  the current self — the `parse = 0`-at-equilibrium argument from
  `PLAN_D3_E4B_SELFPLAY.md` §3 applies unchanged and is *worse* here, because
  `team_first` also pays nothing when both teams cross on the same step.
* **Control noise**: `control_log_std = -1.5`. Non-negotiable; E3 deleted every
  actuator on 3 of 3 seeds at `0`.
* **Budget**: 400 epochs, `min_batch_size` 100,000 (see §6 for why not 50,000).

### 4.2 The metric, and its calibrated null

**Primary statistic.** At each epoch, dump both slots' **mean-action** bodies
(`e4_selfplay.dump_mean_action_body`, under `e3_morph.rng_guard` so the probe
cannot perturb the run it measures) and compute

```
D_role(e)  = SMD( front body , back body )        within the learning team
```

where SMD is `e0_analyse.compare`'s statistic: mean |Δ genome| over **shared
body names**, standardised by the pooled `sampled_genome_std` at that same
epoch, floored at 1e-3 — the identical definition E4 pre-registered.
Secondary: **Jaccard on body-name sets** (the topology channel D2 could not
have), and `|n_bodies_front − n_bodies_back|`, `|n_motors_front − n_motors_back|`.

**The null has three layers and only one of them costs compute.**

| layer | value | status |
|---|---|---|
| **Blind head** | `D_role ≡ 0.000e+00` exactly | **analytic, gated not run** (§3a) |
| **Same-role internal null** `D_same(e)` | SMD(front of team A, front of team B) and SMD(back of A, back of B), averaged | **free** — same run, same epoch, same policy, same role, differing only by which side of the pitch. This is the exactly-matched control: it isolates "two draws from one design distribution" from "two roles". |
| **E3.1 external scale** | within-lineage 40-epoch drift p95 **0.437**; between-seed late-window pooled SMD **mean 0.904, sd 0.086** (n = 1200 rows, 3 pairs); between-seed Jaccard **0.704** | **measured, already committed** — `docs/t2a/e4_null/e31_comparison_set.json` and `e31_crossseed_null.json`, both marked FINAL with all three seeds at epoch 399 |

The E3.1 file also records why an endpoint is not allowed: the cross-seed null
**rises monotonically**, window means `0-49: 0.173 → 50-99: 0.522 → 100-199:
0.744 → 200-299: 0.878 → 300-399: 0.930`. A fixed threshold on a final-epoch
number measures elapsed training.

### 4.3 Pre-registered outcomes

Fixed before launch. All verdicts are on the **window mean over epochs
200-400**, aggregated before comparing, plus a sign-consistency requirement —
because on this project an endpoint has inverted the conclusion at least three
times (`PLAN_D3_E4B_SELFPLAY.md` §5; `DESIGN_2V2.md` §12 on §11d).

Let `Δ(e) = D_role(e) − D_same(e)`.

**Outcome 1 — ROLE-SPECIALISED TOPOLOGIES.** All three must hold.
1. **Distance**: window-mean `Δ ≥ +0.15` **and** `Δ(e) > 0` in ≥ 80% of epochs
   in the window. `0.15` is carried over from E4 with its basis: 3.1 standard
   errors of the measured pair spread (`sd 0.086` over 3 pairs → SE ≈ 0.048).
2. **Scale**: window-mean `D_role ≥ 0.75` — the p05 of the between-seed null,
   i.e. *"as far apart as two independent searches get"*. Below **0.44** (the
   p95 of within-lineage 40-epoch drift) the two roles are the same body
   drifting. Between 0.44 and 0.75 is **AMBIGUOUS and reported as such**, never
   rounded.
3. **Function** (see 4.4): the two bodies are functionally different in the
   predicted direction.

**Outcome 2 — RESCALING ONLY.** `Δ ≥ 0.15` and `D_role ≥ 0.75` but the
**Jaccard on body-name sets ≥ 0.90** in the window and `|Δn_bodies| < 2`. The
roles differ in attribute space and not in topology. *This would reproduce D2's
§15 result in a stack that could have done better, and it is a real finding: it
would say Transform2Act's topological freedom buys nothing over CompetEvo's
scale vector for this question.*

**Outcome 3 — NO SPECIALISATION.** `|Δ| < 0.15`, or sign consistency < 80%, or
`D_role ≤ 0.44`.

**Outcome 4 — UNTESTABLE** (reported separately, never pooled into 1-3). Any of:
* team goal rate < 0.20 over the window — the task was not learned, so nothing
  differentiates the roles. **This is not a remote risk**: D2's three `team`
  seeds scored 94.2% / **47.9%** / 98.6%, a 50.7-point range, with one seed
  never leaving the stalemate;
* timeout + stalemate rate > 0.50 over the window;
* **back-slot crossing share < 0.05** over the window — if the back agent is a
  spectator *in this stack*, there is no role for a body to specialise for and
  the morphological question is vacuous. D2's replicated reference is **54.9%
  mean, 34.5% minimum over six runs**, so < 0.05 would be a genuine regression
  and worth reporting as one.

**What would falsify the rung's premise outright**: Outcome 3 together with a
healthy task (goal ≥ 0.20, back-crossing ≥ 0.34, i.e. D2's floor). That
combination says role-conditioned *behaviour* emerged and role-conditioned
*morphology* did not, in a design space that can express topological
difference — which contradicts the extrapolation from D2's §15 that motivates
E5 and would send E6 back to configured heterogeneity.

### 4.4 The functional test — "suited to interference rather than racing"

**Distance alone cannot support the claim in the ladder.** Two different bodies
are not two *roles*; the ladder's phrasing is functional and needs a functional
measurement. Reuse E3.1's frozen-body diagnostic machinery (`rtg_e31d_s3body`),
which already exists and already worked — it converted "s3 failed" into "s3's
body is fine, its controller was not."

At epochs 200, 300, 400, for each seed, four cross-evaluations at 50 episodes
each:

| probe | what it measures | prediction if Outcome 1 |
|---|---|---|
| **solo time trial**: front body, front slot, no opponents | racing speed | front faster |
| **solo time trial**: back body, front slot, no opponents | racing speed of the specialist | **back ≥ 25% slower** |
| **obstruction**: back body vs a scripted 0.68 m/s opponent, scored on the opponent's *delay* | interference ability | back delays more |
| **obstruction**: front body, same | control | front delays less |

25% is calibrated on E3.1's measured spread of *working* bodies: 4.89 / 3.72 /
2.58 m/s over three distinct topologies, i.e. the slowest working body is 47%
slower than the fastest. 25% sits inside that range and comfortably above the
frozen-ant reference's own noise. **The null for the functional test is the
same-role pair**: front-of-A vs front-of-B on the same probes. If two same-role
bodies already differ by 25% on the time trial, the threshold moves to the
measured same-role p95 before any verdict is read.

### 4.5 Seeds

**Three is the floor and it may not be enough.** Two independent reasons, both
measured:

* T2A's own controller-draw failure rate on this task is **1 in 3** (E3.1: s3
  at goal 0.00 with a body that scored 0.60 under a fresh controller). *"Any
  future claim about design search on this task needs enough seeds to survive
  one dead controller"* (`D3_E31_FIX.md`).
* D2's 2v2 seed variance is **four times worse than its 1v1 variance** — a
  50.7-point goal-rate range within one arm, against 13 points at 1v1
  (`DESIGN_2V2.md` §14). *"A single 2v2 run tells you almost nothing."*

E3.1's early-restart precedent applies and should be pre-registered: **if a
seed's team goal rate is still 0.00 at epoch 150, restart that seed with a new
controller init and keep the run.** E3.1's s3 was diagnosable at epoch ~140.

---

## 5. Shared-code changes E5 needs — written down, not made

Every item below is in a read-only tree (`design_opt/`, `khrylib/`,
`rower_soccer/t2a_port/`, `rower_soccer/competevo_port/`). **None has been
made.** Ordered by risk to the running E4B arm: items 1-3 touch files E4B
executes and must not be applied while it runs.

| # | file | change | why | risk to E4B |
|---|---|---|---|---|
| 1 | `Transform2Act/design_opt/envs/ant.py:261-283` | add an `if 'role' in self.attr_specs:` branch appending a `role_dim` one-hot per body | makes the design heads role-sighted; `attr_fixed_dim` self-sizes at `:48`, so nothing else changes | **none if gated behind the cfg key** — `obs_specs.attr` is `['depth']` in every current cfg, so the branch is dead code for E4B. Still an edit to a file E4B's forks have already imported; apply at a run boundary. |
| 2 | `t2a_port/rtg_scene.py:59-67, 112-143` | `n_agents` parameter; per-agent prefixes; `team_init_pose`-shaped `INIT_POS`/`INIT_EULER`; **replace the binary collision mask with D2's `contype = 1<<i`, `conaffinity = ALL ^ (1<<i)`** | the 2-agent mask is the exact trick `team_scene.py:8-15` proves is broken at four agents | E4B calls `build(base_src, body_path)` per ring archive. A default-preserving signature is required, plus D2's `tests/test_team2v2.py::test_bitmask_matches_theirs_at_two_agents` re-run against the new mask at n = 2. |
| 3 | `Transform2Act/design_opt/envs/run_to_goal.py:110-124, 163-171, 220-243` | name-resolved per-agent qpos/qvel slices instead of `nq = m.nq - qs`; 7 task columns instead of 3; `team_first` instead of `n_reached == 1` | build items 2, 3, 4 | **high** — E4B runs `run_to_goal_sp`, which subclasses this. The 3→7 column change alters `sim_obs_dim` and breaks every checkpoint and the whole ring. Must be a **subclass or a new module**, not an edit. |
| 4 | `Transform2Act/design_opt/envs/run_to_goal_sp.py` | a 2v2 subclass: three prefixed siblings, per-slot `opp_control`, the x-offset generalisation of `qmul_zpi` | build items 1, 2, 6 | low if additive |
| 5 | `khrylib/rl/agents/agent.py:36-83` **or** a new sampler | a two-slot episode loop pushing both slots' transitions | build item 5, the largest | **do not edit `khrylib`.** `transform2act_agent.py:47-97` already overrides this loop; E5's sampler should be a third override in `t2a_port/`, leaving `khrylib` untouched. |
| 6 | `t2a_port/e4r_ring.py:79-124` | team-keyed members: two bodies + one shared policy + a 4-body merged scene | build item 6 | E4B is using this object right now. **New class, not an edit.** |
| 7 | *(none)* | `transform2act_policy.py` | **no change needed** — verified: `attr_fixed_dim` self-sizes and the leading slice at `:170`/`:194` picks up the role columns | — |

### Two D2 recommendations that were never implemented, and what E5 does with them

* **Spawn randomisation** (`DESIGN_2V2.md` §1 recommendation 1: randomise which
  side the back agent is on, a y-offset in ±1 m, and back x in [3.0, 4.0],
  citing Baker et al.'s emergent-phase count 6 → 4 → 2 as randomisation is
  removed). Never shipped; still listed as an unrun ablation at §14.
  **E5 declines it, deliberately.** With a role one-hot the role must mean
  "player 2", not "the ant at x = −4" — which is D2's own §5 note — but adding
  randomisation *and* the role channel *and* topological search in one rung
  makes a null result uninterpretable. Fixed spawns; noted as the first
  robustness ablation if Outcome 1 fires.
* **Death masking** (`DESIGN_2V2.md` §4, MAPPO Appendix C.3: feed the critic an
  all-zeros state carrying only the dead agent's ID). *"Our `frozen` rule
  creates exactly this situation and the port has no death-masking today."*
  E5 uses `down_rule = team_down`, which creates it too. **E5 declines it and
  records the omission**: T2A's critic is per-graph over the learner's own body
  and has no centralised state to mask, so MAPPO's fix does not transfer
  as written. If a downed teammate turns out to poison the value estimate, that
  is a finding, not a silent bug.

**One caveat, verified against the sampler structure**: `env.ring_chosen`
(`train_e4r_gnn.py:350, 411-412` as of 23:1x today) is appended inside workers,
but only worker 0 shares the parent's memory — `khrylib/rl/agents/agent.py:120`
runs pid 0 inline and forks the rest — so ring provenance logged from the
parent reflects **1 of 10 workers**. Any E5 composition logging inherits this
and must either aggregate over the queue or say so.

> **Line-number volatility.** `t2a_port/train_e4r_gnn.py`, `e4r_ring.py`,
> `e4r_tournament.py` and `Transform2Act/design_opt/envs/run_to_goal_sp.py` were
> all modified today by the agent that owns E4B (`gate_e4b.py` is new as of
> 23:11). Line references into those four files in this document were read
> between 22:50 and 23:15 and should be re-resolved by symbol name before any
> of them is acted on. References into `competevo_port/`, `design_opt/envs/{ant,
> run_to_goal}.py`, `design_opt/models/` and `khrylib/` are against files
> untouched today.

---

## 6. Budget

### 6.1 Measured

**Machine, measured now.** CPU quota **10.2 CPUs** (`cpu.cfs_quota_us
1020000` / `cpu.cfs_period_us 100000`) — not the 48 cores `nproc` reports; GPU
**20,475 MiB**; RAM 251 GB.

**T2A 1v1 epoch time, whole series, all four E3.1 arms** (parsed from
`runs/d3_e31_fix/logs/*.log`, `T_sample + T_update + T_eval` per epoch; medians,
because the distributions are right-skewed by contention):

| arm | n epochs | T_sample | T_update | T_eval | **total median** | p10 | p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rtg_e31_s1` | 400 | 31.1 | 89.5 | 7.2 | **130.0** | 100.3 | 170.2 |
| `rtg_e31_s2` | 400 | 32.6 | 88.2 | 7.7 | **128.1** | 99.8 | 166.7 |
| `rtg_e31_s3` | 400 | 49.1 | 72.8 | 14.3 | **137.7** | 111.9 | 172.2 |
| `rtg_e31f_s1` (floor) | 206 | 55.9 | 104.8 | 16.3 | 175.6 | 131.8 | 227.2 |
| `rtg_e31d_s3body` (frozen body) | 297 | 22.0 | 75.7 | 6.1 | 110.5 | 81.8 | 136.8 |

**`T_update` is 55-70% of the epoch, not sampling.** That matters for §6.2: a
second learner slot doubles the buffer and therefore the dominant term.

**T2A cost of a policy-driven opponent, measured** (`D3_E4_PREREQ.md`):

| | s/epoch |
|---|---:|
| E3.1, scripted opponent, 3 concurrent arms | **141** |
| E4 two-lineage, **one** policy-driven opponent | **266-275** |

so **one extra policy-driven body ≈ +129 s/epoch** at 3-arm contention.
Per control step: physics 0.499 ms, opponent obs assembly 0.814 ms, opponent
obs + forward **2.614 ms** = 5.2× physics.

**Recompile is not the problem, measured**: ~4 ms per design; E3.1 s1's epoch
time went **144 → 141 → 139 s** across thirds while episodes per epoch went
**115 → 157 → 392 → 544**. Flat, in fact slightly falling.

**GPU memory is the binding constraint, and it has already cost a seed.**
Three E3.1 arms sustained **19,613 MiB of 20,475 (95.8%)**, per client
**7,228 / 4,602 / 7,032 MiB**, and `rtg_e31_s1` was stopped by stop-file to
save the other two. Body count is bounded at **29** (`1 + 4 + 8 + 16`, from
`min_body_depth 1 / max_body_depth 4 / max_nchild 2`), so the demand terminates
— but 29 is reachable inside one episode's design phase.

**D2's 2v2 trainer, for reference only** (batched torch, fixed topology — a
different algorithm on different hardware paths; quoted because it is the only
measured 2v2 anything, not because it predicts T2A). All nine runs, from
`runs/competevo_port/*/log.json`; 512 worlds, rollout 100, 200 iters,
20.48 M agent-lane steps. `sec` is cumulative in that file — a trap
`PLAN_D2_COMPETEVO.md:113-118` records:

| run | arm | median s/iter | total |
|---|---|---:|---:|
| `t2v2_cold` | `team` s42 | 48.95 | 3.03 h |
| `t2v2_team_s43` | `team` s43 | 44.50 | 2.57 h |
| `t2v2_team_s44` | `team` s44 | 42.28 | 2.60 h |
| `t2v2_scorer` | `scorer` s42 | 48.59 | 3.02 h |
| `t2v2_scorer_s43` | `scorer` s43 | 43.91 | 2.57 h |
| `t2v2_scorer_s44` | `scorer` s44 | 43.21 | 2.55 h |
| `t2v2_role_s42` | **`--role-in-design`** s42 | 60.66 | 3.36 h |
| `t2v2_role_s43` | role s43 | 60.00 | 3.36 h |
| `t2v2_role_s44` | role s44 | 60.06 | 3.34 h |
| `t2h_perslot_s42` | per-slot, quiet box | 14.28 | 0.88 h |

Three readings, and only the first is safe:

* **A 200-epoch D2 2v2 seed costs 2.5-3.4 h.** `DESIGN_2V2.md:1091`'s "~2 h"
  for `t2v2_cold` is wrong against its own log (3.03 h), and §8's pre-run
  "~48 h" estimate was ~15× pessimistic.
* The `--role-in-design` arms are ~24% slower than their contemporaries. That is
  **confounded with box load** — the min/max spread within every run is 3-8× —
  so treat it as an upper bound on the flag's cost, not a measurement of it.
* `t2h_perslot_s42`'s 14.3 s/iter is a quiet box, not a faster architecture.

**Do not read a 2v2-vs-1v1 ratio off this table**, and do not read anything
about T2A off it at all: D2's step is launch-bound on a batched GPU env with a
single compiled model, and T2A's is a per-episode `mujoco_py` recompile with a
GNN forward per body per step.

### 6.2 Estimated — and labelled as such

**This is an estimate. The previous two budgets on this project were wrong —
E4's by >2× because it divided CPU-seconds by 10 workers on a 10.2-CPU quota,
and E4B's earlier one by ~50% because it quoted an unmeasured baseline. The
arithmetic below is shown so the error can be found before it is paid for.**

Anchor on the *measured delta*: adding one policy-driven body to a T2A 1v1
rollout costs **+129 s/epoch**. A 2v2 rollout with one learning team and one
ring team has **three** bodies beyond E3.1's single learner: one extra learner
slot and two ring opponents.

Two brackets, and the difference between them is a **decision, not an unknown**:

| | `min_batch_size` | env steps/epoch | epoch time | 400 epochs |
|---|---|---:|---:|---:|
| (a) keep 50,000 samples | both slots push, so env steps **halve** | 25,000 | `(141 + 3×129)/2` ≈ **264 s** | ~29 h |
| **(b) keep per-slot data constant** | **100,000** | 50,000 | `141 + 3×129` ≈ **528 s** | **~59 h** |

**Take (b).** Under (a) each learner slot gets 25,000 samples per epoch against
E3.1's 50,000 — half the gradient signal per slot on a task whose seed variance
is already the worst in the project. Halving the data to halve the clock is the
wrong trade here.

Two adjustments to (b), both estimates:

* **Recompiles**: 2 learners × 6 design steps = **12** per episode against
  E3.1's 6, on a 4-body scene. At ~4 ms for 2 bodies, call it 6-8 ms for 4:
  `12 × 7 ms = 84 ms/episode` against `6 × 4 ms = 24 ms`. At 544 episodes/epoch
  that is **+33 s/epoch**, ~6% — inside the bracket's own uncertainty.
* **The `+129 s` figure was measured under 3-arm contention.** A single arm on
  a quiet machine is faster (`rtg_e31d_s3body` alone ran at 110.5 s against
  128-138 for the contended trio). This cuts the other way and is not
  quantified.

**So: ~59 h per seed (estimate), with the recompile adjustment ~61 h, and a
downward correction of unknown size if the arm runs alone.**

### 6.3 Concurrency — and this is where E5 gets expensive

**GPU memory, estimated from the measured E3.1 figures.** An E5 arm holds two
learner graphs and two ring policies where E3.1 held one graph and E4R holds
one graph plus one ring policy. Per-client memory was **4.6-7.2 GB** at 12-22
bodies; a 2v2 arm at the same body sizes is plausibly **10-14 GB**.

> **On a 20,475 MiB card that is one E5 arm at a time, possibly two if the
> bodies stay small.** Three concurrent arms — the shape every previous T2A
> rung used — is very likely out of reach, and E3.1 already lost a seed at 95.8%
> occupancy.

| plan | wall clock | confidence |
|---|---:|---|
| 3 seeds **sequential** | **~180 h ≈ 7.5 days** | estimate, from §6.2 |
| 2 concurrent then 1 | ~120 h | estimate, assumes 2 arms fit |
| 3 concurrent | ~60 h | **probably impossible on memory** |

**Plus** the functional probes of §4.4: 3 seeds × 3 epochs × 4 probes × 50
episodes = 1,800 episodes of pure evaluation. At E3.1's `T_eval` of ~7 s for a
50-episode pooled eval, ~4 h total. Small.

**Recommendation: do not launch E5 as a 3-seed wave.** Stage it:

1. **E5.0 — build and gate, no training.** The scene, the two-slot sampler, the
   role channel, the team ring. Plus the free analytic gate of §3a: dump both
   slots' mean-action bodies from an untrained *blind* shared policy and assert
   `0.000e+00`; then from a *sighted* one and assert non-zero. That single gate
   replaces a 60-hour blind control arm.
2. **E5.1 — one seed, 400 epochs, ~60 h.** Read it against Outcome 4's guards
   only. If the task is not learned (D2 says there is a real chance), stop.
3. **E5.2 — seeds 2 and 3**, on the evidence from E5.1.

This is a decision for the user, not a fait accompli: 180 h is ~7.5 days of the
card, against E4B's ~29 h.

---

## 7. What E5 does not test

* **Whether a *behavioural* role division emerges in this stack.** D2 measured
  it (back pair 54.9% of crossings over six runs) in a fixed-topology stack
  with a role one-hot on the control head. E5 inherits the role channel and
  will *report* the crossing split, but as a **guard** (Outcome 4), not as a
  result — one shared arm cannot separate "the role channel changed the body"
  from "the role channel changed the gait", because §3a's cheap route feeds the
  role to the control head and the critic as well.
* **Whether topological specialisation *helps*.** Nothing in §4.3 is a
  performance comparison. D2's §15 found specialisation was free (80.2% vs
  84.6%, indistinguishable at a 50-point range); E5 has no arm that could
  detect a smaller effect than that, and with 3 seeds at 2v2 variance it could
  not detect a 15-point one.
* **Soccer.** No ball, no possession, no passing. E6.
* **Heterogeneous *creatures*.** D2's ant/bug/spider machinery
  (`gate_hetero.py`) is not used; both slots start from the same converted ant.
* **Whether two independently-evolving *teams* diverge.** E5 is shared-weight
  across both slots and, if it follows E4B, shared across both teams. The
  two-lineage question is archived under `docs/t2a/e4_twolineage_archive/`.
* **Anything about the `goal_credit` or `down_rule` choices.** They are adopted
  from D2's recommendations, not re-tested. D2's own three-seed replication
  withdrew the goal-rate half of its `goal_credit` conclusion, so the arms are
  chosen on the *argument* (blocking must be rational for the blocker) rather
  than on that measurement.
* **Cross-play.** E5 samples whole past teams on D2's recommendation, which
  rests entirely on PSRO and Hanabi numbers from other domains. D2's §10 lists
  "no cross-play measurement" as open and E5 does not close it.
* **Spawn randomisation** (§5). Fixed spawns throughout, so E5 cannot
  distinguish "the role channel means player 2" from "the role channel means
  the ant that starts at x = ∓4".
* **Whether the dense reward is what suppresses interference.** §0 shows the
  dense term already pays the back agent *more* for racing. E5 has no arm that
  varies `alpha` or the dense/sparse balance, so a null result cannot be
  attributed between "the architecture cannot" and "the reward will not let
  it".

---

## 8. What I did not check

Stated plainly, as this project requires.

* **I ran nothing.** No smoke, no probe, no microbenchmark — the CPU quota is
  10.2 and an E4R arm holds ten workers, so any timing run of mine would have
  slowed it and produced a contended number anyway. Every timing above is
  either parsed from a completed run's log or quoted from a doc that measured
  it.
* **I did not verify D2's SMD 0.110 → 0.833 by recomputation.** I read §15's
  table, §12's 0.052, §14's replication and §11's spectator claim at source in
  `DESIGN_2V2.md`, and cross-checked them against `slot_policy.py:4-6`,
  `team_policy.py:111-121` and `train_team_selfplay.py:182-187`, which agree.
  But **no committed script computes that SMD**: `role_metrics.py` and
  `score_policies.py` do not, and `fall_analysis.py:162-178`'s SMD helper
  compares fallers against non-fallers, not front against back. The §15 table
  appears to come from an uncommitted ad-hoc script. **E5's equivalent metric
  must therefore be committed code with its own gate, not a notebook** — and
  the two role-hidden numbers in circulation (§12's 0.052 and §15's 0.110) are
  different statistics (mean-action design distance vs largest per-dimension
  SMD) that I did not reconcile.
* **`role_metrics.py`'s CPD docstring is stale**: `:132` still says *"Each group
  is perturbed by the same gaussian jitter"*, while `:157-166` scales by the
  group's own `control_norm` std — which is the fix `DESIGN_2V2.md:966-978`
  applied after the unscaled table gave a wrong ranking. Anyone reusing that
  function should read the code, not the docstring.
* **I did not measure T2A's 2v2 epoch time**, because no 2v2 T2A code exists.
  §6.2 is arithmetic on measured 1v1 deltas and is labelled an estimate.
* **I did not measure the 4-body scene's compile time or its GPU footprint.**
  Both are estimates scaled from 2-body measurements.
* **I did not check whether two `Robot` objects can share one merged XML tree**
  without a `khrylib` change. `swap_opponent`'s docstring says *"`Robot` owns
  the whole tree"*, which is why §2.2 recommends two disjoint subtrees, but I
  did not test it. **This is the single largest unverified assumption in the
  plan and E5.0 should gate it first.**
* **I did not read** `competevo_port/PORT_STATUS.md`, `M2E_VALIDATION.md`, or
  D2's `tests/test_team2v2.py` in full. `PLAY_2V2.md` was checked and is
  **irrelevant to this rung** — it is WS4's human-playable browser demo
  ("Four people, four devices, one 45-second ant match"), a dm_control soccer
  pitch with join codes and MJPEG streaming, with nothing about team lanes,
  morphology or run-to-goal.
* **The running E4R process is a 12-epoch smoke, not the 400-epoch production
  wave**, and `runs/d3_e4r_ring/` is empty. I report this and did not act on it.

---

## Sources

Code, read in this repo and in `/workspace/Transform2Act`:

- `rower_soccer/competevo_port/{team_env,team_scene,team_policy,slot_policy,selfplay,train_team_selfplay,role_metrics,probe_2v2,design,dev_env,gate_hetero,run_to_goal_env,scene}.py`
- `rower_soccer/t2a_port/{rtg_scene,e4r_ring,e4_selfplay,e4_divergence,e4_compset,e4_null_traj,dense_policy}.py`
- `Transform2Act/design_opt/envs/{ant,run_to_goal,run_to_goal_sp}.py`;
  `design_opt/models/{transform2act_policy,transform2act_critic,jsmlp,gnn}.py`;
  `design_opt/agents/transform2act_agent.py`; `khrylib/rl/agents/agent.py`;
  `design_opt/cfg/{rtg_e3_s1,rtg_e31_s1,rtg_e31f_s1,rtg_e4r_s1}.yml`

Measured data:

- `runs/d3_e31_fix/logs/train_{p_s1,p_s2,p_s3,f_s1,d_s3body}.log` — epoch timings, whole series
- `runs/competevo_port/{t2v2_role_s42,t2v2_team_s43,t2h_perslot_s42,m2e_fixed}/log.json` — D2 2v2 iteration times
- `rower_soccer/docs/t2a/e4_null/{e31_comparison_set,e31_crossseed_null}.json` — the SMD calibration, marked FINAL

Documents:

- `DESIGN_2V2.md` (§4, §5, §6, §8, §11-15), `D3_E31_FIX.md`, `D3_E3_ADVERSARIAL.md`,
  `D3_E4_PREREQ.md`, `D3_E4R_SHARED.md`, `PLAN_D3_E4B_SELFPLAY.md`, `PLAN_D3_M3.md`

Literature, via the plans above rather than re-read here: Bansal et al. 2018
(δ = 0 best for Ant); Kuba et al. 2021 (the `(n−1)` credit-variance factor);
MAPPO/IPPO over COMA; PSRO joint policy correlation; Team-PSRO; VDN on
identifiers for parameter-shared agents; Liu et al. 2019 2v2 soccer.
