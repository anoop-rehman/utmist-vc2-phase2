# CompetEvo → mujoco_warp PORT MAP (drives M2 of PLAN_D2_COMPETEVO)

*Written 2026-08-10 from a deep read of `/workspace/competevo` (KJaebye/competevo,
arXiv:2405.18300). All paths below are relative to that clone unless absolute.
Target stack: `rower_soccer/warp_port/` (mujoco_warp batched physics, torch fp32 PPO).
Companion: `docs/repro/COMPETEVO_M1_REPRO_NOTES.md` (= clone's `REPRO_NOTES.md`) for
install/entrypoints/measured numbers.*

**The one-sentence summary an engineer needs first:** the paper's "CompetEvo" agents
(`dev_*`) have a FIXED topology whose 20–40 geometric scale parameters are emitted by
the policy as a special action at step 0 of EVERY episode, after which the env
regenerates the MJCF string and recompiles the MuJoCo model mid-episode — so the GPU
port does not need per-generation model rebuilds at all: one compiled model + per-world
writes of design-derived model fields (sizes, positions, gears, masses/inertias)
replaces their per-episode XML recompile.

---

## 1. ENV LAYER

### 1.1 Construction chain

- `train.py:80-90` picks a runner from `cfg.runner_type`; `runner/base_runner.py:72-78`
  does `gym.make(env_name, cfg=cfg)` (no render_mode during training).
- Env IDs are registered in `competevo/__init__.py` (evolution envs; e.g.
  `run-to-goal-devants-v0` → `MultiDevAgentEnv` with
  `agent_names=['dev_ant','dev_ant']`, `init_pos=[(-1,0,0.75),(1,0,0.75)]`,
  `ini_euler=[(0,0,0),(0,0,180)]`, `max_episode_steps=500`, lines 84-95;
  `robo-sumo-devants-v0` → `RoboSumoDevEnv` with `world_body_arena.xml`,
  `min_radius=2.5, max_radius=4.5`, lines 5-19) and `gym_compete/__init__.py`
  (fixed-morph baselines: `run-to-goal-{ants,bugs,spiders}-v0`, `robo-sumo-*`).

### 1.2 One scene, two agents, one MJCF

- `MultiDevAgentEnv.__init__` (`competevo/evo_envs/multi_dev_agent_env.py:46-99`) maps
  each agent name to `(per-agent body XML, Agent class)` via `AGENT_MAP` (lines 17-42)
  and instantiates agent objects. The scene XML is produced by **merging** the world
  XML and the two agent XMLs into ONE MJCF:
  `create_multiagent_xml` / `create_multiagent_xml_str`
  (`competevo/evo_envs/evo_utils.py:148-250 / 50-146`). Per agent `i` it:
  - prefixes every `name` with `agent{i}/` (`add_prefix`, evo_utils.py:112) — this
    prefix is the ONLY thing that identifies which bodies/joints/actuators belong to
    which agent downstream;
  - overwrites the root body `pos`/`euler` with the registered `init_pos`/`ini_euler`
    (evo_utils.py:93-108);
  - sets geom defaults `conaffinity=str(i)`, `contype=str(1-i)` (evo_utils.py:88-89).
    With floor contype=1/conaff=1 this yields: agent↔agent collide, agent↔floor
    collide, **self-collision disabled for both agents**. NOTE: this bit trick only
    works for exactly 2 agents — 2v2 (M3) needs a proper bitmask scheme
    (e.g. agent i: `contype=1<<i`, `conaffinity=~(1<<i)&0xF`) or explicit excludes.
  - appends all `<actuator>` motors into one global actuator block, in agent order.
- The compiled scene: `MultiAgentScene` (`gym_compete/new_envs/multi_agent_scene.py`,
  gymnasium `MujocoEnv` from a file path, `frame_skip=5`) at init, and
  `MultiEvoAgentScene` (`competevo/evo_envs/multi_evo_agent_scene.py:241-312`, same
  thing but `mujoco.MjModel.from_xml_string` at line 201) for the mid-episode rebuild.
- Physics options come from the world XML, not code: `world_body.xml` (run-to-goal)
  and `world_body_arena.xml:3` (sumo) both use `integrator="RK4" timestep="0.003"
  solver="PGS" iterations="1000"`; frame_skip=5 → control dt = 0.015 s; episodes are
  500 control steps. Actuators: `ctrlrange="-1 1"`, gear 150 (dev ant,
  `evo_envs/assets/dev_ant_body.xml`). Torso body of a dev agent is named `"0"`
  (legs `"1".."4"`, links `"11".."14"`, feet `"111".."114"`).

### 1.3 Per-agent state extraction (what "per-agent" means)

- `Agent` base (`gym_compete/new_envs/agents/agent.py`): after every (re)build,
  `set_env` re-derives per-agent index ranges by filtering global model names on the
  `agent{i}/` prefix — `_set_body`:92-116, `_set_joint`:119-145,
  `_set_other_joint`:147-164. Agent qpos/qvel are **contiguous slices** of the global
  vectors (`get_qpos`:206-211, `get_qvel`:222-227, opponent = the complement,
  `get_other_qpos`:213-220). This is the exact analog of per-agent index slices into
  a batched `qpos[world, :]` on GPU — precompute the slices once.
- Actions: env `_step` concatenates the per-agent action arrays and writes the result
  to `data.ctrl` in one `mj_step` (`multi_agent_scene.py:33-37` → `do_simulation`).
  Global action layout = agent0's actuators then agent1's.

### 1.4 Observations (per agent, run-to-goal dev ant)

`DevAnt._get_obs` (`competevo/evo_envs/agents/dev_ant.py:309-337`) returns a **list of
3 arrays** (not a flat vector): `[stage_flag(1), scale_vector(20),
sim_obs(31 = own qpos 15 + own qvel 14 + opponent root xy 2)]`. `state_dim = 52`,
`action_dim = 20 (design) + 8 (motors) = 28` (dev_ant.py:42-49). Own qpos includes the
global root x,y — there is no ego-centric transform anywhere.
Sumo fighters (`dev_ant_fighter.py:295-334`) use a richer sim_obs: qpos + qvel +
|clipped cfrc_ext| + opponent xy + own torso xmat(9). Fixed-morph `Ant._get_obs`
(`gym_compete/new_envs/agents/ant.py:59-87`) is the flat version: qpos+qvel+opp xy.

### 1.5 Reward and termination plumbing (split agent/env/runner!)

Three layers — port all three or the numbers won't match:
1. **Agent dense reward** in `after_step` (dev_ant.py:280-304): forward progress
   toward goal `(Δx/dt, sign-flipped if moving left)` − `0.5*sum(a²)` ctrl cost −
   `0.5e-3*sum(clip(cfrc_ext,-1,1)²)` contact cost + 1.0 survive; termination when
   not standing (`z ∉ [0.28, 1.2]`). Sumo (`dev_ant_fighter.py:281-291`): only
   `-ctrl_cost` and `alive=2.0` here.
2. **Env parse (sparse) reward**: run-to-goal `goal_rewards`
   (`multi_dev_agent_env.py:218-234`): ±1000 when exactly one agent crosses its goal
   line (x = ±4), sets `infos[i]['winner']`. `_step` (243-272) composes
   `rew = parse + move_reward_weight*dense` and stores `reward_parse`/`reward_dense`
   in infos. Termination `_get_done` (236-241): any agent done, game done, or
   non-finite state → both agents share the same `terminateds`. Truncation at
   `_elapsed_steps >= 500`. Sumo (`robo_sumo_dev.py:163-231`): win +2000 / lose
   −2000 when an opponent falls below `z < 0.29+arena_h` or leaves the radius; draw
   −1000 at the step limit; dense = alive + ctrl + `move_to_opp` (velocity toward
   opponent, coef 10) + `push_opp` (−10·exp(−|opp to center|)).
3. **Runner recombination** — the trained reward is NOT the env reward:
   `MultiEvoAgentRunner.custom_reward` (`runner/multi_evo_agent_runner.py:147-164`)
   implements the exploration curriculum
   `r = α·dense + (1−α)·parse`, `α = max((termination_epoch − epoch)/termination_epoch, 0)`
   (yaml: `use_exploration_curriculum: true`, `termination_epoch: 1000`). Loggers
   record this curriculum reward, not the env reward.

### 1.6 Episode lifecycle (dev envs) — the crucial part

`MultiDevAgentEnv.reset` (341-352) **deletes and recreates the agent objects** (fresh
base XML trees) and rebuilds the init scene, sets `stage='attribute_transform'`, and
randomizes each agent's `scale_vector ~ U(-1,1)` (dev_ant.py:355-365 via reset_agent).
Then `step()` (274-316) branches on stage:
- **Step 0 (`attribute_transform`)**: the incoming action IS the design.
  `agents[i].set_design_params(action[:20])` mutates the agent's lxml tree,
  `load_tmp_mujoco_env` (161-197) merges the two mutated XML strings and compiles a
  **brand-new MjModel/MjData** (`MjModel.from_xml_string`), re-derives all per-agent
  indices, then `transit_execution()`. Returns zero reward, `use_transform_action`
  info, and `design_params` in info (the runner logs 10 random designs/epoch to
  `{run_dir}/0.csv,1.csv`, runner:342-363). No position noise is re-applied for
  run-to-goal — start state is deterministic given the design. Sumo DOES call
  `self._reset()` after the rebuild (random radius + random side/positions,
  `robo_sumo_dev.py:269`) but returns the obs computed *before* that reset
  (lines 266-270) — an off-by-one quirk; don't faithfully port it, flag it.
- **Steps 1..500 (`execution`)**: action tail `action[-8:]` is the motor command
  (`step`:311), stepped as §1.3.

So: **morphology changes once per episode, chosen by the policy, at episode start.**
There is no generation loop and no mutation during execution.

---

## 2. MORPHOLOGY GENOME

### 2.1 `dev_*` agents (the paper's "CompetEvo" agents — what we port)

- Representation: a flat vector `scale_vector ∈ [-1,1]^k`, k = 20 (ant), 30 (bug),
  40 (spider) (`dev_ant.py:26`, `dev_bug.py:26`, `dev_spider.py:26`). It is
  simultaneously (a) part of the observation, (b) the first-step action, (c) the
  genome. Scaled by `SCALE_MAX` (ant 0.3, bug 0.5, spider 1.2; **fighter variants
  differ**: ant_fighter 0.5, bug_fighter 0.3 — see `dev_*_fighter.py:10`).
- Genome → MJCF: `DevAnt.set_design_params` (dev_ant.py:53-269) computes
  `a = 1 + 0.3·s` (geometry) and `b = 1 + 0.15·s` (gears), then **multiplies** base-XML
  attributes: per leg chain, param triplet (length of upper link `fromto`, radius
  `size` of mid link, length of mid `fromto`, radius+length of foot), with child body
  `pos` scaled by the same factor as the parent geom's `fromto` (so links stay
  attached), and motor `gear` scaled by `b` for the corresponding joints. For the ant:
  params 0-4 = leg 1 (bodies `1/11/111`), 5-9 = leg 2, 10-14 = leg 3, 15-19 = leg 4.
  Everything else (joint ranges, axes, densities, topology, actuator count) is fixed.
- Because the compiler has `inertiafromgeom="true"` (density 5.0), rescaling geoms
  also rescales **masses and inertias** at compile time — a GPU port that only writes
  geom sizes is wrong; see §5.2.
- When: start of every episode (§1.6). Multiplication is w.r.t. the base XML because
  `reset()` re-instantiates the agents; it never compounds.

### 2.2 `evo_*` agents (Transform2Act-style; present but NOT the paper's headline)

- Representation: an actual body **graph** — `Robot`/`Body`/`Joint`/`Geom`/`Actuator`
  classes over the XML tree (`competevo/evo_envs/robot/xml_robot.py:270-632`), with
  per-body continuous params (bone direction/length, size, gear; normalized to [-1,1]
  with 'sin'/'clip' mapping, `set/get_params`:584-618) and GNN edges = parent-child
  pairs (`get_gnn_edges`:624-632).
- Episode staging (`multi_evo_agent_env.py:259-343`): `skel_transform_nsteps` (=5)
  steps of per-body discrete add/remove-child actions (`EvoAnt.apply_skel_action`,
  `evo_ant.py:204-214`), then 1 attribute step (delta or absolute design,
  `robot_param_scale`), then execution with per-joint 1-dim actions gathered from
  graph nodes (torso node's action discarded, env:337). XML recompiled after EVERY
  transform step.
- Obs per node: `[attr_fixed | sim_obs(13+2) | attr_design]` + edge list + stage +
  num_nodes (`evo_ant.py:333-351`).
- **Port implication**: nq/nv/nu change within an episode → a compiled batched model
  cannot follow. If we ever port `evo_*`, batch-by-skeleton: freeze skeleton decisions
  at reset, bucket worlds by resulting topology, one compiled model per bucket (their
  own loop only changes topology in the first 5 steps, so "batch by
  morphology-at-reset" is faithful up to those 5 free steps). **Recommendation: defer
  evo entirely; dev is the paper's competitive-co-evolution result.**

---

## 3. POLICY

### 3.1 Dev policy (what we port first) — NOT a GNN

`DevPolicy` (`custom/models/dev_actor.py`): two independent MLP heads behind
`RunningNorm` input normalization:
- **scale head**: `RunningNorm(20) → MLP[64,64] tanh → Linear(20)`, diagonal Gaussian
  with learned log_std init 0 but std divided by 5 at distribution build (line 91);
  sampled design clamped to [-1,1] (line 125). Input is ONLY the current
  `scale_vector` (i.e. the random init vector) — not sim state.
- **control head**: `RunningNorm(31) → MLP[64,128,64] tanh → Linear(8)`, DiagGaussian,
  learned log_std (init 0). Input is only `sim_obs` (`use_entire_obs: false`).
- `forward` (66-112) partitions a mixed batch by the per-sample stage flag
  (`design_mask`) and runs each head on its sub-batch; `select_action` (114-140)
  returns a 28-dim action = [scale(20) | control(8)] with the inactive part zeroed;
  `get_log_prob` (142-170) scatters per-stage log-probs back into one column. This
  mixed-stage batching maps 1:1 onto a batched GPU policy where per-world stage flags
  select the head.
- Critic `DevValue` (`custom/models/dev_critic.py`): one MLP[64,64,64] on the FULL
  52-dim state (stage flag + scale + sim_obs).
- `RunningNorm` (`lib/rl/core/running_norm.py:5-42`) updates running mean/var
  **inside forward when module.training** — i.e. during the PPO update pass, not
  during sampling. Preserve update cadence or curves shift.

### 3.2 Fixed-morph baseline policy

`NormalPolicy` (`custom/models/normal_actor.py`): RunningNorm → MLP[128,128] tanh →
Gaussian head, per yaml `policy_specs` (`config/run-to-goal-ants-v0.yaml`). Critic
`NormalValue`: MLP[512,256]. Same select_action contract, flat obs.

### 3.3 GNN policy (evo only)

`Transform2ActPolicy` (`custom/models/transform2act_actor.py`): three sub-policies
(skel Categorical / attr Gaussian / control Gaussian), each optionally
`GNNSimple` (`custom/models/gnn.py`, torch_geometric `GraphConv/GCNConv/...` stack
over the body graph, weight-shared across nodes) plus per-body-index `JSMLP`.
Batches variable-size graphs by concatenating nodes and offsetting edge indices
(`batch_data`:115-128); episode log-prob per agent = sum over its nodes
(cumsum trick, 242-270). Only needed if/when evo is ported; torch_geometric would be
replaced by a padded dense adjacency batched over worlds (small graphs, ≤ ~13 nodes).

### 3.4 Distributions

`DiagGaussian` (`lib/rl/core/distributions.py`) = torch Normal, `log_prob` summed to
one column, `mean_sample()` = loc (used for ALL eval actions). Everything runs in
**torch.float64** (`train.py:61-62`) — the warp stack is fp32; accept fp32 (validate
via curve shape, not bit-match).

---

## 4. TRAINING LOOP (`runner/multi_evo_agent_runner.py`)

### 4.1 Iteration structure (`train.py:93-99`, `optimize_policy`:76-106)

Per epoch: (1) sample `min_batch_size=50k` env steps (both agents in the same envs);
(2) `learner.update_params(batch_i)` for each agent (`DevLearner`, PPO, clip 0.2,
GAE γ=0.995 τ=0.95, 10 optim epochs × minibatch 2048, Adam policy 5e-5 / value 3e-4,
grad-clip 40, critic L2 reg 1e-3 — `custom/learners/dev_learner.py:129-199`);
(3) eval-sample `eval_batch_size=10k` steps with `mean_action=True` on 10 workers;
(4) checkpoint. Learners are per-agent — two full policy+critic sets, no sharing.

**Bug to NOT reproduce blindly**: the fixed-morph `Learner`'s critic update is
commented out (`custom/learners/learner.py:218-219,228-229`) — the baseline's value
net is never trained (advantages come from a frozen random critic). `DevLearner`
trains its critic properly (`ppo_step`:172-181). Match their behavior only if a
baseline reproduction refuses to line up.

### 4.2 Worker parallelism (`sample`:310-461)

`multiprocessing.Process` fork-workers, each with a **copy of the whole env**, each
rolling out `min_batch_size/nthreads` steps of full episodes (`sample_worker`:
166-308). Memory tuple per step: `(states, actions, mask, next_states, reward, exp)`;
mask=0 only on terminal steps (`sample_worker`:284-292) — note truncation leaves
mask=1, so GAE (`lib/rl/core/common.py:5-25`, sequential scan over the concatenated
buffer, advantages globally std-normalized) bootstraps across episode boundaries on
draws. The transform step is stored as a normal step (reward 0, mask 1) — that is
how the design action gets PPO credit (via GAE bootstrap from later rewards). On GPU
this whole layer collapses into the batched rollout; keep the "design step is
step 0 of the trajectory" convention.

### 4.3 Opponent sampling — load-bearing (PLAN flags it), via pickle files on disk

Two modes inside `sample_worker` (191-225):
- **Eval / epoch 0 / `use_opponent_sample: false`**: both agents load checkpoint
  `epoch_%04d.p` where ckpt = current epoch; wrapped in try/except **pass** — at
  epoch 0 no file exists, so eval runs with freshly initialized random sampler nets
  (explains iter-0 eval reward ≈ 430 in our sanity run).
- **Training with `use_opponent_sample: true` (all dev configs)**: TWO worker fleets
  per iteration (`sample`:377-461), fleet `idx∈{0,1}`: ego agent `idx` loads its
  CURRENT epoch checkpoint; opponent `1-idx` loads a uniformly sampled checkpoint
  from `[max(1, floor(delta·epoch)), epoch]`, `delta=0.5` (dev; fixed-morph ants use
  `delta: 0` = full history, Bansal-style). Only ego's half of each fleet's data is
  kept (`b = [ma_buffer_0[0], ma_buffer_1[1]]`, line 457). So each PPO update sees
  50k ego steps collected against a *mixture* of past opponents, resampled **per
  worker per rollout** (once per worker-batch, not per episode — the sampler loop
  reassigns samplers each outer `while` iteration, line 179-225).
  `multi_agent_runner.py:243` additionally makes the opponent act deterministically
  (`mean_action`) in the fixed-morph runner; the evo runner does not.
- GPU translation: keep an in-memory ring of opponent state_dicts (one per epoch,
  epochs ≥ floor(δ·epoch)); split the world batch into K opponent-blocks, sample one
  opponent ckpt per block per iteration, run K opponent forward passes (or vmapped
  stacked params). Halve the batch for ego=0 / ego=1 fleets — same envs, roles
  swapped in which half's data is kept.

### 4.4 Checkpoints & logging

- Per agent per epoch: `tmp/<env>/<ts>/models/agent_{i}/epoch_%04d.p` (pickle of
  `{'policy_dict','value_dict','running_state'(=None),'best_reward','epoch'}`,
  float64 tensors) + `best.p` on best eval reward OR best win rate
  (`save_checkpoint`:482-504, `log_optimize_policy`:114-132). `save_model_interval: 1`
  — every epoch is kept (opponent sampling depends on this).
- TB scalars (:126-132): `train_R_eps_avg_{i}`, `eval_R_eps_avg_{i}`,
  `eval_win_rate_{i}`, `episode_length`. Design params: 10 random per epoch into
  `{run_dir}/{0,1}.csv`.
- **Win-rate definition** (`sample_worker`:284-299 + `sample`:369-372): during a
  sampling pass, every finished episode increments `games`; agent i's wins increment
  when `terminateds[0]` and `'winner' in infos[i]` (run-to-goal: crossed the goal
  line, ±1000; sumo: opponent fell/left arena). Truncated episodes are draws and
  **count in the denominator**. `win_rate[i] = wins_i / games`. The logged number is
  from the EVAL pass (mean actions, 10k steps ≈ 20+ episodes). Reproduce exactly this
  quotient — including draws in the denominator — or comparisons to their curves are
  meaningless.

---

## 5. PORT PLAN

### 5.1 Target shape (what we already have)

`warp_port/worm_env_base.py` (`WormEnv`): num_worlds batched env over a
`PhysicsBackend` (`backend.py` — owns `mujoco_warp` put_model/put_data/graph capture),
scene compiled ONCE from generated XML (`scene.py`), torch-tensor obs/rewards on GPU,
pluggable `RewardStrategy`, per-world auto-reset. `warp_port/ppo.py`
(`PPOTrainer`/`ActorCritic`): fp32, rollout segments (rollout_len≈64), GAE on GPU,
`.pt` checkpoints. `train_kick_warp.py`: CLI trainer + one-world eval env pattern.

### 5.2 What maps 1:1

| theirs | ours |
|---|---|
| merged 2-agent MJCF (`create_multiagent_xml`) | run once offline → one scene XML fed to a new `CompeteScene` builder (like `scene.py`); bake `init_pos/euler`, contype/conaffinity trick as-is |
| per-agent qpos/qvel slices (`agent.py:_set_body/_set_joint`) | precomputed index tensors; obs = gather + concat per §1.4, batched `[nworld, 52]` per agent |
| concat actions → `data.ctrl` | write both agents' action tensors into ctrl slices, one physics step |
| dense/parse rewards, termination (§1.5) | pure tensor ops on qpos/xpos slices — a `RewardStrategy` per arena; curriculum α computed from iteration counter in the trainer |
| DevPolicy/DevValue MLPs | `ActorCritic` variant with two heads + stage mask; RunningNorm exists conceptually as obs-norm — replicate the update-during-training-pass semantics |
| PPO (clip 0.2, GAE 0.995/0.95, minibatch 2048×10) | `PPOTrainer` hyperparameters; keep their values for validation runs |
| eval win-rate pass | dedicated eval-env fleet run to episode end with mean actions; win/draw counters per §4.4 |

### 5.3 New machinery (the actual work)

1. **Two-policy trainer.** One env batch, two learners, per-agent trajectories.
   Smallest delta from `PPOTrainer`: hold two ActorCritics + two optimizers, collect
   both agents' transitions from the same rollout, update sequentially (their order:
   agent 0 then agent 1, `optimize_policy`:91-92).
2. **Opponent checkpoint ring + block sampling** (§4.3). Also the ego-fleet split
   (half worlds keep agent-0 data, half keep agent-1 data).
3. **Vectorized design→model-fields writer** (replaces per-episode XML recompile).
   Topology is fixed, so ONE compiled model serves all worlds; at each world's reset,
   after the policy emits `s ∈ [-1,1]^20`, write per-world model fields implied by
   §2.1: `geom_size` (radius, half-length), `geom_pos`/`geom_quat` (capsule `fromto`
   → frame; pure scaling keeps the quat, scales the midpoint), `body_pos`,
   `actuator_gear`, and — because `inertiafromgeom` — `body_mass`, `body_inertia`,
   `body_ipos` recomputed analytically for capsules/spheres at density 5.0. Verify
   which fields our pinned mujoco_warp exposes per-world (model-field batching /
   domain-randomization support); anything not batchable per-world must be checked
   against `mjwarp`'s Model layout before committing (fallback: expand that field
   ourselves — sizes are tiny). **Gate this component in isolation**: pick 10 random
   design vectors, compare one-world mjwarp rollouts against their CPU env
   (`MultiDevAgentEnv` with fixed action sequences) for qpos trajectories at matched
   integrator settings before training anything.
4. **Per-world stage flags.** Worlds reset asynchronously; a world's first post-reset
   action is its design action (stage flag in obs, zero reward, model-field write,
   then execution). DevPolicy's design_mask batching (§3.1) already supports mixed
   batches. PPO must include that step in the trajectory (it is how design gets
   credit).
5. **Full-episode eval fleet** for win rates (training rollouts can stay segmented;
   eval cannot).

### 5.4 First task to port: `run-to-goal-devants-v0`

Justification: (a) it is the paper's headline dev matchup AND our live M1 sanity run
(`sanity_run.log`, run dir `tmp/run-to-goal-devants-v0/20260810_211247/`) — the only
config with a fresh their-code reference curve to diff against; (b) simplest reward
and termination (no arena-radius randomization, no contact-force obs — the run-to-goal
dev obs skips cfrc entirely, §1.4, so no cfrc_ext dependence on the GPU side);
(c) flat-ground locomotion is exactly what the warp stack already does; (d) the
fixed-morph twin `run-to-goal-ants-v0` shares the arena, giving Stage 0 for free.
Sumo (`robo-sumo-devants-v0`) is Stage 3: adds per-world arena radius (a per-world
`geom_size` write we'll already have), the cfrc_ext/xmat obs block, and the
win/lose/draw structure.

### 5.5 Staged port + validation gates

- **Stage 0 — fixed-morph harness** (`run-to-goal-ants-v0`, no evolution): 2-agent
  scene, slices, rewards, two-policy PPO, opponent ring. Gates: iter-0 eval reward
  ≈ 490-510 per agent with win rate 0.00 (their measured smoke numbers,
  REPRO_NOTES); win rate leaves 0 and eval reward trend matches a their-code CPU run
  of the same config over the first ~50 epochs (cheap: ~4.5 min/iter on the pod).
- **Stage 1 — design machinery offline**: design→fields writer vs their env,
  trajectory-level (gate in §5.3.3).
- **Stage 2 — `run-to-goal-devants-v0` end-to-end**: Gates: iter-0 eval ≈ 428-440 /
  win rate 0.00 (matches sanity_run.log iters 0-9); then the M1 sanity run's
  `eval_R_eps_avg`/`eval_win_rate` TB curves over its first ~50-100 epochs (same
  hyperparams, same curriculum α schedule); design-param CSVs show the same
  qualitative convergence (their 0.csv/1.csv vs ours). Full gate (= M2's "validation
  IS the paper numbers"): 1000-epoch run, win-rate trajectories shaped like paper
  Fig. for dev-vs-dev, then cross-play eval devant-vs-fixed-ant reproducing the
  paper's reported dev advantage (exact table values in the paper PDF; cite them in
  the eval report when run).
- **Stage 3 — `robo-sumo-devants-v0`**: same gates against a short their-code CPU
  reference run (needs to be launched then; none exists yet).
- **Stage 4 (M3 prep)** — 2v2 scene: new contype bitmask scheme (§1.2), n_agents
  generalization of slices/opponent-obs (their `get_other_qpos` already loops
  agents, `agent.py:147-177`).

---

## 6. RISKS (things in their code that will not survive batching)

1. **Per-episode MJCF recompile** (`load_tmp_mujoco_env`, lxml string manipulation,
   `MjModel.from_xml_string` twice per episode per worker). Cannot exist on GPU.
   → §5.3.3 vectorized field writer; fixed topology makes it sound.
2. **`inertiafromgeom` side effects**: naive port that only scales geom sizes gets
   the kinematics right and the dynamics wrong (mass/inertia/ipos stale).
   → analytic capsule/sphere mass properties at density 5.0; gate §5.3.3.
3. **RK4 + PGS(iterations=1000) + margin 0.01**: mujoco_warp does not cover this
   combination (Euler/implicit + Newton/CG are its lane). → switch scene options to
   `implicitfast` + Newton; ALSO run their CPU env with the same switched options for
   a few epochs to isolate "integrator delta" from "port bugs" before comparing
   against original curves. Expect reward-curve-shape equivalence, not trajectory
   equality.
4. **Python-loop reward/termination/contact code**: per-agent `after_step`, sumo's
   `get_agent_contacts` iterating `data.contact` (`robo_sumo_dev.py:64-80`, only used
   for diagnostics), goal_rewards. → tensor rewrites; nothing here actually needs
   contact enumeration (dense rewards use cfrc_ext-free formulas for run-to-goal;
   sumo win condition is position-based). The fighter OBS does need |cfrc_ext| —
   mjwarp exposes body external force accumulators; verify availability at Stage 3.
5. **Opponent weights via pickle files re-read per rollout per worker** → in-memory
   ring (§4.3). Disk layout compat matters only for resuming their checkpoints:
   loading their `.p` into the GPU policy needs a float64→32 + key-map shim
   (worth writing — lets us validate the ported env with their trained agents before
   training anything: replay their `best.p` pair in our env and check win rates.
   NOTE: their sanity run is mid-flight producing exactly these files).
6. **float64 everywhere** vs fp32 warp: fine for MLPs; the ±1000/±2000 sparse rewards
   and 500-step returns are well within fp32; keep advantage normalization.
7. **GAE quirks**: mask=1 on truncation (bootstraps across episode boundary),
   globally std-normalized advantages, buffer is a concat of whole episodes vs our
   segmented rollouts. For validation-grade runs, use full-episode rollouts
   (500-step horizon × nworld ≫ 50k anyway) and replicate their mask semantics;
   fix the truncation bug only after curves match.
8. **Fixed-morph Learner never trains its critic** (`learner.py:218-228` commented).
   Reproduce (frozen critic) for Stage 0 gate fidelity, then fix; DevLearner is
   correct so Stage 2 is unaffected.
9. **Eval-at-epoch-0 loads no weights** (try/except pass → random nets): harmless,
   but explains iter-0 numbers; replicate by evaluating the untrained nets.
10. **Reset determinism**: run-to-goal dev episodes have NO state noise after the
    rebuild (§1.6) — diversity comes only from the random initial `scale_vector` obs
    and stochastic actions. Don't "helpfully" add reset noise before curves match.
    Sumo, by contrast, randomizes radius and spawn sides per episode
    (`robo_sumo_dev.py:98-131`) — per-world tensors.
11. **2-agent-only contact bitmask** (§1.2) — must be redesigned for M3's 2v2.
12. **Their sumo transform-step obs off-by-one** (`robo_sumo_dev.py:266-270`, obs
    predate the post-transform reset): do NOT replicate; document the deviation when
    Stage 3 curves are compared.
