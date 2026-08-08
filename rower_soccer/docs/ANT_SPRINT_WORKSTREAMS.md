# Ant sprint — parallel workstreams (5 agents max)

*Operationalizes [ANT_SPRINT_PLAN.md](ANT_SPRINT_PLAN.md) into parallel agent
workstreams. Method skeleton follows the 2022 DeepMind humanoid football paper
(Liu et al., Science Robotics 2022) as adapted by
[PIPELINE_V2.md](PIPELINE_V2.md). Status: DRAFT — awaiting approval.*

## Milestone (unchanged)

4 humans on LAN play a 45-s 2v2 **ant** soccer match in the browser that feels
controllable; a demo file records all four players and replays. Then swap in
rower/worm. Self-play training and BC training are out of scope (the demo
*format* is in scope — it is the BC dataset).

## Mapping to the 2022 paper

| DeepMind stage | Ours | Sprint status |
|---|---|---|
| 1. Low-level NPMP from mocap | z-bottleneck follow, no mocap; frozen decoder = the low-level controller | **DONE** — `follow_ant_v1` fitness 0.981 @56M, video approved, weights in GCS |
| 2. Mid-level drill experts (dribble, shoot, …) | dribble / kick / shoot experts on the frozen follow decoder | dribble next; kick/shoot envs to build |
| (2b) Distill experts → one prior | **replaced**: all experts share the frozen decoder's z-space by construction | — |
| 3. Multi-agent RL w/ priors + PBT | **replaced for this sprint**: 4 humans ARE the high-level policy (play server); their demos feed BC next sprint | the game build |

Note on prior art in GCS: `fetch_ant_small*` are dm_control **quadruped**
fetch runs (`"plain": true`, no z-space, different body/obs, old sha) —
evidence ant-class ball tasks train in this stack, but the weights are not
reusable. Dribble trains fresh.

## Ground rules

- **One GPU.** All training is serialized through WS1's queue; other
  workstreams are code-only and must not launch GPU jobs. (If a second pod is
  approved, WS1 splits its queue across pods — nothing else changes.)
- **Merge discipline.** Each agent works in its own git worktree, commits
  small, rebases on `rower-gear-fix` often. File ownership below is designed
  to make overlap ~zero; anything cross-cutting goes through the orchestrator.
- **Contracts are law.** obs layout (proprio 65 + task), z-space (dim 16), and
  the demo schema (WS4 defines, others consume) change only by orchestrator
  sign-off recorded in this doc.
- Every trained artifact: `--gcs-bucket vc2-2026-checkpoints`, wandb project
  `creature-soccer`. Every gate has a video.

## Workstreams

### WS1 — Drill training + GPU queue (agent 1)
**Scope:** own the single-GPU queue end to end. Launch/monitor/gate each drill;
export/verify checkpoints; keep wandb + GCS tidy.
**Queue:** 1) `dribble_ant_v1` — `train_dribble_warp --init-from
runs_v2/follow_ant_v1/best.pt --freeze-decoder` (fallback per plan: unfrozen if
the frozen decoder stalls it; decide on curves, not vibes). 2) `kick_ant_v1`,
3) `shoot_ant_v1` as WS2 delivers envs. Re-runs/sweeps as needed.
**Files:** `runs_v2/*` (gitignored), no library code.
**Gates:** dribble: ball driven to target region reliably in eval video.
kick/shoot: the video looks like the named skill.

### WS2 — Kick + shoot envs (agent 2)
**Scope:** two new warp envs + trainers, modeled on `dribble_env.py` /
`train_dribble_warp.py` (same scene/ball machinery, same obs contract:
proprio 65 + task block).
- `kick`: reward = ball speed toward commanded direction at contact-break;
  episode segments on contact+separation.
- `shoot`: kick + goal geometry + scoring termination (goal specs already in
  `scene.py`'s pitch).
**Files:** `warp_port/kick_env.py`, `warp_port/shoot_env.py`,
`warp_port/train_kick_warp.py`, `warp_port/train_shoot_warp.py`; may extract
shared ball-task helpers into `warp_port/ball_task.py` (new file, not edits to
dribble's).
**Gate:** envs pass a 256-world random-torque smoke (no divergence, obs
finite, reward sane) → handoff to WS1.

### WS3 — High-level skill interface (agent 3)
**Scope:** the piece the 2022 paper puts between "drill priors" and "game": a
`SkillController` that, given (skill_id, target_xy) and the game obs, builds
the drill's exact obs vector, runs the right expert head, and emits the action
through the shared frozen decoder. One creature-agnostic module used by BOTH
the play server (human picks skill+target) and, next sprint, BC/self-play.
Includes: per-skill obs adapters (reuse `soccer_bridge.py`'s reconstruction
trick), checkpoint loading/caching, and a `scripted` baseline skill (chase
ball) for filling empty player slots.
**Files:** `rower_soccer/skills/` (new package), tests under it.
**Gate:** in the CPU soccer env, a SkillController-driven ant executes
follow-to-point on command; switching skill mid-episode does not glitch.
(Starts with follow only; dribble/kick/shoot slot in as WS1 lands them.)

### WS4 — Multiplayer game server (agent 4)
**Scope:** extend `play_server.py` into the 4-human LAN game:
- Authoritative CPU dm_control soccer sim (4 ants via `envs/build.py`;
  ant added to `CREATURE_XMLS` — coordinate one-line touch with WS5), fixed
  control-rate loop with real-time pacing.
- Lobby: 4 player slots (home/away × 1/2) + spectators; reconnect-safe.
- Client: browser page, topdown view, click-drag target + skill keys
  (follow/dribble/kick/shoot), per-slot input routing → WS3's SkillController.
- **Demo recording** (the BC dataset): per player per tick — obs, active
  skill, target, z, action; game events (touches, goals); versioned schema +
  writer/reader + deterministic replay renderer.
**Files:** `warp_port/play_server.py` (owns), `rower_soccer/game/` (new: lobby,
recording, replay), static client assets.
**Gate = sprint milestone** (with WS5): 4 browsers on LAN, 45-s match,
demo records and replays.

### WS5 — Integration, QA, and docs (agent 5)
**Scope:** ant into `CREATURE_XMLS`; verify 4-ant CPU soccer env steps at
≥ realtime on a laptop-class CPU (else flag hosting on the pod + port-forward;
measure input latency both ways); end-to-end tests (env smoke, SkillController
in soccer, demo record→replay byte-stability); keep STATUS/plan docs current;
produce the eval videos for every gate; final integration test choreography.
**Files:** `rower_soccer/envs/build.py` (one-line ant entry), `tests/`,
docs. Touches others' modules only via review, not edits.
**Gate:** the integration checklist below fully green.

## Dependency graph (critical path in bold)

```
WS2 kick/shoot envs ──▶ WS1 trains kick/shoot ─┐
**follow decoder ──▶ WS1 dribble** ────────────┼─▶ WS3 all-skills controller ─▶ **WS4 game + demos** ─▶ milestone
WS3 follow-only controller ◀── follow (done) ──┘         ▲
WS5 ant-in-soccer + perf check ──────────────────────────┘
```

WS4 and WS3 start immediately (follow exists). WS2 starts immediately.
WS1's queue runs continuously. Nothing blocks on kick/shoot except their own
training — the game is playable with follow+dribble first, kick/shoot hot-add.

## Sequencing & estimates

| Phase | Days (calendar, part-time) | What lands |
|---|---|---|
| P1 | 1-2 | dribble trained; kick/shoot envs smoke-tested; SkillController(follow) in soccer env; lobby skeleton serving 4 clients |
| P2 | 2-4 | kick/shoot trained; full SkillController; playable 2v2 with mixed human/scripted slots; demo recording round-trips |
| P3 | 1-2 | 4-human LAN session, latency measured, milestone gate, docs + retro |

## Integration checklist (WS5 owns)

- [x] ant in `CREATURE_XMLS`; 4-ant soccer env steps ≥ realtime on target host
      — **2.8x realtime idle** (9.0 ms/control step), **1.8x with 640×480
      top-down rendering in the loop**, on a Xeon E5-2650 v4 @ 2.2 GHz. That
      CPU is slower per core than any current laptop, so **host the server on
      the laptop; no pod + port-forward is needed**. (Under heavy contention
      from four concurrent agents the same bench read 1.16x — still above
      realtime, but the game host should not be sharing a box with training.)
      Shadows/MSAA must stay off (four 8192² shadowmaps ≈ 100 ms/frame).
- [x] GPU→CPU sim2sim gap measured and closed out — see the section below
- [x] `soccer_bridge` / SkillController can actually be driven inside soccer
      (needed `absolute_root_pos`/`absolute_root_mat` enabled on the soccer
      walkers; it raised `KeyError` before — fixed in `envs/build.py`)
- [x] env smokes: `MUJOCO_GL=egl .venv/bin/python -m tests.integration_smoke`
      (6/6; add `--warp` for the GPU/CPU observation-parity check)
- [ ] SkillController drives all 4 skills in CPU soccer; mid-episode switching clean
- [ ] 4 slots claimable from 4 devices on LAN; inputs isolated per slot
- [ ] demo file: record → replay is deterministic; schema versioned
- [ ] 45-s match completes with goals scored; video captured
- [ ] all drill checkpoints + configs in GCS; wandb links in STATUS doc

## The GPU→CPU sim2sim gap (affects WS4's backend choice)

Documented in `warp_port/render.py`'s header and `scene.py`'s solref comment:
mujoco_warp resolves contacts **~6.7x softer** than MuJoCo CPU on
byte-identical parameters (floor penetration 2.28 cm vs 0.34 cm); warp
therefore runs a stiffened `solref=0.005` against CPU's `0.02`. `noslip` was a
second, smaller contributor (fixed in `53971b6`). Consequence: a policy trained
in warp is not guaranteed to behave the same in the CPU dm_control soccer env
the play server uses — this is exactly why drill scoring was moved into warp in
the first place.

**Why it may not bite here:** the measured 0.4 m/0.6 s divergence was on the
*worm*, which topples chaotically. The ant is statically stable on four legs —
the property that makes it easy to train should also make it far less
chaos-sensitive. Untested assumption, so test it early and cheaply.

### PROBE RESULT — 2026-08-08 (WS5). Verdict: **no escalation. Build the CPU dm_control game as planned.**

`follow_ant_v1/best.pt`, deterministic (distribution mean, never sampled), 15 s
episodes, 6 episodes, matched initial states — warp's root pose, root velocity,
joint angles, joint velocities, target and target velocity all copied into the
dm_control drill.

| arm | fitness (median) | root-trajectory divergence vs warp (median @1 s / @5 s / @15 s) |
|---|---|---|
| **warp** (training sim) | **0.905** | — |
| **CPU dm_control**, closed loop | **0.892** | 0.48 m / 0.03 m / **0.047 m** |
| CPU, warp's actions replayed open-loop | 0.375 | 0.27 m / 1.43 m / 1.23 m |
| CPU with the pre-fix observation (below) | 0.284 | 0.46 m / 1.96 m / 4.65 m |

Read the first two rows together with the third. The backends really do differ
— replay warp's exact action sequence open-loop on CPU and the paths separate
by metres, so the contacts are as different as documented. **It does not matter
closed-loop.** Per-episode CPU-vs-warp separation after a full 15 s, all six
episodes: **0.066, 0.125, 0.026, 0.061, 0.025, 0.033 m**. Per-episode fitness
(warp → CPU): 0.917→0.899, 0.929→0.948, 0.899→0.866, 0.911→0.910,
0.086→0.086, 0.897→0.886. The two sims separate over the first second (the
spawn transient) and the policy then pulls them back together and holds them
within centimetres. The worm's 0.4 m in 0.6 s was a body that topples; the ant
is statically stable and its feedback loop absorbs the gap. The bet in the
paragraph above was right.

The 0.086 pair is the freeze in (3) below — and note it fired on the **same
seed in both backends, to three decimal places**. Even the policy's failure
mode is reproduced identically across the sim2sim boundary.

Artifacts: `runs_v2/follow_ant_v1/videos/ws5_sim2sim_warp_vs_cpu.mp4`
(side-by-side, left warp / right dm_control, same policy, same initial state —
warp 0.939 vs CPU 0.932 on that episode) and
`ws5_ant_follow_in_soccer_dithered.mp4` (the ant driven by `soccer_bridge`
inside the real CPU soccer env). Both also in
`gs://vc2-2026-checkpoints/follow_ant_v1/videos/`.

Consequences, in order of importance:

1. **WS4 is unblocked**: keep the authoritative CPU dm_control soccer sim. No
   solref stiffening, no CPU finetune, no warp 2v2 backend this sprint.
2. **The gap we were actually carrying was an OBSERVATION bug, not physics.**
   `warp_port/follow_env.py` scales the accelerometer `/100` and clips to
   ±50 (it is otherwise the only unbounded input, spiking to ~5,700 m/s² on
   contact); `creature.py` returned it raw, so the dm_control path fed the same
   policy a different vector. That alone cost fitness 0.89 → 0.28 and 4.6 m of
   divergence — i.e. every "sim2sim gap" number taken on the CPU drill was
   dominated by it. Fixed in `creature.py: CreatureObservables.sensors_accelerometer`
   (the scaling now lives with the observation, so drills, `soccer_bridge`,
   the play server and WS3's SkillController all inherit it). All live
   checkpoints post-date the warp-side scaling (2026-07-15), so nothing needs
   retraining. `tests/integration_smoke.py` asserts the contract.
3. **A real robustness bug, in BOTH backends** (so: not sim2sim, and it will
   bite the game): `follow_ant_v1` has an absorbing "sit down and do nothing"
   fixed point — torso height collapses to 0.35 m (vs 0.40-0.47 m walking),
   the action goes constant, and the ant never moves again (root path 0.27 m in
   15 s). It fired on **1 of 6 probe seeds, in both backends identically**, and
   **deterministically whenever the commanded target is
   within ~1° of dead ahead**, which is exactly what a human clicking "go
   there" produces. Bearing sweep, 3 m away, CPU drill, **static** target,
   400 steps (fitness = mean `exp(-0.5·d)`; torso height 0.35 m is the frozen
   pose, ~0.42-0.51 m is walking):

   | bearing | 0° | 0.5° | 1° | 2° | 5° | 10° | 30° | 90° | 180° |
   |---|---|---|---|---|---|---|---|---|---|
   | fitness | **0.23** | **0.23** | 0.64 | 0.81 | 0.84 | 0.83 | 0.81 | 0.83 | 0.76 |
   | height  | 0.35 | 0.36 | 0.48 | 0.42 | 0.47 | 0.41 | 0.40 | 0.44 | 0.51 |

   The ant is bilaterally symmetric, so a target with zero lateral offset is a
   symmetric observation and the policy answers it with a symmetric,
   net-zero-thrust action. A static target is fine at every other bearing
   (0.64-0.84) — this is *not* "the drill only trained on moving targets".
   Mitigations measured at bearing 0°:

   | mitigation | fitness |
   |---|---|
   | none | 0.23 |
   | target drifts 0.05 m/s sideways | 0.23 (insufficient) |
   | target drifts 0.20 m/s sideways | 0.71 |
   | Gaussian action noise σ=0.05 | 0.23 (insufficient) |
   | Gaussian action noise σ=0.20 | 0.69 |
   | command the bearing 2° off instead | 0.81 |

   **WS3/WS4 action** (cheapest and best of the three): `SkillController` must
   never hand the follow expert a target with `|target_ego_y| ≈ 0` — rotate the
   commanded bearing by ~3° when it is under that. 3° costs 5 cm of aim at 1 m,
   i.e. nothing. **Validated end to end in the real CPU soccer env** via
   `soccer_bridge`, ant commanded 3 m dead ahead:

   | | t=0 | 2 s | 4 s | 6 s | 15 s |
   |---|---|---|---|---|---|
   | no dither | 2.99 m | 2.94 | 2.94 | 2.94 | 2.94 (frozen, h 0.35) |
   | 3° dither | 2.99 m | 0.58 | 0.11 | 0.02 | 0.17 (arrived, holds) |

   **WS1 action**: worth one retrain arm with stationary targets in the
   curriculum; today's checkpoint is usable *with* the dither and unusable
   without it.
4. Note for `play_server.py`: it already pins `env.target_vel[0] = 0.0` (a
   static target). That is safe for the worm and unsafe for the ant — see 3.

Repro: the probe scripts are the ones described above; the CPU/warp
observation-parity assertion is `tests/integration_smoke.py --warp`.

## Open question (needs answer before launch)

1. **Second GPU pod?** Single-GPU serializes dribble→kick→shoot (~3-5
   GPU-hours total — likely fine). A second pod (~$0.3-0.7/hr) would let
   kick/shoot train in parallel with dribble re-runs, compressing P2 by a day
   or two. Not required; queue is the default.
