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

- [ ] ant in `CREATURE_XMLS`; 4-ant soccer env steps ≥ realtime on target host
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

**WS5 owns an early gap probe** (do this in P1, before WS4 commits to a
backend): run `follow_ant_v1/best.pt` deterministically in the warp follow env
and in the CPU drill/soccer path from matched initial states; report
divergence over 15 s and fitness in both. Then:

- gap small → CPU dm_control game as planned (simplest, laptop-hostable).
- gap large → escalate, in order of cost: (a) stiffen/soften the CPU side's
  solref toward warp's effective contact behaviour, (b) brief CPU-side
  finetune of the drill policies, (c) build a warp 2v2 game backend (the
  expensive option this sprint was scoped to avoid — and which self-play will
  need eventually anyway, so it is not wasted work, just not now).

## Open question (needs answer before launch)

1. **Second GPU pod?** Single-GPU serializes dribble→kick→shoot (~3-5
   GPU-hours total — likely fine). A second pod (~$0.3-0.7/hr) would let
   kick/shoot train in parallel with dribble re-runs, compressing P2 by a day
   or two. Not required; queue is the default.
