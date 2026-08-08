# Ant sprint — validate the pipeline end-to-end before the creature run

*Decided 2026-08-08. Scope: run the full pipeline on the MuJoCo Ant up to a
**playable 4-human LAN 2v2 soccer game** that records BC demos. Stop there,
evaluate, then swap in the rower/worm. Self-play training is explicitly out of
scope for this sprint. Supersedes the sprint order in
[STATUS_2026-08-08.md](STATUS_2026-08-08.md) (creature drills move to after
ant validation).*

## Why the ant

- Empirically fast to train; symmetric, statically stable, clear upright, can
  strafe without turning — every property that made the rower drills hard, the
  ant lacks.
- **Same 8-actuator count as the rower.** The whole warp stack sizes its obs
  contract from the creature; act_dim 8 means the eventual creature swap keeps
  every architecture dimension (decoder shapes, z-dim, task widths) identical.
  The validation genuinely de-risks the creature run.
- No mocap/reference needed (style is a non-goal per the pivot): follow trains
  the z-bottleneck directly, as the worm did.

## Architecture (paper-shaped, minus what the pivot made unnecessary)

Frozen-decoder skill chain, replacing DeepMind's distillation stage:

1. Train `follow` with the z-bottleneck ActorCritic (expert → z → decoder).
2. Freeze the decoder (+ action_net). Train `dribble`, `kick`, `shoot` as
   z-emitting experts on the SAME frozen decoder.
3. Shared z-space for all skills falls out by construction — no distillation.

Evidence this works for task performance: the NPMP ablation
([NPMP_SMP_POSTMORTEM.md](NPMP_SMP_POSTMORTEM.md)) — five frozen-decoder arms,
all fitness 0.92-0.97. Its only failure mode was style, which is dropped.
Fallback if the frozen decoder handicaps a ball skill (dribble history
suggests watching for this): per-task unfrozen policies + BC on actions
instead of z. Decide on evidence, not upfront.

## Steps and gates

### 0. Ant asset (~half a day)
Build `creature_configs/ant.xml` in our creature format (free-joint root,
8 hinge actuators, touch sensors + torso velocimeter/gyro/accelerometer, the
naming conventions `warp_port/scene.py` and `creature.py` expect — crib from
`three_seg_worm.xml`). Standard MuJoCo ant geometry, gear scaled via the
torque-margin logic learned from the gear_scale bug.
**Gate:** loads in both the warp scene builder and the CPU `Creature` walker;
`probe_speed` shows it walks under random torque without divergence.

### 1. Ant `follow` + decoder (~1 GPU-hour + eval)
`train_follow_warp` unchanged, `--creature-xml creature_configs/ant.xml`.
**Gate:** fitness ≥ 0.9; eval video shows purposeful locomotion to a moving
target. Export decoder (run-scoped; `--publish-decoder` equivalent flow).

### 2. Ant `dribble` on the frozen decoder (budget: days)
`train_dribble_warp --init-from <ant decoder> --freeze-decoder`. This is the
step the rower fought hardest; the ant's stability is the bet.
**Gate:** dribbles ball toward a target region reliably in eval video.

### 3. `kick` + `shoot` envs + training (~2-4 days)
New env variants on dribble's ball machinery (`dribble_env.py`, `scene.py`):
- `kick`: impart velocity to ball toward target direction; episode ends on
  contact+separation; reward = ball speed toward target.
- `shoot`: kick specialization with goal geometry + scoring termination.
**Gate:** per-skill eval videos look like the named skill.

### 4. Playable 2v2 — 4-human LAN multiplayer (~3-5 days, the new build)
CPU dm_control soccer env (`envs/build.py` + `soccer_bridge.py` machinery —
realtime 1-env throughput is trivial; **no warp 2v2 needed for this
milestone**), ant XML added to `CREATURE_XMLS`.

Extend `play_server.py`:
- One authoritative sim process ticking at control rate; 4 browser clients
  over LAN (players on shared wifi; server on a laptop, or port-forwarded pod
  — decide by measured input latency, laptop CPU is sufficient for 1 env).
- Lobby: client claims a player slot (home/away × 1/2); spectator = extra.
- Controls per PIPELINE_V2 stage 4: click-drag target + skill keys
  (follow/dribble/kick/shoot) — inputs drive the high-level (which expert runs
  + its target), experts drive z, frozen decoder drives the body.
- **Demo recording**: per player per tick — obs, active skill, skill target,
  emitted z, action; plus game events (touches, goals). This is the BC
  dataset format; versioned from day one.

**Gate (= sprint milestone):** 4 humans on LAN play a 45-s 2v2 ant match that
feels controllable; a demo file records all four players and replays.

## Explicitly out of scope
- Self-play training (and therefore the warp 2v2 env)
- BC training itself (data format must be ready; training is the next sprint)
- Any style/gait constraint
- Creature (rower/worm) versions of any of this — after the milestone

## Budget
~1.5-2 weeks calendar at current pace; GPU cost trivially small (<$20-class
on the current pod — drills are ~1 GPU-hour each at 38-53k fps). The
engineering weight is step 4's multiplayer server, then step 3's envs.
