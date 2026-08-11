# Direction 1 — Ant skills → human demos → BC → RL: the 2v2 video

*Written 2026-08-10. The continuation of the post-pivot sprint (STATUS_2026-08-08.md);
supersedes nothing, but pins the road ahead so we don't get lost.*

## Motivation

The club promised a 2v2 creature-soccer video three years ago. The NPMP/SMP style-
transfer track died (NPMP_SMP_POSTMORTEM.md); what survived the pivot is the
DeepMind humanoid-football recipe (Liu et al. 2022, Science Robotics): train drill
skills, collect human demonstrations, behavior-clone them into a prior, then
RL-fine-tune 2v2 self-play anchored to that prior. Motion style no longer matters —
task competence does. The ant is the validation body: every architecture dimension
(65 proprio / 8 act / z=16) matches the rower byte-for-byte, so everything ports by
swapping the XML.

## Goal

A watchable 2v2 creature-soccer match — first with ants, then with the evolved
creatures (rower + worm) — driven by policies that learned from human play.
Deliverable: the video, plus the pipeline that made it.

## Approach

DeepMind's pipeline, one shared frozen decoder per body:

1. **Drills** (DONE, training deeper now): follow / dribble / kick / shoot, all on
   `_decoder_ant_final.pt`. Checkpoints + GCS links in the skills registry.
2. **Human demos**: the 2v2 game server records per tick: exact expert obs vector,
   z latent, action, skill label, target — already verified bit-exact-replayable
   (`replay.py --mode controller`).
3. **BC**: train a policy `game obs → (skill, target)` (high level) and/or
   `obs → z` (low level, reusing the frozen decoder) from the demo corpus.
4. **RL fine-tune**: 2v2 self-play in the warp soccer env, KL-anchored to the BC
   prior so play stays human-flavored while win-rate climbs.
5. **Creature swap**: retrain drills on rower/worm (same trainers, same registry),
   re-collect a small demo set, repeat 3-4. The 0.52 ball/torso proportion rule
   (scale the BALL and PITCH to the creature, never the creature to the ball)
   carries over.

## Implementation plan (parallelizable units)

| unit | depends on | status |
|---|---|---|
| 1a. Drill training to saturation (48 h ceiling, we interrupt) | — | RUNNING |
| 1b. Demo-collection sessions (4 humans, play_online.sh) | 1a good-enough | ready |
| 1c. BC dataset builder (demos → training tensors) | demo format (fixed) | can start NOW |
| 1d. BC trainer + eval harness | 1c | after 1c |
| 1e. Warp 2v2 self-play env (4 creatures + ball on one pitch) | — | BUILT (see below) |
| 1f. RL fine-tune w/ KL anchor | 1d + 1e | later |
| 1g. Creature swap (rower drills) | 1a validates recipe | later |

1c and 1e are independent of everything running and of each other — good agent-
sized chunks on separate branches/worktrees.

## Eval method

- **Drills**: per-skill warp fitness + the CPU-soccer transfer evals
  (`demo_follow_soccer` gate, `eval_dribble_soccer`) — both already scripted.
  Rule learned five times over: **watch the eval videos**; metrics lie, videos don't.
- **BC**: held-out demo action agreement + controller-replay determinism check +
  "does a BC-driven 2v2 look like play" (human eval).
- **RL**: win rate vs scripted baseline and vs BC-only; KL to prior as the
  style-drift gauge.
- **End**: the video itself, judged by humans.

## Unit 1e — the batched 2v2 self-play env (built 2026-08-11)

`rower_soccer/warp_port/soccer2v2_env.py` — `WarpSoccer2v2Env`. N parallel 2v2
matches: four ants and one drill ball (`BallSpec(radius=0.15, mass=0.045)`) on
the uniformly scaled dm_soccer pitch (`pitch_scale` 0.3125 → 30 × 22.5 m, goal
line |x| = 13.33, goal 7.4 m wide, crossbar 1.67 m), both goals present.

**Batching / interface.** Every tensor is flattened over (world, player), world
major: `obs [n*4, 99]`, `act [n*4, 8]`, `rew [n*4]`, and one `done` bool for the
whole batch — the drills' trainer contract, so a PPO loop needs no new plumbing.
`act_dim` is per player; the reshape to the model's 32 actuators *is* the slot
routing (actuators are creature-major in slot order). Slots are `match.py`'s:
`(home_1, home_2, away_1, away_2)`, MJCF prefixes `p0-`…`p3-`.

**Observation** (proprio-first, contiguous task block, 65 + 34 = 99 for the ant):

```
proprio(65) | ball_ego(6) | opp_goal_mid(3) | opp_post_left(2) | opp_post_right(2)
            | own_goal_mid(3) | teammate(3+3) | opp_a(3+3) | opp_b(3+3)
```

The first 13 task entries are `shoot`'s task block verbatim (with "the goal" =
the goal this team attacks), so a shoot checkpoint warm-starts the task encoder,
not just the decoder.

**Proprio is the drills' own function, not a copy.** `worm_env_base` now exposes
`proprio_index` / `proprio_obs` / `to_ego3` / `vec_to_ego3` as module-level
definitions; `WormEnv._proprio_obs` delegates to them and the 2v2 env calls them
per player. Verified byte-identical to the pre-refactor code on a `shoot`
rollout (5 steps × 2 worlds × 78 dims, `max|diff| = 0.0`), and `scene.py`'s
per-creature meta extraction is likewise one function (`creature_meta`) shared
by `build_creature_scene` and the new `build_soccer_scene`. As a second guard
on the refactor, the existing drill gate `tests/test_drill_v4.py` was re-run in
full (physics group included, on GPU): **15/15 PASS**.

`n_per_team` is a parameter — a 1v1 variant builds and steps (obs 87 = 65 + 22,
the team-mate block simply absent) — but only 2v2 is gated.

**Team symmetry.** The 180° rotation `M(x, y) = (-x, -y)` swaps the goals and is
a proper rotation (a reflection would swap the ant's left and right legs). Every
task entry is egocentric, so `M` leaves the numbers alone and the symmetry is
carried entirely by role assignment — which goal is `opp_goal`, which players
are teammate/opponents. `mirror_state()` / `mirror_actions()` make the transform
executable (also the free BC augmentation unit 1c wants).

**Rules**, transcribed from dm_control (which is what `match.py` reads): goal =
ball centre strictly inside the goal box, counted on a rising edge per world,
`-x` goal credits AWAY; after a goal `MultiturnTask` re-kicks-off that world and
does NOT terminate; out of play = outside dm_control's inverted `field`
detector → throw-in (`ball_xy * U[0.7, 0.9]`, velocity zeroed); the only thing
that ends an episode is the 45 s clock. Reward defaults to dm_soccer's
`get_reward` exactly (+1 scorers / −1 conceders); `w_player_to_ball` and
`w_ball_to_goal` shaping exist and default to **0**.

### The gate — `tests/test_soccer2v2.py`, 12/12 PASS

`MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m tests.test_soccer2v2 [--gpu]`.
Everything but the last check runs on the CPU MuJoCo backend (1–2 worlds).

| check | real result |
|---|---|
| **proprio BYTE-IDENTICAL to the drill's** (contact-free) | PASS — 65-wide proprio *and* the 6-wide ball_ego identical in all four slots, `max|diff|` exactly 0 vs a `shoot` env at the same creature state |
| proprio vs the drill in contact (measured) | PASS — also exactly 0 |
| obs lanes are per-creature, a slot swap is detectable | PASS — per-(world, slot) body-height marker + independently recomputed ball_ego; every pairwise swap asserted distinguishable |
| action lanes are per-creature | PASS — driven slot moves ≥ 8.7 rad/s, the other three ≤ 4.5e-19 (measured as a difference against a zero-action rollout) |
| mirrored state → mirrored obs | PASS — `max|diff| = 0.0` over 4 slots × 99 dims from a state stepped 10 steps away from the symmetric kickoff; without the slot swap the same comparison is 5.6 |
| mirrored state + mirrored action → mirrored step | PASS — dqpos 0.0, dqvel 0.0 after a full 10-substep control step |
| goal / off-court boxes vs dm_control's own detector | PASS — boxes equal to 1e-6 against a `Pitch` built at this geometry; 13 constructed positions (on the line, inside/outside the post, over the bar, on the floor, behind the wall, over the touchline) all agree with `PositionDetector._is_in_zone` |
| goals: rising edge + dm_soccer's ±1 | PASS — three steps with the ball in the goal score once, reward `[+1, +1, −1, −1]`, ball re-spotted |
| out of play → throw-in | PASS — shrink (0.752, 0.886) ∈ U[0.7, 0.9], ball velocity zeroed, players untouched |
| time limit | PASS — one `done`, at step 1800/45 s (tested at 40/1 s) |
| N worlds × M steps, finite, 0 diverged (cpu) | PASS |
| same on warp/GPU | PASS — 32 worlds × 200 steps, 0 diverged, obs (128, 99) finite, 340 world-steps/s on a card already running five drills |

Two of these failed on the first run and both were real: goals were counted
three times because the test held the ball in the goal while the re-spawn kept
re-arming the edge (fixed by adding `goal_respawn=False`, which is what makes
the latch observable at all), and the mirror was off by 0.03 in three obs
entries — the **accelerometer**, because `qacc` includes the actuator forces and
the mirrored state was being forwarded with un-permuted `ctrl`. `set_state` now
takes `ctrl`, and the mirror is bit-exact.

### The picture — `runs/soccer2v2_1e/{contact_sheet.png, clip.mp4}`

`python -m rower_soccer.warp_port.probe_soccer2v2 --out runs/soccer2v2_1e`.
Looked at: four ants, two per half at kickoff, on the textured pitch with both
goals and their posts, drifting sensibly under random torque; the ball sits on
the centre spot. The sixth panel is a deliberate close-up on the ball beside an
ant, because from overhead a correct 0.15 m ball is three pixels and "the ball
is the wrong size" is one of the two bugs this render exists to catch. The
close-up shows it about 60% of the ant's torso sphere, which is the intended
0.52-class proportion.

### What I could NOT verify

* **The scene and the CPU game do not share a goal.** The warp pitch scales
  dm_soccer uniformly, so the goal scales too; `match.py` builds a dm_control
  `RandomizedPitch` at `pitch_half=(15, 11)` and dm_control's goal DEPTH and
  HEIGHT are the absolute constant `_SIDE_WIDTH/2 = 2.667`, not a ratio. So the
  game's goal line is at |x| = 9.67 with a 5.33 m crossbar, against 13.33 and
  1.67 here; only the half-width rule (0.33 × pitch half-y) agrees. The gate
  therefore compares against dm_control's detector *at this scene's geometry*,
  which pins the RULE and not the pitch. Reconciling the two pitches is a real
  decision (it also affects `shoot`, which trains against the scaled goal) and
  it was out of scope for this unit. No `MatchSim` was instantiated.
* **No policy, no training run, and the throughput number does not scale.**
  340 world-steps/s at 32 worlds is ~3 ms per batched step, i.e. dominated by
  per-step launch cost on a card already running five drills and two CompetEvo
  jobs -- and it varied 201-340 between two runs of the same command, purely
  with the neighbours' load. It says "runs and does not diverge"; it says
  nothing about what this costs at the hundreds of worlds 1f will want, and
  nothing was measured there.
  `nconmax`/`njmax` default to 256/2048 (4× the drills'), sized from a measured
  peak of ~4 contacts / 33 constraints per world under random torque — margin,
  not a fit, and not stress-tested at hundreds of worlds.
* **Contact realism between four creatures is unmeasured.** The ants never
  reached the ball under random torque, so creature–creature and
  creature–ball–creature contacts have not been exercised by a competent
  policy; the warp/CPU contact-softness gap is unmeasured for this scene.
* The touch-sensor ordering keeps `worm_env_base`'s lexicographic sort rather
  than `scene.touch_slices`' numeric one. They agree for every creature that
  exists (≤ 9 single-digit segments) and byte-identity with trained checkpoints
  required keeping it; a 10+-segment creature would diverge.
* Episode length: 45 s = 1800 control steps per episode is the game's clock, not
  a claim that it is the right RL horizon for 1f.

## Notes / risks

- Shoot trained on the +x goal only; the −x mirror transform is still TODO in the
  game (documented in `fields._post_ego`).
- Shoot's fitness selects on accuracy only while its reward pays for power — fine
  so far, revisit if best.pt looks timid at saturation.
- Demo volume: DeepMind used ~hours of human play; we'll have far less. BC may need
  heavy augmentation (mirroring exploits the pitch symmetry) — design 1c with that in mind.
- GPU is shared with nothing (drills own it); CPU yields to Direction 2 on conflict.
