# Interactive Multi-Model Human-Control Mode

**Status:** implemented; the drill scene and obs plumbing are verified in-tree against the current
obs contract (§3, §7). Two blockers stand between this tree and a live model-driven run:

1. **No in-repo checkpoint matches the current contract.** `runs_v2/follow_drill_model.pt` predates
   the proprio swap (obs 41 vs current 33) and is rejected by the loader's dimension guard.
2. **`--env soccer` currently crashes** in `soccer_bridge`: the soccer per-player obs no longer
   carries `absolute_root_pos`/`absolute_root_mat`, which the bridge reads to synthesize
   `target_ego`. See §8.

The pygame window on a physical display and a one-time click→world calibration also remain
unexercised.

Report covering `rower_soccer/play_interactive.py` — an inference-only mode where one creature stands
on an effectively infinite episode and a human drives it like a game character, switching trained skill
policies on the fly. It runs on **two scenes**, selected with `--env`:

- **`--env drill`** (default): the flat-floor drill scene (`InteractivePlayTask`).
- **`--env soccer`**: the real soccer pitch (`RandomizedPitch`, 40×30), driven by the same
  drill-trained **follow** model through the `soccer_bridge` observation adapter. See
  [`SOCCER_BRIDGE.md`](SOCCER_BRIDGE.md) for the drill↔soccer obs mapping. **Currently broken**
  (§8).

---

## 1. What it does

| Input | Effect |
|---|---|
| **Q** then left-click | Arm FOLLOW; the click sets a destination and the **follow** policy drives the creature there |
| **W** then left-click | Arm DRIBBLE; the click dribbles the ball there — **drill env only** (inert on soccer, and while `ENABLE_DRIBBLE=False`) |
| left-click (already active) | Retarget the current skill to the new point |
| **ESC** | Quit |
| *(no command yet)* | Creature sits still (zero action) |

The program **never stops on its own**. After a command completes, the active policy keeps running and
the creature simply holds at the target until the next command — exactly as requested.

---

## 2. Files changed

| File | Change |
|---|---|
| `rower_soccer/play_interactive.py` | The whole feature: task subclass, env factory, obs/camera/unprojection helpers, `.pt` loader, the `DrillScene`/`SoccerScene` adapters, and the pygame control loop. |
| `rower_soccer/soccer_bridge.py` | **Reused** by `SoccerScene` (`reference_follow_layout`, `soccer_to_drill_follow_dict`, `drill_follow_obs`) to reconstruct the drill FOLLOW vector from soccer obs. |

No existing modules were modified. `play_interactive` reuses the drill/creature/soccer/model machinery
read-only. It is a self-contained, multi-scene, human-driven inference mode. The pygame event/render
loop is env-agnostic: each scene adapter hides the env-specific plumbing (obs keying, action wrapping,
target marker, camera), so the same loop drives both the drill floor and the soccer pitch.

Environment note (not a code change): the drill envs require `mujoco==3.1.3` (a newer `mujoco 3.10.0`
is incompatible with `dm-control 1.0.16` — fails on `flex_xvert0`). The project `.venv` is already on
`mujoco 3.1.3`, so no realignment was needed here.

---

## 3. Key design insight — why one scene serves both skills

`DribbleTask` **subclasses `FollowTask`** (`rower_soccer/drills/dribble.py:23`) and adds only a
`SoccerBall` + a `ball_ego` observable. Both skills "aim at" the task's internal `self._target_xy`.

Therefore:

- **One shared scene** (the dribble scene: floor + target marker + ball + rower) drives both skills.
- **Switching skill ≡ switching which model runs and which observation vector we build.** The physics
  scene never changes — no rebuild, no reset.
- Each trained model carries its own `proprio_indices` / `task_indices` (baked into `policy_kwargs`
  at train time), so we only feed each model a flat obs vector in the **same sorted-key order it
  trained on**; the model slices proprio-vs-task internally.

### Observation layout (verified on the current code)

Keys are sorted; proprio keys are `creature/`-prefixed, task keys are not.

| Skill | Keys | Dim (worm) | Dim (rower) | Difference |
|---|---|---|---|---|
| FOLLOW | 11 (all except `ball_ego`) | **33** | **69** | — |
| DRIBBLE | 12 (all) | **39** | **75** | `+ ball_ego` (6) |

```
sorted order: ball_ego, creature/bodies_pos, creature/body_height, creature/joints_pos,
              creature/joints_vel, creature/sensors_accelerometer, creature/sensors_gyro,
              creature/sensors_velocimeter, creature/touch_sensors, creature/world_zaxis,
              target_ego, target_ego_future
```

The live scene emits all 12 keys; the FOLLOW vector is exactly the 11-key subset, which is identical
to the standalone follow env (which has no ball) — so trained weights transfer unchanged.

Worm FOLLOW split within the flat vector: proprio indices 0..28 (29 dims), task indices 29..32
(`target_ego` 2 + `target_ego_future` 2). Rower: proprio 65 / task 4 (FOLLOW), task 10 (DRIBBLE).

### Contract change (supersedes the old 41/45 worm, 77/81 rower layout)

`creature.py`'s `proprioception` now returns `world_zaxis` (3) + `body_height` (1) where it used to
return `absolute_root_mat` (9) + `absolute_root_pos` (3) — proprio 37 → 29 dims. Rationale (see the
comment at `creature.py:265`): proprio is the shared low-level controller's entire input contract, so
it may only contain features that (a) exist in the dm_soccer 2v2 per-player obs and (b) are invariant
to where on the pitch the creature is; the absolute root pose fails both tests.

`ball_ego` also grew 4 → 6 dims: 3-D egocentric position + 3-D egocentric velocity, matching the 2v2
game's `ball_ego_position` + `ball_ego_linear_velocity`. The old 2-D obs additionally reported the
velocity in the **world** frame, breaking egocentric invariance.

Checkpoints exported before this change (e.g. `runs_v2/follow_drill_model.pt`, obs 41) cannot be used
or index-remapped: the features themselves differ, not just their order or count.

---

## 4. Architecture

```
                    ┌─────────────────────────────────────────────┐
                    │            main() control loop (40 Hz)       │
                    │                                              │
   keyboard/mouse   │   pygame events                              │
   ───────────────► │     Q/W  -> armed skill                      │
                    │     click -> pixel_to_world() -> world (x,y)  │
                    │              env.task.set_command_target()    │
                    │              armed -> active                  │
                    │                                              │
                    │   per step:                                  │
                    │     obs_vec = build_obs(obs, KEYS[active])    │
                    │     action  = models[active].predict(obs_vec) │  ← FOLLOW/DRIBBLE PPO (LatentActorCriticPolicy)
                    │              (or zero_action if idle)         │
                    │     ts = env.step(action);  obs = ts.obs      │
                    │     frame = physics.render('topdown')         │
                    │     blit -> pygame window -> flip -> tick(40)  │
                    └───────────────────┬──────────────────────────┘
                                        │
                    ┌───────────────────▼──────────────────────────┐
                    │   InteractivePlayTask(DribbleTask)            │
                    │     • floor + target marker + ball + rower    │
                    │     • 'topdown' camera (pixel↔world mapping)   │
                    │     • time_limit = 1e9  (never truncates)     │
                    │     • set_command_target(): _target_xy = xy,  │
                    │       _target_vel = 0 (static), move marker   │
                    └──────────────────────────────────────────────┘
```

The diagram shows the **drill** scene. `--env soccer` swaps the bottom box for `SoccerScene`
(`make_soccer_env` + `soccer_bridge` obs adapter + a `topdown` camera/marker added to the pitch); the
`main()` loop above is unchanged because both scenes share one adapter interface. `obs_vec` is built by
`scene.obs_vector(obs, active)` (drill: `build_obs`; soccer: `soccer_bridge.drill_follow_obs`), and
`step`/`render`/`set_target` are scene methods.

### Components in `play_interactive.py`

| Symbol | Role |
|---|---|
| `InteractivePlayTask(DribbleTask)` | Adapts the dribble scene: adds a straight-down `topdown` camera (fovy derived from `cam_height`/`view_half`), sets an infinite episode, and provides `set_command_target()` for a **static** target (`_target_vel = 0`, so the inherited `after_step` leaves it fixed and `target_ego_future == target_ego`). |
| `make_interactive_env()` | Mirror of `make_dribble_env` but instantiating `InteractivePlayTask`. |
| `build_obs(obs_dict, keys)` | Flatten selected observables in sorted-key order — the same contract as `drills/gym_wrap.py:44-46` (`DrillGymEnv._flatten`), restricted to the active skill's key list. |
| `resolve_topdown_camera(physics)` | Finds the camera id whose name ends in `topdown` (handles model-prefixing). |
| `pixel_to_world(px, py, W, H, view_half)` | Affine unprojection for the straight-down camera: center→origin, corners→±`view_half`. |
| `DrillScene` | Scene adapter over `InteractivePlayTask`: `obs = ts.observation`, `step(action)`, `set_command_target`, `build_obs`. FOLLOW (+ DRIBBLE when `ENABLE_DRIBBLE`). |
| `SoccerScene` | Scene adapter over `make_soccer_env(home_team=(creature,), n_away=0)`. Adds a `topdown` camera + `play_target` marker geom to `task.arena.mjcf_model.worldbody` before reset; `obs = ts.observation[0]`; `step([action])`; `set_target` moves the marker via `physics.bind(marker).pos` and builds the obs vector with `soccer_bridge.drill_follow_obs`. FOLLOW only. |
| `main()` | Picks the scene from `--env`, loads one policy per enabled skill via the scene's `policy_layout`, runs the env-agnostic pygame loop. Heavy imports (`stable_baselines3`, `pygame`) are lazy. |

The two scenes share one interface (`init_obs`, `cam_id`, `act_dim`, `zero_action`, `control_hz`,
`view_half`, `skills`; `obs_vector`, `policy_layout`, `set_target`, `step`, `render`), so the loop in
`main()` never branches on the env.

---

## 5. Key design decisions

1. **Static target instead of moving target.** The human's click sets a fixed point; `_target_vel = 0`
   makes it stationary and collapses `target_ego_future` onto `target_ego`. (Mildly out-of-distribution
   for FOLLOW, which trained on targets moving 0.03–0.21 m/s — a stopped target ≈ a very slow one, so
   the policy should still converge to it.)
2. **Ball always present** (your choice). One compiled scene; FOLLOW ignores the ball in its obs but
   may physically nudge it. DRIBBLE fetches the ball from wherever it currently rests.
3. **Infinite episode** via `time_limit = 1e9` s; we drive our own loop and never reset, so the rower
   persists across commands.
4. **Idle = zero action** (clipped into the actuator ctrl range). Assumes zero torque settles the
   rower; swap for a neutral hold-pose if it collapses.
5. **Top-down camera for exact pixel→world.** A straight-down perspective camera makes the floor-plane
   unprojection a simple affine, avoiding version-fragile `engine.Camera` matrix internals.
6. **Local pygame window** (your choice). Creature frames render offscreen (EGL/GPU); pygame/SDL
   provides the visible window independently, so it works wherever there's a display.

---

## 6. How to run

Requires a machine **with a display**, plus `stable_baselines3` + `torch` + `pygame`.

**Current temporary setup (dribble disabled, worm follow).** The defaults are `--creature worm
--follow-model runs_v2/follow_drill_model.pt` with dribble off via the `ENABLE_DRIBBLE = False`
toggle in `play_interactive.py` — **but that checkpoint predates the obs-contract change (§3)**: it is
obs 41 where the live worm scene is obs 33, so the loader rejects it and the run stops at startup with
a dimension-mismatch error. Running end-to-end again requires a worm follow export trained against the
current contract (the Warp training path, `warp_port/follow_env.py`, is kept in lockstep with
`creature.py`):

```bash
MUJOCO_GL=egl .venv/bin/python -m rower_soccer.play_interactive \
    --follow-model <current-contract worm follow .pt>
```

Only the **Q** (follow) command is live; **W** is inert until `ENABLE_DRIBBLE` is flipped back on and
a dribble checkpoint is passed via `--dribble-model`.

**Play on the soccer pitch** — currently **blocked**: `--env soccer` crashes in `soccer_bridge`
before the first step (§8). Once the bridge is fixed, the same worm follow model is driven through
`soccer_bridge`:

```bash
MUJOCO_GL=egl .venv/bin/python -m rower_soccer.play_interactive --env soccer
# worm on RandomizedPitch (40×30); Q + click steers it; W is inert (dribble N/A on soccer)
```

Soccer notes: the top-down camera defaults to a wider frame (`--cam-height 32 --view-half 22`) to fit
the pitch; the ball, walls, and goalposts are physically present but ignored. **Shadows/MSAA are
disabled by default** — the `RandomizedPitch` ships 4 lights each rendering an 8192×8192 shadowmap
every frame (~100 ms/frame, a fixed cost independent of window size that capped the UI near ~9 FPS);
turning them off is purely cosmetic and drops the frame to ~11 ms (smooth 40 Hz). Pass `--shadows` to
restore them (e.g. for recording). The soccer physics
timestep is **matched to the drill's 0.0025 by default** — soccer's native Task uses 0.005 (5
substeps), but the follow policy trained at 0.0025 (10 substeps), so we set the pitch to 0.0025 to feed
the creature the same integration it was optimized on (control rate is 40 Hz either way). Pass
`--no-match-physics-dt` to keep soccer's native 0.005. The follow policy is egocentric, so it homes
onto reachable clicks; targets tens of metres away (or across a wall) are out-of-distribution and may
not converge.

Once trained **rower** weights exist, run e.g.:

```bash
MUJOCO_GL=egl .venv/bin/python -m rower_soccer.play_interactive \
    --creature rower --follow-model runs_v2/follow_rower.pt
```

The follow loader validates the checkpoint's dims against the live scene and errors clearly on a
mismatch (obs=33/act=2 → worm, obs=69/act=8 → rower), so a wrong `--creature` fails fast.

Useful flags: `--window` (square render/window px, default 800), `--cam-height` / `--view-half`
(top-down framing), `--device {cuda,cpu}`.

---

## 7. Verification status

Verified in-tree at HEAD against the current obs contract:

- ✅ Drill env builds; `topdown` camera resolves (id 1, name `topdown`).
- ✅ Obs dims (worm): FOLLOW **33** (proprio 29, idx 0..28 / task 4, idx 29..32), DRIBBLE **39**
  (proprio 29 / task 10). Rower: FOLLOW **69** / DRIBBLE **75** (proprio 65 / task 4 or 10).
  FOLLOW ⊂ DRIBBLE by exactly the 6-dim `ball_ego`.
- ✅ Static target holds: set `[5,-3]` → stays `[5,-3]` with zero velocity across steps, and
  `target_ego_future == target_ego`.
- ✅ `pixel_to_world`: center → origin, corners → ±view_half.
- ✅ Offscreen render returns frames (EGL), drill scene.
- ✅ Loader guard: `runs_v2/follow_drill_model.pt` (obs 41/act 2, pre-contract-change) is rejected
  with a clear dimension-mismatch error against the live worm scene (obs 33/act 2).
- ❌ `--env soccer`: `SoccerScene.policy_layout` raises `KeyError: 'absolute_root_pos'`. The bridge
  synthesizes `target_ego` from `absolute_root_pos`/`absolute_root_mat`, which left the soccer
  per-player obs with the proprio swap — they were never dm_soccer observations; they previously
  leaked into the soccer obs through the walker's proprioception set. Scene construction itself
  (env build, `topdown` camera resolution) still succeeds; the crash is in the obs path.

Still unexercised (needs a physical display, and trained current-contract weights):

- ⬜ `model.predict` end-to-end with any current-contract checkpoint. (The path was verified at
  `9ec2f87` with the 41-dim worm follow export, which the contract change has since invalidated.)
- ⬜ The pygame window loop on a live display (keyboard/mouse interaction).
- ⬜ One-time calibration of click→world axis signs against a visible landmark.

---

## 8. Open items / caveats

- **No compatible checkpoints yet.** `runs_v2/follow_drill_model.pt` (worm follow, obs 41) predates
  the proprio contract change and fails the loader's dimension guard; legacy
  `trained_creatures/*.zip` use the old policy class + obs and will not load. The mode is blocked on
  training/exporting follow (and later dribble) weights against the current contract.
- **Soccer bridge root-pose source.** `soccer_to_drill_follow_dict` reads `absolute_root_pos` /
  `absolute_root_mat` from the soccer per-player obs to compute `target_ego`; those keys no longer
  exist there. The root pose must instead come from the physics state directly
  (`physics.bind(walker.root_body).xpos` / `.xmat` — the scene holds `env.physics`, and the 2v2 game
  never exposed global pose in obs anyway). Until that is fixed, `--env soccer` cannot run.
- **Click→world sign check.** Verify a click lands the target marker under the cursor; flip signs in
  `pixel_to_world` if mirrored.
- **Idle behavior.** Confirm zero action doesn't make the rower collapse oddly.
