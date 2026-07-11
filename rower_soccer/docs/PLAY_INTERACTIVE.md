# Interactive Multi-Model Human-Control Mode

**Status:** implemented and verified in-tree against the real dm_control + SB3 stack. The full
inference path (obs plumbing → `model.predict` with the custom `LatentActorCriticPolicy` → env step
→ top-down render → pygame blit) is exercised end-to-end; only the live pygame window on a physical
display and a one-time click→world calibration remain unexercised. No trained rower follow/dribble
checkpoints exist in-repo yet, so a real model-driven run is still pending those weights.

Report covering the new `rower_soccer/play_interactive.py` — an inference-only mode where one rower
stands in the drill arena on an effectively infinite episode and a human drives it like a game
character, switching trained skill policies on the fly.

---

## 1. What it does

| Input | Effect |
|---|---|
| **Q** then left-click | Arm FOLLOW; the click sets a destination and the **follow** policy drives the rower there |
| **W** then left-click | Arm DRIBBLE; the click sets a destination and the **dribble** policy dribbles the ball there |
| left-click (already active) | Retarget the current skill to the new point |
| **ESC** | Quit |
| *(no command yet)* | Rower sits still (zero action) |

The program **never stops on its own**. After a command completes, the active policy keeps running and
the rower simply holds at the target until the next command — exactly as requested.

---

## 2. Files changed

| File | Change |
|---|---|
| `rower_soccer/play_interactive.py` | **New.** The entire feature (task subclass, env factory, obs builder, camera/unprojection helpers, pygame control loop). |

No existing modules were modified. `play_interactive` reuses the drill/creature/model machinery
read-only. It is a self-contained, multi-model, infinite-episode, human-driven inference mode.

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

### Observation layout (verified on the rower)

Keys are sorted; proprio keys are `creature/`-prefixed, task keys are not.

| Skill | Keys | Dim (rower) | Difference |
|---|---|---|---|
| FOLLOW | 11 (all except `ball_ego`) | **77** | — |
| DRIBBLE | 12 (all) | **81** | `+ ball_ego` (4) |

```
sorted order: ball_ego, creature/absolute_root_mat, creature/absolute_root_pos,
              creature/bodies_pos, creature/joints_pos, creature/joints_vel,
              creature/sensors_accelerometer, creature/sensors_gyro,
              creature/sensors_velocimeter, creature/touch_sensors,
              target_ego, target_ego_future
```

The live scene emits all 12 keys; the FOLLOW vector is exactly the 11-key subset, which is identical
to the standalone follow env (which has no ball) — so trained weights transfer unchanged.

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

### Components in `play_interactive.py`

| Symbol | Role |
|---|---|
| `InteractivePlayTask(DribbleTask)` | Adapts the dribble scene: adds a straight-down `topdown` camera (fovy derived from `cam_height`/`view_half`), sets an infinite episode, and provides `set_command_target()` for a **static** target (`_target_vel = 0`, so the inherited `after_step` leaves it fixed and `target_ego_future == target_ego`). |
| `make_interactive_env()` | Mirror of `make_dribble_env` but instantiating `InteractivePlayTask`. |
| `build_obs(obs_dict, keys)` | Flatten selected observables in sorted-key order — the same contract as `drills/gym_wrap.py:44-46` (`DrillGymEnv._flatten`), restricted to the active skill's key list. |
| `resolve_topdown_camera(physics)` | Finds the camera id whose name ends in `topdown` (handles model-prefixing). |
| `pixel_to_world(px, py, W, H, view_half)` | Affine unprojection for the straight-down camera: center→origin, corners→±`view_half`. |
| `main()` | Loads both checkpoints once, computes the two key layouts off the live scene, runs the pygame loop. Heavy imports (`stable_baselines3`, `pygame`) are lazy so the module's env/task logic is importable without them. |

---

## 5. Key design decisions

1. **Static target instead of moving target.** The human's click sets a fixed point; `_target_vel = 0`
   makes it stationary and collapses `target_ego_future` onto `target_ego`. (Mildly out-of-distribution
   for FOLLOW, which trained on targets moving 0.25–2.0 m/s — a stopped target ≈ a very slow one, so
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

**Current temporary setup (dribble disabled, worm follow).** The only trained checkpoint in-repo is
`runs_v2/follow_drill_model.pt` — a **worm** follow policy (obs 41, act 2), exported in the
`warp_port.ppo.export_sb3_compatible` layout and loaded via `load_into_sb3_policy`. Dribble is off
via the `ENABLE_DRIBBLE = False` toggle in `play_interactive.py`. Defaults match this, so:

```bash
MUJOCO_GL=egl .venv/bin/python -m rower_soccer.play_interactive
# == --creature worm --follow-model runs_v2/follow_drill_model.pt
```

Only the **Q** (follow) command is live; **W** is inert until `ENABLE_DRIBBLE` is flipped back on and
a dribble checkpoint is passed via `--dribble-model`.

Once trained **rower** weights exist, run e.g.:

```bash
MUJOCO_GL=egl .venv/bin/python -m rower_soccer.play_interactive \
    --creature rower --follow-model runs_v2/follow_rower.pt
```

The follow loader validates the checkpoint's dims against the live scene and errors clearly on a
mismatch (obs=41/act=2 → worm, obs=77/act=8 → rower), so a wrong `--creature` fails fast.

Useful flags: `--window` (square render/window px, default 800), `--cam-height` / `--view-half`
(top-down framing), `--device {cuda,cpu}`.

---

## 7. Verification status

Verified in-tree against the real dm_control + SB3 stack:

- ✅ Env builds; `topdown` camera resolves (id 1, name `topdown`).
- ✅ Obs dims: FOLLOW **77**, DRIBBLE **81** (differ by exactly the 4-dim `ball_ego`).
- ✅ Static target holds: set `[5,-3]` → stays `[5,-3]` with zero velocity across steps, and
  `target_ego_future == target_ego`.
- ✅ `pixel_to_world`: center → origin, corners → ±view_half.
- ✅ Offscreen render returns frames (EGL).
- ✅ `model.predict` path end-to-end with the actual `LatentActorCriticPolicy`: the real
  `runs_v2/follow_drill_model.pt` (worm follow) loads via `load_into_sb3_policy` and produces
  in-range 2-dim actions fed from `build_obs`. A rower-scene load of the same file is correctly
  rejected with a dimension-mismatch error.
- ✅ pygame surface path: `physics.render` frame → `surfarray.make_surface` → blit (headless SDL).

Still unexercised (needs a physical display, and trained rower weights for the rower path):

- ⬜ `model.predict` with trained *rower* follow/dribble weights (worm follow is the only real net).
- ⬜ The pygame window loop on a live display (keyboard/mouse interaction).
- ⬜ One-time calibration of click→world axis signs against a visible landmark.

---

## 8. Open items / caveats

- **No compatible checkpoints yet.** `runs_v2/` is empty; legacy `trained_creatures/*.zip` use the old
  policy class + obs and will not load. The mode is blocked on training rower follow/dribble weights.
- **Click→world sign check.** Verify a click lands the target marker under the cursor; flip signs in
  `pixel_to_world` if mirrored.
- **Idle behavior.** Confirm zero action doesn't make the rower collapse oddly.
