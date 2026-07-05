# WS1 — Low-Level Control (branch: `ws1-low-level`)

**Goal:** deliver `RowerController` and `WormController` — command-conditioned
policies that make each creature drivable via the Command interface
(`docs/CONTRACTS.md` §1–§3): W/S acceleration, A/D rotation, kick burst.

**Deliverable:** `checkpoints/low_level/rower/` and `.../worm/`, each
`policy.pt` + `meta.json`, loadable by `load_controller()` (CONTRACTS §2), plus
a rendered demo video (figure-8 command script + kicks) per creature.

## Assets you have

- Creatures: `creature_configs/two_arm_rower_blueprint.xml` (8 actuators,
  hand-tuned known-good gears) and `creature_configs/three_seg_worm.xml`
  (2 actuators, generated gears — free to retune via
  `rower_soccer/tools/unity2mujoco.py --gear-scale`, or edit XML directly;
  physics realism is explicitly not a goal, controllability is).
- Walker class: `creature.py: Creature` (repo root) with proprio observables.
- Env factory: `rower_soccer/envs/build.py`. Prior evidence the rower trains:
  April runs (see `trained_creatures/*/model_card.md`), PPO, 192 envs.
- `rower_soccer/render_video.py` for demo clips (MUJOCO_GL=egl).

## Suggested approach (adapt freely)

1. `rower_soccer/envs/low_level_task.py`: single creature + ball on pitch.
   Episode ~20s. Command resampled every 2–8s from a distribution favoring
   motion (forward-biased; include pure-rotation and coast segments). Kick
   trigger scheduled when near ball (place ball in path periodically).
2. Reward per 0.025s step, all terms
   command-conditioned:
   - locomotion: `a_cmd * v_forward` (yaw-frame forward velocity) +
     `r_cmd * yaw_rate`, each clipped/normalized; small penalty for lateral
     drift and for motion when `a_cmd==0 and r_cmd==0`.
   - kick: within the 0.4s burst window, reward ball speed gain along the
     creature's facing; small cost per accepted trigger (discourage spam).
   - regularizers: action smoothness (Δaction²), uprightness if relevant.
3. PPO (sb3 2.6.0 in `.venv`), MLP 256×256, SubprocVecEnv. **This pod
   throttles beyond ~8 concurrent env processes** — benchmark first
   (`python -m rower_soccer.bench_env`), consider 8 procs × longer, or a
   separate RunPod CPU pod. 1v0-creature-only env runs ~500 steps/s/core.
4. Train rower and worm as **separate runs of the same code** (different
   obs/act dims; `meta.json` records layout). Worm may need gear-scale
   iteration before it moves at all — check with random actions first.
5. Wrap checkpoints in `rower_soccer/controllers.py` implementations and
   verify `scripts/drive_demo.py` (write it): scripted command sequence →
   video. Tracking error < visually-obviously-responsive. Kick must visibly
   launch the ball.

## Gotchas

- The rower rotates by differential arm-rowing; W/S semantics = acceleration
  *intent*, not velocity tracking — reward velocity, let the gait emerge.
- Creature root orientation: "forward" = root frame +x after settling; verify
  visually before wiring the reward (render a clip with a constant W command
  and axis overlay; the worm's forward axis especially is an empirical fact).
- Keep the 0.1s command hold (4 low-level steps) in training too, so the
  controller sees realistic command dynamics.
- Don't touch `envs/commands.py` semantics (CONTRACT). Cooldown constants are
  tunable at env construction, not in the dataclass defaults.
