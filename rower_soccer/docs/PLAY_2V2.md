# The 2v2 LAN game and the demo format (WS4)

Four people, four devices, one wifi, one 45-second ant match — and a file that
records all four of them well enough to train on. The file is the point: the
match is the sprint's demo, the demo file is next sprint's BC dataset.

---

## 1. Run it

```bash
cd /workspace/utmist-vc2-phase2
MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m rower_soccer.game.server --port 8090
```

It prints the URL to hand out:

```
[game] ant 2v2 | pitch (15.0, 11.0) half-extents | skills ('follow', 'idle', 'scripted')
[game] open on any device on this wifi:  http://10.0.0.42:8090/
```

Each player opens that URL, types a name, taps a free seat (**home 1/2**,
**away 1/2**), and plays. Anyone can hit **start 45s match**. Extra people who
join without a seat are spectators — they see everything and can drive nothing.
Seats nobody claims are driven by the `scripted` chase-the-ball baseline, so the
match works with 1–4 humans.

| Input | Effect |
|---|---|
| drag on the pitch | target = where you released; aim = the drag direction |
| tap | target, zero aim |
| keys `1`–`5` | follow / dribble / kick / shoot / scripted |
| `Esc` | idle (zero torque) |

Skills with no trained checkpoint are greyed out and rejected by the server —
they are not silently downgraded (§4).

No display needed anywhere: the server renders offscreen with EGL and streams
MJPEG. If it runs on the pod rather than on the LAN, forward the port
(`ssh -L 8090:localhost:8090 pod`) and open `http://localhost:8090/`.

Flags worth knowing:

| Flag | Default | Why |
|---|---|---|
| `--pitch-half X Y` | `15 11` | half-extents in metres. The repo's stock pitch is `40 30` — an **80×60 m** field, which an ant crossing at ~0.5 m/s cannot traverse in 45 s. Recorded in the demo either way. |
| `--match-seconds` | `45` | |
| `--physics-dt` | `0.0025` | the dt the drills trained at. `0.005` is soccer's native dt and ~1.5× cheaper. |
| `--action-mode` | `auto` | see §4 |
| `--fill scripted\|idle` | `scripted` | who drives unclaimed seats |
| `--auto-start N` | `0` | start (and restart) automatically once N seats are claimed |
| `--demo-dir` | `demos` | `''` disables recording |
| `--render-hz` | `20` | frames, not ticks — the sim always runs at 40 Hz |
| `--torch-threads` | `1` | **do not raise this**, see §6 |

## 2. Prove it works without four humans

```bash
MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m rower_soccer.game.sim_client --selftest
```

Boots the server in-process, drives it with four scripted HTTP clients over the
*public* endpoints (no back door into the sim), and then verifies the demo the
match produced. It checks seat exclusivity, spectator lockout, reconnect, the
tick count, the labels, and all three replay modes.

Against an already-running server, four bots that claim seats and chase the ball:

```bash
PYTHONPATH=. .venv/bin/python -m rower_soccer.game.sim_client \
    --url http://localhost:8090 --seconds 45 --start
```

Unit tests (schema + lobby), no simulator, no pytest needed:

```bash
PYTHONPATH=. .venv/bin/python -m rower_soccer.game.tests.run_tests
```

## 3. Demo schema v1

One `.npz` per match. Numpy only — memory-mappable, and `obs[t, p]` is already
the tensor a BC dataloader wants. `meta_json` and `events_json` are JSON strings
stored as npz entries.

```bash
PYTHONPATH=. .venv/bin/python -m rower_soccer.game.recording demos/*.npz --events
```

**Per-tick arrays** (`T` ticks, `P` players, home first):

| name | dtype | shape | |
|---|---|---|---|
| `tick`, `t` | int64, float32 | `[T]` | control tick and sim time |
| `obs` | float32 | `[T,P,O]` | the dm_soccer per-player observation, flat, in `meta.obs_keys` order |
| `skill` | int8 | `[T,P]` | index into `meta.skill_vocab` — **the skill that ran** |
| `skill_req` | int8 | `[T,P]` | the skill that was asked for |
| `target` | float32 | `[T,P,2]` | world target actually used (for `scripted`, the ball) |
| `aim` | float32 | `[T,P,2]` | drag direction, unit or zero |
| `z` | float32 | `[T,P,Z]` | the latent the expert emitted; NaN where no expert ran |
| `action` | float32 | `[T,P,A]` | torques applied, in [-1, 1] |
| `skill_obs` | float32 | `[T,P,Om]` | the exact vector fed to the expert, NaN-padded |
| `skill_obs_n` | int16 | `[T,P]` | how much of it is real |
| `ctrl_tick` | int32 | `[T,P]` | the SkillController's own counter (§4) |
| `player_pos` | float32 | `[T,P,3]` | root world position |
| `player_mat` | float32 | `[T,P,9]` | root world rotation, row-major |
| `ball_pos`, `ball_vel` | float32 | `[T,3]` | |
| `score` | int16 | `[T,2]` | cumulative (home, away) |
| `qpos`, `qvel` | **float64** | `[T,Q]`, `[T,V]` | full physics state (§5) |

`player_pos` + `player_mat` are there because dm_soccer's observation
deliberately omits the root pose (`creature.py`'s `proprioception` drops it so
the shared decoder can never learn a position-dependent gait). Together with
`obs` they are exactly a `skills.PlayerFrame`, which is what lets a demo be
re-run through a `SkillController` with no simulator at all.

**`meta_json`** carries everything needed to interpret and reproduce the file:
schema version, match id, git sha, seed, `control_dt`/`physics_dt`, `pitch_half`,
per-player records (`slot`, `team`, `creature`, `controller` ∈ human/scripted/idle,
display name), `obs_keys`/`obs_sizes`, `skill_vocab`, `available_skills`,
per-skill field lists, checkpoint paths **with sha256**, the camera/affine, the
skill backend and action mode, and the RNG state (§5).

**`skill_vocab` is append-only.** `("idle", "follow", "dribble", "kick", "shoot",
"scripted")` — these are `rower_soccer.skills`' skill_ids. The indices are baked
into every file ever recorded, so reordering silently relabels historical data.

**Events** (`events_json`, each with `tick`, `t`, `type`): `match_start`,
`match_end`, `goal` (team, scorer), `ball_touch` (player, repossessed,
intercepted), `skill_change`, `target_set`, `slot_claim`, `slot_release`.

**Reading it:**

```python
from rower_soccer.game.recording import read_demo
d = read_demo("demos/x.demo.npz")
d.arrays["obs"].shape          # (T, 4, O)
d.obs_dict(0)["ball_ego_position"]      # player 0's obs, split by key
d.skill_names(0)[:5]           # ['follow', 'follow', ...]
d.events_of("goal")
pairs = d.bc_pairs(skills=["follow"])   # obs/target/skill/z/action/player
```

## 4. What is recorded is what happened

Two rules, both about not lying to next sprint's training run:

* **The skill recorded is the skill that ran.** Press `3` (kick) before WS1 has
  trained a kick expert and the tick records `idle` — never a `follow` tick
  wearing a kick label. The server also rejects the request with the list of
  available skills so the client can grey the button out.
* **`action_mode` and its inputs are recorded.** `follow_ant_v1` trained with
  `ent_ceil = 0`, so its action std sits at ~1.0 — the whole action range — and
  PPO scored the *sampled* policy. WS3's controller can therefore run either the
  distribution mean or `mean + std·ε` where ε is a pure function of
  `(seed, player_index, controller tick)`. Both are reproducible, but only if the
  demo stores all three, which is why `skill_seed` is in the header and
  `ctrl_tick` is a per-row array — the controller resets its counter on a skill
  switch, so it cannot be derived from the match tick.

## 5. Replay, three ways

```bash
MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m rower_soccer.game.replay \
    demos/x.demo.npz --mode state --video match.mp4
PYTHONPATH=. .venv/bin/python -m rower_soccer.game.replay demos/x.demo.npz --mode all
```

| mode | what it does | what it proves |
|---|---|---|
| `state` | writes recorded `qpos` into a fresh env and renders | the video. Exact by construction |
| `action` | rebuilds the env, restores the recorded start state, re-steps with the recorded **actions**, diffs `qpos` | the demo describes the match it claims to |
| `controller` | re-runs the skill layer over the recorded **observations** and diffs actions/z/obs | the demo carries everything BC needs, without a simulator |

Two things must be restored or `action` mode drifts, and neither is obvious:

* **The RNG state, at the first recorded tick.** dm_soccer's `Task.before_step`
  calls `_throw_in` — which draws from the env's `random_state` — every time the
  ball leaves the field, and `MultiturnTask.after_step` re-spawns everyone from
  it on a goal. The seed alone is not enough, because the server steps the sim
  during the lobby and the countdown too, so by kickoff the stream sits at an
  offset that depends on how long people took to join.
* **float64 `qpos`/`qvel`.** A legged creature in contact is chaotic: rounding
  the initial state to float32 (~1e-7 relative) grows to ~0.5 m of divergence
  inside one second of re-simulation. Measured, not assumed.

With both, `action` replay is bit-exact: `max|Δqpos| = 0.0` over a full match.
`controller` replay lands ~5e-7 on the action, which is the float32 storage of
the observations; the `max_obs_err` column compares against the stored
`skill_obs` and is the check that is exact.

## 6. Performance

Measured on the shared 48-core dev box, 2v2 ant, `pitch_half (9,7)`:

| | `physics_dt` 0.005 | `physics_dt` 0.0025 |
|---|---|---|
| tick (4 policies + physics) | ~8.6 ms | ~13 ms |
| render 960×640 | ~6.7 ms | ~6.7 ms |
| headroom in a 25 ms tick | ~2.9× realtime | ~1.9× realtime |

**`torch.set_num_threads(1)` is load-bearing.** With the default intra-op pool a
single 256-wide `Linear` takes 8 ms on this host and one 4-player tick takes
**536 ms — 20× slower than realtime**. Pinned to one thread the same tick is
13 ms. The experts are tiny MLPs; parallelising them is pure synchronisation
overhead. The server sets this itself; `--torch-threads` exists only to
re-measure it.

When the host is busy the loop **drops frames, never ticks**: the physics tick is
the authoritative state four humans and the demo depend on, a frame is only a
picture of it. `/state` reports `frames_dropped` and `late`.

## 7. How the pieces fit

```
browser  --POST /input {token, skill, u, v}-->  server (request thread)
                                                   | writes to a locked inbox
                                                   v
                        sim thread: seats -> control -> inputs -> MatchSim.step
                                                   |        |
                        SkillController per player -+        +-> DemoWriter row
                                                   v
                        physics.render -> JPEG -> /stream (MJPEG) -> browsers
```

* **One sim thread owns the GPU/EGL context, the env and the policies.** MuJoCo's
  EGL context must be created and used on one thread, and keeping the HTTP
  handlers away from it stops a request from stepping physics inline. Same
  reasoning as `warp_port/play_server.py`.
* **Input isolation is structural, not checked.** A client sends a *token*, never
  a slot id; the server derives the slot. There is no request a client can
  construct that reaches another player's creature — which matters more for the
  dataset than for fairness, because a cross-slot input would mislabel a demo row.
* **Reconnect is by token, not by socket.** A phone that sleeps and comes back
  resumes its seat; a seat silent for `--slot-timeout` seconds becomes claimable,
  but only when somebody actually asks for it.
* **Inputs apply at a tick boundary**, in the order seats → match control →
  inputs, so a claim that lands in the same tick as `start` is already visible
  when the demo header snapshots who is playing.
* **No flask.** It is not installed in the project venv; stdlib
  `ThreadingHTTPServer` serves the page and the MJPEG stream fine for 4–8
  clients, and the headline demo does not depend on a pip install.
* **A failing tick does not kill the server.** An exception inside `sim.step()`
  ends the match (keeping whatever it recorded), returns everyone to the lobby,
  and publishes the traceback on `/state.tick_error`. Without this the loop dies,
  HTTP keeps serving the last frame, and four people stare at a still picture
  wondering whose wifi broke.

## 8. Cross-workstream dependency: the accelerometer override

The game requires `creature.py`'s `CreatureObservables.sensors_accelerometer` to
apply the training-time transform (`raw / 100`, clipped to ±50). WS3 owns that
edit and it is the right fix: `warp_port/follow_env.py` applies it before the
accelerometer reaches the network ("any future body must apply the same scaling
at deployment — it is part of the obs contract"), while dm_control's base walker
returns the sensor raw, so without the override the CPU path feeds the same
policy a different observation — an obs bug that reads as a sim2sim gap.

Until it lands on this branch, the play server raises
`ObservationContractError: sensors_accelerometer ... exceeds the contract's clip`
on the first hard footfall. The server survives it (§ the tick guard below) and
returns to the lobby rather than freezing, but no match completes. Verified: with
WS3's pending `creature.py` in place, the full self test and a mixed
human/scripted match both pass.

## 9. Known gaps

* **dribble / kick / shoot** are registered in WS3's skill registry with no
  checkpoints, so they appear greyed out and are rejected until WS1 lands them.
  Adding one is `--dribble runs_v2/dribble_ant_v1/best.pt`, or a registry entry
  on WS3's side. No code change here.
* **`aim`** (the drag direction) is recorded on every tick but no current skill
  consumes it — `follow` only needs a point. It exists so kick/shoot have a
  direction to use the day they arrive, and so the demos recorded before then
  still carry it.
* **The ant is not in `envs/build.py`'s `CREATURE_XMLS`** on this branch; the
  game registers it at runtime (`match.register_ant`), which becomes a no-op the
  moment WS5's one-line entry lands.
* **dm_control's EGL teardown** prints an `EGL_BAD_ACCESS` traceback from an
  `atexit` handler when the process exits with a renderer alive on a non-main
  thread. Cosmetic, after all work is done, upstream.
