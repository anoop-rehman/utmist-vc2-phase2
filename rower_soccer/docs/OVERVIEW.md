# Rower Soccer — Project Overview

**Goal:** a compelling video of heterogeneous creature teams playing 2v2 soccer in the
dm_control soccer environment, produced in days on minimal compute.

**Teams (fixed, heterogeneous):** each team = 1 **two-arm rower** (attacker) + 1
**three-segment worm** (defender/goalkeeper). Both are GA-evolved Unity creatures already
converted to MuJoCo XML (`creature_configs/`).

## Method (mini-AlphaStar, 3-stage pipeline)

1. **Low-level control (WS1):** per-morphology command-conditioned RL controllers make each
   creature drivable like a game character: `W/S` = forward/back acceleration, `A/D` = CCW/CW
   rotation, `Space` = kick burst (discrete, ~2s cooldown — like the BoxHead "jump" channel
   in DeepMind's 2019 soccer paper).
2. **Human demonstrations → BC (WS2):** a browser-based online-multiplayer soccer game
   (server-authoritative sim streamed from this pod, up to 4 humans in their own browsers,
   optional slow-motion factor) records trajectories; behavior cloning distills human tactics
   (passing, defending) into per-role policies. Humans supply the tactics because RL at our
   scale converges to tactic-less ball-chasing (see Samtani et al. 2021 vs Liu et al. 2019).
3. **RL finetune (WS3):** PPO self-play at the command level, initialized from BC, with a
   KL-anchor to the BC policy (AlphaStar trick — prevents RL from erasing human-like play)
   and an opponent pool of past checkpoints (poor-man's league). Final checkpoint selected
   by *watching footage*, not win rate.

## Why this design (1-paragraph history)

DeepMind 2019 ("Emergent Coordination Through Competition") got passing/defense to *emerge*
from population-based co-play — at 40–80B frames. Curriculum shortcuts (Samtani 2021, ~40M
steps) lose the tactics because frozen opponents never punish ball-chasing. Our factorization
(frozen low-level + human-primed, KL-anchored self-play at ~2–10Hz command level) is
DeepMind's own 2022 humanoid-football architecture in miniature: it moves the learning
problem to strategy space where our compute budget is meaningful, and imports tactics that
are not discoverable at low skill equilibria.

## Parallel workstreams

The pipeline stages are independent given the **frozen interface contracts** in
`CONTRACTS.md` (read it first — it is the integration boundary). Each stream works on its
own branch and uses proxies so nobody blocks on anybody:

| Stream | Branch | Doc | Proxy that unblocks it |
|---|---|---|---|
| WS1 low-level control | `ws1-low-level` | `WS1_LOW_LEVEL.md` | none needed (start of pipeline) |
| WS2 data collection | `ws2-data-collection` | `WS2_DATA_COLLECTION.md` | **BoxHead** walkers (native actions ≈ our command space, zero training needed), then dm_control **Ant** |
| WS3 RL finetune | `ws3-rl-finetune` | `WS3_RL_FINETUNE.md` | BoxHead env + random/scripted/self-collected mini-BC data |

**Proxies are for building the recipe, not the weights.** Nothing trained on a proxy is
expected to transfer; what transfers is code, data schemas, and configs. When WS1 delivers
real controllers and WS2 delivers real human data, the same infrastructure re-runs unchanged.

## Repo layout (new code lives in `rower_soccer/`)

```
rower_soccer/
  docs/                # you are here
  envs/
    build.py           # env factory (heterogeneous teams, proxy walkers)
    commands.py        # command spec + kick cooldown state machine (FROZEN CONTRACT)
    obs.py             # omniscient high-level obs builder (FROZEN CONTRACT)   [WS2 owns]
    low_level_task.py  # command-following training env                        [WS1 owns]
    hl_soccer.py       # command-level soccer env (frozen low-level inside)    [WS3 owns]
    scripted.py        # command-level scripted bots                           [WS3 owns, WS2 uses]
  controllers.py       # LowLevelController interface + BoxHeadPassthrough (FROZEN CONTRACT)
  train_low.py         # WS1
  play_server/         # WS2
  train_bc.py          # WS2
  train_selfplay.py    # WS3
  eval.py              # WS3
  render_video.py      # shared (offscreen video; works today)
  bench_env.py         # shared
```

## Environment facts (verified)

- Env: dm_control soccer via `custom_soccer_env.create_soccer_env` (repo root), 40×30 pitch,
  arena.xml monkey-patch (physics dt 0.0025). Control dt 0.025s. Multi-agent API:
  `env.step([a0..a3])`, `timestep.observation[i]`, `env.action_spec()[i]`.
- Works on dm-control 1.0.43 / mujoco 3.10 (fresh `.venv` at repo root; use
  `uv pip install --python .venv/bin/python ...` for speed).
- Headless rendering: `MUJOCO_GL=egl` works on this pod (`render_video.py`).
- This pod throttles CPU beyond ~8 concurrent processes; plan vectorization accordingly.
- GPU: RTX 4000 Ada 20GB. Torch must be the cu124 build (driver is CUDA 12.4).

## Cadence

Integration branch: `rower-soccer-v2`. Merge from workstream branches whenever a contract-
compatible milestone lands. Never edit `CONTRACTS.md` unilaterally — contract changes need
all three streams to agree (it's cheaper to talk than to rebase).
