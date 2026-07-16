---
title: Repo layout
---

# Repo layout

The active codebase is `rower_soccer/`. Everything else at the repo root is
legacy (pre-2026 PPO experiments) or scratch (CPU/top logs, notebooks).

```
rower_soccer/              current codebase
  docs/                    canonical design docs (rendered under "Design docs")
  warp_port/               MuJoCo Warp GPU envs, PPO, play server  ← active
    scene.py               builds the creature+ball+pitch scene; solref tuning
    render.py              render-only MjModel; Warp qpos -> picture
    follow_env.py          follow drill env (33-dim obs)
    dribble_env.py         dribble drill env (39-dim obs, 3D ball)
    ppo.py                 latent ActorCritic, SimpleActorCritic, PPOTrainer
    train_follow_warp.py   follow trainer CLI
    train_dribble_warp.py  dribble trainer CLI (curriculum flags)
    play_server.py         browser interactive play (Warp physics)
    gcs.py                 checkpoint sync
    probe_speed.py         measure achievable worm speed (target calibration)
  tools/                   .creature (Unity NRBF genotype) -> MuJoCo XML converter
  envs/                    soccer env factory (heterogeneous teams), commands, obs
  drills/                  CPU drill tasks + gymnasium wrapper (reference path)
  models/                  latent-bottleneck policy (expert -> z -> shared decoder)
  train_drill.py           CPU PPO drill trainer (reference)
  monitor.py               fps/ETA console lines + periodic eval videos
  render_video.py          offscreen match/drill rendering
  play_interactive.py      CPU/dm_control interactive mode
  soccer_bridge.py         drill<->soccer observation adapter

creature_configs/          creature XMLs + original .creature genotype files
scripts/                   run_play.sh, gcs_pull_run.sh
sky/                       SkyPilot sweep infra + local fan-out launchers
runs_v2/                   Warp training runs (checkpoints; gitignored)
trained_creatures/         legacy checkpoints (pre-2026)
docs/                      this documentation site (MkDocs)
mkdocs.yml, .readthedocs.yaml   docs build config
```

## Docs map

| Doc | Where | What |
|---|---|---|
| `README.md` | root | project pitch + quickstart |
| `POD_SETUP.md` | root | fresh-pod runbook ([here](../getting-started/pod-setup.md)) |
| `PIPELINE_V2.md` | `rower_soccer/docs/` | north-star method design |
| `STAGE2_MULTITASK.md` | `rower_soccer/docs/` | stages 1–2 engineering + the obs contract |
| `CONTRACTS.md` | `rower_soccer/docs/` | frozen interface boundaries |
| `OVERVIEW.md` | `rower_soccer/docs/` | project overview + workstreams |
| `WS1_LOW_LEVEL.md` | `rower_soccer/docs/` | low-level control workstream |
| `PLAY_INTERACTIVE.md` | `rower_soccer/docs/` | CPU interactive mode |
| `SOCCER_BRIDGE.md` | `rower_soccer/docs/` | drill↔soccer obs mapping |
