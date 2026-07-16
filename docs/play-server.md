---
title: Interactive play server
---

# Interactive play server

Drive the worm in a browser, live, on [Warp physics](architecture/warp-backend.md)
— the same physics the policies trained in, so there is no sim2sim gap. It runs
headless (EGL rendering streamed over HTTP), so it works on a display-less pod;
reach it through an SSH-forwarded port.

Code: `rower_soccer/warp_port/play_server.py`.

## Run it

```bash
bash scripts/run_play.sh 8085
# forward the port, then open http://localhost:8085
```

The launcher wraps:

```bash
MUJOCO_GL=egl .venv/bin/python -m rower_soccer.warp_port.play_server \
    --follow runs_v2/follow_base/latest.pt \
    --dribble runs_v2/cur1_ours_ws/latest.pt --port 8085
```

Startup takes ~40 s the first time (CUDA-graph capture + Warp JIT) before it binds
the port.

## Controls

| Input | Effect |
|---|---|
| **Q** / *Follow* button | the follow policy drives the worm to the target |
| **W** / *Dribble* button | the dribble policy shepherds the ball to the target |
| **click the arena** | set the target for the active skill (red sphere) |
| **Space** / *Stop* | sit still (zero action) |
| **R** / *Reset* | re-scatter the worm, ball, and target |

The top-down camera is fixed, so a screen click maps to stable world coordinates:
`x = (px/w·2−1)·VIEW_HALF`, `y = (1−py/h·2)·VIEW_HALF`, with `VIEW_HALF = 12 m`.

One env serves both skills: it's a dribble env (39-dim obs), and the follow
policy is driven with `obs[:, 6:]` to drop the 6 ball dims down to its 33-dim
input.

## The threading model (both halves are load-bearing)

The server exists in its current shape because of a hard constraint: **mujoco's
EGL context and Warp's CUDA context (including a captured CUDA graph) must be
created *and* used on the same thread**, or `eglMakeCurrent` throws
`EGL_BAD_ACCESS`.

- **One background thread owns all GPU work.** It creates the Warp env (num
  worlds = 1, `use_graph=True`), the EGL renderer, and the policies, then loops
  *step → render → publish JPEG* at 40 Hz. Isolating the CUDA graph here also
  keeps it away from werkzeug, which 500s the render if it drives the graph
  inline.
- **Flask runs in the main thread and never touches the GPU.** It only reads the
  latest JPEG (`/stream`, MJPEG) and writes command state (`/skill`, `/click`,
  `/reset`). State is shared under a lock; the reset request just sets a flag the
  sim loop picks up at the top of its next iteration.

## Performance

CUDA-graph capture cuts the single-world step from ~95 ms to ~5.5 ms, and the
background-thread architecture pushes the server to **40 fps real-time** (an
earlier single-threaded, non-graph version ran at ~10 fps and crashed the render
inside Flask).

!!! tip "If the server won't bind"
    Repeatedly launching and killing on the same port leaves sockets in
    `TIME_WAIT` and new launches hit `ConnectionRefusedError`. Use a fresh port
    (e.g. 8092+) and give it ~45 s to come up.

## Relation to the pipeline

This is an early, Warp-native version of the **stage-4 play UI**. It currently
supports the worm with follow + dribble (stage 1). The eventual human-demo
recorder (stage 5) records `(football_obs, z)` through this same interface — see
[the pipeline](pipeline.md). A separate CPU/dm_control interactive mode also
exists (`play_interactive.py`, [design doc](design/play-interactive.md)).
