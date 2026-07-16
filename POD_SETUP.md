# Pod setup runbook (for a fresh RunPod pod)

This is a checklist for bringing a new pod up to a working training state. It
exists because a rebuilt container loses everything installed outside the repo:
the Python venv, system `apt` packages, and cloud credentials all have to be
re-established. The Python deps are pinned and reproducible; the three things
below are the ones that bite because they live *outside* pip.

> **Claude: run these checks first, top to bottom.** Each has a verify command.
> Only install what a verify step shows missing — on some base images EGL and
> gcloud are already present. Don't reinstall blindly.

Working directory throughout: `/workspace/utmist-vc2-phase2`. The venv is
`.venv/` (Python 3.11); always invoke it explicitly as `.venv/bin/python`.

---

## 1. System library for headless rendering: `libegl1`

**Why this matters and why it's easy to miss:** the training loop renders
transfer-eval videos with `MUJOCO_GL=egl`. mujoco → PyOpenGL then `dlopen`s
`libEGL.so.1`, the vendor-neutral GLVND dispatch stub. Many CUDA base images
ship the NVIDIA *driver* half (`libEGL_nvidia.so.0`) and its ICD manifest but
**not** `libEGL.so.1` itself. When it's missing, PyOpenGL silently sets its EGL
backend to `None` and you get a baffling error at first render:

```
AttributeError: 'NoneType' object has no attribute 'eglQueryString'
```

That traceback names neither EGL nor the missing package — do not chase it into
mujoco or PyOpenGL. The fix is one apt package.

```bash
apt-get update                    # REQUIRED first — a stale index makes the
                                  # next line fail with "Unable to locate package"
apt-get install -y libegl1        # provides /lib/x86_64-linux-gnu/libEGL.so.1
# Optional, only if you ever use the software renderer MUJOCO_GL=osmesa:
# apt-get install -y libosmesa6
```

`ffmpeg` is **not** needed as a system package — video encoding uses the
pip-bundled `imageio-ffmpeg` inside the venv. Only `libEGL` must come from apt.

**Verify** (renders a real frame on the GPU and prints its pixel count):

```bash
MUJOCO_GL=egl .venv/bin/python - <<'PY'
import mujoco, numpy as np
ctx = mujoco.GLContext(64, 64); ctx.make_current()
m = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><light pos="0 0 3"/>'
    '<geom type="box" size=".1 .1 .1" rgba="1 0 0 1"/></worldbody></mujoco>')
d = mujoco.MjData(m); mujoco.mj_forward(m, d)
r = mujoco.Renderer(m, 64, 64); r.update_scene(d)
px = r.render()
print("EGL render OK:", px.shape, "nonzero px:", int((px > 0).sum()))
PY
```

Expect `EGL render OK: (64, 64, 3) nonzero px: <a few hundred+>`. An
`EGLError` printed *only* under `Exception ignored in ... __del__` at exit is a
harmless teardown quirk — ignore it as long as the "render OK" line printed.

---

## 2. Python environment

If `.venv/` survived on the persistent volume, skip to the verify step. If it's
gone, recreate it (mirrors README "Setup"):

```bash
python3 -m venv .venv
uv pip install --python .venv/bin/python dm_control mujoco gymnasium \
    stable-baselines3 wandb imageio[ffmpeg] warp-lang mujoco-warp \
    "torch==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124
```

> **Install torch LAST, and check what you actually got.** If you split this into
> two commands (torch from the PyTorch index, then the rest from PyPI), the second
> command **silently replaces your cu124 torch** — `stable-baselines3` depends on
> `torch`, uv resolves it from PyPI, and PyPI's default wheel is a *newer CUDA*
> build. You then get:
>
> ```
> RuntimeError: The NVIDIA driver on your system is too old (found version 12040)
> ```
>
> which names the driver, not the real culprit. The pod's driver is CUDA 12.4;
> the clobbering wheel was `torch 2.12.1+cu130`. Always verify the build, not just
> that torch imports:
>
> ```bash
> .venv/bin/python -c "import torch; print(torch.__version__, torch.version.cuda)"
> # want: 2.6.0+cu124 12.4     NOT: 2.12.1+cu130 13.0
> ```
>
> Fix is simply to reinstall torch from the PyTorch index afterwards.

Also needed on a fresh pod, and neither survives a rebuild:

```bash
.venv/bin/wandb login          # interactive; training defaults to --wandb-project creature-soccer
gh auth login -h github.com -p https -w && gh auth setup-git   # for pushes
```

A frozen, known-good pin set is in `runpod_requirements.txt` if you need exact
versions (this session ran torch 2.6.0, mujoco 3.1.3, warp 1.14.0, wandb 0.28.0,
Python 3.11.10).

**Verify** the GPU stack imports and CUDA is visible:

```bash
.venv/bin/python -c "import torch, warp, mujoco_warp; \
print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0)); \
print('warp', warp.config.version)"
```

First Warp run JIT-compiles kernels and can sit silent for a few minutes before
the first `[setup] worlds=...` line — that's normal, not a hang. Kernels are
cached afterward, so later runs start quickly.

---

## 3. Cloud auth + checkpoint backup (GCS)

The gcloud SDK lives at `/workspace/google-cloud-sdk` (not on `PATH`; call it by
full path or `source /workspace/google-cloud-sdk/path.bash.inc`). Auth is a
credential, so it does **not** survive a rebuild.

**Verify** you're still authed:

```bash
/workspace/google-cloud-sdk/bin/gcloud auth list 2>&1 | grep -q ACTIVE \
  && echo "gcloud: authed" || echo "gcloud: NEEDS LOGIN"
```

If it needs login, the user must run this themselves (interactive/browser) —
Claude cannot. Suggest they type it into the session with the `!` prefix:

```
! /workspace/google-cloud-sdk/bin/gcloud auth login
```

Project is `vc2-2026`; the checkpoint bucket is `gs://vc2-2026-checkpoints`.

> **gcloud is slow on RunPod.** `gcloud storage cp` can take ~2 min per file and
> `ls`/`rm` sometimes hang for a minute. This is a known pod quirk, not a
> failure — uploads still land. The trainer's sync layer is built around it
> (uploads to the same object coalesce; the end-of-run flush blocks so nothing
> is lost on exit). Don't "fix" a slow upload by killing it.

---

## Launching a run

Always set `MUJOCO_GL=egl` and always pass `--gcs-bucket` so checkpoints are
backed up (pods die without warning):

```bash
MUJOCO_GL=egl .venv/bin/python -m rower_soccer.warp_port.train_follow_warp \
    --run-name <unique-name> \
    --steps 2000000000 --worlds 2048 \
    --gcs-bucket vc2-2026-checkpoints \
    --resume            # add only when continuing an existing run
```

**To survive an SSH disconnect** (i.e. any overnight run), detach it with
`setsid nohup ... &` so it is reparented to init and a hangup cannot kill it:

```bash
mkdir -p logs
setsid nohup env MUJOCO_GL=egl .venv/bin/python -m rower_soccer.warp_port.train_follow_warp \
    --run-name <unique-name> --steps 800000000 --worlds 2048 \
    --ent-floor -1.2 --ent-ceil 0.0 \
    --first-video-secs 60 --video-secs 900 --ckpt-secs 1800 \
    --gcs-bucket vc2-2026-checkpoints > logs/<unique-name>.log 2>&1 < /dev/null &
```

Confirm it actually detached — `PPID` must be **1**:

```bash
ps -eo pid,ppid,cmd | grep train_.*_warp | grep -v grep
```

`--first-video-secs 60` puts the first transfer-eval video ~1 minute in (rather
than one full `--video-secs` interval), so a run with a dead reward or a broken
obs layout is visible almost immediately instead of the next morning. Watch that
first video before walking away.

**Two runs fit on one A4000** (follow + dribble at 2048 worlds each): ~1.5 GB of
16 GB, ~60k and ~45k steps/s respectively when sharing the GPU.

Checkpoint conventions this repo follows (see the checkpointing commit for the
reasoning) — worth knowing before you touch the files in a bucket:

- **`checkpoint.pt`** — full state incl. optimizer; the file to `--resume` from.
- **`latest.pt`** — weights-only export, kept in lockstep with `checkpoint.pt`;
  this is the one to load for **inference / eval** (take the action
  distribution's `.mean`, not `.sample()`).
- **`final.pt`** — only written on a *clean* run exit. Equals `latest.pt` at
  that point. A `final.pt` present at the *start* of a run is stale (belongs to
  an earlier run of the same name) and is auto-deleted at launch.
- **`checkpoint_mid.pt`** — one rollback copy, written once past
  `--mid-ckpt-frac` (default 0.5 of `--steps`).
- **Run names must be unique.** Launching into a non-empty `runs_v2/<name>/`
  without `--resume` is refused, to stop two runs' artifacts mixing in one
  directory / GCS prefix. Each resume leg records its own `config_resume_N.json`
  rather than overwriting the original `config.json`.

---

## One-shot check

Paste-and-run to see everything at a glance:

```bash
cd /workspace/utmist-vc2-phase2
echo "libEGL: $(ldconfig -p | grep -q 'libEGL.so.1' && echo OK || echo MISSING-run-apt)"
echo "venv:   $(.venv/bin/python -c 'import torch,warp,mujoco_warp' 2>/dev/null && echo OK || echo BROKEN)"
echo "cuda:   $(.venv/bin/python -c 'import torch;print(torch.cuda.is_available())' 2>/dev/null)"
/workspace/google-cloud-sdk/bin/gcloud auth list 2>&1 | grep -q ACTIVE \
  && echo "gcloud: authed" || echo "gcloud: NEEDS LOGIN"
```
