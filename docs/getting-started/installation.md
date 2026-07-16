---
title: Installation
---

# Installation

The project runs on Linux with an NVIDIA GPU (developed on an **RTX 4000 Ada,
20 GB**; the CUDA-12.4 toolchain is assumed throughout — the driver is 12.4).

## Python environment

```bash
python3 -m venv .venv
uv pip install --python .venv/bin/python dm_control mujoco gymnasium \
    "stable-baselines3==2.6.0" wandb imageio[ffmpeg] warp-lang mujoco-warp \
    "torch==2.6.0+cu124" --index-url https://download.pytorch.org/whl/cu124
```

Always invoke the interpreter explicitly as `.venv/bin/python` — nothing here
relies on an activated shell.

!!! danger "Install the cu124 torch **last**"
    Installing SB3 / other deps *after* torch pulls a newer torch from PyPI
    (e.g. `2.12.1+cu130`), which does not match the CUDA-12.4 driver and breaks
    every GPU kernel at runtime. Pin `torch==2.6.0+cu124` from the cu124 index
    and install it in the final step. See the
    [pod setup runbook](pod-setup.md) for the full trap.

## Headless rendering

Rendering (drill videos, match renders, the play server) is offscreen via EGL:

```bash
sudo apt-get install -y libegl1 libosmesa6 ffmpeg
export MUJOCO_GL=egl
```

Every rendering command in these docs is prefixed with `MUJOCO_GL=egl`.

## Cloud credentials (optional)

Training syncs checkpoints to GCS and logs to Weights & Biases. Neither is
required to run locally, but the training scripts expect them when enabled:

```bash
gcloud auth application-default login     # checkpoint sync to GCS
wandb login                               # metrics + eval videos
```

- **wandb** project: `creature-soccer`, entity `team-anoop`.
- On a **fresh pod**, follow the [pod setup runbook](pod-setup.md) — a rebuilt
  container loses the venv, apt packages, and credentials.

## Verify

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect: 2.6.0+cu124 True

MUJOCO_GL=egl .venv/bin/python -c "import mujoco, mujoco_warp; print('warp ok')"
```

Next: [Quickstart](quickstart.md).
