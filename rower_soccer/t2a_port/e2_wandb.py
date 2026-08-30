"""D3 M3 E2: inline wandb for both arms, metrics and video in ONE run.

E0 and E1 shipped metrics post-hoc and rendered video in a second pass, so the
video landed at a step BEHIND the run's current one and wandb silently dropped
it; `<name>_media` runs were the fix. E2 has no such problem because both
trainers log from the training process as the epoch finishes, so the video goes
in the SAME `wandb.log` call as that epoch's metrics -- the pattern
`train_soccer2v2_warp` and `train_t2a.py` already use.

Two environment facts this file exists to hide:

  * `/workspace/Transform2Act/.venv-gpu` has mujoco-py and no wandb. wandb is
    installed beside it at `/workspace/t2a_pylibs` and put FIRST on sys.path,
    because it needs protobuf >= 4 and the venv pins 3.x;
  * that in turn breaks `tensorboardX`'s protobuf-3 generated code, so
    `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` is forced before any
    protobuf import. Pure-Python protobuf is slower and is used for nothing on
    the hot path.

Nothing here may ever end a training run: every call is wrapped, prints why it
failed, and returns.
"""

import os
import sys

T2A_PYLIBS = "/workspace/t2a_pylibs"


def _prepare():
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    os.environ.setdefault("WANDB_SILENT", "true")
    if os.path.isdir(T2A_PYLIBS) and T2A_PYLIBS not in sys.path:
        sys.path.insert(0, T2A_PYLIBS)


class Run:
    """A wandb run, or a no-op that says why."""

    def __init__(self, name, project="creature-soccer", config=None,
                 tags=None, notes=None, enabled=True, log=print):
        self.wb = None
        self.log = log
        self.fails = 0
        if not enabled:
            return
        try:
            _prepare()
            import wandb
            if not hasattr(wandb, "init"):
                raise RuntimeError("`wandb` resolved to a namespace package -- "
                                   "a `wandb/` directory is shadowing it")
            self.wandb = wandb
            self.wb = wandb.init(project=project, name=name, id=name,
                                 resume="allow", config=config or {},
                                 tags=tags or ["d3", "e2"], notes=notes)
            wandb.define_metric("epoch")
            wandb.define_metric("*", step_metric="epoch")
            self.log(f"wandb: {self.wb.url}")
        except Exception as e:
            self.wb = None
            self.log(f"wandb DISABLED ({e!r}) -- training continues")

    def log_epoch(self, epoch, payload, video=None, fps=25):
        """One call per epoch. `video` is an mp4 path; it goes in the SAME log
        call as the metrics, so it can never land at an earlier step."""
        if self.wb is None:
            return
        try:
            body = dict(payload)
            body["epoch"] = int(epoch)
            if video:
                body["video/best_median_worst"] = self.wandb.Video(
                    video, fps=fps, format="mp4")
            self.wandb.log(body)
            self.fails = 0
        except Exception as e:
            self.fails += 1
            self.log(f"wandb log FAILED at epoch {epoch} ({e!r})")
            if self.fails >= 5:
                self.log("wandb: five consecutive failures -- disabling")
                self.wb = None

    def finish(self):
        if self.wb is not None:
            try:
                self.wb.finish()
            except Exception:
                pass
