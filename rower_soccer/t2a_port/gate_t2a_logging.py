"""Gate: wandb metrics and periodic video may never change, or end, a run.

    cd /workspace/utmist-vc2-phase2
    MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m \
        rower_soccer.t2a_port.gate_t2a_logging

CPU-only and opens NO CUDA context, like `gate_policy_init.py` and for the same
reason: nothing that needs a 200k-parameter forward pass and an offscreen GL
context should be a client on a card that has live MPS jobs on it. It takes
about four minutes.

Why these five checks
---------------------
The user's requirement for the logging added to `train_t2a.py` was "this stuff
can be lazily logged, it's just nice to be able to track this stuff", which is
a hard constraint, not a soft one: **nothing added for observability may ever
take a training run down, and nothing added for observability may change what
the run computes.** Those two are what checks 1-3 measure. Checks 4 and 5 are
the two ways the output can be confidently WRONG rather than absent:

1. **Training is bit-identical.** Not "identical to itself" -- identical to the
   file at git HEAD, i.e. to the trainer as it was before any of this was
   added. The baseline is loaded straight out of `git show` and run beside the
   current one, with video ON in a third arm. The video draws from its own
   generator with every global stream snapshotted around it, and the third arm
   is what proves that rather than assuming it.
2. **A failed `wandb.init` prints and continues.**
3. **A render exception prints and continues.**
4. **The video shows the CURRENT design.** D3's whole subject is that the body
   changes over training, so a clip of a cached starting model would look
   entirely plausible and be worthless. The check mutates the skeleton policy
   and asserts the compiled models behind the panels change body count and
   geometry.
5. **Episode length is logged on the right convention.** Their `LoggerRL`
   counts the 5 skeleton and 1 attribute steps (reward 0) that the port's does
   not, so `train_R_eps / train_R` means two different things on the two sides.
   That exact off-by-six sent an agent chasing a phantom MuJoCo physics
   discrepancy for hours. Both the native path and `scripts/wandb_ship.py` are
   checked.

Every check has a negative control that puts the corresponding bug back and
asserts the check FAILS. A gate whose controls are not run is a gate that
might be asserting nothing.
"""

import argparse
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import types

import numpy as np
import torch

from rower_soccer.t2a_port import train_t2a as CUR

REPO = "/workspace/utmist-vc2-phase2"
REL = "rower_soccer/t2a_port/train_t2a.py"
OUT = "/tmp/gate_t2a_logging"

# Small enough to run four times on the CPU, big enough that a PPO step
# actually happens: `mini` is overridden to 256 on BOTH arms so a ~900-step
# batch yields four minibatches instead of zero.
MINI = 256


def base_args(run, **kw):
    a = types.SimpleNamespace(
        cfg="hopper_gpu_s2", run=run, outdir=OUT, seed=0,
        batch_steps=900, min_worlds=8, max_worlds=24, eval_worlds=2,
        epochs=0, device="cpu", backend="cpu", fp32=True,
        save_interval=10 ** 6, mempool_mb=-1, stop_file="",
        batch_design=None,
        wandb=False, wandb_project="gate", wandb_tags=["gate"],
        video_secs=0.0, video_worlds=6, video_frames=90,
        video_panel=(240, 180), video_fps=40, video_mean_action=False,
        video_budget_frac=0.0, video_max_steps=0)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def head_module():
    """`train_t2a.py` as it stands at git HEAD -- the trainer BEFORE any of
    this was added, which is the only honest baseline for "unchanged".

    Loaded from a temp file rather than from the tree, so the check cannot be
    fooled by an uncommitted edit sitting in the package.
    """
    src = subprocess.run(["git", "-C", REPO, "show", f"HEAD:{REL}"],
                         capture_output=True, check=True).stdout
    fd, path = tempfile.mkstemp(suffix="_head_train_t2a.py")
    os.write(fd, src)
    os.close(fd)
    spec = importlib.util.spec_from_file_location("head_train_t2a", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.__gate_src_len = len(src)
    return mod


def digest(tr):
    """Every parameter of both networks, plus the running-norm statistics --
    which are state the update writes and a plain `parameters()` misses."""
    h = []
    for net in (tr.policy, tr.value):
        for k, v in sorted(net.state_dict().items()):
            h.append((k, float(v.double().sum()), float(v.double().abs().max()),
                      tuple(v.shape)))
    return h


def rows_of(run):
    """The reproducible half of the log: rewards and lengths, not timings.

    `T_sample`, `build_s` and `steps_per_s_sample` are wall clock and differ
    run to run by construction; comparing them would make the check flaky
    rather than strict.
    """
    out = []
    with open(os.path.join(OUT, run, "log_train.txt")) as f:
        for line in f:
            line = line.strip()
            if not line.startswith("{"):
                continue
            d = json.loads(line)
            out.append({k: d[k] for k in
                        ("epoch", "batch_steps", "n_train_eps", "train_len",
                         "eval_len", "gens", "groups", "v_loss", "p_loss")
                        if k in d})
    return out


def run_trainer(mod, run, epochs=2, **kw):
    a = base_args(run, **kw)
    path = os.path.join(OUT, run, "log_train.txt")
    if os.path.exists(path):
        os.remove(path)
    tr = mod.Trainer(a)
    tr.mini = MINI
    tr.train(epochs)
    return tr


# ---------------------------------------------------------------------------
# 1. the training path is bit-identical to the file at git HEAD
# ---------------------------------------------------------------------------
def check_identical(quiet=True):
    head = head_module()
    with _hush(quiet):
        base = run_trainer(head, "id_head")
        off = run_trainer(CUR, "id_off")
        on = run_trainer(CUR, "id_video", video_secs=1e-9)
    d0, d1, d2 = digest(base), digest(off), digest(on)
    r0, r1, r2 = rows_of("id_head"), rows_of("id_off"), rows_of("id_video")
    n_vid = sum("  video " in l for l in
                open(os.path.join(OUT, "id_video", "log_train.txt")))
    assert n_vid >= 1, "the video arm never fired a video; check 1 is vacuous"
    assert d0 == d1, "wandb/video OFF changed the trained parameters"
    assert r0 == r1, f"wandb/video OFF changed the log rows\n{r0}\n{r1}"
    assert d0 == d2, ("the video event perturbed training -- the RNG snapshot "
                      "or the private generator is not doing its job")
    assert r0 == r2, f"the video event changed the log rows\n{r0}\n{r2}"
    return {"head_src_bytes": head.__gate_src_len, "epochs": len(r0),
            "videos_fired": n_vid,
            "train_len": [r["train_len"] for r in r0],
            "tensors_compared": len(d0)}


def control_identical(quiet=True):
    """Put the bug back: let the video draw from the TRAINING generator and
    skip the global-stream snapshot. Training must then diverge."""
    orig = CUR.Trainer.maybe_video

    def leaky(self, epoch):
        secs = float(getattr(self.args, "video_secs", 0.0) or 0.0)
        if secs <= 0:
            return
        if self._next_video is not None and \
                __import__("time").time() < self._next_video:
            return
        self._next_video = __import__("time").time() + secs
        path = os.path.join(self.out, "videos", f"epoch_{epoch:05d}.mp4")
        self.render_best_median_worst(path)      # no gen swap, no RNG restore

    CUR.Trainer.maybe_video = leaky
    try:
        with _hush(quiet):
            base = run_trainer(CUR, "id_off")
            leak = run_trainer(CUR, "id_leak", video_secs=1e-9)
        return digest(base) != digest(leak)
    finally:
        CUR.Trainer.maybe_video = orig


# ---------------------------------------------------------------------------
# 2. a failed wandb.init prints and training continues
# ---------------------------------------------------------------------------
class FakeWandb(types.ModuleType):
    """Stands in for the real client so the gate needs no network and no key.

    `mode="fail"` is the injected failure; `mode="ok"` is the positive control
    that also captures the payloads checks 5a reads.
    """

    def __init__(self, mode):
        super().__init__("wandb")
        self.mode = mode
        self.calls = []
        self.videos = []
        self.init_kwargs = {}
        outer = self

        def _log(d, step=None):
            if outer.mode == "log_fail":
                raise RuntimeError("injected wandb.log failure")
            outer.calls.append((step, d))

        class _Run:
            url = "https://wandb.test/fake/run"

            def log(self, d, step=None):
                _log(d, step)

            def finish(self):
                pass

        def init(**kw):
            if outer.mode == "init_fail":
                raise RuntimeError("injected wandb.init failure")
            outer.init_kwargs = kw
            return _Run()

        self.init = init
        # `scripts/wandb_ship.py` logs through the MODULE, the trainer through
        # the run object; both land in `calls`.
        self.log = _log
        self.define_metric = lambda *a, **k: None
        self.Video = lambda p, **k: outer.videos.append(p) or {"video": p}


class _hush:
    """Swallow the trainers' stdout; the gate prints its own summary."""

    def __init__(self, on=True):
        self.on = on

    def __enter__(self):
        if self.on:
            self._old, sys.stdout = sys.stdout, io.StringIO()

    def __exit__(self, *a):
        if self.on:
            sys.stdout = self._old


class _fake_wandb:
    def __init__(self, mode):
        self.mod = FakeWandb(mode)

    def __enter__(self):
        self._old = sys.modules.get("wandb")
        sys.modules["wandb"] = self.mod
        return self.mod

    def __exit__(self, *a):
        if self._old is None:
            sys.modules.pop("wandb", None)
        else:
            sys.modules["wandb"] = self._old


def check_wandb_failure(quiet=True):
    with _fake_wandb("init_fail"):
        with _hush(quiet):
            tr = run_trainer(CUR, "wb_fail", epochs=1, wandb=True)
    log = open(os.path.join(OUT, "wb_fail", "log_train.txt")).read()
    assert tr.wb is None, "a raising wandb.init left a run object behind"
    assert "wandb DISABLED" in log, "the failure was not reported"
    assert "training done!" in log, "training did not survive the failure"
    assert rows_of("wb_fail"), "no epoch was logged after the failure"
    return {"epochs_after_failure": len(rows_of("wb_fail"))}


def check_wandb_ok(quiet=True):
    """Positive control AND the source for check 5a: with a working client the
    payload must actually carry the numbers."""
    with _fake_wandb("ok") as wb:
        with _hush(quiet):
            tr = run_trainer(CUR, "wb_ok", epochs=1, wandb=True,
                             video_secs=1e-9)
    assert tr.wb is not None, "a working wandb.init produced no run"
    assert wb.init_kwargs["id"] == wb.init_kwargs["name"] == "wb_ok", (
        "id must be the run name so a resume REATTACHES")
    assert wb.init_kwargs["resume"] == "allow"
    steps = [s for s, _ in wb.calls]
    assert steps == [0, 0], f"expected two logs at epoch 0, got {steps}"
    metrics, video = wb.calls[0][1], wb.calls[1][1]
    want = ["t2a/train_R", "t2a/train_R_eps", "t2a/exec_R", "t2a/exec_R_eps",
            "t2a/T_sample", "t2a/T_update", "t2a/T_eval",
            "t2a/train_ep_len", "t2a/exec_ep_len",
            "port/batch_steps", "port/gens", "port/gen_fill",
            "port/n_train_eps", "port/len_est", "port/train_len",
            "port/eval_len", "port/groups", "port/buckets", "port/gpu_mib",
            "port/g_skel", "port/g_attr", "port/g_control",
            "port/frozen_skel", "port/frozen_attr", "port/frozen_control"]
    missing = [k for k in want if k not in metrics]
    assert not missing, f"metrics missing from the wandb payload: {missing}"
    assert "video/best_median_worst" in video, "the clip was not logged"
    for k in ("video/best_R", "video/median_R", "video/worst_R",
              "video/cost_s", "video/episodes"):
        assert k in video, f"video payload missing {k}"
    assert video["video/best_R"] >= video["video/median_R"] >= \
        video["video/worst_R"], "best/median/worst are not ordered by reward"
    assert wb.videos and os.path.getsize(wb.videos[0]) > 1000, (
        "the mp4 handed to wandb.Video is missing or empty")
    return metrics, video


# ---------------------------------------------------------------------------
# 3. a render exception prints and training continues
# ---------------------------------------------------------------------------
BOOM = "injected render failure"


def check_render_failure(quiet=True):
    orig = CUR.Trainer.render_best_median_worst

    def boom(self, path):
        raise RuntimeError(BOOM)

    CUR.Trainer.render_best_median_worst = boom
    try:
        # The negative control lives here rather than in its own function: the
        # injected exception must be REAL, i.e. it must propagate when it is
        # not behind the guard. If it did not, this check would pass on a
        # trainer that never called the renderer at all.
        raised = False
        try:
            boom(None, "/dev/null")
        except RuntimeError:
            raised = True
        with _fake_wandb("ok"):
            with _hush(quiet):
                tr = run_trainer(CUR, "vid_fail", epochs=2, wandb=True,
                                 video_secs=1e-9)
    finally:
        CUR.Trainer.render_best_median_worst = orig
    log = open(os.path.join(OUT, "vid_fail", "log_train.txt")).read()
    assert raised, "the injected exception does not raise; check 3 is vacuous"
    assert BOOM in log, "the render failure was not reported"
    assert "video FAILED" in log
    assert "training done!" in log, "training did not survive the render failure"
    assert len(rows_of("vid_fail")) == 2, "an epoch was lost to the failure"
    assert tr.wb is not None, "the render failure took wandb down with it"
    return {"epochs_after_failure": len(rows_of("vid_fail")),
            "failures_logged": log.count("video FAILED")}


def check_wandb_log_failure(quiet=True):
    """A wandb backend that raises on every `log` must not cost an epoch, and
    must switch itself off rather than printing a stack trace per epoch."""
    with _fake_wandb("log_fail"):
        with _hush(quiet):
            tr = run_trainer(CUR, "wb_logfail", epochs=2, wandb=True)
    log = open(os.path.join(OUT, "wb_logfail", "log_train.txt")).read()
    assert "wandb log FAILED" in log
    assert "training done!" in log
    assert len(rows_of("wb_logfail")) == 2
    return {"log_failures": log.count("wandb log FAILED")}


# ---------------------------------------------------------------------------
# 4. the video shows the CURRENT design
# ---------------------------------------------------------------------------
SKEL_ADD = 1


def _fingerprint(stats):
    return tuple((stats[f"{k}_bodies"], stats[f"{k}_geom_size"])
                 for k in ("best", "median", "worst"))


def _bias_skeleton_to_add(tr):
    """Make the LIVE policy grow bodies. `skel.ind_mlp.linear.b` is the
    skeleton head's per-body-type output bias (`dense_policy.IndexLinear`), so
    a large constant on the ADD column makes ADD the sampled action at every
    node -- a deterministic, visible change to the design distribution."""
    with torch.no_grad():
        tr.policy.skel.ind_mlp.linear.b[:, SKEL_ADD] += 50.0


def check_current_design(quiet=True, run="design"):
    with _hush(quiet):
        tr = CUR.Trainer(base_args(run, video_worlds=6, video_frames=60))
        _, before = tr.render_best_median_worst(f"{OUT}/{run}/v_before.mp4")
        _bias_skeleton_to_add(tr)
        _, after = tr.render_best_median_worst(f"{OUT}/{run}/v_after.mp4")

    # (a) The COMPILED model behind each panel is that episode's own design.
    #     `compile_design` puts the world body at index 0, so the model's node
    #     count is exactly the DesignWorld's -- measured, not assumed
    #     (`model.nbody == len(robot.bodies) + 1` over sampled designs). A
    #     cached model would hold `_model_bodies` fixed while `_bodies` moved.
    for st, tag in ((before, "before"), (after, "after")):
        for k in ("best", "median", "worst"):
            assert st[f"{k}_model_bodies"] == st[f"{k}_bodies"], (
                f"{tag}: the model rendered for the {k} panel has "
                f"{st[f'{k}_model_bodies']} bodies but the episode's design "
                f"has {st[f'{k}_bodies']} -- the panel is not that episode")

    # (b) A change to the LIVE policy moves the design distribution the clip
    #     samples. This is the population statistic, not a per-pick one, so it
    #     cannot be satisfied by the ranking merely reshuffling.
    assert after["mean_bodies"] > before["mean_bodies"], (
        f"biasing the skeleton head to ADD did not grow the body: "
        f"{before['mean_bodies']} -> {after['mean_bodies']}")
    fb, fa = _fingerprint(before), _fingerprint(after)
    assert fb != fa, ("the rendered design did not change when the policy did "
                      "-- the clip is of a cached model, not the current one")
    # (c) Three panels must be three DESIGNS, not one morphology three times.
    assert len(set(fa)) > 1 or len(set(fb)) > 1, (
        "every panel rendered the identical design; the video is sampling "
        "one body, which on D3 throws away the thing being studied")
    return {"before": {"bodies": before["mean_bodies"], "fingerprint": fb},
            "after": {"bodies": after["mean_bodies"], "fingerprint": fa}}


def control_current_design(quiet=True):
    """Put the bug back: cache the worlds from the first call and reuse them,
    which is exactly "render the starting model". The CHECK must then fail --
    not some weaker restatement of it, the same function."""
    orig = CUR.Trainer.design_phase
    cache = {}

    def cached(self, n_worlds, batch, mean_action, world_offset=0):
        if "w" not in cache:
            cache["w"] = orig(self, n_worlds, batch, mean_action, world_offset)
        return cache["w"]

    CUR.Trainer.design_phase = cached
    try:
        check_current_design(quiet, run="design_ctl")
        return False                      # the check survived the bug
    except AssertionError:
        return True
    finally:
        CUR.Trainer.design_phase = orig


# ---------------------------------------------------------------------------
# 5. episode length, on the right convention for port and reference
# ---------------------------------------------------------------------------
REF_ROW = ("0\tT_sample 1.00\tT_update 2.00\tT_eval 0.50\ttrain_R 4.00\t"
           "train_R_eps 400.00\texec_R 5.00\texec_R_eps 500.00\thopper_gpu\n")
PORT_STARTUP = ("run p  cfg hopper_gpu_s2  seed 1  batch_design True "
                "(cfg agent_specs.batch_design True, --batch-design None)  "
                "dtype torch.float32\n")
PORT_SIDECAR = '  {"epoch": 0, "train_len": 100.0, "eval_len": 100.0}\n'


def _ship(text, name):
    """Run `scripts/wandb_ship.py`'s t2a path over `text` with a fake client."""
    spec = importlib.util.spec_from_file_location(
        "wandb_ship", os.path.join(REPO, "scripts", "wandb_ship.py"))
    ship = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ship)
    path = os.path.join(OUT, f"{name}.log")
    os.makedirs(OUT, exist_ok=True)
    open(path, "w").write(text)
    a = types.SimpleNamespace(t2a_log=path, name=name, project="gate",
                              tags=["gate"], config=[], notes=None,
                              design_steps=6, follow=False, poll=1.0,
                              idle_giveup=1.0, step_key="steps")
    with _fake_wandb("ok") as wb:
        with _hush(True):
            ship.ship_t2a(a)
    # `ship_t2a` calls the module-level `wandb.log`, which the fake exposes as
    # the run's method; grab whatever it recorded.
    return dict(wb.calls[0][1]) if wb.calls else {}


def check_ep_len(metrics, quiet=True):
    out = {}

    # ---- 5a: the native path -------------------------------------------
    side = rows_of("wb_ok")[0]
    ratio = metrics["t2a/train_R_eps"] / metrics["t2a/train_R"]
    assert abs(metrics["t2a/train_ep_len"] - ratio) < 1e-9, (
        "the port's train_ep_len must be train_R_eps / train_R")
    assert abs(metrics["t2a/train_ep_len"] - side["train_len"]) < 0.05, (
        f"train_ep_len {metrics['t2a/train_ep_len']} disagrees with the "
        f"sidecar's train_len {side['train_len']}")
    assert abs(metrics["t2a/train_ep_len_all_stages"]
               - metrics["t2a/train_ep_len"] - 6.0) < 1e-9, (
        "train_ep_len_all_stages must be the port's length plus the 5 skeleton "
        "and 1 attribute steps their logger counts")
    assert abs(metrics["t2a/exec_ep_len"] - side["eval_len"]) < 0.05
    out["native"] = {"train_ep_len": round(metrics["t2a/train_ep_len"], 3),
                     "sidecar_train_len": side["train_len"],
                     "all_stages": round(
                         metrics["t2a/train_ep_len_all_stages"], 3)}

    # ---- 5b: the shipper, both provenances -----------------------------
    ref = _ship(REF_ROW, "ship_ref")
    port_full = _ship(PORT_STARTUP + REF_ROW + PORT_SIDECAR, "ship_port")
    # The race the startup-line mark fixes: a `--follow` poll that lands
    # between the monitor line and its sidecar.
    port_race = _ship(PORT_STARTUP + REF_ROW, "ship_race")
    assert abs(ref["t2a/train_ep_len"] - 94.0) < 1e-6, (
        f"a REFERENCE log must have the 6 design steps removed, got "
        f"{ref['t2a/train_ep_len']}")
    assert abs(ref["t2a/train_ep_len_all_stages"] - 100.0) < 1e-6
    assert abs(ref["t2a/exec_ep_len"] - 100.0) < 1e-6, (
        "exec_R is execution-only on both sides and needs no correction")
    assert abs(port_full["t2a/train_ep_len"] - 100.0) < 1e-6, (
        f"a PORT log must NOT be corrected, got "
        f"{port_full['t2a/train_ep_len']}")
    assert abs(port_race["t2a/train_ep_len"] - 100.0) < 1e-6, (
        f"a port log whose sidecar has not been written yet was corrected as "
        f"if it were the reference: {port_race['t2a/train_ep_len']}")
    out["shipped"] = {"reference": ref["t2a/train_ep_len"],
                      "port": port_full["t2a/train_ep_len"],
                      "port_before_sidecar": port_race["t2a/train_ep_len"]}
    return out


def control_ep_len():
    """Strip BOTH provenance marks and the same rows must be corrected as a
    reference log -- so the marks are what decide it, not luck."""
    bare = _ship(REF_ROW, "ship_bare")
    return abs(bare["t2a/train_ep_len"] - 94.0) < 1e-6


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verbose", action="store_true",
                   help="let the trainers print")
    p.add_argument("--only", nargs="*", default=None,
                   help="subset of 1 2 3 4 5")
    args = p.parse_args()
    quiet = not args.verbose
    want = set(args.only or ["1", "2", "3", "4", "5"])
    os.makedirs(OUT, exist_ok=True)
    np.random.seed(0)
    torch.manual_seed(0)
    fails = []

    def report(n, ok, detail=""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {n}  {detail}", flush=True)
        if not ok:
            fails.append(n)

    print("gate_t2a_logging", flush=True)
    metrics = None

    if "2" in want or "5" in want:
        print("2. wandb", flush=True)
        try:
            r = check_wandb_failure(quiet)
            report("2 init failure prints and training continues", True,
                   f"{r['epochs_after_failure']} epoch(s) after the failure")
        except AssertionError as e:
            report("2 init failure prints and training continues", False, str(e))
        try:
            metrics, video = check_wandb_ok(quiet)
            report("2c metrics populate (positive control)", True,
                   f"{len(metrics)} scalars, video "
                   f"best {video['video/best_R']:.1f} / median "
                   f"{video['video/median_R']:.1f} / worst "
                   f"{video['video/worst_R']:.1f}")
        except AssertionError as e:
            report("2c metrics populate (positive control)", False, str(e))
        try:
            r = check_wandb_log_failure(quiet)
            report("2d a raising wandb.log costs no epoch", True, str(r))
        except AssertionError as e:
            report("2d a raising wandb.log costs no epoch", False, str(e))

    if "3" in want:
        print("3. video failure", flush=True)
        try:
            r = check_render_failure(quiet)
            report("3 render exception prints and training continues", True,
                   str(r))
        except AssertionError as e:
            report("3 render exception prints and training continues", False,
                   str(e))

    if "4" in want:
        print("4. the video shows the current design", flush=True)
        try:
            r = check_current_design(quiet)
            report("4 design change reaches the rendered model", True,
                   f"{r['before']['bodies']} -> {r['after']['bodies']} bodies")
        except AssertionError as e:
            report("4 design change reaches the rendered model", False, str(e))
        ok = control_current_design(quiet)
        report("4-control cached designs make the check FAIL", ok,
               "fingerprints identical when the worlds are reused"
               if ok else "the control did not reproduce the bug")

    if "5" in want:
        print("5. episode length convention", flush=True)
        if metrics is None:
            metrics, _ = check_wandb_ok(quiet)
        try:
            r = check_ep_len(metrics, quiet)
            report("5 port and reference land on one axis", True,
                   json.dumps(r))
        except AssertionError as e:
            report("5 port and reference land on one axis", False, str(e))
        ok = control_ep_len()
        report("5-control an unmarked log is corrected as the reference", ok)

    if "1" in want:
        print("1. training is bit-identical to git HEAD", flush=True)
        try:
            r = check_identical(quiet)
            report("1 wandb/video off and on both reproduce HEAD", True,
                   json.dumps(r))
        except AssertionError as e:
            report("1 wandb/video off and on both reproduce HEAD", False,
                   str(e))
        ok = control_identical(quiet)
        report("1-control an unisolated video DOES perturb training", ok,
               "" if ok else "the control did not diverge; check 1 may be "
                             "asserting nothing")

    print(f"\n{'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
