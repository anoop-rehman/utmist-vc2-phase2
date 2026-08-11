"""Run the bc tests with or without pytest installed.

Same shim as `game/tests/run_tests.py` (the project venv is a training image and
has no pytest), so the suite runs today and drops into a real runner unchanged.

    PYTHONPATH=. .venv/bin/python -m rower_soccer.bc.tests.run_tests            # fast
    MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m rower_soccer.bc.tests.run_tests --slow

`--slow` adds the two modules that build a real 4-ant dm_soccer env:

  * `test_mirror_physics` actually PROVES the mirror — it steps MuJoCo on a
    mirrored state and compares against the mirror of the unmirrored rollout,
    and it checks every one of dm_soccer's 47 observation keys against the env's
    own answer. Nothing in this package should be trusted for a training run
    without it passing.
  * `test_eval_rollout` checks the behavioural half of the eval harness: the 2v2
    env, the BC and scripted agents driving their slots, and the video writer.
"""

import argparse
import importlib
import inspect
import pathlib
import sys
import tempfile
import traceback


def _install_pytest_shim():
    try:
        import pytest  # noqa: F401
        return False
    except ImportError:
        pass
    import types

    mod = types.ModuleType("pytest")

    class _Raises:
        def __init__(self, exc, match=None):
            self.exc, self.match = exc, match

        def __enter__(self):
            return self

        def __exit__(self, et, ev, tb):
            if et is None:
                raise AssertionError(f"expected {self.exc.__name__}, nothing raised")
            if not issubclass(et, self.exc):
                return False
            if self.match is not None:
                import re
                if not re.search(self.match, str(ev)):
                    raise AssertionError(f"{ev!r} does not match {self.match!r}")
            self.value = ev
            return True

    class _Approx:
        def __init__(self, v, abs=None, rel=None):
            self.v, self.abs, self.rel = v, abs, rel

        def __eq__(self, other):
            tol = self.abs if self.abs is not None else max(1e-6, abs(self.v) * (self.rel or 1e-6))
            return abs(other - self.v) <= tol

        def __repr__(self):
            return f"approx({self.v})"

    class _Mark:
        def __getattr__(self, name):
            def deco(fn=None, **kw):
                if fn is None:
                    return lambda f: f
                return fn
            return deco

    mod.raises = _Raises
    mod.approx = _Approx
    mod.mark = _Mark()
    mod.skip = lambda *a, **k: None
    sys.modules["pytest"] = mod
    return True


FAST = ["test_dataset", "test_augment", "test_model", "test_train", "test_eval"]
SLOW = ["test_mirror_physics", "test_eval_rollout"]


def run(modules):
    passed, failed = [], []
    for name in modules:
        m = importlib.import_module(f"rower_soccer.bc.tests.{name}")
        for fname, fn in sorted(vars(m).items()):
            if not fname.startswith("test_") or not inspect.isfunction(fn):
                continue
            kwargs, tmp = {}, None
            if "tmp_path" in inspect.signature(fn).parameters:
                tmp = tempfile.TemporaryDirectory()
                kwargs["tmp_path"] = pathlib.Path(tmp.name)
            label = f"{name}.{fname}"
            try:
                fn(**kwargs)
                passed.append(label)
                print(f"  PASS {label}", flush=True)
            except Exception:                       # noqa: BLE001
                failed.append(label)
                print(f"  FAIL {label}", flush=True)
                traceback.print_exc()
            finally:
                if tmp:
                    tmp.cleanup()
    print(f"\n{len(passed)} passed, {len(failed)} failed")
    return 1 if failed else 0


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--slow", action="store_true",
                   help="also run the dm_soccer mirror verification")
    p.add_argument("--only", default=None)
    a = p.parse_args(argv)
    if _install_pytest_shim():
        print("[tests] pytest not installed; using the built-in shim")
    mods = [a.only] if a.only else (FAST + SLOW if a.slow else FAST)
    raise SystemExit(run(mods))


if __name__ == "__main__":
    main()
