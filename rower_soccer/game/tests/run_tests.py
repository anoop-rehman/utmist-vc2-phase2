"""Run the game tests with or without pytest installed.

The project venv has no pytest (it is a training image, not a dev image), but the
tests are written as ordinary pytest modules so they drop straight into whatever
runner WS5 stands up.  This shim supplies the two pytest features they use
(`raises`, `approx`, `mark`) and a `tmp_path` fixture, so the suite is runnable
today with nothing installed:

    PYTHONPATH=. .venv/bin/python -m rower_soccer.game.tests.run_tests          # fast
    MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m rower_soccer.game.tests.run_tests --slow
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


FAST = ["test_recording", "test_lobby"]
SLOW = ["test_endtoend"]


def run(modules):
    passed, failed = [], []
    for name in modules:
        m = importlib.import_module(f"rower_soccer.game.tests.{name}")
        for fname, fn in sorted(vars(m).items()):
            if not fname.startswith("test_") or not inspect.isfunction(fn):
                continue
            kwargs = {}
            tmp = None
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
    p.add_argument("--slow", action="store_true", help="also run the end-to-end match")
    p.add_argument("--only", default=None)
    a = p.parse_args(argv)
    shimmed = _install_pytest_shim()
    if shimmed:
        print("[tests] pytest not installed; using the built-in shim")
    mods = [a.only] if a.only else (FAST + SLOW if a.slow else FAST)
    raise SystemExit(run(mods))


if __name__ == "__main__":
    main()
