"""Best-effort GCS sync for checkpoints (never blocks/kills training).

Mid-run uploads happen on background threads; if gcloud/network is unavailable
the error is logged and training continues. Uses `gcloud storage` via subprocess
so no extra Python deps are needed.

Two rules keep the remote copy trustworthy:

- At most one in-flight upload per destination. `gcloud storage cp` can take
  minutes on a slow pod, so without this a checkpoint interval shorter than the
  upload time stacks up concurrent writes to the same object, and an older one
  can land *after* a newer one.
- The end-of-run flush drains those background uploads first, then uploads
  synchronously (`sync_blocking`), so the final bytes are written last and the
  process cannot exit mid-transfer.
"""

import os
import shutil
import subprocess
import threading
import time

_GCLOUD = shutil.which("gcloud") or "/workspace/google-cloud-sdk/bin/gcloud"

_threads = []
_inflight = set()
_lock = threading.Lock()


def _dest(local_path, bucket, run_name):
    base = os.path.basename(local_path)
    return f"gs://{bucket.removeprefix('gs://')}/{run_name}/{base}"


def _upload(local_path, dest):
    try:
        subprocess.run([_GCLOUD, "storage", "cp", local_path, dest],
                       check=True, capture_output=True, timeout=300)
        print(f"[gcs] synced {os.path.basename(local_path)} -> {dest}", flush=True)
    except Exception as e:  # noqa: BLE001 - never let sync crash training
        print(f"[gcs] WARN sync failed ({os.path.basename(local_path)}): {e}",
              flush=True)
    finally:
        with _lock:
            _inflight.discard(dest)


def sync_async(local_path, bucket, run_name):
    """Fire-and-forget upload to gs://<bucket>/<run_name>/.

    Skipped if an upload to the same destination is still running: the next
    checkpoint interval will carry newer bytes anyway.
    """
    if not bucket:
        return
    dest = _dest(local_path, bucket, run_name)
    with _lock:
        if dest in _inflight:
            print(f"[gcs] skip {os.path.basename(local_path)}: previous upload "
                  f"still in flight", flush=True)
            return
        _inflight.add(dest)
    t = threading.Thread(target=_upload, args=(local_path, dest), daemon=True)
    t.start()
    _threads.append(t)


def sync_blocking(local_path, bucket, run_name):
    """Upload on the calling thread. Use after wait_all() for the final flush."""
    if not bucket:
        return
    dest = _dest(local_path, bucket, run_name)
    with _lock:
        _inflight.add(dest)
    _upload(local_path, dest)


def wait_all(timeout=600):
    """Block until in-flight uploads finish.

    Upload threads are daemons, so the interpreter kills them on exit. Any
    sync_async issued just before the process ends -- i.e. the end-of-run
    checkpoint, the one most worth keeping -- is silently dropped unless it is
    joined first.
    """
    deadline = time.monotonic() + timeout
    for t in _threads:
        t.join(max(0.0, deadline - time.monotonic()))
    alive = [t for t in _threads if t.is_alive()]
    if alive:
        print(f"[gcs] WARN {len(alive)} upload(s) unfinished after {timeout}s; "
              f"they will be killed at exit", flush=True)
    _threads[:] = alive
    return not alive
