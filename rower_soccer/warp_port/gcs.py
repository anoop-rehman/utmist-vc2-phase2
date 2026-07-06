"""Best-effort GCS sync for checkpoints (never blocks/kills training).

Uploads happen in a background thread; if gcloud/network is unavailable the
error is logged and training continues. Uses `gcloud storage` (falls back to
gsutil) via subprocess so no extra Python deps are needed.
"""

import os
import shutil
import subprocess
import threading

_GCLOUD = shutil.which("gcloud") or "/workspace/google-cloud-sdk/bin/gcloud"


def _upload(local_path, dest):
    try:
        subprocess.run([_GCLOUD, "storage", "cp", local_path, dest],
                       check=True, capture_output=True, timeout=300)
        print(f"[gcs] synced {os.path.basename(local_path)} -> {dest}", flush=True)
    except Exception as e:  # noqa: BLE001 - never let sync crash training
        print(f"[gcs] WARN sync failed ({os.path.basename(local_path)}): {e}",
              flush=True)


def sync_async(local_path, bucket, run_name):
    """Fire-and-forget upload of local_path to gs://<bucket>/<run_name>/."""
    if not bucket:
        return
    dest = f"gs://{bucket.removeprefix('gs://')}/{run_name}/{os.path.basename(local_path)}"
    threading.Thread(target=_upload, args=(local_path, dest), daemon=True).start()
