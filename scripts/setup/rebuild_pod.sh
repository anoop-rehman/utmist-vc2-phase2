#!/usr/bin/env bash
# Rebuild a working pod from scratch. Written 2026-08-23, after a pod
# replacement cost an afternoon of guesswork because nothing recorded how the
# warp venv had been built.
#
#   bash scripts/setup/rebuild_pod.sh
#
# Assumes: the repo is cloned at /workspace/utmist-vc2-phase2, gcloud is
# installed at /workspace/google-cloud-sdk, and you have run `gcloud auth login`
# yourself. Credentials are never restored from the bucket by design.
set -euo pipefail
REPO=/workspace/utmist-vc2-phase2
cd "$REPO"
export PATH=/workspace/google-cloud-sdk/bin:$PATH

echo "== 1/6 system GL (offscreen rendering) =="
# EGL was broken on the 2026-08-23 pod (libEGL_nvidia present, eglQueryString
# fails); osmesa is the software fallback that works. Try egl first at runtime.
DEBIAN_FRONTEND=noninteractive apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  libegl1 libgles2 libglvnd0 libopengl0 libosmesa6

echo "== 2/6 python venv =="
python3.11 -m venv "$REPO/.venv"
P="$REPO/.venv/bin/python"
$P -m pip install -q --upgrade pip setuptools wheel
# torch MUST be the cu124 build: the driver is CUDA 12.8 and a newer default
# wheel (cu130) fails with "NVIDIA driver ... too old". Installing anything that
# depends on torch AFTER this can silently upgrade it -- re-check afterwards.
$P -m pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
$P -m pip install -q -r "$REPO/scripts/setup/requirements-warp.lock.txt" || true
$P -m pip install -q torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
$P -c "import torch;assert torch.cuda.is_available(), 'CUDA unavailable';print('torch',torch.__version__)"

echo "== 3/6 sibling repos (transcripts reference these ABSOLUTE paths) =="
[ -d /workspace/competevo ]     || git clone -q https://github.com/KJaebye/competevo /workspace/competevo
[ -d /workspace/Transform2Act ] || git clone -q https://github.com/Khrylx/Transform2Act /workspace/Transform2Act

echo "== 4/6 chat persistence =="
cp scripts/persistence/post-commit.hook .git/hooks/post-commit
chmod +x .git/hooks/post-commit
git config user.name  "$(git log -1 --pretty=%an)"
git config user.email "$(git log -1 --pretty=%ae)"

echo "== 5/6 MPS (worth ~3.4x when several GPU processes share the card) =="
export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d 2>/dev/null || true
# Verify by the SOCKET, not by pgrep: `pgrep -f mps` matches its own shell and
# will happily report a daemon that is not there.
[ -S $CUDA_MPS_PIPE_DIRECTORY/control ] && echo "  MPS up" || echo "  MPS NOT up"

echo "== 6/6 what you must do by hand =="
cat <<'MSG'
  * .env is gitignored and does NOT survive a pod: recreate it with
        WANDB_API_KEY=...
    or every training launcher dies at startup on "No API key configured".
  * Restore checkpoints you need:
        gcloud storage rsync gs://vc2-2026-checkpoints/<run> runs_v2/<run>
MSG
