#!/usr/bin/env bash
# Rebuild CompetEvo's virtualenv from nothing.
#
# The companion to rebuild_t2a_venv.sh, and for the same reason: /workspace/
# competevo is an upstream checkout, so the venv dies with the pod. Recipe and
# pin rationale are in docs/repro/COMPETEVO_M1_REPRO_NOTES.md; this is its
# executable form.
#
#   bash scripts/setup/rebuild_competevo_venv.sh
#   CE_SKIP_APT=1 bash scripts/setup/rebuild_competevo_venv.sh
#
# Much easier than Transform2Act's: modern mujoco bindings and gymnasium, so
# no mujoco210 tarball, no LD_LIBRARY_PATH, no patchelf, no cython build.
# ~5 minutes.
set -euo pipefail

CE=${CE_DIR:-/workspace/competevo}
VENV=$CE/.venv

[ -d "$CE/competevo" ] || { echo "no CompetEvo checkout at $CE"; exit 1; }
UV=$(command -v uv || echo "$HOME/.local/bin/uv")
[ -x "$UV" ] || { echo "uv not found"; exit 1; }

if [ "${CE_SKIP_APT:-0}" != "1" ]; then
  # box2d-py builds from an sdist and needs swig. Nothing in competevo uses
  # box2d -- every env is MuJoCo -- but it is in their requirements.txt, so
  # the install fails without it.
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq swig
fi

echo "=== venv (python 3.8, their version)"
rm -rf "$VENV"
"$UV" venv --python 3.8 "$VENV"

echo "=== torch 1.12.0, CPU build"
# Their dockerfile pins +cu113; CPU-only here, same version numbers. The GPU
# on this pod belongs to the warp port and to Transform2Act.
VIRTUAL_ENV=$VENV "$UV" pip install -q \
  torch==1.12.0+cpu torchvision==0.13.0+cpu torchaudio==0.12.0+cpu \
  --index-url https://download.pytorch.org/whl/cpu

echo "=== their requirements, at their pins"
VIRTUAL_ENV=$VENV "$UV" pip install -q -r "$CE/docker/requirements.txt"
# gym_compete/new_envs/agents/agent.py imports six and requirements.txt does
# not list it; in their docker image it arrived as a transitive dependency.
VIRTUAL_ENV=$VENV "$UV" pip install -q six

echo "=== verify"
cd "$CE"
PYTHONPATH=. MUJOCO_GL=osmesa "$VENV/bin/python" - <<'PY'
import torch, mujoco, gymnasium, torch_geometric, six
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("mujoco", mujoco.__version__, "gymnasium", gymnasium.__version__,
      "torch_geometric", torch_geometric.__version__)
# Importing the runner is the real check: it is what train.py loads, and it
# pulls in gym_compete, the custom envs and the registration side effects.
from runner.multi_evo_agent_runner import MultiEvoAgentRunner  # noqa: F401
print("MultiEvoAgentRunner imports OK")
PY

echo
echo "=== done. $VENV"
echo "    cd $CE && PYTHONPATH=. MUJOCO_GL=osmesa OMP_NUM_THREADS=1 \\"
echo "      .venv/bin/python train.py --cfg config/run-to-goal-devants-v0.yaml \\"
echo "      --use_cuda false --num_threads 24"
