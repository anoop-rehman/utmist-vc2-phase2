#!/usr/bin/env bash
# Stand up a THROWAWAY pod that only runs the 2v2 game server, with GPU
# rendering. No training, no research venvs, no GCS -- this box exists to answer
# one question and then be deleted.
#
#   bash scripts/setup/render_pod.sh check     # is GPU rendering available?
#   bash scripts/setup/render_pod.sh install   # deps (only if check passed)
#   bash scripts/setup/render_pod.sh serve     # run the game on :8090
#
# WHY THIS EXISTS. On the research pod `mjr_render` costs ~46 ms because it runs
# on OSMesa, a SOFTWARE rasteriser, so four per-player camera views cost ~200 ms
# against a 50 ms budget at 20 Hz. Everything needed for GPU rendering is
# installed there -- libEGL_nvidia, the vendor ICD, /dev/nvidia0,
# libnvidia-glcore -- but `eglQueryDevicesEXT` reports 0 devices, which is the
# signature of a container started WITHOUT the NVIDIA `graphics` capability.
# CUDA works, OpenGL does not, and that is set at container launch.
#
# So the pod running this must be created with:
#
#     NVIDIA_DRIVER_CAPABILITIES=all      (or compute,utility,graphics)
#
# `check` verifies that BEFORE anything is installed, because if it fails there
# is no point continuing and the answer is the pod config, not the code.
set -euo pipefail
CMD="${1:-check}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VENV="$REPO/.venv-render"

check() {
    echo "== GPU =="
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || {
        echo "  NO GPU VISIBLE -- wrong pod type"; exit 1; }

    echo "== the libraries EGL needs =="
    for f in /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so.0 \
             /usr/share/glvnd/egl_vendor.d/10_nvidia.json \
             /dev/nvidia0; do
        [ -e "$f" ] && echo "  ok   $f" || echo "  MISSING $f"
    done
    # ADVISORY ONLY. It is tempting to treat libnvidia-glcore as the tell for
    # the graphics capability, but it is present on the research pod where EGL
    # still enumerates 0 devices -- so its presence proves nothing and using it
    # as a gate would wave a bad pod through. The eglQueryDevicesEXT probe
    # below is the only reliable answer.
    if ls /usr/lib/x86_64-linux-gnu/libnvidia-glcore.so.* >/dev/null 2>&1; then
        echo "  ok   libnvidia-glcore present (necessary, NOT sufficient)"
    else
        echo "  note libnvidia-glcore absent"
    fi

    echo "== does EGL actually enumerate a device? =="
    # Written to a file rather than piped, so the same probe can run under
    # whichever python exists (the venv's before or after `install`).
    cat > /tmp/_eglprobe.py <<'PY'
import ctypes, sys
try:
    from OpenGL.EGL.EXT.device_enumeration import eglQueryDevicesEXT
    from OpenGL import EGL
except Exception as e:
    print(f"  (PyOpenGL missing: {e}) -- run `install`, then re-run `check`")
    sys.exit(0)
n = EGL.EGLint(); devs = (EGL.EGLDeviceEXT * 16)()
ok = eglQueryDevicesEXT(16, devs, ctypes.byref(n))
print(f"  eglQueryDevicesEXT ok={ok} devices={n.value}")
print("  DEVICES > 0 -> GPU rendering AVAILABLE" if n.value else
      "  0 DEVICES -> no graphics capability; recreate the pod")
PY
    # Prefer a venv that already has the deps: this pod's own .venv works for
    # the probe even before `install` builds .venv-render.
    PY_BIN=python3
    for c in "$VENV/bin/python" "$REPO/.venv/bin/python"; do
        [ -x "$c" ] && "$c" -c "import OpenGL, mujoco" 2>/dev/null && { PY_BIN="$c"; break; }
    done
    echo "  (probing with $PY_BIN)"
    "$PY_BIN" /tmp/_eglprobe.py

    echo "== can MuJoCo actually render on it? =="
    cat > /tmp/_mjprobe.py <<'PY'
import os, time
os.environ["MUJOCO_GL"] = "egl"
try:
    import mujoco
except Exception as e:
    print(f"  (mujoco missing: {e}) -- run `install` first"); raise SystemExit(0)
# The offscreen framebuffer defaults to 640x480 and Renderer refuses a larger
# image, so the probe declares the size it wants -- otherwise this fails on a
# framebuffer limit and reads like an EGL failure.
m = mujoco.MjModel.from_xml_string(
    "<mujoco><visual><global offwidth='960' offheight='640'/></visual>"
    "<worldbody><body><geom size='.1'/></body></worldbody></mujoco>")
r = mujoco.Renderer(m, 640, 960); d = mujoco.MjData(m)
r.update_scene(d); r.render()
t0 = time.time()
for _ in range(20): r.update_scene(d); r.render()
ms = (time.time() - t0) / 20 * 1000
print(f"  EGL render OK: {ms:.2f} ms/frame at 960x640")
print(f"  -> four per-player views ~{4*ms:.0f} ms "
      f"({'FITS' if 4*ms < 50 else 'over'} the 50 ms budget at 20 Hz)")
PY
    "$PY_BIN" /tmp/_mjprobe.py || echo "  EGL render FAILED (see the error above)"
}

install() {
    apt-get update -qq && apt-get install -y -qq python3-venv curl >/dev/null
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q --upgrade pip
    # dm_control 1.0.45 is the first release that works with mujoco 3.12: 3.12
    # renamed MjData.qM to .M and older dm_control dies on it at sim start.
    "$VENV/bin/pip" install -q "mujoco>=3.12" "dm_control==1.0.45" \
        numpy pillow pyopengl
    echo "installed -> $VENV"
}

serve() {
    # egl, not osmesa: the whole point of this pod.
    export MUJOCO_GL=egl PYTHONPATH="$REPO"
    cd "$REPO"
    exec "$VENV/bin/python" -m rower_soccer.game.server --port 8090 "${@:2}"
}

case "$CMD" in
    check) check ;;
    install) install; check ;;
    serve) serve "$@" ;;
    *) echo "usage: $0 check|install|serve"; exit 2 ;;
esac
