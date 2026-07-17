"""Label which way is UP for a creature -- a one-click browser tool.

The fetch reward multiplies by an uprightness factor (dm_control:
torso_upright = torso z-axis . world z). The quadruped has an obvious "up";
our GA-evolved worm does not -- its belly is whatever the GA said it was. This
tool renders the worm rolled about its long axis in 30-degree increments; you
click the one that is right-side-up, and the chosen orientation's local up
vector is saved to creature_configs/<name>_up_axis.json. The worm-fetch env
then scores upright = (1 + R@up_local . z)/2, exactly the dm_control formula
with "torso z-axis" replaced by the labeled axis.

    MUJOCO_GL=egl .venv/bin/python -m rower_soccer.warp_port.label_up --port 8096
"""
import argparse
import base64
import io
import json
import os

import mujoco
import numpy as np
from flask import Flask, request

from rower_soccer.warp_port.scene import build_creature_scene

N_ROLLS = 12
FLOAT_Z = 0.8   # render the worm floating so no roll intersects the floor


def roll_quat(axis, t):
    q = np.zeros(4)
    q[0] = np.cos(t / 2)
    q[1:] = np.sin(t / 2) * axis
    return q


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


def render_rolls(creature_xml):
    """Returns (list of (roll_deg, png_bytes), long_axis, model_name)."""
    model, meta = build_creature_scene(creature_xml, ball=None)
    data = mujoco.MjData(model)
    ren = mujoco.Renderer(model, height=240, width=320)

    # Long axis = rest-pose direction from first to last creature body.
    mujoco.mj_forward(model, data)
    body_xpos = data.xpos[meta.body_ids]
    axis = body_xpos[-1] - body_xpos[0]
    axis[2] = 0.0
    axis /= np.linalg.norm(axis)

    qr = meta.qpos_root
    shots = []
    for k in range(N_ROLLS):
        t = 2 * np.pi * k / N_ROLLS
        data.qpos[:] = 0.0
        data.qpos[qr + 0:qr + 3] = 0.0, 0.0, FLOAT_Z
        data.qpos[qr + 3:qr + 7] = roll_quat(axis, t)
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)

        views = []
        for az, el in ((90.0, -10.0), (5.0, -10.0)):   # side view + down-the-axis
            cam = mujoco.MjvCamera()
            cam.lookat[:] = data.xpos[meta.root_body]
            cam.distance = 3.0
            cam.azimuth = az
            cam.elevation = el
            ren.update_scene(data, camera=cam)
            views.append(ren.render())
        frame = np.hstack(views)
        from PIL import Image
        b = io.BytesIO()
        Image.fromarray(frame).save(b, format="PNG")
        shots.append((int(round(np.rad2deg(t))), b.getvalue()))
    return shots, axis


PAGE = """<!doctype html><html><head><title>label up</title>
<style>body{background:#111;color:#eee;font-family:sans-serif;text-align:center}
img{border:3px solid #333;margin:4px;cursor:pointer}img:hover{border-color:#5cb85c}
#msg{font-size:18px;color:#5cb85c;min-height:24px}</style></head><body>
<h3>Click the image where the worm is RIGHT SIDE UP</h3>
<p>each tile: side view (left) + looking down the body axis (right)</p>
<div id="msg"></div>
<div>{{TILES}}</div>
<script>
function pick(deg){fetch('/label',{method:'POST',headers:{'Content-Type':'application/json'},
 body:JSON.stringify({roll_deg:deg})}).then(r=>r.json()).then(d=>{
 document.getElementById('msg').textContent='saved: roll '+deg+'deg -> up_local '+JSON.stringify(d.up_local);});}
</script></body></html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--creature-xml", default="creature_configs/three_seg_worm.xml")
    p.add_argument("--out", default=None,
                   help="default: creature_configs/<stem>_up_axis.json")
    p.add_argument("--port", type=int, default=8096)
    args = p.parse_args()
    out = args.out or os.path.join(
        "creature_configs",
        os.path.splitext(os.path.basename(args.creature_xml))[0] + "_up_axis.json")

    shots, axis = render_rolls(args.creature_xml)
    tiles = "".join(
        f'<img src="data:image/png;base64,{base64.b64encode(png).decode()}" '
        f'onclick="pick({deg})" title="roll {deg}deg">'
        for deg, png in shots)

    app = Flask(__name__)

    @app.route("/")
    def index():
        return PAGE.replace("{{TILES}}", tiles)

    @app.route("/label", methods=["POST"])
    def label():
        deg = float(request.get_json()["roll_deg"])
        t = np.deg2rad(deg)
        # If R(roll about axis) is right-side-up, the body axis that points at
        # world +z is R^T z.
        up_local = quat_to_mat(roll_quat(axis, t)).T @ np.array([0.0, 0.0, 1.0])
        payload = {"creature_xml": args.creature_xml,
                   "roll_deg": deg,
                   "long_axis": [round(float(v), 6) for v in axis],
                   "up_local": [round(float(v), 6) for v in up_local]}
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[label] saved {out}: {payload}", flush=True)
        return json.dumps({"ok": True, "up_local": payload["up_local"]})

    print(f"[label] serving on http://0.0.0.0:{args.port} -> writes {out}",
          flush=True)
    app.run(host="0.0.0.0", port=args.port, threaded=False, use_reloader=False)


if __name__ == "__main__":
    main()
