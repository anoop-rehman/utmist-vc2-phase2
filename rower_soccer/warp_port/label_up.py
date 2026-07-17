"""Label a creature's rest pose + which way is UP -- interactive posing tool.

Pose the creature limb by limb and place the whole body in 6-DOF, then Save:

  * CLICK a limb in the render to select it (MuJoCo's own mjv_select ray
    pick); its joint sliders appear and the limb highlights yellow.
  * Slide the selected limb's joint(s), the whole-body position/rotation,
    and the camera (orbit) freely -- the render is live.
  * Save writes creature_configs/<name>_up_axis.json with the root pose,
    every joint angle, and the local up vector (R^T z). The fetch env spawns
    the creature in exactly this pose (joints + orientation x random yaw) and
    scores upright = (1 + R@up_local . z)/2 -- dm_control's formula with the
    labeled axis standing in for the quadruped's torso z.

    MUJOCO_GL=egl .venv/bin/python -m rower_soccer.warp_port.label_up \
        --creature-xml creature_configs/two_arm_rower_scaled.xml --port 8098
"""
import argparse
import base64
import io
import json
import os

import mujoco
import numpy as np
from flask import Flask, request, jsonify

from rower_soccer.warp_port.scene import build_creature_scene

W, H = 640, 480


def rpy_to_quat(roll, pitch, yaw):
    """World-frame z(yaw) * y(pitch) * x(roll), radians -> wxyz quaternion."""
    def axis_quat(ax, t):
        q = np.zeros(4)
        q[0] = np.cos(t / 2)
        q[1 + ax] = np.sin(t / 2)
        return q
    q = axis_quat(2, yaw)
    for ax, t in ((1, pitch), (0, roll)):
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, q, axis_quat(ax, t))
        q = out
    return q


def quat_to_mat(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, q)
    return m.reshape(3, 3)


PAGE = """<!doctype html><html><head><title>pose + label up</title>
<style>body{background:#111;color:#eee;font-family:sans-serif;margin:0;padding:14px;display:flex;gap:18px}
#left{flex:0 0 660px}#right{flex:1;max-width:430px;overflow-y:auto;max-height:96vh}
img{border:2px solid #333;width:640px;height:480px;cursor:crosshair}
label{display:block;margin:8px 0 2px;color:#aaa;font-size:13px}
input[type=range]{width:100%}
.val{color:#5cb85c;font-family:monospace}
button{font-size:16px;padding:10px 22px;margin-top:14px;border:0;border-radius:6px;cursor:pointer;background:#5cb85c;color:#fff}
#msg{color:#5cb85c;min-height:22px;font-size:13px;margin-top:8px;font-family:monospace;word-break:break-all}
#sel{color:#f1c40f;font-family:monospace;min-height:20px}
h4{margin:12px 0 2px;color:#ddd}#jpanel{border:1px solid #333;border-radius:6px;padding:6px 10px;min-height:60px}</style></head><body>
<div id="left"><img id="view" src="" onclick="pick(event)"><div id="msg"></div></div>
<div id="right">
<h3>Pose the creature, then save</h3>
<h4>selected limb (click the image)</h4>
<div id="jpanel"><div id="sel">nothing selected -- click a limb</div><div id="jsliders"></div></div>
<h4>whole-body position</h4>
<label>x <span class="val" id="vx">0</span></label><input type="range" id="x" min="-2" max="2" step="0.02" value="0">
<label>y <span class="val" id="vy">0</span></label><input type="range" id="y" min="-2" max="2" step="0.02" value="0">
<label>z <span class="val" id="vz">0.6</span></label><input type="range" id="z" min="0" max="2.5" step="0.01" value="0.6">
<h4>whole-body rotation (deg)</h4>
<label>roll <span class="val" id="vroll">0</span></label><input type="range" id="roll" min="-180" max="180" step="1" value="0">
<label>pitch <span class="val" id="vpitch">0</span></label><input type="range" id="pitch" min="-180" max="180" step="1" value="0">
<label>yaw <span class="val" id="vyaw">0</span></label><input type="range" id="yaw" min="-180" max="180" step="1" value="0">
<h4>camera</h4>
<label>azimuth <span class="val" id="vcaz">90</span></label><input type="range" id="caz" min="0" max="360" step="2" value="90">
<label>elevation <span class="val" id="vcel">-15</span></label><input type="range" id="cel" min="-89" max="10" step="1" value="-15">
<label>distance <span class="val" id="vcd">3</span></label><input type="range" id="cd" min="0.5" max="8" step="0.1" value="3">
<button onclick="save()">Save: this pose is right side up</button>
</div>
<script>
const ids=['x','y','z','roll','pitch','yaw','caz','cel','cd'];
let joints={};          // name -> deg (every joint, persisted client-side)
let selected=[];        // [{name,lo,hi}] currently shown sliders
let pending=false,dirty=false;
function state(){const o={joints:joints};for(const i of ids)o[i]=parseFloat(document.getElementById(i).value);return o;}
function labels(){for(const i of ids)document.getElementById('v'+i).textContent=document.getElementById(i).value;}
function render(){
 if(pending){dirty=true;return;} pending=true;
 fetch('/render',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(state())})
 .then(r=>r.json()).then(d=>{document.getElementById('view').src='data:image/jpeg;base64,'+d.img;
  pending=false;if(dirty){dirty=false;render();}});
}
function pick(e){
 const r=e.target.getBoundingClientRect();
 const body=state(); body.px=(e.clientX-r.left)/r.width; body.py=(e.clientY-r.top)/r.height;
 fetch('/pick',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
 .then(r=>r.json()).then(d=>{
  selected=d.joints;
  document.getElementById('sel').textContent=d.body?('selected: '+d.body):'nothing selected -- click a limb';
  const div=document.getElementById('jsliders');div.innerHTML='';
  for(const j of d.joints){
   const cur=(joints[j.name]||0);
   div.innerHTML+=`<label>${j.name} <span class="val" id="vj_${j.name}">${cur}</span> (range ${j.lo}..${j.hi})</label>
    <input type="range" id="j_${j.name}" min="${j.lo}" max="${j.hi}" step="1" value="${cur}"
     oninput="jmove('${j.name}')">`;
  }
  render();
 });
}
function jmove(n){const v=document.getElementById('j_'+n).value;joints[n]=parseFloat(v);
 document.getElementById('vj_'+n).textContent=v;render();}
function save(){
 fetch('/label',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(state())})
 .then(r=>r.json()).then(d=>{document.getElementById('msg').textContent='saved! up_local='+JSON.stringify(d.up_local);});
}
for(const i of ids)document.getElementById(i).addEventListener('input',()=>{labels();render();});
labels();render();
</script></body></html>"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--creature-xml", default="creature_configs/three_seg_worm.xml")
    p.add_argument("--out", default=None)
    p.add_argument("--port", type=int, default=8096)
    args = p.parse_args()
    out = args.out or os.path.join(
        "creature_configs",
        os.path.splitext(os.path.basename(args.creature_xml))[0] + "_up_axis.json")

    model, meta = build_creature_scene(args.creature_xml, ball=None)
    data = mujoco.MjData(model)
    ren = mujoco.Renderer(model, height=H, width=W)
    qr = meta.qpos_root
    base_rgba = model.geom_rgba.copy()

    # Hinge joints of the creature: name -> qpos addr + range (deg).
    hinges = {}
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            jm = model.joint(j)
            lo, hi = np.rad2deg(jm.range) if jm.limited else (-180.0, 180.0)
            hinges[jm.name] = {"adr": int(jm.qposadr[0]),
                               "body": int(model.jnt_bodyid[j]),
                               "lo": float(np.floor(lo)), "hi": float(np.ceil(hi))}
    creature_bodies = set(int(b) for b in meta.body_ids)
    sel_state = {"body": None}

    def apply(v):
        data.qpos[:] = 0.0
        data.qvel[:] = 0.0
        data.qpos[qr:qr + 3] = v["x"], v["y"], v["z"]
        data.qpos[qr + 3:qr + 7] = rpy_to_quat(
            *np.deg2rad([v["roll"], v["pitch"], v["yaw"]]))
        for name, deg in (v.get("joints") or {}).items():
            if name in hinges:
                data.qpos[hinges[name]["adr"]] = np.deg2rad(float(deg))
        mujoco.mj_forward(model, data)

    def camera(v):
        cam = mujoco.MjvCamera()
        cam.lookat[:] = data.xpos[meta.root_body]
        cam.azimuth, cam.elevation, cam.distance = v["caz"], v["cel"], v["cd"]
        return cam

    def shot(v):
        apply(v)
        # highlight the selected body's geoms
        model.geom_rgba[:] = base_rgba
        if sel_state["body"] is not None:
            gsel = np.nonzero(model.geom_bodyid == sel_state["body"])[0]
            model.geom_rgba[gsel] = [1.0, 0.85, 0.1, 1.0]
        ren.update_scene(data, camera=camera(v))
        from PIL import Image
        b = io.BytesIO()
        Image.fromarray(ren.render()).save(b, format="JPEG", quality=85)
        return base64.b64encode(b.getvalue()).decode()

    app = Flask(__name__)

    @app.route("/")
    def index():
        return PAGE

    @app.route("/render", methods=["POST"])
    def render():
        return jsonify(img=shot(request.get_json()))

    @app.route("/pick", methods=["POST"])
    def pick():
        v = request.get_json()
        apply(v)
        scn = mujoco.MjvScene(model, maxgeom=2000)
        opt = mujoco.MjvOption()
        cam = camera(v)
        mujoco.mjv_updateScene(model, data, opt, None, cam,
                               mujoco.mjtCatBit.mjCAT_ALL, scn)
        selpnt = np.zeros(3)
        geomid = np.array([-1], dtype=np.int32)
        flexid = np.array([-1], dtype=np.int32)
        skinid = np.array([-1], dtype=np.int32)
        body = mujoco.mjv_select(model, data, opt, W / H,
                                 float(v["px"]), 1.0 - float(v["py"]),
                                 scn, selpnt, geomid, flexid, skinid)
        joints = []
        if body in creature_bodies:
            sel_state["body"] = int(body)
            # joints on the picked body; if none (e.g. clicked the root), walk
            # DOWN is ambiguous, so offer the root's directly-attached hinges.
            names = [n for n, hj in hinges.items() if hj["body"] == body]
            if not names:
                names = [n for n, hj in hinges.items()
                         if int(model.body_parentid[hj["body"]]) == body]
            joints = [{"name": n, "lo": hinges[n]["lo"], "hi": hinges[n]["hi"]}
                      for n in names]
            bname = model.body(body).name
        else:
            sel_state["body"] = None
            bname = None
        return jsonify(body=bname, joints=joints)

    @app.route("/label", methods=["POST"])
    def label():
        v = request.get_json()
        q = rpy_to_quat(*np.deg2rad([v["roll"], v["pitch"], v["yaw"]]))
        up_local = quat_to_mat(q).T @ np.array([0.0, 0.0, 1.0])
        joints_rad = {n: float(np.deg2rad(float(d)))
                      for n, d in (v.get("joints") or {}).items() if n in hinges}
        payload = {"creature_xml": args.creature_xml,
                   "pose": {k: float(v[k]) for k in
                            ("x", "y", "z", "roll", "pitch", "yaw")},
                   "quat_wxyz": [round(float(x), 6) for x in q],
                   "joints": {n: round(r, 6) for n, r in joints_rad.items()},
                   "up_local": [round(float(x), 6) for x in up_local]}
        with open(out, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[label] saved {out}: {payload}", flush=True)
        return jsonify(ok=True, up_local=payload["up_local"])

    print(f"[label] serving on http://0.0.0.0:{args.port} -> writes {out}",
          flush=True)
    app.run(host="0.0.0.0", port=args.port, threaded=False, use_reloader=False)


if __name__ == "__main__":
    main()
