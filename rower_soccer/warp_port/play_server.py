"""Browser-served interactive play: drive the worm with trained policies, live.

Warp physics (1 world), EGL top-down rendering, streamed to a browser over HTTP.

    Q / "Follow" button  -> follow policy (follow_base) drives the worm
    W / "Dribble" button -> dribble policy (cur1_ws) shepherds the ball
    click the arena       -> set the target for the active skill
    Space / "Stop"        -> sit still (zero action)

Warp is ground truth: the interactive sim runs the SAME physics the policies trained
in, so there is no sim2sim gap in the demo. It is headless -- renders via EGL and
serves frames over HTTP -- so it runs on a display-less pod; reach it through an
SSH-forwarded port.

    MUJOCO_GL=egl .venv/bin/python -m rower_soccer.warp_port.play_server \
        --follow runs_v2/follow_base/latest.pt \
        --dribble runs_v2/cur1_ours_ws/latest.pt --port 8080

Follow is polished; dribble is a curriculum-stage-1 snapshot, so it dribbles best
when the target keeps the ball roughly ahead of the worm.
"""
import argparse
import io
import time

import numpy as np
import torch
from PIL import Image
from flask import Flask, Response, request, jsonify

from rower_soccer.warp_port.dribble_env import WarpDribbleEnv
from rower_soccer.warp_port.render import WarpRenderer
from rower_soccer.warp_port.ppo import ActorCritic, load_pretrained

VIEW_HALF = 12.0        # metres from centre to frame edge (matches the topdown cam)
CAM_HEIGHT = 25.0
PX = 640                # square frame

# --- shared state. The server is SINGLE-THREADED (Flask threaded=False): every
# request runs in the one main thread, which is mandatory here -- mujoco's EGL
# context and Warp's CUDA context share the GPU and a second live thread anywhere in
# the process makes eglMakeCurrent fail with EGL_BAD_ACCESS. So there is no
# background sim loop: /frame steps the sim and renders inline. -------------------
state = {"skill": "none", "target": np.zeros(2, dtype=np.float64)}


def pixel_to_world(px, py, w, h):
    """Top-down affine: image-right -> world +x, image-up -> world +y, centre = origin."""
    x = (px / w * 2.0 - 1.0) * VIEW_HALF
    y = (1.0 - py / h * 2.0) * VIEW_HALF
    return float(x), float(y)


def step_and_render(env, ren, follow_ac, dribble_ac):
    """Advance one control step under the active skill, return a JPEG frame."""
    tgt = torch.tensor(state["target"], dtype=torch.float32, device="cuda")
    env.target_xy[0] = tgt
    env.target_vel[0] = 0.0                     # user-commanded target is static

    obs = env._obs()                            # 39-dim dribble obs
    with torch.no_grad():
        if state["skill"] == "dribble":
            a = dribble_ac.dist(obs.float()).mean.clamp(-1, 1)
        elif state["skill"] == "follow":
            # follow obs = dribble obs minus the leading 6 ball dims (33-dim)
            a = follow_ac.dist(obs[:, 6:].float()).mean.clamp(-1, 1)
        else:
            a = torch.zeros(1, env.act_dim, device="cuda")
    env.step(a)

    frame = ren.frame(env, w=0)
    b = io.BytesIO(); Image.fromarray(frame).save(b, format="JPEG", quality=80)
    return b.getvalue()


PAGE = """<!doctype html><html><head><title>worm play</title>
<style>body{background:#111;color:#eee;font-family:sans-serif;text-align:center;margin:0;padding:12px}
img{border:2px solid #333;cursor:crosshair;touch-action:none}
button{font-size:16px;padding:8px 16px;margin:4px;border:0;border-radius:6px;cursor:pointer}
#follow{background:#4a90d9;color:#fff}#dribble{background:#5cb85c;color:#fff}#stop{background:#666;color:#fff}
.on{outline:3px solid #fff}#hint{color:#888;font-size:13px}</style></head>
<body>
<h3>worm interactive play &mdash; Warp physics</h3>
<div><button id="follow" onclick="setSkill('follow')">Q &middot; Follow</button>
<button id="dribble" onclick="setSkill('dribble')">W &middot; Dribble</button>
<button id="stop" onclick="setSkill('none')">Space &middot; Stop</button></div>
<div><img id="view" width="640" height="640" onclick="click_(event)"></div>
<p id="hint">Pick a skill, then click the arena to set a target. Red sphere = target.</p>
<script>
const view=document.getElementById('view');
function setSkill(s){fetch('/skill',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({skill:s})});
 for(const id of ['follow','dribble','stop']){document.getElementById(id).classList.toggle('on',(id=='stop'?s=='none':id==s));}}
function click_(e){const r=e.target.getBoundingClientRect();
 fetch('/click',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({px:e.clientX-r.left,py:e.clientY-r.top,w:r.width,h:r.height})});}
document.addEventListener('keydown',e=>{if(e.key=='q'||e.key=='Q')setSkill('follow');
 else if(e.key=='w'||e.key=='W')setSkill('dribble');else if(e.key==' ')setSkill('none');});
// Poll /frame: each request advances the sim one step and returns a JPEG. Chaining
// on load keeps it as fast as the render allows without piling up requests.
function tick(){fetch('/frame').then(r=>r.blob()).then(b=>{
  view.src=URL.createObjectURL(b);});}
view.onload=()=>{URL.revokeObjectURL(view.src);requestAnimationFrame(tick);};
setSkill('none'); tick();
</script></body></html>"""


def make_app(env, ren, follow_ac, dribble_ac):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return PAGE

    @app.route("/frame")
    def frame():
        jpeg = step_and_render(env, ren, follow_ac, dribble_ac)
        return Response(jpeg, mimetype="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.route("/skill", methods=["POST"])
    def skill():
        s = request.get_json()["skill"]
        state["skill"] = s if s in ("follow", "dribble", "none") else "none"
        return jsonify(ok=True)

    @app.route("/click", methods=["POST"])
    def click():
        d = request.get_json()
        x, y = pixel_to_world(d["px"], d["py"], d["w"], d["h"])
        state["target"] = np.array([x, y], dtype=np.float64)
        return jsonify(x=x, y=y)

    return app


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--follow", required=True, help="follow policy .pt (33-dim)")
    p.add_argument("--dribble", required=True, help="dribble policy .pt (39-dim)")
    p.add_argument("--creature-xml", default="creature_configs/three_seg_worm.xml")
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    # One dribble scene (has the ball) serves both skills; follow just ignores it.
    env = WarpDribbleEnv(num_worlds=1, use_graph=False, seed=0,
                         creature_xml=args.creature_xml, episode_seconds=1e6)
    ren = WarpRenderer(args.creature_xml, has_ball=True, width=PX, height=PX,
                       topdown=True, view_half=VIEW_HALF, cam_height=CAM_HEIGHT)

    dribble_ac = ActorCritic(env.obs_dim, env.act_dim,
                             proprio_indices=env.proprio_indices.tolist(),
                             task_indices=env.task_indices.tolist(), z_dim=16).cuda()
    load_pretrained(dribble_ac, args.dribble, device="cuda")
    dribble_ac.eval()

    # follow: 33-dim obs = dribble obs minus ball_ego, so proprio [0:29], task [29:33].
    follow_ac = ActorCritic(33, env.act_dim, proprio_indices=list(range(0, 29)),
                            task_indices=list(range(29, 33)), z_dim=16).cuda()
    load_pretrained(follow_ac, args.follow, device="cuda")
    follow_ac.eval()

    env.reset()
    state["target"] = env.target_xy[0].detach().cpu().numpy().copy()

    # threaded=False: every request runs in this one main thread, so the EGL render
    # context is never touched from a second thread (see the note above). The browser
    # drives the sim by polling /frame.
    print(f"[play] serving on http://0.0.0.0:{args.port}  "
          f"(forward this port and open http://localhost:{args.port})", flush=True)
    make_app(env, ren, follow_ac, dribble_ac).run(
        host="0.0.0.0", port=args.port, threaded=False, use_reloader=False)


if __name__ == "__main__":
    main()
