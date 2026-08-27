"""Pick team colours against the ACTUAL renderer, not against a swatch.

A hex code does not tell you what a creature will look like on the pitch: the
scene has four lights, the material has specular and shininess, and the turf
is a saturated green that everything is seen against. `#faaca5` is a pale pink
in a colour picker and reads as YELLOW on an ant. Guessing hex codes and
re-rendering by hand is the slow way to find that out.

So: a page with colour inputs, and a live render of the four creatures on the
pitch in those colours. Choose by looking.

    MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python scripts/color_picker.py --port 8099

Nothing here writes to the project: it renders a throwaway MatchSim and returns
JPEGs. When you have the colours you want, they go in `match.PLAYER_HEX`.
"""

import argparse
import io
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

PAGE = """<!doctype html><meta charset=utf-8>
<title>team colours</title>
<style>
 body{background:#14161a;color:#dfe3ea;font:14px/1.5 system-ui,sans-serif;
      margin:0;padding:18px}
 h1{font-size:15px;font-weight:600;letter-spacing:.02em;margin:0 0 14px}
 .row{display:flex;gap:18px;align-items:center;flex-wrap:wrap;margin-bottom:14px}
 label{display:flex;gap:8px;align-items:center}
 input[type=color]{width:52px;height:34px;border:1px solid #2c313a;border-radius:6px;
      background:none;padding:2px;cursor:pointer}
 input[type=text]{width:88px;background:#1c1f26;color:#dfe3ea;border:1px solid #2c313a;
      border-radius:6px;padding:6px 8px;font-family:ui-monospace,monospace}
 img{width:100%;max-width:1000px;border-radius:8px;border:1px solid #2c313a;display:block}
 .hint{color:#8b93a1;font-size:12px;margin-top:10px}
 code{background:#1c1f26;padding:2px 6px;border-radius:4px}
</style>
<h1>team colours &mdash; rendered on the pitch</h1>
<div class=row id=inputs></div>
<img id=shot alt="pitch">
<div class=hint>Colours apply to the creature, its target disc and its aim line.
Copy the four values into <code>PLAYER_HEX</code> in
<code>rower_soccer/game/match.py</code>.</div>
<script>
const DEF = ["#598eff", "#59d0ff", "#faaca5", "#c91000"];
const NAMES = ["home 1", "home 2", "away 1", "away 2"];
const box = document.getElementById("inputs");
const shot = document.getElementById("shot");
const state = DEF.slice();
DEF.forEach((hex, i) => {
  const wrap = document.createElement("label");
  wrap.innerHTML = `<span>${NAMES[i]}</span>`;
  const c = document.createElement("input"); c.type = "color"; c.value = hex;
  const t = document.createElement("input"); t.type = "text"; t.value = hex;
  const sync = (v) => { state[i] = v; c.value = v; t.value = v; refresh(); };
  c.oninput = () => sync(c.value);
  t.onchange = () => { if (/^#[0-9a-f]{6}$/i.test(t.value)) sync(t.value); };
  wrap.append(c, t); box.append(wrap);
});
let pending = null;
function refresh() {
  // One render in flight at a time: dragging a colour picker fires continuously
  // and each render is a real MuJoCo frame.
  if (pending) { pending = "again"; return; }
  pending = "busy";
  const q = state.map((h, i) => `c${i}=${encodeURIComponent(h)}`).join("&");
  const img = new Image();
  img.onload = img.onerror = () => {
    shot.src = img.src;
    const again = pending === "again"; pending = null;
    if (again) refresh();
  };
  img.src = "/render?" + q + "&t=" + Date.now();
}
refresh();
</script>
"""


class Picker(BaseHTTPRequestHandler):
    sim = None
    lock = threading.Lock()

    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if u.path == "/render":
            q = parse_qs(u.query)
            hexes = [(q.get(f"c{i}") or ["#888888"])[0] for i in range(4)]
            jpeg = self.render(hexes)
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(jpeg)))
            self.end_headers()
            self.wfile.write(jpeg)
            return
        self.send_response(404)
        self.end_headers()

    def render(self, hexes):
        from PIL import Image
        with Picker.lock:
            sim = Picker.sim
            m = sim.physics.model
            for i, hx in enumerate(hexes):
                rgba = _rgba(hx)
                # Both the material AND the geoms, for the reason recorded in
                # match.marker_rgba: where a geom has a material MuJoCo renders
                # the material, so painting one leaves the other stale.
                for mid in sim._preview_matids[i]:
                    m.mat_rgba[mid] = rgba
                for gid in sim._preview_geomids[i]:
                    m.geom_rgba[gid] = rgba
                sim._markers[i].rgba = rgba
                for g in sim._dashes[i]:
                    g.rgba = rgba
            sim.set_camera("topdown")
            frame = sim.render()
            buf = io.BytesIO()
            Image.fromarray(frame).save(buf, format="JPEG", quality=88)
            return buf.getvalue()


def _rgba(hx):
    hx = hx.lstrip("#")
    return np.array([int(hx[k:k + 2], 16) / 255.0 for k in (0, 2, 4)] + [1.0])


def main():
    import mujoco
    from rower_soccer.game.match import SLOTS, MatchSim

    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8099)
    args = p.parse_args()

    sim = MatchSim(render_size=(1000, 660), shadows=False, seed=0)
    sim.start_match()
    for c in sim.commands:
        c.skill = "scripted"
    for i, c in enumerate(sim.commands):
        c.target = [(7.0, 3.5), (-4.0, 4.5), (3.0, -4.5), (-6.5, -2.5)][i]
    for _ in range(140):
        sim.step()

    # Resolve each player's material and geom ids ONCE; the request path only
    # writes colours, so dragging a picker never recompiles anything.
    m = sim.physics.model
    sim._preview_matids, sim._preview_geomids = [], []
    for i, pl in enumerate(sim.task.players):
        prefix = pl.walker.mjcf_model.model + "/"
        gids = [g for g in range(m.ngeom)
                if (mujoco.mj_id2name(m.ptr, mujoco.mjtObj.mjOBJ_GEOM, g) or "")
                .startswith(prefix)]
        sim._preview_geomids.append(gids)
        sim._preview_matids.append(sorted({int(m.geom_matid[g]) for g in gids
                                           if m.geom_matid[g] >= 0}))
    Picker.sim = sim
    # SINGLE-threaded on purpose. A GL context belongs to the thread that made
    # it current, and a threading server hands each request to a fresh worker:
    # the first one claims the context, dies, and every later request fails with
    # "already current on another thread". One thread, one context, one render
    # at a time -- which is all a colour picker needs.
    srv = HTTPServer(("0.0.0.0", args.port), Picker)
    print(f"[picker] http://localhost:{args.port}/", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
