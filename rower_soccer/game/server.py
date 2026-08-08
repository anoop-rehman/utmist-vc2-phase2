"""LAN play server: 4 browsers, one authoritative 2v2 match, one demo file.

    MUJOCO_GL=egl .venv/bin/python -m rower_soccer.game.server \
        --follow runs_v2/follow_ant_v1/best.pt --port 8090

Everyone on the wifi opens http://<host-lan-ip>:8090/ , types a name, claims a seat
(home 1/2, away 1/2), and plays.  Unclaimed seats are driven by the built-in
chase-the-ball baseline so a match is playable with 1-4 humans.

Threading (both halves are load-bearing, same reason as warp_port/play_server.py):

  * ONE sim thread creates and owns the dm_control env, the EGL render context and
    the torch policies, then steps + renders forever.  MuJoCo's EGL context must be
    created AND used on one thread, and keeping the HTTP handlers away from it also
    stops a request from stepping physics inline.
  * Request threads (stdlib ThreadingHTTPServer -- no flask, which is not installed
    in the project venv) touch only small locked state: the command inbox (writes)
    and the latest JPEG / snapshot (reads).  They never touch physics.

Inputs are high-level only: a click sets a world target, a key picks a skill.  The
torques come from the skill experts through the shared frozen decoder -- humans are
the high-level policy, which is the whole point (the 2022 paper's stage 3, with
people standing in for the multi-agent RL that trains next sprint).
"""

from __future__ import annotations

import argparse
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

from rower_soccer.game.lobby import Lobby, SLOTS
from rower_soccer.game import match as M

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


class GameServer:
    """Everything shared between the sim thread and the request threads."""

    def __init__(self, args):
        self.args = args
        self.lobby = Lobby(claim_timeout=args.slot_timeout)
        self.sim: M.MatchSim | None = None
        self.ready = threading.Event()
        self.stop_flag = threading.Event()
        self.error = None       # fatal: the sim thread never started
        self.tick_error = None  # non-fatal: one tick blew up, the server stayed up

        self._cmd_lock = threading.Lock()
        self._inbox: dict[str, dict] = {}         # slot -> pending {skill,target,aim}
        self._control: list = []                  # queued match-control actions

        self._frame_lock = threading.Condition()
        self._jpeg = None
        self._frame_no = 0
        self._snap = {"phase": "starting"}
        self._events: list = []
        self.demos: list = []
        self.stats = {"tick_ms": 0.0, "render_ms": 0.0, "realtime": 1.0, "late": 0,
                      "frames_dropped": 0}

    # -- request-thread API ------------------------------------------------
    def push_input(self, slot, **fields):
        with self._cmd_lock:
            self._inbox.setdefault(slot, {}).update(fields)

    def push_control(self, action, **kw):
        with self._cmd_lock:
            self._control.append(dict(action=action, **kw))

    def snapshot(self, token=None):
        with self._frame_lock:
            snap = dict(self._snap)
            events = list(self._events)
            frame_no = self._frame_no
        out = dict(snap)
        out["lobby"] = self.lobby.state()
        out["events"] = events
        out["frame"] = frame_no
        out["stats"] = dict(self.stats)
        out["tick_error"] = self.tick_error
        out["demos"] = self.demos[-5:]
        out["available_skills"] = list(self._skills)
        out["match_seconds"] = self.args.match_seconds
        if token:
            c = self.lobby.get(token)
            out["me"] = None if c is None else dict(name=c.name, slot=c.slot)
        return out

    def wait_frame(self, last_no, timeout=2.0):
        with self._frame_lock:
            if self._frame_no == last_no:
                self._frame_lock.wait(timeout)
            return self._jpeg, self._frame_no

    # -- sim thread --------------------------------------------------------
    def run_sim(self):
        try:
            self._run_sim()
        except BaseException as exc:            # noqa: BLE001 - surface it, don't hang
            import traceback
            self.error = "".join(traceback.format_exception(exc))
            print(self.error, flush=True)
            self.ready.set()

    def _build(self):
        from PIL import Image
        import torch
        from rower_soccer.game.skills import build_controller
        a = self.args

        # MEASURED, not superstition: on a 48-core host, torch's default intra-op
        # pool turns a 256-wide Linear into an 8 ms call, and one tick (4 players x
        # ~9 matmuls) into 536 ms -- 20x SLOWER than realtime. Pinned to one thread
        # the same tick is 13 ms. The experts are tiny MLPs; parallelising them is
        # pure synchronisation overhead.
        torch.set_num_threads(max(1, int(a.torch_threads)))

        sim = M.MatchSim(creature=a.creature, pitch_half=tuple(a.pitch_half),
                         match_seconds=a.match_seconds, seed=a.seed,
                         physics_dt=a.physics_dt,
                         render_size=(a.width, a.height), shadows=a.shadows,
                         countdown=a.countdown)
        ck = {k: v for k, v in dict(follow=a.follow, dribble=a.dribble,
                                    kick=a.kick, shoot=a.shoot).items() if v}
        sim.controller = build_controller(sim.env, creature=a.creature,
                                          checkpoints=ck, device=a.device,
                                          target_clip=a.target_clip,
                                          action_mode=a.action_mode, seed=a.seed)
        sim.controller.bind(sim.env)
        self._skills = tuple(sim.controller.skills)
        self.sim = sim
        self._Image = Image
        return sim

    def _run_sim(self):
        sim = self._build()
        self._publish_frame(sim)
        self.ready.set()

        dt = sim.control_dt
        render_every = max(1, int(round((1.0 / dt) / max(1, self.args.render_hz))))
        next_t = time.perf_counter()
        i = 0
        while not self.stop_flag.is_set():
            t0 = time.perf_counter()
            # Order matters. Seats first: `start` snapshots who is playing into the
            # demo header, so a claim that landed in this same tick must already be
            # visible or the demo labels a human seat "scripted". Inputs last:
            # `start` resets every command, and a click that arrived with the start
            # request should survive it.
            self._autofill(sim)
            self._apply_control(sim)
            self._apply_inputs(sim)
            try:
                sim.step()
            except Exception as exc:            # noqa: BLE001
                # A bad tick must not silently freeze the world: without this the
                # loop dies, the HTTP layer keeps serving the last frame, and four
                # people stare at a still picture wondering whose wifi broke. Keep
                # whatever the match recorded, say so on /state, and stay up.
                import traceback
                self.tick_error = "".join(traceback.format_exception(exc))
                print(self.tick_error, flush=True)
                kept = sim.end_match("error")
                if kept:
                    self.demos.append(kept)
                sim.phase = M.PHASE_LOBBY
                for c in sim.commands:
                    c.skill = "idle"
                self._publish_state(sim)
                time.sleep(0.5)
                continue
            t1 = time.perf_counter()
            # DROP FRAMES, NEVER TICKS. The physics tick is the authoritative state
            # that four humans and the demo file all depend on; a rendered frame is
            # a picture of it. When the host is busy (a shared box with a training
            # job on it is the normal case here), skipping the picture keeps the
            # match on the wall clock, which is what "feels controllable" means.
            behind = time.perf_counter() - next_t
            if i % render_every == 0 and behind < dt:
                self._publish_frame(sim)
            elif i % render_every == 0:
                self.stats["frames_dropped"] += 1
            t2 = time.perf_counter()
            self._publish_state(sim)

            self.stats["tick_ms"] = round((t1 - t0) * 1000, 2)
            if t2 - t1 > 1e-4:
                self.stats["render_ms"] = round((t2 - t1) * 1000, 2)
            i += 1
            next_t += dt
            rem = next_t - time.perf_counter()
            if rem < -0.5:
                # More than half a second behind: give up on catching up, or the
                # loop sprints through ticks nobody saw and the match "skips".
                self.stats["late"] += 1
                next_t = time.perf_counter()
            self.stats["realtime"] = round(dt / max(1e-6, time.perf_counter() - t0), 2)
            if rem > 0:
                time.sleep(rem)
        sim.abort()

    def _apply_control(self, sim):
        with self._cmd_lock:
            ctl, self._control = self._control, []
        a = self.args
        for c in ctl:
            if c["action"] == "start":
                path = None
                if a.demo_dir:
                    os.makedirs(a.demo_dir, exist_ok=True)
                    path = os.path.join(
                        a.demo_dir,
                        f"{time.strftime('%Y%m%d-%H%M%S')}_{a.creature}_2v2")
                sim.start_match(demo_path=path)
                print(f"[game] match {sim.match_id} started -> {sim.demo_path}", flush=True)
            elif c["action"] == "stop":
                p = sim.end_match("stopped")
                if p:
                    self.demos.append(p)
        # auto: start when the seats are ready, restart after the whistle
        if sim.phase == M.PHASE_ENDED:
            if sim.last_demo and (not self.demos or self.demos[-1] != sim.last_demo):
                self.demos.append(sim.last_demo)
                print(f"[game] demo written: {sim.last_demo}", flush=True)
        if a.auto_start and sim.phase in (M.PHASE_LOBBY, M.PHASE_ENDED):
            humans = sum(1 for s in SLOTS if self.lobby.occupant(s))
            ready = humans >= a.auto_start
            if sim.phase == M.PHASE_ENDED:
                self._ended_at = getattr(self, "_ended_at", None) or time.time()
                ready = ready and (time.time() - self._ended_at) > a.restart_delay
            if ready:
                self._ended_at = None
                self.push_control("start")
        if sim.phase != M.PHASE_ENDED:
            self._ended_at = None

    def _apply_inputs(self, sim):
        with self._cmd_lock:
            inbox, self._inbox = self._inbox, {}
        for slot, d in inbox.items():
            p = SLOTS.index(slot)
            cmd = sim.commands[p]
            if "target" in d:
                cmd.target = np.asarray(d["target"], np.float64)
                cmd.aim = np.asarray(d.get("aim", (0.0, 0.0)), np.float64)
                sim._emit("target_set", player=p, slot=slot,
                          target=[float(x) for x in cmd.target],
                          aim=[float(x) for x in cmd.aim])
            if "skill" in d and d["skill"] != cmd.skill:
                cmd.skill = d["skill"]
                sim._emit("skill_change", player=p, slot=slot, skill=cmd.skill,
                          target=[float(x) for x in cmd.target])

    def _autofill(self, sim):
        """Seats with no live client are driven by the built-in baseline, so a match
        is playable with 1 human and still records 4 usable trajectories."""
        fill = self.args.fill
        for p, slot in enumerate(SLOTS):
            c = self.lobby.occupant(slot)
            cmd = sim.commands[p]
            if c is not None:
                if cmd.controller != "human":
                    cmd.controller, cmd.name = "human", c.name
                    if cmd.skill == "scripted":
                        cmd.skill = "idle"
                elif cmd.name != c.name:
                    cmd.name = c.name
            elif cmd.controller != fill:
                cmd.controller, cmd.name = fill, ""
                cmd.skill = "scripted" if fill == "scripted" else "idle"

    def _publish_frame(self, sim):
        frame = sim.render()
        b = io.BytesIO()
        self._Image.fromarray(frame).save(b, format="JPEG", quality=self.args.quality)
        with self._frame_lock:
            self._jpeg = b.getvalue()
            self._frame_no += 1
            self._frame_lock.notify_all()

    def _publish_state(self, sim):
        snap = sim.snapshot()
        evs = sim.drain_events()
        with self._frame_lock:
            self._snap = snap
            if evs:
                self._events = (self._events + evs)[-30:]


# --------------------------------------------------------------------------
# HTTP -- stdlib only
# --------------------------------------------------------------------------
# No flask/werkzeug: they are not in the project venv (warp_port/play_server.py's
# import of flask does not actually resolve there), and the sprint's headline
# deliverable should not be blocked on a pip install on somebody's laptop.
# `ThreadingHTTPServer` gives a thread per connection, which is what the
# long-lived MJPEG streams want anyway, and 4-8 clients is nothing.

class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    gs: GameServer = None          # set on the subclass in make_httpd

    # -- plumbing ----------------------------------------------------------
    def log_message(self, fmt, *args):
        pass                        # one line per poll x 4 clients x 5 Hz = noise

    def _send(self, code, body=b"", ctype="application/json", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if body and self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj, default=str).encode(), "application/json")

    def _body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if not n:
            return {}
        try:
            return json.loads(self.rfile.read(n).decode() or "{}")
        except ValueError:
            return {}

    def _static(self, name):
        # basename() is the whole path-traversal defence: this only ever serves the
        # three files next to this module.
        path = os.path.join(STATIC_DIR, os.path.basename(name))
        if not os.path.isfile(path):
            return self._send(404, b"not found", "text/plain")
        ctype = {".html": "text/html", ".js": "text/javascript",
                 ".css": "text/css"}.get(os.path.splitext(path)[1], "text/plain")
        with open(path, "rb") as f:
            self._send(200, f.read(), ctype + "; charset=utf-8")

    # -- routes ------------------------------------------------------------
    def do_GET(self):
        gs = self.gs
        u = urlparse(self.path)
        q = parse_qs(u.query)
        token = (q.get("token") or [None])[0]
        if u.path == "/":
            return self._static("index.html")
        if u.path.startswith("/static/"):
            return self._static(u.path[len("/static/"):])
        if u.path == "/health":
            return self._json(dict(ok=gs.error is None, ready=gs.ready.is_set(),
                                   error=gs.error))
        if u.path == "/state":
            return self._json(gs.snapshot(token))
        if u.path == "/frame":
            jpeg, _ = gs.wait_frame(-1, timeout=0.0)
            if jpeg is None:
                return self._send(503, b"no frame yet", "text/plain")
            return self._send(200, jpeg, "image/jpeg")
        if u.path == "/stream":
            return self._stream()
        return self._send(404, b"not found", "text/plain")

    def _stream(self):
        """MJPEG. One frame per published frame -- no polling, no duplicate JPEGs:
        `wait_frame` blocks on the sim thread's condition variable."""
        self.send_response(200)
        self.send_header("Content-Type",
                         "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        last = -1
        try:
            while not self.gs.stop_flag.is_set():
                jpeg, last = self.gs.wait_frame(last, timeout=2.0)
                if jpeg is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                 b"Content-Length: " + str(len(jpeg)).encode()
                                 + b"\r\n\r\n" + jpeg + b"\r\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass                     # a client closed the tab; entirely normal

    def do_POST(self):
        gs = self.gs
        u = urlparse(self.path)
        d = self._body()
        token = d.get("token") or self.headers.get("X-Token")

        if u.path == "/join":
            c = gs.lobby.join(d.get("name", ""), token)
            return self._json(dict(token=c.token, name=c.name, slot=c.slot,
                                   slots=list(SLOTS)))

        c = gs.lobby.get(token) if token else None
        if c is None:
            return self._json(dict(ok=False, error="unknown token; join first"), 403)

        if u.path == "/claim":
            slot = d.get("slot")
            ok, msg = gs.lobby.claim(token, slot)
            if ok and gs.sim is not None:
                gs.sim._emit("slot_claim", slot=slot, player=SLOTS.index(slot),
                             name=c.name)
            return self._json(dict(ok=ok, error=None if ok else msg, slot=c.slot))

        if u.path == "/release":
            slot = gs.lobby.release(token)
            if slot and gs.sim is not None:
                gs.sim._emit("slot_release", slot=slot, player=SLOTS.index(slot),
                             name=c.name)
            return self._json(dict(ok=True, slot=None))

        if u.path == "/input":
            return self._json(*_handle_input(gs, c, d))

        if u.path == "/control":
            if d.get("action") not in ("start", "stop"):
                return self._json(dict(ok=False, error="action must be start|stop"), 400)
            gs.push_control(d["action"])
            return self._json(dict(ok=True))

        return self._send(404, b"not found", "text/plain")


def _handle_input(gs, client, d):
    """The ONLY way a human moves a creature. Returns (payload, status).

    The slot comes from the token, server-side; the request has no field that could
    name a different one. That is what makes cross-slot input impossible rather than
    merely checked -- and a mislabelled demo row is worse than a dropped input.
    """
    if client.slot is None:
        return dict(ok=False, error="spectators cannot send inputs"), 403
    if gs.sim is None:
        return dict(ok=False, error="sim not ready"), 503
    fields = {}
    if "skill" in d:
        s = str(d["skill"])
        if s not in gs._skills and s != "idle":
            return dict(ok=False, error=f"skill {s!r} is not trained yet",
                        available=list(gs._skills)), 400
        fields["skill"] = s
    if "u" in d and "v" in d:
        # Normalized click coordinates, so the affine lives here (one place) and a
        # resized window / rotated phone / different render resolution cannot
        # desync input from picture.
        u = min(max(float(d["u"]), 0.0), 1.0)
        v = min(max(float(d["v"]), 0.0), 1.0)
        fields["target"] = gs.sim.uv_to_world(u, v)
        ax = float(d.get("aim_u", 0.0)) * gs.sim.half_x * 2.0
        ay = -float(d.get("aim_v", 0.0)) * gs.sim.half_y * 2.0
        n = float(np.hypot(ax, ay))
        fields["aim"] = (ax / n, ay / n) if n > 1e-6 else (0.0, 0.0)
    elif "x" in d and "y" in d:                  # world coords, for scripted clients
        fields["target"] = (float(d["x"]), float(d["y"]))
    if not fields:
        return dict(ok=False, error="nothing to do"), 400
    gs.push_input(client.slot, **fields)
    out = {k: (list(v) if isinstance(v, tuple) else v) for k, v in fields.items()}
    return dict(ok=True, slot=client.slot, **out), 200


def make_httpd(gs: GameServer, host="0.0.0.0", port=8090):
    handler = type("Handler", (_Handler,), {"gs": gs})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True     # a hung stream must never block shutdown
    return httpd


def lan_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def build_parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # Checkpoints default to whatever rower_soccer.skills' registry knows; these
    # flags only OVERRIDE it, so a fresh WS1 run needs no code change on either side.
    p.add_argument("--follow", default=None, help="override the follow expert .pt")
    p.add_argument("--dribble", default=None, help="dribble expert .pt (WS1)")
    p.add_argument("--kick", default=None, help="kick expert .pt (WS1+WS2)")
    p.add_argument("--shoot", default=None, help="shoot expert .pt (WS1+WS2)")
    p.add_argument("--action-mode", default="auto", choices=["auto", "mean", "noise"],
                   help="how an expert turns its distribution into torques. `auto` "
                        "picks `noise` for a checkpoint whose action std fills the "
                        "action range -- follow_ant_v1 does, and its MEAN does not "
                        "walk at all. All three replay bit-exact")
    p.add_argument("--target-clip", type=float, default=None,
                   help="metres; re-aim a far click to a nearer waypoint on the same "
                        "bearing (the drills trained inside a +/-10 m box). 0 disables")
    p.add_argument("--creature", default="ant")
    p.add_argument("--pitch-half", type=float, nargs=2, default=(15.0, 11.0),
                   metavar=("X", "Y"),
                   help="pitch HALF extents in m (stock repo pitch is 40 30, which is "
                        "an 80x60 m field -- far too big for a 45 s ant match)")
    p.add_argument("--match-seconds", type=float, default=45.0)
    p.add_argument("--countdown", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--physics-dt", type=float, default=M.PHYSICS_DT,
                   help="0.0025 matches the drills the policies trained on; 0.005 is "
                        "soccer's native dt and ~1.5x cheaper")
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=640)
    p.add_argument("--render-hz", type=int, default=20)
    p.add_argument("--quality", type=int, default=75)
    p.add_argument("--shadows", action="store_true")
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--torch-threads", type=int, default=1,
                   help="1 is right and 20x faster than the default on a many-core "
                        "host; see GameServer._build")
    p.add_argument("--fill", default="scripted", choices=["scripted", "idle"],
                   help="who drives unclaimed seats")
    p.add_argument("--auto-start", type=int, default=0, metavar="N",
                   help="start (and restart) a match once N seats are claimed; 0 = manual")
    p.add_argument("--restart-delay", type=float, default=8.0)
    p.add_argument("--demo-dir", default="demos", help="'' disables recording")
    p.add_argument("--slot-timeout", type=float, default=25.0)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8090)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    gs = GameServer(args)
    t = threading.Thread(target=gs.run_sim, name="sim", daemon=True)
    t.start()
    gs.ready.wait()
    if gs.error:
        raise SystemExit("sim thread failed to start; see traceback above")
    ip = lan_ip()
    print(f"[game] {args.creature} 2v2 | pitch {tuple(args.pitch_half)} half-extents | "
          f"skills {gs._skills}", flush=True)
    print(f"[game] open on any device on this wifi:  http://{ip}:{args.port}/",
          flush=True)
    print(f"[game] (same machine: http://localhost:{args.port}/ ; over ssh, forward "
          f"port {args.port})", flush=True)
    httpd = make_httpd(gs, args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[game] stopping; any match in progress keeps its demo", flush=True)
    finally:
        gs.stop_flag.set()
        httpd.server_close()
        t.join(timeout=5)
        if gs.demos:
            print(f"[game] demos: {gs.demos}", flush=True)


if __name__ == "__main__":
    main()
