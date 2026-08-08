"""Scripted HTTP clients -- four humans, without four humans.

The milestone is "4 people on 4 devices play a 45 s match".  You cannot get four
people into a room to find out that `/claim` 500s, so the whole loop has to be
drivable by a program that talks the same HTTP the browser talks: join, claim a
seat, press skill keys, drag targets, reconnect after a drop.  Everything here goes
through the public endpoints -- no back door into the sim -- so a green run here is
evidence about the real thing, not about a mock.

    # against a running server
    .venv/bin/python -m rower_soccer.game.sim_client --url http://localhost:8090 \
        --slots home_1,home_2,away_1,away_2 --seconds 45 --start

    # the whole thing, server included, no browser (this is the CI check)
    MUJOCO_GL=egl .venv/bin/python -m rower_soccer.game.sim_client --selftest
"""

from __future__ import annotations

import argparse
import json
import random
import threading
import time
import urllib.error
import urllib.request

SLOTS = ("home_1", "home_2", "away_1", "away_2")


class BotClient:
    """One simulated player.  Same endpoints, same token discipline as the browser."""

    def __init__(self, url, name, slot=None, seed=0, skills=("follow",)):
        self.url = url.rstrip("/")
        self.name = name
        self.want_slot = slot
        self.slot = None
        self.token = None
        self.rng = random.Random(seed)
        self.skills = tuple(skills)
        self.errors = []
        self.inputs_sent = 0

    # -- transport ---------------------------------------------------------
    def _req(self, path, body=None, timeout=10.0):
        data = None
        if body is not None:
            body = dict(body)
            body.setdefault("token", self.token)
            data = json.dumps(body).encode()
        req = urllib.request.Request(
            self.url + path, data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if data is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return json.loads(e.read().decode() or "{}")

    # -- lobby -------------------------------------------------------------
    def join(self):
        r = self._req("/join", {"name": self.name, "token": self.token})
        self.token, self.slot = r["token"], r.get("slot")
        return r

    def claim(self, slot=None):
        slot = slot or self.want_slot
        r = self._req("/claim", {"slot": slot})
        if r.get("ok"):
            self.slot = r.get("slot") or slot
        else:
            self.errors.append(f"claim {slot}: {r.get('error')}")
        return r

    def release(self):
        return self._req("/release", {})

    def reconnect(self):
        """Exactly what a phone does after a sleep: same token, new HTTP session."""
        return self.join()

    def state(self):
        return self._req(f"/state?token={self.token}")

    def start_match(self):
        return self._req("/control", {"action": "start"})

    def stop_match(self):
        return self._req("/control", {"action": "stop"})

    # -- play --------------------------------------------------------------
    def set_skill(self, skill):
        r = self._req("/input", {"skill": skill})
        if not r.get("ok"):
            self.errors.append(f"skill {skill}: {r.get('error')}")
        return r

    def click(self, u, v, aim_u=0.0, aim_v=0.0):
        r = self._req("/input", {"u": u, "v": v, "aim_u": aim_u, "aim_v": aim_v})
        if r.get("ok"):
            self.inputs_sent += 1
        else:
            self.errors.append(f"click: {r.get('error')}")
        return r

    def chase_ball(self, st=None):
        """The most human thing a bot can do: drag toward the ball."""
        st = st or self.state()
        b = st.get("ball") or {"u": 0.5, "v": 0.5}
        du = self.rng.uniform(-0.03, 0.03)
        dv = self.rng.uniform(-0.03, 0.03)
        return self.click(min(max(b["u"] + du, 0), 1), min(max(b["v"] + dv, 0), 1),
                          du, dv)

    def play(self, seconds, hz=2.0, stop_event=None):
        """Claim, pick a skill, and keep retargeting until the whistle."""
        self.join()
        if self.want_slot:
            self.claim()
        self.set_skill(self.skills[0])
        t_end = time.time() + seconds
        i = 0
        while time.time() < t_end and not (stop_event and stop_event.is_set()):
            st = self.state()
            if st.get("phase") == "ended" and i > 4:
                break
            self.chase_ball(st)
            if len(self.skills) > 1 and i % 20 == 19:
                self.set_skill(self.skills[(i // 20) % len(self.skills)])
            i += 1
            time.sleep(1.0 / hz)
        return self


def run_bots(url, slots=SLOTS, seconds=45.0, hz=2.0, start=False, skills=("follow",)):
    """Four bots in four threads -- concurrent, like four phones."""
    bots = [BotClient(url, f"bot_{s}", slot=s, seed=i, skills=skills)
            for i, s in enumerate(slots)]
    if start:
        b0 = BotClient(url, "starter"); b0.join()
        for b in bots:
            b.join()
            b.claim()
        b0.start_match()
    stop = threading.Event()
    ts = [threading.Thread(target=b.play, args=(seconds,), kwargs=dict(hz=hz, stop_event=stop))
          for b in bots]
    for t in ts:
        t.start()
    for t in ts:
        t.join(seconds + 30)
    stop.set()
    return bots


# --------------------------------------------------------------------------
# self test: server + bots + demo verification, no browser, no humans
# --------------------------------------------------------------------------

def selftest(seconds=8.0, port=0, pitch_half=(9.0, 7.0), physics_dt=0.005,
             demo_dir=None, verbose=True, check_replay=True):
    """Start a server in-process, play a short match with 4 bots, verify the demo.

    Returns a report dict; raises AssertionError on anything the milestone needs.
    """
    import os
    import socket
    import tempfile
    from rower_soccer.game import server as SV
    from rower_soccer.game.recording import read_demo, summarize

    if port == 0:
        s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    demo_dir = demo_dir or tempfile.mkdtemp(prefix="demo_")
    argv = ["--port", str(port), "--host", "127.0.0.1",
            "--pitch-half", str(pitch_half[0]), str(pitch_half[1]),
            "--match-seconds", str(seconds), "--countdown", "0.5",
            "--physics-dt", str(physics_dt), "--demo-dir", demo_dir,
            "--width", "480", "--height", "320", "--render-hz", "10"]
    args = SV.build_parser().parse_args(argv)
    gs = SV.GameServer(args)
    sim_t = threading.Thread(target=gs.run_sim, daemon=True); sim_t.start()
    gs.ready.wait(180)
    if gs.error:
        raise AssertionError(gs.error)

    httpd = SV.make_httpd(gs, "127.0.0.1", port)
    http_t = threading.Thread(target=httpd.serve_forever, daemon=True); http_t.start()
    url = f"http://127.0.0.1:{port}"
    report = {"url": url, "demo_dir": demo_dir}
    try:
        # 1. four independent clients claim four seats
        bots = [BotClient(url, f"bot{i}", slot=s, seed=i) for i, s in enumerate(SLOTS)]
        for b in bots:
            b.join()
            assert b.claim().get("ok"), b.errors
        # 2. a fifth client cannot take a seat, and cannot drive one
        spec = BotClient(url, "spectator"); spec.join()
        assert not spec.claim("home_1").get("ok"), "a taken seat was claimable"
        r = spec._req("/input", {"skill": "follow"})
        assert not r.get("ok"), "a spectator drove a creature"
        report["isolation"] = "ok"
        # 3. reconnect keeps the seat
        tok = bots[2].token
        bots[2].reconnect()
        assert bots[2].token == tok and bots[2].state()["me"]["slot"] == "away_1"
        report["reconnect"] = "ok"
        # 4. play
        bots[0].start_match()
        time.sleep(0.4)
        stop = threading.Event()
        ts = [threading.Thread(target=b.play, args=(seconds + 3,), kwargs=dict(
            hz=4.0, stop_event=stop)) for b in bots]
        for t in ts:
            t.start()
        # Generous: this runs on a shared box alongside training jobs, where a
        # render can spike to 700 ms. The server drops frames rather than ticks, so
        # a busy host makes the match take longer in wall time, not shorter in sim
        # time -- what we are asserting is that it completes and records, not that
        # the host was idle.
        deadline = time.time() + max(60.0, seconds * 8)
        while time.time() < deadline and gs.sim.phase != "ended":
            time.sleep(0.25)
        stop.set()
        for t in ts:
            t.join(10)
        assert gs.sim.phase == "ended", f"match did not finish (phase={gs.sim.phase})"
        report["errors"] = sum((b.errors for b in bots), [])
        report["inputs_sent"] = sum(b.inputs_sent for b in bots)
        assert not report["errors"], report["errors"]
        assert report["inputs_sent"] > 0

        # 5. the demo
        demos = sorted(os.path.join(demo_dir, f) for f in os.listdir(demo_dir))
        assert demos, "no demo written"
        path = demos[-1]
        d = read_demo(path)
        report["demo"] = path
        report["n_ticks"] = d.n_ticks
        report["summary"] = summarize(path)
        exp = int(round(seconds / d.meta.control_dt))
        assert abs(d.n_ticks - exp) <= 2, f"{d.n_ticks} ticks, expected ~{exp}"
        for k in ("obs", "skill", "target", "z", "action", "qpos", "player_pos"):
            assert k in d.arrays, f"demo is missing {k}"
        assert d.arrays["obs"].shape[:2] == (d.n_ticks, 4)
        assert d.meta.version == 1 and d.meta.obs_keys
        # every seat was a human, and at least one tick of a real skill was recorded
        assert all(p.controller == "human" for p in d.meta.players), \
            [p.controller for p in d.meta.players]
        import numpy as np
        ran = np.unique(d.arrays["skill"])
        report["skills_recorded"] = [d.meta.skill_vocab[i] for i in ran]
        assert any(d.meta.skill_vocab[i] != "idle" for i in ran), "no skill ever ran"
        report["events"] = {e["type"]: sum(1 for x in d.events if x["type"] == e["type"])
                            for e in d.events}
        if check_replay:
            from rower_soccer.game.replay import replay_actions, replay_controller
            report["action_replay"] = replay_actions(d)
            assert report["action_replay"]["deterministic"], report["action_replay"]
            report["controller_replay"] = replay_controller(d, max_ticks=120)
            assert report["controller_replay"]["ok"], report["controller_replay"]
    finally:
        httpd.shutdown()
        gs.stop_flag.set()
    if verbose:
        print(json.dumps({k: v for k, v in report.items() if k != "summary"},
                         indent=1, default=str))
        print(report.get("summary", ""))
    return report


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://localhost:8090")
    p.add_argument("--slots", default=",".join(SLOTS))
    p.add_argument("--seconds", type=float, default=45.0)
    p.add_argument("--hz", type=float, default=2.0)
    p.add_argument("--skills", default="follow")
    p.add_argument("--start", action="store_true", help="also start the match")
    p.add_argument("--selftest", action="store_true",
                   help="run server+bots+demo verification in one process")
    p.add_argument("--selftest-seconds", type=float, default=8.0)
    a = p.parse_args(argv)
    if a.selftest:
        selftest(seconds=a.selftest_seconds)
        return
    bots = run_bots(a.url, tuple(a.slots.split(",")), a.seconds, a.hz, a.start,
                    tuple(a.skills.split(",")))
    for b in bots:
        print(f"{b.name:12s} slot={b.slot} inputs={b.inputs_sent} errors={b.errors}")


if __name__ == "__main__":
    main()
