"""The join gate, over real HTTP, on the real handler.

`test_lobby.py` proves the gate at the data-structure level.  This module proves
it at the level that is actually exposed to the internet: a socket, a URL and a
JSON body.  The two are not the same claim -- a lobby that refuses a join is
useless if `/input` never asks the lobby anything.

No simulator here.  `GameServer` is constructed but its sim thread is never
started, which is exactly the state the server is in for the ~10 s the scene takes
to compile -- so these also check that the gate holds while the sim is booting,
the window in which the URL is already live.
"""

import json
import threading
import urllib.error
import urllib.request

from rower_soccer.game import server as SV

CODE = "correct-horse"


class _Server:
    """A real ThreadingHTTPServer on a real port, with no sim behind it."""

    def __init__(self, *extra):
        argv = ["--host", "127.0.0.1", "--port", "0", "--demo-dir", "", *extra]
        self.args = SV.build_parser().parse_args(argv)
        self.gs = SV.GameServer(self.args)
        self.httpd = SV.make_httpd(self.gs, "127.0.0.1", 0)
        self.port = self.httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}"
        self.t = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.t.start()

    def close(self):
        self.gs.stop_flag.set()
        self.httpd.shutdown()
        self.httpd.server_close()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # -- the two verbs, returning (status, parsed body) --------------------
    def get(self, path, read=True):
        req = urllib.request.Request(self.url + path)
        try:
            r = urllib.request.urlopen(req, timeout=10)
            body = r.read() if read else b""
            return r.status, body
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def post(self, path, body):
        req = urllib.request.Request(
            self.url + path, data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            r = urllib.request.urlopen(req, timeout=10)
            return r.status, json.loads(r.read().decode() or "{}")
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode() or "{}")


# -- LAN mode is untouched --------------------------------------------------

def test_open_server_still_joins_with_no_code():
    with _Server("--join-code", "") as s:
        assert s.gs.lobby.join_code is None
        code, r = s.post("/join", {"name": "lan"})
        assert code == 200 and r["token"]
        assert s.get("/state")[0] == 200            # spectating needs nothing


def test_join_code_comes_from_the_environment_too(monkeypatch=None):
    """$ROWER_JOIN_CODE is the form to prefer: an argv shows up in `ps`."""
    import os
    old = os.environ.get("ROWER_JOIN_CODE")
    os.environ["ROWER_JOIN_CODE"] = "from-env"
    try:
        args = SV.build_parser().parse_args([])
        assert SV.join_code_of(args) == "from-env"
        # ...but an explicit flag wins, and an explicit empty flag means "open"
        args = SV.build_parser().parse_args(["--join-code", "from-argv"])
        assert SV.join_code_of(args) == "from-argv"
        args = SV.build_parser().parse_args(["--join-code", ""])
        assert SV.join_code_of(args) is None
    finally:
        if old is None:
            os.environ.pop("ROWER_JOIN_CODE", None)
        else:
            os.environ["ROWER_JOIN_CODE"] = old


# -- the gate ---------------------------------------------------------------

def test_no_code_no_token():
    with _Server("--join-code", CODE) as s:
        for body in ({}, {"code": ""}, {"code": "correct-hors"},
                     {"code": CODE + " "}, {"code": None}):
            code, r = s.post("/join", body)
            assert code == 403, (body, code, r)
            assert "token" not in r and r.get("need_code")
        assert s.gs.lobby.state()["n_clients"] == 0


def test_right_code_gets_a_token_and_a_seat():
    with _Server("--join-code", CODE) as s:
        code, r = s.post("/join", {"name": "friend", "code": CODE})
        assert code == 200 and r["token"]
        tok = r["token"]
        assert s.post("/claim", {"token": tok, "slot": "home_1"})[1]["ok"]
        assert s.get(f"/state?token={tok}")[0] == 200


def test_a_stranger_cannot_claim_or_drive():
    """The seat and the creature are both keyed on a token only /join hands out."""
    with _Server("--join-code", CODE) as s:
        friend = s.post("/join", {"name": "friend", "code": CODE})[1]["token"]
        s.post("/claim", {"token": friend, "slot": "away_1"})
        s.post("/input", {"token": friend, "skill": "idle"})
        s.gs._inbox.clear()
        for tok in (None, "", "made-up", friend[:-1] + "x"):
            assert s.post("/claim", {"token": tok, "slot": "away_2"})[0] == 403
            assert s.post("/input", {"token": tok, "skill": "follow"})[0] == 403
            assert s.post("/input", {"token": tok, "u": 0.5, "v": 0.5})[0] == 403
            assert s.post("/release", {"token": tok})[0] == 403
            assert s.post("/control", {"token": tok, "action": "start"})[0] == 403
        assert s.gs._inbox == {}, "a rejected client reached the sim inbox"
        assert s.gs.lobby.occupant("away_1").name == "friend"
        assert s.gs.lobby.occupant("away_2") is None


def test_watching_is_gated_too_but_the_page_and_health_are_not():
    with _Server("--join-code", CODE) as s:
        for path in ("/state", "/frame", "/stream", "/state?token=nope"):
            assert s.get(path)[0] == 403, path
        # ...while the things you need in order to be asked for a code stay open
        assert s.get("/")[0] == 200
        assert s.get("/static/client.js")[0] == 200
        st, body = s.get("/health")
        assert st == 200 and json.loads(body)["gated"] is True


def test_reconnect_needs_only_the_token():
    with _Server("--join-code", CODE) as s:
        tok = s.post("/join", {"name": "phone", "code": CODE})[1]["token"]
        s.post("/claim", {"token": tok, "slot": "home_2"})
        code, r = s.post("/join", {"name": "phone", "token": tok})   # no code
        assert code == 200 and r["token"] == tok and r["slot"] == "home_2"


def test_the_code_never_appears_in_a_response():
    with _Server("--join-code", CODE) as s:
        tok = s.post("/join", {"name": "friend", "code": CODE})[1]["token"]
        for path in ("/health", f"/state?token={tok}", "/"):
            assert CODE.encode() not in s.get(path)[1], path


# -- surviving the internet -------------------------------------------------

def test_too_many_viewers_is_refused_not_queued():
    """Each MJPEG viewer is a thread and ~7 Mbit/s off the same uplink the four
    players are using; a public URL needs a ceiling on that."""
    with _Server("--join-code", "", "--max-streams", "1") as s:
        held = urllib.request.urlopen(s.url + "/stream", timeout=10)
        try:
            assert s.gs._streams == 1
            assert s.get("/stream", read=False)[0] == 503
        finally:
            held.close()
