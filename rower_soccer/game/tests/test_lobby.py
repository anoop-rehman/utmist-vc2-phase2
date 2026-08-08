"""Lobby tests. Seat exclusivity and reconnect are the only stateful bits of the
server, and both are exactly the kind of thing 4 humans find in the first minute."""

import time

from rower_soccer.game.lobby import Lobby


def test_claim_is_exclusive():
    lb = Lobby()
    a, b = lb.join("a"), lb.join("b")
    assert lb.claim(a.token, "home_1")[0]
    ok, msg = lb.claim(b.token, "home_1")
    assert not ok and "taken" in msg
    assert lb.claim(b.token, "home_2")[0]
    assert lb.slot_of(a.token) == "home_1"
    assert lb.slot_of(b.token) == "home_2"


def test_one_seat_per_client():
    lb = Lobby()
    a = lb.join("a")
    lb.claim(a.token, "home_1")
    lb.claim(a.token, "away_2")
    assert lb.slot_of(a.token) == "away_2"
    assert lb.occupant("home_1") is None


def test_reconnect_keeps_the_seat():
    lb = Lobby()
    a = lb.join("phone")
    lb.claim(a.token, "away_1")
    again = lb.join("phone", token=a.token)     # what a woken phone sends
    assert again.token == a.token and again.slot == "away_1"


def test_stale_seat_is_reclaimable_but_only_on_demand():
    lb = Lobby(claim_timeout=0.05)
    a, b = lb.join("a"), lb.join("b")
    lb.claim(a.token, "home_1")
    time.sleep(0.06)
    assert lb.occupant("home_1") is None            # reads as empty
    assert lb.state()["seats"][0]["stale"] is True
    assert lb.claim(b.token, "home_1")[0]
    assert lb.slot_of(a.token) is None


def test_unknown_token_cannot_claim():
    lb = Lobby()
    ok, msg = lb.claim("garbage", "home_1")
    assert not ok and "unknown" in msg


def test_leave_frees_the_seat():
    lb = Lobby()
    a = lb.join("a")
    lb.claim(a.token, "home_2")
    lb.leave(a.token)
    assert lb.occupant("home_2") is None
    assert lb.state()["n_clients"] == 0


def test_spectators_have_no_slot():
    lb = Lobby()
    lb.join("watcher")
    st = lb.state()
    assert st["spectators"] == ["watcher"]
    assert all(not s["taken"] for s in st["seats"])
