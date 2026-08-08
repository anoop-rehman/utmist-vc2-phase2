"""Lobby tests. Seat exclusivity and reconnect are the only stateful bits of the
server, and both are exactly the kind of thing 4 humans find in the first minute."""

import time

import pytest

from rower_soccer.game.lobby import JoinRefused, Lobby


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


# -- the join code (public URL) --------------------------------------------
# The gate is on token issuance because a token is the only key to a seat: if no
# token is minted, there is nothing to claim with and nothing to send input with.

def test_no_join_code_means_lan_behaviour():
    lb = Lobby()
    assert lb.join_code is None
    assert lb.join("a").token                        # no code, still works
    assert lb.check_code(None) and lb.check_code("anything")


def test_wrong_or_missing_code_gets_no_token():
    lb = Lobby(join_code="hunter2")
    for bad in (None, "", "hunter", "hunter2 ", "HUNTER2"):
        with pytest.raises(JoinRefused):
            lb.join("intruder", code=bad)
    assert lb.state()["n_clients"] == 0              # nothing was created


def test_right_code_joins_and_can_claim():
    lb = Lobby(join_code="hunter2")
    c = lb.join("friend", code="hunter2")
    assert c.token and lb.claim(c.token, "home_1")[0]


def test_a_refused_join_cannot_reach_a_seat():
    """The whole point: no token, no seat, no input -- structurally."""
    lb = Lobby(join_code="hunter2")
    good = lb.join("friend", code="hunter2")
    lb.claim(good.token, "away_1")
    try:
        lb.join("intruder", code="guess")
    except JoinRefused:
        pass
    # every door into a slot is keyed on a token the intruder does not have
    assert not lb.claim("made-up-token", "away_2")[0]
    assert lb.slot_of("made-up-token") is None
    assert lb.occupant("away_1").name == "friend"


def test_reconnect_does_not_re_ask_for_the_code():
    lb = Lobby(join_code="hunter2")
    a = lb.join("phone", code="hunter2")
    lb.claim(a.token, "home_2")
    again = lb.join("phone", token=a.token)          # no code: a woken phone
    assert again.token == a.token and again.slot == "home_2"


def test_the_code_is_not_in_any_response():
    lb = Lobby(join_code="hunter2")
    c = lb.join("friend", code="hunter2")
    assert "hunter2" not in repr(c)
    assert "hunter2" not in repr(lb.state())


def test_seatless_clients_are_reaped_but_seated_ones_are_not():
    """A public URL mints a client per page load; without a TTL that dict grows
    forever and the spectator list fills with strangers."""
    lb = Lobby(claim_timeout=999.0, client_ttl=0.05)
    seated = lb.join("player")
    lb.claim(seated.token, "home_1")
    lb.join("passer-by")
    time.sleep(0.06)
    lb.join("someone-new")                           # a join is what triggers a reap
    st = lb.state()
    assert st["spectators"] == ["someone-new"]
    assert lb.get(seated.token) is not None and lb.occupant("home_1").name == "player"


def test_client_ttl_zero_remembers_forever():
    lb = Lobby(client_ttl=0.0)
    lb.join("ghost")
    time.sleep(0.02)
    lb.join("other")
    assert lb.state()["n_clients"] == 2
