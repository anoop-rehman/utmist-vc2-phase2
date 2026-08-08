"""The milestone, minus the four humans.

Boots a real server in-process, drives it with four scripted HTTP clients over the
same endpoints a browser uses, and then checks the demo file the match produced:
it exists, it is the right length, every seat is labelled, and it replays both
through physics (recorded actions -> recorded states) and through the skill layer
(recorded obs -> recorded actions).

Slow (builds a dm_control scene and loads torch): ~1-2 minutes.  Run with
`-m "not slow"` to skip.
"""

import pytest

pytestmark = pytest.mark.slow


def test_full_match_records_and_replays():
    from rower_soccer.game.sim_client import selftest
    r = selftest(seconds=6.0, pitch_half=(9.0, 7.0), physics_dt=0.005, verbose=False)
    assert r["isolation"] == "ok"
    assert r["reconnect"] == "ok"
    assert r["inputs_sent"] > 0
    assert not r["errors"]
    assert r["n_ticks"] == pytest.approx(240, abs=2)
    assert "follow" in r["skills_recorded"] or "scripted" in r["skills_recorded"]
    assert r["action_replay"]["deterministic"], r["action_replay"]
    assert r["controller_replay"]["ok"], r["controller_replay"]
