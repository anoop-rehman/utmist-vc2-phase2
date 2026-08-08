"""Schema tests: no simulator, no torch, no network. These must stay fast."""

import json

import numpy as np
import pytest

from rower_soccer.game import recording as R


def _meta(**kw):
    m = R.DemoMeta(match_id="abc", n_players=2, z_dim=4, act_dim=3,
                   obs_keys=["a", "b"], obs_sizes=[2, 3],
                   players=[R.PlayerMeta(0, "home_1", "home", "ant", "human", "u", 3),
                            R.PlayerMeta(1, "home_2", "home", "ant", "scripted", "", 3)])
    for k, v in kw.items():
        setattr(m, k, v)
    return m


def _write(tmp_path, n=5):
    w = R.DemoWriter(str(tmp_path / "d"), _meta())
    rng = np.random.default_rng(0)
    for t in range(n):
        w.record_tick(tick=t, t=t * 0.025,
                      obs=rng.normal(size=(2, 5)).astype(np.float32),
                      skill=np.array([1, 5], np.int8),
                      skill_req=np.array([1, 5], np.int8),
                      target=rng.normal(size=(2, 2)).astype(np.float32),
                      aim=np.zeros((2, 2), np.float32),
                      z=rng.normal(size=(2, 4)).astype(np.float32),
                      action=rng.normal(size=(2, 3)).astype(np.float32),
                      qpos=rng.normal(size=9).astype(np.float32),
                      score=np.array([0, 0], np.int16))
    w.add_event("goal", 3, 0.075, team="home", scorer=0)
    return w.close(), w


def test_writer_roundtrip(tmp_path):
    path, w = _write(tmp_path)
    assert path.endswith(".npz")
    d = R.read_demo(path)
    assert d.n_ticks == 5 and d.n_players == 2
    assert d.arrays["obs"].shape == (5, 2, 5)
    assert d.arrays["tick"].dtype == np.int64
    assert d.arrays["skill"].dtype == np.int8
    assert d.arrays["score"].dtype == np.int16
    assert d.meta.version == R.SCHEMA_VERSION
    assert d.meta.obs_keys == ["a", "b"]
    assert [p.slot for p in d.meta.players] == ["home_1", "home_2"]
    assert d.events_of("goal")[0]["team"] == "home"


def test_skill_vocab_is_append_only():
    """Indices are baked into every file ever recorded. Reordering silently
    relabels historical BC data, so this list is a contract, not a convenience."""
    assert R.SKILL_VOCAB[:6] == ("idle", "follow", "dribble", "kick", "shoot",
                                 "scripted")
    assert R.SKILL_INDEX["idle"] == 0
    assert len(set(R.SKILL_VOCAB)) == len(R.SKILL_VOCAB)


def test_version_mismatch_is_loud(tmp_path):
    path, _ = _write(tmp_path)
    with np.load(path, allow_pickle=False) as f:
        arrays = {k: f[k] for k in f.files}
    meta = json.loads(str(arrays.pop("meta_json")))
    meta["version"] = 99
    events = arrays.pop("events_json")
    p2 = str(tmp_path / "bad.npz")
    with open(p2, "wb") as fh:
        np.savez_compressed(fh, meta_json=np.array(json.dumps(meta)),
                            events_json=events, **arrays)
    with pytest.raises(ValueError, match="schema v99"):
        R.read_demo(p2)


def test_obs_split_is_exact_inverse():
    obs = {"a": np.arange(2.0), "b": np.arange(3.0) + 10}
    keys, sizes = R.obs_layout(obs)
    vec = R.flatten_obs(obs, keys)
    back = R.split_obs(vec, keys, sizes)
    for k in keys:
        assert np.allclose(back[k], np.ravel(obs[k]))


def test_flatten_handles_singleton_buffer_dim():
    """dm_soccer keeps a leading (1, n) buffer dim; the flattener must ravel it."""
    obs = {"a": np.zeros((1, 4))}
    assert R.flatten_obs(obs, ["a"]).shape == (4,)


def test_bc_pairs(tmp_path):
    path, _ = _write(tmp_path, n=6)
    d = R.read_demo(path)
    pairs = d.bc_pairs()
    assert pairs["obs"].shape == (12, 5)
    assert pairs["action"].shape == (12, 3)
    assert pairs["z"].shape == (12, 4)
    only = d.bc_pairs(skills=["follow"])
    assert only["obs"].shape[0] == 6      # player 0 only
    assert set(np.unique(only["skill"])) == {R.SKILL_INDEX["follow"]}


def test_unknown_event_type_rejected(tmp_path):
    w = R.DemoWriter(str(tmp_path / "d"), _meta())
    with pytest.raises(ValueError):
        w.add_event("teleport", 0, 0.0)


def test_close_is_idempotent(tmp_path):
    path, w = _write(tmp_path)
    assert w.close() == path
