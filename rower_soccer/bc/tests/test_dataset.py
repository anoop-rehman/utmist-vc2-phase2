"""Dataset builder: format round-trip, split determinism, filters, provenance.

No simulator, no torch, no network — these must stay fast.
"""

import glob
import os

import numpy as np
import pytest

from rower_soccer.bc import dataset as D
from rower_soccer.bc.tests import synth

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))


def _corpus(tmp_path, n=4, n_ticks=24, **kw):
    return [synth.make_demo(tmp_path / f"d{i}", match_id=f"match{i}", seed=i,
                            n_ticks=n_ticks, **kw) for i in range(n)]


# --- round trip -------------------------------------------------------------

def test_build_and_roundtrip(tmp_path):
    ds = D.build_dataset(_corpus(tmp_path))
    # 4 demos x 24 ticks x 3 non-idle players
    assert len(ds) == 4 * 24 * 3
    assert ds.arrays["obs"].shape[1] == sum(synth.OBS_SIZES)
    assert ds.arrays["action"].shape[1] == 8
    assert ds.meta["schema"] == D.SCHEMA_NAME

    path = ds.save(tmp_path / "ds.npz")
    back = D.load_dataset(path)
    assert len(back) == len(ds)
    assert back.meta == ds.meta
    for k, v in ds.arrays.items():
        assert k in back.arrays, k
        assert back.arrays[k].dtype == v.dtype, k
        np.testing.assert_array_equal(np.nan_to_num(back.arrays[k], nan=-7.0),
                                      np.nan_to_num(v, nan=-7.0), err_msg=k)


def test_load_rejects_foreign_npz(tmp_path):
    p = str(tmp_path / "junk.npz")
    np.savez(p, meta_json=np.array('{"schema": "nope"}'))
    with pytest.raises(ValueError, match="not a"):
        D.load_dataset(p)


def test_select_keeps_metadata_and_landmarks(tmp_path):
    ds = D.build_dataset(_corpus(tmp_path))
    sub = ds.select(ds.arrays["skill"] == ds.skill_vocab.index("follow"))
    assert 0 < len(sub) < len(ds)
    # landmarks are per-demo, not per-sample: selection must not slice them
    assert sub.arrays["landmarks"].shape == ds.arrays["landmarks"].shape
    assert sub.meta["demos"] == ds.meta["demos"]


# --- split ------------------------------------------------------------------

def test_split_is_by_match_not_by_tick(tmp_path):
    ds = D.build_dataset(_corpus(tmp_path, n=4))
    for d in ds.meta["demos"]:
        m = ds.arrays["demo"] == d["index"]
        assert len(set(ds.arrays["split"][m].tolist())) == 1, \
            f"{d['file']} was split across train and val"
    assert set(ds.arrays["split"].tolist()) == {D.SPLIT_TRAIN, D.SPLIT_VAL}


def test_split_is_deterministic_and_order_independent(tmp_path):
    paths = _corpus(tmp_path, n=4)
    a = D.build_dataset(paths)
    b = D.build_dataset(list(reversed(paths)))
    sa = {d["match_id"]: d["split"] for d in a.meta["demos"]}
    sb = {d["match_id"]: d["split"] for d in b.meta["demos"]}
    assert sa == sb
    # ...and rebuilding from scratch gives the same answer again
    c = D.build_dataset(paths)
    assert {d["match_id"]: d["split"] for d in c.meta["demos"]} == sa


def test_split_modes():
    ids = [f"m{i}" for i in range(8)]
    q = [D.split_of_match(m, 0.25, "", "quota", ids) for m in ids]
    assert sum(q) == 2                       # exactly 25% of eight matches
    # the hash mode never depends on the corpus
    h1 = D.split_of_match("m3", 0.25, "", "hash")
    h2 = D.split_of_match("m3", 0.25, "", "hash", ids[:2])
    assert h1 == h2
    # a salt changes the assignment, deterministically
    salted = [D.split_of_match(m, 0.25, "x", "quota", ids) for m in ids]
    assert sum(salted) == 2
    # never leaves train empty
    assert D.split_of_match("a", 1.0, "", "quota", ["a", "b"]) + \
        D.split_of_match("b", 1.0, "", "quota", ["a", "b"]) == 1


def test_train_val_partition(tmp_path):
    ds = D.build_dataset(_corpus(tmp_path))
    tr, va = ds.train(), ds.val()
    assert len(tr) + len(va) == len(ds)
    assert not (set(tr.arrays["demo"].tolist()) & set(va.arrays["demo"].tolist()))


# --- filters ----------------------------------------------------------------

def test_idle_is_dropped_and_counted(tmp_path):
    paths = _corpus(tmp_path, n=2)
    ds = D.build_dataset(paths)
    idle = ds.skill_vocab.index("idle")
    assert not (ds.arrays["skill"] == idle).any()
    assert ds.meta["dropped"]["idle"] == 2 * 24        # one idle seat per demo

    keep = D.build_dataset(paths, drop_idle=False)
    assert (keep.arrays["skill"] == idle).sum() == 2 * 24
    assert keep.meta["dropped"]["idle"] == 0


def test_scripted_is_kept_and_tagged(tmp_path):
    ds = D.build_dataset(_corpus(tmp_path, n=1))
    ctrl = ds.controller_names()
    assert set(ctrl) == {"human", "scripted"}
    assert (ctrl == "scripted").sum() == 2 * 24        # two scripted seats
    # ...and can be excluded on demand, with the drop counted
    only = D.build_dataset([p for p in glob.glob(str(ds.meta["demos"][0]["path"]))],
                           keep_controllers=["human"])
    assert set(only.controller_names()) == {"human"}
    assert only.meta["dropped"]["controller"] == 2 * 24


def test_playing_phase_filter(tmp_path):
    # a match_end at tick 9 makes ticks 10.. non-playing
    p = synth.make_demo(tmp_path / "short", match_id="m", n_ticks=24, end_tick=9)
    ds = D.build_dataset([p])
    assert int(ds.arrays["tick"].max()) == 9
    assert ds.meta["dropped"]["non_playing"] == 14 * 4
    allt = D.build_dataset([p], playing_only=False)
    assert int(allt.arrays["tick"].max()) == 23


def test_require_expert_obs(tmp_path):
    paths = _corpus(tmp_path, n=1)
    a = D.build_dataset(paths, drop_idle=False)
    b = D.build_dataset(paths, drop_idle=False, require_expert_obs=True)
    assert len(a) - len(b) == 24                       # the idle seat has none
    assert b.meta["dropped"]["no_expert_obs"] == 24
    assert (b.arrays["expert_obs_n"] > 0).all()


# --- layouts, provenance, landmarks -----------------------------------------

def test_mixed_obs_contracts_get_separate_layout_ids(tmp_path):
    new = synth.make_demo(tmp_path / "new", match_id="new", seed=1)
    old = synth.make_demo(tmp_path / "old", match_id="old", seed=2,
                          follow_fields=synth.FOLLOW_V1)
    ds = D.build_dataset([new, old])
    dims = {l["skill"]: [] for l in ds.meta["layouts"]}
    for l in ds.meta["layouts"]:
        dims[l["skill"]].append(l["obs_dim"])
    assert sorted(dims["follow"]) == [69, 71]
    # the narrow rows are NaN-padded to the corpus width, and say so
    assert ds.arrays["expert_obs"].shape[1] == 71
    narrow = ds.arrays["expert_obs_n"] == 69
    assert narrow.any()
    assert np.isnan(ds.arrays["expert_obs"][narrow, 69:]).all()
    assert np.isfinite(ds.arrays["expert_obs"][narrow, :69]).all()


def test_demo_with_a_different_obs_layout_is_skipped_not_fatal(tmp_path):
    good = synth.make_demo(tmp_path / "good", match_id="g", seed=1)
    odd = synth.make_demo(tmp_path / "odd", match_id="o", seed=2,
                          obs_keys=synth.OBS_KEYS + ["extra"],
                          obs_sizes=synth.OBS_SIZES + [4])
    ds = D.build_dataset([good, odd])
    assert len(ds.meta["demos"]) == 1
    assert len(ds.meta["skipped"]) == 1
    assert "obs layout" in ds.meta["skipped"][0]["reason"]


def test_provenance_columns_point_back_at_the_demo(tmp_path):
    paths = _corpus(tmp_path, n=2)
    ds = D.build_dataset(paths)
    for i in (0, len(ds) // 3, len(ds) - 1):
        rec_ = ds.meta["demos"][int(ds.arrays["demo"][i])]
        assert os.path.exists(rec_["path"])
        assert 0 <= int(ds.arrays["tick"][i]) < rec_["n_ticks"]
        assert 0 <= int(ds.arrays["player"][i]) < rec_["n_players"]
        pm = rec_["players"][int(ds.arrays["player"][i])]
        assert ds.controller_vocab[int(ds.arrays["controller"][i])] == pm["controller"]
        assert ds.meta["team_vocab"][int(ds.arrays["team"][i])] == pm["team"]


def test_landmarks_are_recovered(tmp_path):
    ds = D.build_dataset(_corpus(tmp_path, n=1, n_ticks=40))
    home = ds.landmarks_for(0, ds.meta["team_vocab"].index("home"))
    away = ds.landmarks_for(0, ds.meta["team_vocab"].index("away"))
    for k, w in synth.HOME_LANDMARKS.items():
        np.testing.assert_allclose(home[k], w, atol=1e-4)
        np.testing.assert_allclose(away[k], synth.AWAY_LANDMARKS[k], atol=1e-4)
    for d in ds.meta["demos"]:
        for team, res in d["landmark_residual"].items():
            assert res < 1e-3, (team, res)


# --- the real corpus --------------------------------------------------------

def _real_demos():
    return sorted(glob.glob(os.path.join(REPO, "demos", "*.demo.npz")))


def test_real_demos_load():
    paths = _real_demos()
    if not paths:
        return                                   # nothing recorded on this checkout
    ds = D.build_dataset(paths)
    assert len(ds) > 1000
    assert ds.arrays["obs"].shape[1] == 186
    assert ds.meta["act_dim"] == 8 and ds.meta["z_dim"] == 16
    assert not ds.meta["skipped"], ds.meta["skipped"]
    # the two Aug-8 matches predate the v3 skill contract: both widths present
    widths = {l["obs_dim"] for l in ds.meta["layouts"]}
    assert {69, 71}.issubset(widths)
    # every sample's expert vector is exactly as wide as its layout says
    for l in ds.meta["layouts"]:
        m = ds.arrays["layout"] == l["id"]
        if m.any():
            assert set(ds.arrays["expert_obs_n"][m].tolist()) == {l["obs_dim"]}
    assert np.isfinite(ds.arrays["action"]).all()
    assert np.abs(ds.arrays["action"]).max() <= 1.0 + 1e-6
    for d in ds.meta["demos"]:
        for team, res in d["landmark_residual"].items():
            assert res < 1e-3, (d["file"], team, res)


def test_dataset_rows_reproduce_the_recorded_expert_input():
    """A sample must carry everything `replay.py --mode controller` needs.

    That mode is what certifies a demo: re-run the skill layer over the
    recorded rows and get the same expert input (and hence the same action)
    with no simulator. A BC dataset that dropped one of the inputs would still
    train — on rows whose labels no longer follow from their features. So:
    rebuild each sampled row's expert vector from the DATASET's own columns
    (game obs + root pose + ball world state + target + the layout's field
    order) and require it to equal the `expert_obs` column.
    """
    paths = _real_demos()
    if not paths:
        return
    from rower_soccer.skills.api import PlayerFrame
    from rower_soccer.skills.fields import FieldContext, get_field
    from rower_soccer.skills.registry import DEFAULT_TARGET_CLIP

    ds = D.build_dataset(paths[-2:])
    a = ds.arrays
    off = ds.obs_offsets()
    keys = ds.meta["obs_keys"]
    layout = {int(l["id"]): tuple(l["fields"]) for l in ds.meta["layouts"]}
    rng = np.random.default_rng(0)
    seen = set()
    for i in rng.integers(0, len(ds), size=150):
        i = int(i)
        n = int(a["expert_obs_n"][i])
        if not n:
            continue
        fields = layout[int(a["layout"][i])]
        frame = PlayerFrame(obs={k: a["obs"][i][off[k]] for k in keys},
                            root_pos=a["root_pos"][i], root_mat=a["root_mat"][i],
                            ball_pos=a["ball_pos"][i], ball_vel=a["ball_vel"][i])
        ctx = FieldContext(frame=frame, target_xy=np.asarray(a["target"][i], float),
                           target_clip=DEFAULT_TARGET_CLIP)
        vec = np.concatenate([np.asarray(get_field(f).build(ctx), np.float32).ravel()
                              for f in fields])
        np.testing.assert_allclose(vec, a["expert_obs"][i, :n], atol=2e-6,
                                   err_msg=f"sample {i} ({fields})")
        seen.add(ds.skill_vocab[int(a["skill"][i])])
    assert len(seen) >= 3, seen
