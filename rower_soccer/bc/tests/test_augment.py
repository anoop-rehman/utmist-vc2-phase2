"""Mirror augmentation: the algebra, the refusals, and the dataset plumbing.

The PHYSICS proof of the actuator map (and the key-by-key proof of the game
observation map) lives in `test_mirror_physics.py`, which needs dm_soccer. What
is here needs nothing but numpy: derivation from the MJCF, involution, the
goalpost swap, and the loud failures.
"""

import glob
import os

import numpy as np
import pytest

from rower_soccer.bc import augment as A
from rower_soccer.bc import dataset as D
from rower_soccer.bc.tests import synth

REPO = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))))
ANT_XML = os.path.join(REPO, "creature_configs", "ant.xml")


# --- deriving the ant's symmetry from its XML --------------------------------

def test_ant_mirror_matches_the_hand_derivation():
    bm = A.body_mirror("ant")
    # legs: 1<->4 (+x+y <-> +x-y) and 2<->3 (-x+y <-> -x-y); shins follow.
    # find_all('body') is DFS pre-order: seg0 seg1 seg5 seg2 seg6 seg3 seg7 seg4 seg8
    assert bm.body_names == ("seg0", "seg1", "seg5", "seg2", "seg6",
                             "seg3", "seg7", "seg4", "seg8")
    np.testing.assert_array_equal(bm.body_perm, [0, 7, 8, 5, 6, 3, 4, 1, 2])
    # touch sensors are declared seg0..seg8
    np.testing.assert_array_equal(bm.touch_perm, [0, 4, 3, 2, 1, 8, 7, 6, 5])
    # actuators: hips negate (axis is world z, an axial vector), ankles do not
    np.testing.assert_array_equal(bm.act_perm, [6, 7, 4, 5, 2, 3, 0, 1])
    np.testing.assert_array_equal(bm.act_sign, [-1, 1, -1, 1, -1, 1, -1, 1])
    assert bm.n_bodies == 9 and bm.n_joints == 8 and bm.n_touch == 9


def test_all_perms_are_involutions():
    bm = A.body_mirror("ant")
    for perm in (bm.body_perm, bm.act_perm, bm.touch_perm):
        np.testing.assert_array_equal(perm[perm], np.arange(len(perm)))


def test_refuses_an_asymmetric_body(tmp_path):
    xml = open(ANT_XML).read().replace(
        '<body name="seg4" pos="0.2 -0.2 0">', '<body name="seg4" pos="0.2 -0.3 0">')
    p = tmp_path / "lopsided.xml"
    p.write_text(xml)
    with pytest.raises(A.MirrorError, match="mirror image"):
        A.derive_body_mirror(str(p), "lopsided")


def test_refuses_a_rotated_body_frame(tmp_path):
    xml = open(ANT_XML).read().replace(
        '<body name="seg1" pos="0.2 0.2 0">',
        '<body name="seg1" pos="0.2 0.2 0" quat="1 0 0 0">')
    p = tmp_path / "rotated.xml"
    p.write_text(xml)
    with pytest.raises(A.MirrorError, match="quat"):
        A.derive_body_mirror(str(p), "rotated")


def test_refuses_a_mismatched_gear(tmp_path):
    xml = open(ANT_XML).read().replace(
        '<motor name="motor0_to_4" joint="seg0_to_4" />',
        '<motor name="motor0_to_4" joint="seg0_to_4" gear="99" />')
    p = tmp_path / "geared.xml"
    p.write_text(xml)
    with pytest.raises(A.MirrorError, match="gear"):
        A.derive_body_mirror(str(p), "geared")


# --- action ------------------------------------------------------------------

def test_action_mirror_is_the_expected_permutation():
    bm = A.body_mirror("ant")
    a = np.arange(1.0, 9.0)
    m = A.mirror_action(a, bm)
    # motor order: 0_to_1, 1_to_5, 0_to_2, 2_to_6, 0_to_3, 3_to_7, 0_to_4, 4_to_8
    np.testing.assert_allclose(m, [-a[6], a[7], -a[4], a[5], -a[2], a[3], -a[0], a[1]])


def test_action_mirror_is_an_involution():
    bm = A.body_mirror("ant")
    rng = np.random.default_rng(0)
    a = rng.uniform(-1, 1, size=(64, 8))
    np.testing.assert_allclose(A.mirror_action(A.mirror_action(a, bm), bm), a)


def test_action_mirror_preserves_the_range_and_rejects_a_wrong_width():
    bm = A.body_mirror("ant")
    rng = np.random.default_rng(1)
    a = rng.uniform(-1, 1, size=(32, 8))
    assert np.abs(A.mirror_action(a, bm)).max() <= 1.0
    with pytest.raises(A.MirrorError, match="wide"):
        A.mirror_action(np.zeros(7), bm)


def test_a_left_right_symmetric_action_is_its_own_mirror():
    """A gait with no turn should be a fixed point (up to the hip sign)."""
    bm = A.body_mirror("ant")
    # hips zero, ankles equal on mirror-partner legs
    a = np.array([0.0, 0.4, 0.0, -0.3, 0.0, -0.3, 0.0, 0.4])
    np.testing.assert_allclose(A.mirror_action(a, bm), a)


# --- the expert observation vector ------------------------------------------

def _layouts():
    from rower_soccer.skills import registry as R
    out = {s: spec.fields for s, spec in R.SKILLS.items() if spec.fields}
    out["follow_v1"] = synth.FOLLOW_V1          # the pre-v3 contract, still in demos
    return out


def test_expert_obs_mirror_is_an_involution_for_every_layout():
    bm = A.body_mirror("ant")
    rng = np.random.default_rng(2)
    for name, fields in _layouts().items():
        off = A.expert_field_offsets(fields, bm)
        w = sum(s.stop - s.start for s in off.values())
        v = rng.normal(size=(16, w)).astype(np.float32)
        m = A.mirror_expert_obs(v, fields, bm)
        np.testing.assert_allclose(A.mirror_expert_obs(m, fields, bm), v,
                                   atol=1e-6, err_msg=name)
        assert not np.allclose(m, v), name        # it actually does something


def test_expert_obs_widths_match_the_registry():
    """The width table must agree with what the demos actually recorded."""
    bm = A.body_mirror("ant")
    from rower_soccer.skills import registry as R
    for skill, want in (("follow", 71), ("dribble", 77), ("kick", 77), ("shoot", 78)):
        off = A.expert_field_offsets(R.SKILLS[skill].fields, bm)
        assert sum(s.stop - s.start for s in off.values()) == want, skill
        assert off["bodies_pos"] == slice(0, 27)      # proprio 65 comes first
        assert off["world_zaxis"].stop == 65
    assert sum(s.stop - s.start
               for s in A.expert_field_offsets(synth.FOLLOW_V1, bm).values()) == 69


def test_proprio_block_mirrors_field_by_field():
    bm = A.body_mirror("ant")
    from rower_soccer.skills import registry as R
    fields = R.SKILLS["follow"].fields
    off = A.expert_field_offsets(fields, bm)
    rng = np.random.default_rng(3)
    v = rng.normal(size=71)
    m = A.mirror_expert_obs(v, fields, bm)

    bodies = v[off["bodies_pos"]].reshape(9, 3)
    np.testing.assert_allclose(m[off["bodies_pos"]].reshape(9, 3),
                               bodies[bm.body_perm] * [1, -1, 1])
    np.testing.assert_allclose(m[off["body_height"]], v[off["body_height"]])
    np.testing.assert_allclose(m[off["joints_pos"]],
                               A.mirror_action(v[off["joints_pos"]], bm))
    np.testing.assert_allclose(m[off["sensors_gyro"]],
                               v[off["sensors_gyro"]] * [-1, 1, -1])   # axial
    np.testing.assert_allclose(m[off["sensors_velocimeter"]],
                               v[off["sensors_velocimeter"]] * [1, -1, 1])
    np.testing.assert_allclose(m[off["touch_sensors"]],
                               v[off["touch_sensors"]][bm.touch_perm])
    np.testing.assert_allclose(m[off["world_zaxis"]],
                               v[off["world_zaxis"]] * [1, -1, 1])
    np.testing.assert_allclose(m[off["target_ego3"]],
                               v[off["target_ego3"]] * [1, -1, 1])


def test_goalposts_swap_not_just_negate():
    """shoot's two posts exchange places under the mirror. Negating each in
    place would hand the expert a goal whose left post is to its right."""
    bm = A.body_mirror("ant")
    from rower_soccer.skills import registry as R
    fields = R.SKILLS["shoot"].fields
    off = A.expert_field_offsets(fields, bm)
    v = np.arange(78.0)
    m = A.mirror_expert_obs(v, fields, bm)
    np.testing.assert_allclose(m[off["post_left_ego"]],
                               v[off["post_right_ego"]] * [1, -1])
    np.testing.assert_allclose(m[off["post_right_ego"]],
                               v[off["post_left_ego"]] * [1, -1])


def test_expert_obs_refuses_unknown_fields_and_bad_widths():
    bm = A.body_mirror("ant")
    from rower_soccer.skills import registry as R
    fields = R.SKILLS["follow"].fields
    with pytest.raises(A.MirrorError, match="wide"):
        A.mirror_expert_obs(np.zeros(70), fields, bm)
    A.FIELD_MIRROR.pop("body_height")
    try:
        with pytest.raises(A.MirrorError, match="no mirror is defined"):
            A.mirror_expert_obs(np.zeros(71), fields, bm)
    finally:
        A.FIELD_MIRROR["body_height"] = lambda v, bm_: np.asarray(v)
    # a layout with only one of the two posts cannot be mirrored
    with pytest.raises(A.MirrorError, match="mirrors onto"):
        A.mirror_expert_obs(np.zeros(76), tuple(R.SKILLS["shoot"].fields[:-1]), bm)


# --- the game observation ----------------------------------------------------

def _ego_obs(R, x, landmarks, keys, sizes):
    """Build a synthetic game-observation row the way dm_soccer would."""
    row, i = np.zeros(int(np.sum(sizes)), np.float64), 0
    for k, n in zip(keys, sizes):
        if k == "absolute_root_mat":
            row[i:i + 9] = R.ravel()
        elif k == "absolute_root_pos":
            row[i:i + 3] = x
        elif k in landmarks:
            row[i:i + 2] = R[:2, :2].T @ (np.asarray(landmarks[k]) - x[:2])
        i += n
    return row


def test_game_obs_landmark_keys_match_the_ground_truth():
    """The corner keys are recomputed from the landmark's WORLD point; check
    that against a directly-constructed mirrored world."""
    bm = A.body_mirror("ant")
    keys, sizes = synth.OBS_KEYS, synth.OBS_SIZES
    rng = np.random.default_rng(4)
    for _ in range(20):
        R = synth.rot(rng.uniform(-np.pi, np.pi), rng.uniform(-0.4, 0.4))
        x = np.array([rng.uniform(-9, 9), rng.uniform(-5, 5), 0.75])
        obs = _ego_obs(R, x, synth.HOME_LANDMARKS, keys, sizes)
        got = A.mirror_game_obs(obs, keys, sizes, bm, synth.HOME_LANDMARKS)
        xm, Rm = A.mirror_world_pose(x, R)
        want = _ego_obs(Rm, xm, synth.HOME_LANDMARKS, keys, sizes)
        np.testing.assert_allclose(got, want, atol=1e-9)


def test_game_obs_mirror_is_an_involution():
    bm = A.body_mirror("ant")
    keys, sizes = synth.OBS_KEYS, synth.OBS_SIZES
    rng = np.random.default_rng(5)
    R = synth.rot(0.7, 0.2)
    x = np.array([3.0, -2.0, 0.75])
    obs = _ego_obs(R, x, synth.HOME_LANDMARKS, keys, sizes)
    once = A.mirror_game_obs(obs, keys, sizes, bm, synth.HOME_LANDMARKS)
    twice = A.mirror_game_obs(once, keys, sizes, bm, synth.HOME_LANDMARKS)
    np.testing.assert_allclose(twice, obs, atol=1e-9)


def test_game_obs_without_landmarks_refuses():
    bm = A.body_mirror("ant")
    obs = np.zeros(int(np.sum(synth.OBS_SIZES)))
    with pytest.raises(A.MirrorError, match="landmark"):
        A.mirror_game_obs(obs, synth.OBS_KEYS, synth.OBS_SIZES, bm)


def test_game_obs_refuses_an_unknown_key():
    bm = A.body_mirror("ant")
    with pytest.raises(A.MirrorError, match="no mirror is defined"):
        A.game_obs_ops(list(synth.OBS_KEYS) + ["mystery_sensor"],
                       list(synth.OBS_SIZES) + [3], bm)


# --- dataset level -----------------------------------------------------------

def _ds(tmp_path, n=3):
    paths = [synth.make_demo(tmp_path / f"d{i}", match_id=f"m{i}", seed=i)
             for i in range(n)]
    return D.build_dataset(paths)


def test_mirror_dataset_doubles_and_tags(tmp_path):
    ds = _ds(tmp_path)
    aug = A.mirror_dataset(ds)
    assert len(aug) == 2 * len(ds)
    assert (aug.arrays["mirrored"][:len(ds)] == 0).all()
    assert (aug.arrays["mirrored"][len(ds):] == 1).all()
    # the mirrored half carries no latent (the decoder is not equivariant)
    assert np.isnan(aug.arrays["z"][len(ds):]).all()
    assert np.isfinite(aug.arrays["z"][:len(ds)]).all()
    # provenance and split ride along, so a val match cannot leak into train
    np.testing.assert_array_equal(aug.arrays["split"][len(ds):],
                                  ds.arrays["split"])
    np.testing.assert_array_equal(aug.arrays["demo"][len(ds):], ds.arrays["demo"])
    with pytest.raises(A.MirrorError, match="already contains mirrored"):
        A.mirror_dataset(aug)


def test_mirror_dataset_is_an_involution(tmp_path):
    ds = _ds(tmp_path)
    once = A.mirror_dataset(ds, append=False)
    twice = A.mirror_dataset(once, append=False)
    for k in ("obs", "action", "expert_obs", "target", "aim", "root_pos",
              "root_mat", "ball_pos", "ball_vel"):
        np.testing.assert_allclose(np.nan_to_num(twice.arrays[k], nan=0.0),
                                   np.nan_to_num(ds.arrays[k], nan=0.0),
                                   atol=1e-5, err_msg=k)
    for k in ("skill", "controller", "layout", "tick", "demo", "player", "team",
              "split", "expert_obs_n"):
        np.testing.assert_array_equal(twice.arrays[k], ds.arrays[k], err_msg=k)


def test_mirror_dataset_actually_changes_the_data(tmp_path):
    ds = _ds(tmp_path, n=1)
    m = A.mirror_dataset(ds, append=False)
    assert not np.allclose(m.arrays["action"], ds.arrays["action"])
    assert not np.allclose(m.arrays["obs"], ds.arrays["obs"])
    np.testing.assert_allclose(m.arrays["root_pos"][:, 1], -ds.arrays["root_pos"][:, 1])
    np.testing.assert_allclose(m.arrays["target"][:, 1], -ds.arrays["target"][:, 1])


def test_mirror_handles_mixed_layouts(tmp_path):
    new = synth.make_demo(tmp_path / "new", match_id="new", seed=1)
    old = synth.make_demo(tmp_path / "old", match_id="old", seed=2,
                          follow_fields=synth.FOLLOW_V1)
    ds = D.build_dataset([new, old])
    m = A.mirror_dataset(ds, append=False)
    # the 69-wide rows keep their NaN tail and are mirrored on their own layout
    narrow = ds.arrays["expert_obs_n"] == 69
    assert narrow.any()
    assert np.isnan(m.arrays["expert_obs"][narrow, 69:]).all()
    assert np.isfinite(m.arrays["expert_obs"][narrow, :69]).all()
    back = A.mirror_dataset(m, append=False)
    np.testing.assert_allclose(np.nan_to_num(back.arrays["expert_obs"], nan=0.0),
                               np.nan_to_num(ds.arrays["expert_obs"], nan=0.0),
                               atol=1e-5)


# --- the real corpus ---------------------------------------------------------

def test_expert_obs_mirror_matches_the_field_builders():
    """The strongest non-simulator check: rebuild the expert vector from a
    MIRRORED frame with `skills.fields` itself and compare against
    `mirror_expert_obs`.

    Step one proves the reconstruction is the real thing (it reproduces the
    `skill_obs` the live match recorded, which is what `replay.py --mode
    controller` certifies). Step two then holds `mirror_expert_obs` to the same
    builders, so a wrong sign anywhere in the task block shows up here rather
    than in a BC run three days later.
    """
    paths = sorted(glob.glob(os.path.join(REPO, "demos", "*.demo.npz")))
    if not paths:
        return
    from rower_soccer.game.recording import read_demo
    from rower_soccer.skills.api import PlayerFrame
    from rower_soccer.skills.fields import FieldContext, get_field
    from rower_soccer.skills.registry import DEFAULT_TARGET_CLIP

    bm = A.body_mirror("ant")
    demo = read_demo(paths[-1])
    a, m = demo.arrays, demo.meta
    keys, sizes = list(m.obs_keys), [int(s) for s in m.obs_sizes]
    off = D.key_offsets(keys, sizes)
    lm = {}
    for team in ("home", "away"):
        idx = [p for p in range(demo.n_players) if m.players[p].team == team]
        block = np.concatenate([a["obs"][:, p, :] for p in idx], axis=0)
        lm[team], _ = D.recover_landmarks(block, off)

    def build(fields, frame, target):
        ctx = FieldContext(frame=frame, target_xy=np.asarray(target, float),
                           target_clip=DEFAULT_TARGET_CLIP)
        return np.concatenate([np.asarray(get_field(f).build(ctx), np.float32).ravel()
                               for f in fields])

    rng = np.random.default_rng(0)
    seen = set()
    for _ in range(200):
        t, p = int(rng.integers(demo.n_ticks)), int(rng.integers(demo.n_players))
        n_o = int(a["skill_obs_n"][t, p])
        if n_o == 0:
            continue
        name = m.skill_vocab[int(a["skill"][t, p])]
        fields = tuple(m.skill_obs[name]["fields"])
        obs_row = a["obs"][t, p]
        frame = PlayerFrame(obs={k: obs_row[off[k]] for k in keys},
                            root_pos=a["player_pos"][t, p],
                            root_mat=a["player_mat"][t, p],
                            ball_pos=a["ball_pos"][t], ball_vel=a["ball_vel"][t])
        vec = build(fields, frame, a["target"][t, p])
        np.testing.assert_allclose(vec, a["skill_obs"][t, p, :n_o], atol=2e-6,
                                   err_msg=f"{name} @ {t},{p}")

        team = m.players[p].team
        obs_m = A.mirror_game_obs(obs_row, keys, sizes, bm, lm[team])
        rp, rm = A.mirror_world_pose(a["player_pos"][t, p], a["player_mat"][t, p])
        frame_m = PlayerFrame(obs={k: obs_m[off[k]] for k in keys},
                              root_pos=rp, root_mat=rm,
                              ball_pos=a["ball_pos"][t] * A.M3,
                              ball_vel=a["ball_vel"][t] * A.M3)
        want = build(fields, frame_m, a["target"][t, p] * A.M2)
        got = A.mirror_expert_obs(vec, fields, bm)
        np.testing.assert_allclose(got, want, atol=1e-5,
                                   err_msg=f"mirrored {name} @ {t},{p}")
        seen.add(name)
    assert len(seen) >= 2, seen


def test_mirror_the_real_corpus_if_present():
    paths = sorted(glob.glob(os.path.join(REPO, "demos", "*.demo.npz")))
    if not paths:
        return
    ds = D.build_dataset(paths[:2])
    aug = A.mirror_dataset(ds)
    assert len(aug) == 2 * len(ds)
    act = aug.arrays["action"]
    assert np.isfinite(act).all() and np.abs(act).max() <= 1.0 + 1e-6
    n = len(ds)
    # the mirror of a mirror is the original, on real data too
    back = A.mirror_dataset(A.mirror_dataset(ds, append=False), append=False)
    np.testing.assert_allclose(back.arrays["obs"], ds.arrays["obs"], atol=2e-3)
    np.testing.assert_allclose(back.arrays["action"], ds.arrays["action"], atol=1e-6)
    assert np.isnan(aug.arrays["z"][n:]).all()
