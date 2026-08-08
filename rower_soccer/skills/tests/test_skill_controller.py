"""Tests for `rower_soccer.skills`.

Runnable two ways. There is no pytest in this project's venv, so the file ships
its own runner:

    PYTHONPATH=<repo> MUJOCO_GL=egl python -m rower_soccer.skills.tests.test_skill_controller
    PYTHONPATH=<repo> MUJOCO_GL=egl python -m ...test_skill_controller --slow   # + locomotion
    PYTHONPATH=<repo> MUJOCO_GL=egl pytest rower_soccer/skills/tests -q          # if installed

Everything is a zero-argument `test_*` function using plain `assert`, so pytest
collects it unchanged; the shared soccer env is a cached module-level getter
rather than a fixture, for the same reason.

Three layers:
  * pure — registry/layout/contract arithmetic. No sim, no checkpoint.
  * checkpoint — loading and validation. Skipped when `follow_ant_v1` is absent.
  * soccer — against the CPU dm_control soccer env. `--slow` adds the locomotion
    gate, which is ~1 min of simulated walking.

The load-bearing test is `test_obs_matches_warp_formula`: it rebuilds the drill
observation two independent ways — through `SkillController.build_obs` from the
soccer observation DICT, and directly from `physics` with the arithmetic of
`warp_port/follow_env.py:_obs` used at training time — and requires them equal.
That is the whole premise of this package, and it is what would fail silently if
dm_soccer reordered `bodies_pos`, stopped pre-scaling touch, or the creature grew
a body.
"""

import contextlib
import os
import sys
import traceback

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rower_soccer.skills import (CheckpointMismatch, PROPRIO_V1, SkillController,
                                 SkillUnavailable, UnknownSkill, contract_for,
                                 get_spec, resolve_checkpoint)
from rower_soccer.skills.api import ObservationContractError, PlayerFrame
from rower_soccer.skills.fields import ACCEL_CLIP, ACCEL_SCALE

ANT_FOLLOW = "runs_v2/follow_ant_v1/best.pt"


# --- tiny harness ----------------------------------------------------------

class Skip(Exception):
    """Raised by a test that cannot run here. Reported, not failed."""


@contextlib.contextmanager
def raises(exc_type, contains=()):
    """Assert the block raises `exc_type`, and that its message mentions each of
    `contains` — error messages are the product here, so they are asserted on."""
    try:
        yield
    except exc_type as e:
        for frag in ((contains,) if isinstance(contains, str) else contains):
            assert frag in str(e), f"{frag!r} not in error message: {e}"
        return
    raise AssertionError(f"expected {exc_type.__name__}, nothing was raised")


def slow(fn):
    fn.slow = True
    return fn


def need_checkpoint():
    try:
        if os.path.exists(resolve_checkpoint(ANT_FOLLOW)):
            return
    except Exception:
        pass
    raise Skip(f"{ANT_FOLLOW} not found (set $VC2_CHECKPOINT_ROOT)")


_SOCCER = []


def soccer():
    """The shared 1-ant soccer env, built once per process (~15 s)."""
    if not _SOCCER:
        try:
            from rower_soccer.skills.soccer import (SoccerFrameSource,
                                                    make_skill_soccer_env)
        except ImportError as e:
            raise Skip(f"dm_control unavailable: {e}")
        env = make_skill_soccer_env(home=("ant",), time_limit=1e6, random_state=0)
        _SOCCER.append((env, SoccerFrameSource(env)))
    return _SOCCER[0]


def stepped_frame(steps=25, seed=0):
    """A frame from a pose that is NOT the spawn pose, so the parity check has
    something to compare (at the spawn pose most of the vector is zero)."""
    env, src = soccer()
    ts = env.reset()
    rng = np.random.default_rng(seed)
    for _ in range(steps):
        ts = env.step([rng.uniform(-1, 1, 8)])
    return env, src, ts, src.frame(ts, 0)


# --- pure ------------------------------------------------------------------

def test_ant_contract_widths():
    c = contract_for("ant")
    assert (c.n_bodies, c.n_joints, c.n_touch, c.act_dim) == (9, 8, 9, 8)
    assert c.proprio_dim == 65          # 3*9 + 1 + 8 + 8 + 9 + 9 + 3


def test_worm_contract_widths():
    """The worm is the body every earlier checkpoint used; its 29/2 is the
    regression anchor for the width formula."""
    c = contract_for("worm")
    assert (c.n_bodies, c.n_joints, c.n_touch, c.act_dim) == (3, 2, 3, 2)
    assert c.proprio_dim == 29


def test_follow_layout_is_proprio_then_task():
    obs_dim, p_idx, t_idx = get_spec("follow").layout(contract_for("ant"))
    assert obs_dim == 69
    assert p_idx == list(range(65))
    assert t_idx == list(range(65, 69))


def test_dribble_layout_puts_ball_ego_first():
    """dribble_env replicates dm_control's SORTED-key order, where "ball_ego"
    sorts ahead of "creature/*". The registry must encode that rather than assume
    proprio-first — this is the case a naive design gets wrong."""
    obs_dim, p_idx, t_idx = get_spec("dribble").layout(contract_for("ant"))
    assert obs_dim == 75
    assert t_idx[:6] == [0, 1, 2, 3, 4, 5]
    assert p_idx == list(range(6, 71))
    assert t_idx[6:] == [71, 72, 73, 74]


def test_proprio_block_is_the_decoder_contract():
    """Every skill riding the shared frozen decoder must present it the identical
    proprio block, or the decoder is fed a permutation of its own input."""
    for sid in ("follow", "scripted", "dribble", "kick", "shoot"):
        assert get_spec(sid).proprio_fields() == PROPRIO_V1


def test_adding_a_skill_is_configuration_not_code():
    from dataclasses import replace
    from rower_soccer.skills import SKILLS, register_skill

    assert not SKILLS["dribble"].is_available("ant")
    spec = replace(SKILLS["dribble"], checkpoints={"ant": "/nowhere/best.pt"})
    try:
        register_skill(spec, replace_existing=True)
        assert SKILLS["dribble"].is_available("ant")
        assert SKILLS["dribble"].checkpoint_for("ant") == "/nowhere/best.pt"
    finally:
        register_skill(replace(spec, checkpoints={}), replace_existing=True)
    assert not SKILLS["dribble"].is_available("ant")


def test_scripted_inherits_follows_weights():
    """`weights_from` keeps one source of truth for a checkpoint path."""
    assert (get_spec("scripted").checkpoint_for("ant")
            == get_spec("follow").checkpoint_for("ant"))


def test_unknown_and_unavailable_skills_raise():
    ctrl = SkillController("ant", quiet=True)
    with raises(UnknownSkill, "teleport"):
        ctrl.set_command("teleport", (0.0, 0.0))
    with raises(SkillUnavailable, "not trained yet"):
        ctrl.set_command("shoot", (0.0, 0.0))      # registered, never trained
    with raises(SkillUnavailable, "needs a target_xy"):
        ctrl.set_command("follow")


def test_idle_needs_no_checkpoint_at_all():
    """A slot must always be fillable, even for a creature with zero weights."""
    ctrl = SkillController("worm", quiet=True)
    assert "idle" in ctrl.available_skills()
    ctrl.set_command("idle")
    out = ctrl.act(PlayerFrame(obs={}, root_pos=np.zeros(3), root_mat=np.eye(3)))
    assert out.action.shape == (2,)
    assert not out.action.any()
    assert out.z is None


def test_uncommanded_controller_stands_still():
    """An unclaimed slot must not stop the match."""
    ctrl = SkillController("ant", quiet=True)
    out = ctrl.act(PlayerFrame(obs={}, root_pos=np.zeros(3), root_mat=np.eye(3)))
    assert out.action.shape == (8,) and not out.action.any()


def test_ego_transform_matches_the_drill_env():
    """`to_ego_xy` must use the UNNORMALISED xy-projections of the body axes, as
    `follow_env._to_ego` does; normalising would be a different observation."""
    from rower_soccer.skills import to_ego_xy

    ang = np.deg2rad(60.0)                 # pitched 60 deg about y
    mat = np.array([[np.cos(ang), 0.0, np.sin(ang)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(ang), 0.0, np.cos(ang)]])
    ego = to_ego_xy(np.zeros(3), mat, (2.0, 0.0))
    assert np.allclose(ego, [2.0 * np.cos(ang), 0.0], atol=1e-6)


def test_target_clip_preserves_bearing():
    """A click at the far end of a 96 x 72 m pitch is ~10x outside the drill's
    training box; the re-aim must shorten the vector without turning it."""
    from rower_soccer.skills.fields import FieldContext, get_field

    frame = PlayerFrame(obs={}, root_pos=np.zeros(3), root_mat=np.eye(3))
    build = get_field("target_ego").build
    raw = build(FieldContext(frame, np.array([30.0, 40.0]), 0.0))
    clipped = build(FieldContext(frame, np.array([30.0, 40.0]), 10.0))
    assert abs(np.linalg.norm(raw) - 50.0) < 1e-4
    assert abs(np.linalg.norm(clipped) - 10.0) < 1e-4
    assert abs(np.dot(raw / 50.0, clipped / 10.0) - 1.0) < 1e-6


def test_raw_accelerometer_is_rejected():
    """`creature.py` pre-scales the accelerometer (/100, clip 50) to match the
    warp contract. Serving the raw sensor instead is exactly the bug WS5 found
    behind the apparent ant sim2sim gap, so an out-of-range value must raise
    rather than be silently scaled a second time."""
    from rower_soccer.skills.fields import FieldContext, get_field

    ok = PlayerFrame(obs={"sensors_accelerometer": np.array([1.0, -2.0, 3.0])},
                     root_pos=np.zeros(3), root_mat=np.eye(3))
    raw = PlayerFrame(obs={"sensors_accelerometer": np.array([0.0, 0.0, 5700.0])},
                      root_pos=np.zeros(3), root_mat=np.eye(3))
    build = get_field("sensors_accelerometer").build
    assert np.allclose(build(FieldContext(ok, None, 0.0)), [1.0, -2.0, 3.0])
    with raises(ObservationContractError, "RAW"):
        build(FieldContext(raw, None, 0.0))


def test_frame_from_obs_matches_frame_from_physics():
    """A replay must be able to rebuild a frame from the recorded observation
    alone (plus the ball), with no simulator."""
    _, src, ts, frame = stepped_frame(steps=3)
    replayed = PlayerFrame.from_obs(ts.observation[0],
                                    ball_pos=frame.ball_pos,
                                    ball_vel=frame.ball_vel)
    assert np.allclose(replayed.root_pos, frame.root_pos, atol=1e-9)
    assert np.allclose(replayed.root_mat, frame.root_mat, atol=1e-9)

    ctrl = SkillController("ant", quiet=True)
    spec = get_spec("follow")
    np.testing.assert_array_equal(
        ctrl.build_obs(spec, frame, np.array([2.0, 1.0])),
        ctrl.build_obs(spec, replayed, np.array([2.0, 1.0])))


def test_uprightness_reads_cos_tilt():
    from rower_soccer.skills import uprightness

    up = PlayerFrame(obs={"world_zaxis": np.array([0.0, 0.0, 1.0])},
                     root_pos=np.zeros(3), root_mat=np.eye(3))
    over = PlayerFrame(obs={"world_zaxis": np.array([0.0, 0.0, -0.9])},
                       root_pos=np.zeros(3), root_mat=np.eye(3))
    assert uprightness(up) == 1.0
    assert uprightness(over) < -0.5


def test_ball_fields_need_the_world_ball_state():
    """dm_soccer's `ball_ego_position` is in MuJoCo's INERTIAL frame while the
    drills' `ball_ego` is in the BODY frame, so a frame without `ball_pos` must
    refuse rather than quietly substitute the wrong one."""
    from rower_soccer.skills.fields import FieldContext, ball_world_xy, get_field

    bare = PlayerFrame(obs={"ball_ego_position": np.ones(3)},
                       root_pos=np.zeros(3), root_mat=np.eye(3))
    with raises(ObservationContractError, "INERTIAL frame"):
        ball_world_xy(bare)
    with raises(ObservationContractError, "ball_ego"):
        get_field("ball_ego").build(FieldContext(bare, None, 0.0))


def test_ball_ego_uses_the_drills_body_frame():
    """`ball_ego` must equal `_to_ego3`/`_vec_to_ego3` of the world ball state."""
    from rower_soccer.skills import vec_to_ego3, world_to_ego3
    from rower_soccer.skills.fields import FieldContext, ball_world_xy, get_field

    ang = 0.7
    mat = np.array([[np.cos(ang), -np.sin(ang), 0.0],
                    [np.sin(ang), np.cos(ang), 0.0],
                    [0.0, 0.0, 1.0]])
    root = np.array([3.0, -4.0, 0.75])
    ball = np.array([-8.0, 11.0, 0.35])
    vel = np.array([1.5, -0.5, 0.2])
    frame = PlayerFrame(obs={}, root_pos=root, root_mat=mat,
                        ball_pos=ball, ball_vel=vel)
    got = get_field("ball_ego").build(FieldContext(frame, None, 0.0))
    assert got.shape == (6,)
    assert np.allclose(got[:3], world_to_ego3(root, mat, ball), atol=1e-6)
    assert np.allclose(got[3:], vec_to_ego3(mat, vel), atol=1e-6)
    assert np.allclose(ball_world_xy(frame), ball[:2], atol=1e-12)


def test_soccer_ball_ego_differs_from_the_drill_frame():
    """Documents the trap, on the real env: dm_soccer's egocentric ball vector is
    NOT the drills'. If this ever starts passing as 'equal', the frames have
    converged and `_ball_ego` can be simplified — until then, do not."""
    _, src, _, frame = stepped_frame()
    from rower_soccer.skills import world_to_ego3

    drill = world_to_ego3(frame.root_pos, frame.root_mat, frame.ball_pos)
    game = np.asarray(frame.obs["ball_ego_position"]).ravel()
    assert np.abs(drill - game).max() > 1.0, (
        f"drill {drill} vs game {game}: frames agree — re-check _ball_ego")


# --- checkpoint ------------------------------------------------------------

def test_checkpoint_layout_matches_derived_layout():
    """The checkpoint records the obs layout it trained on; ours must equal it."""
    need_checkpoint()
    import torch

    sd = torch.load(resolve_checkpoint(ANT_FOLLOW), map_location="cpu",
                    weights_only=True)
    _, p_idx, t_idx = get_spec("follow").layout(contract_for("ant"))
    assert sd["mlp_extractor"]["p_idx"].numpy().tolist() == p_idx
    assert sd["mlp_extractor"]["t_idx"].numpy().tolist() == t_idx


def test_wrong_creature_fails_loudly():
    """The failure that has cost this project two runs: a checkpoint quietly
    loaded into the wrong body. It must raise, and name the mismatch."""
    need_checkpoint()
    from rower_soccer.skills import load_policy

    _, p_idx, t_idx = get_spec("follow").layout(contract_for("worm"))
    with raises(CheckpointMismatch, ("proprio", "65", "29", "action width")):
        load_policy(ANT_FOLLOW, proprio_indices=p_idx, task_indices=t_idx,
                    act_dim=2, device="cpu", label="worm-vs-ant")


def test_task_block_in_the_wrong_place_fails_loudly():
    """A skill spec whose task block sits somewhere other than where the trained
    env put it is rejected. The widths still add up, so only comparing the index
    ARRAYS catches it — and this is the realistic error, because dribble's task
    block really does start at column 0 while follow's is at the end."""
    need_checkpoint()
    from dataclasses import replace
    from rower_soccer.skills import SKILLS, load_policy

    moved = replace(SKILLS["follow"],
                    fields=("target_ego",) + PROPRIO_V1 + ("target_ego_future",))
    dim, p_idx, t_idx = moved.layout(contract_for("ant"))
    assert dim == 69 and len(p_idx) == 65 and len(t_idx) == 4   # same widths
    with raises(CheckpointMismatch, ("different order", "slot 0")):
        load_policy(ANT_FOLLOW, proprio_indices=p_idx, task_indices=t_idx,
                    act_dim=8, device="cpu", label="task-block-moved")


def test_proprio_field_order_is_pinned_by_a_golden_value():
    """The one layout error a checkpoint CANNOT catch.

    `p_idx` records which COLUMNS are proprio, not which field is in each column.
    Permuting fields inside the proprio block therefore leaves `p_idx` as
    `range(65)` and validation passes while the decoder silently receives a
    permuted input. Nothing in the checkpoint format can detect that, so the
    order is pinned here instead: if you change PROPRIO_V1, this test fails and
    you have to prove the change is intentional and matched by retraining."""
    assert PROPRIO_V1 == (
        "bodies_pos",
        "body_height",
        "joints_pos",
        "joints_vel",
        "sensors_accelerometer",
        "sensors_gyro",
        "sensors_velocimeter",
        "touch_sensors",
        "world_zaxis",
    ), "PROPRIO_V1 must stay byte-identical to warp_port/follow_env.py:_obs"


def test_policy_cache_shares_weights_between_players():
    need_checkpoint()
    from rower_soccer.skills import clear_policy_cache
    from rower_soccer.skills.policy import policy_cache_size

    clear_policy_cache()
    a = SkillController("ant", quiet=True, preload=("follow",))
    b = SkillController("ant", quiet=True, preload=("follow",))
    assert policy_cache_size() == 1
    assert a._expert("follow") is b._expert("follow")


def test_default_mode_is_the_mean():
    """Gameplay and replay run the action MEAN — the same policy the drills' own
    `eval_video` scores. A wide action std is reported, never acted on."""
    need_checkpoint()
    from rower_soccer.skills import MODE_MEAN

    ctrl = SkillController("ant", quiet=True, preload=("follow",))
    assert ctrl.action_mode == MODE_MEAN
    expert = ctrl._expert("follow")
    assert expert.info.action_std > 0.9      # ent_ceil=0 pinned log_std at 0
    assert expert.wide_std                   # noticed...
    assert ctrl.action_mode == MODE_MEAN     # ...and not acted on


def test_follow_default_checkpoint_is_final_not_best():
    """`best.pt` is whichever checkpoint scored highest on the WARP eval — for
    follow_ant_v1 the 55.8M-step one. Measured in the CPU soccer env it is much
    worse than `final.pt` (77% upright / 1.44 m vs 99.9% / 0.56 m over six 45-s
    episodes) and has a symmetric-input fixed point. See registry.py."""
    assert get_spec("follow").checkpoint_for("ant").endswith("final.pt")


# --- soccer env ------------------------------------------------------------

def _warp_formula_obs(env, walker, target_xy):
    """Recompute the drill observation straight from physics, using exactly the
    arithmetic in `warp_port/follow_env.py:_obs`. Deliberately shares no code with
    `fields.py` — that is what makes it a check and not a tautology."""
    ph = env.physics
    root = ph.bind(walker.root_body)
    pos = np.array(root.xpos)
    rot = np.array(root.xmat).reshape(3, 3)

    bodies = np.array([ph.bind(b).xpos for b in walker.bodies])
    bodies_ego = ((bodies - pos) @ rot).reshape(-1)          # R^T (x - p)
    jq = np.array([ph.bind(j).qpos for j in walker.observable_joints]).ravel()
    jv = np.array([ph.bind(j).qvel for j in walker.observable_joints]).ravel()

    sensor = walker.mjcf_model.sensor
    sa = np.clip(np.array(ph.bind(sensor.accelerometer).sensordata).ravel()
                 / ACCEL_SCALE, -ACCEL_CLIP, ACCEL_CLIP)
    sg = np.array(ph.bind(sensor.gyro).sensordata).ravel()
    sv = np.array(ph.bind(sensor.velocimeter).sensordata).ravel()
    touch = np.array(ph.bind(walker.touch_sensors).sensordata).ravel() / 10000.0
    world_zaxis = rot.reshape(9)[6:9]

    d = np.asarray(target_xy, dtype=np.float64) - pos[:2]
    tgt = np.array([d @ rot[:2, 0], d @ rot[:2, 1]])
    return np.concatenate([bodies_ego, pos[2:3], jq, jv, sa, sg, sv, touch,
                           world_zaxis, tgt, tgt]).astype(np.float32)


def test_obs_matches_warp_formula():
    """THE test: the observation rebuilt from the soccer obs dict must equal the
    one the warp trainer would have emitted for the same physical state."""
    env, src, _, frame = stepped_frame()
    ctrl = SkillController("ant", quiet=True, target_clip=0.0)   # no re-aim
    target = np.array([4.0, -3.0])

    mine = ctrl.build_obs(get_spec("follow"), frame, target)
    theirs = _warp_formula_obs(env, src.walkers[0], target)
    assert mine.shape == theirs.shape == (69,)
    assert np.abs(mine - theirs).max() < 1e-6, np.abs(mine - theirs).max()


def test_observation_width_mismatch_is_caught():
    """Driving an ant slot with a worm controller must fail on the first tick."""
    _, _, _, frame = stepped_frame(steps=1)
    from rower_soccer.skills import SkillCommand

    ctrl = SkillController("worm", quiet=True)
    ctrl.set_command("idle")
    ctrl._command = SkillCommand("scripted", None)     # bypass the checkpoint load
    with raises(ObservationContractError, ("bodies_pos", "Wrong creature")):
        ctrl.act(frame)


def test_repeated_act_is_bit_identical():
    """Replay determinism: same frame, same tick -> same torques, every time."""
    need_checkpoint()
    _, _, _, frame = stepped_frame(steps=1)

    def once():
        c = SkillController("ant", quiet=True, seed=7)
        c.set_command("follow", (5.0, 5.0))
        return c.act(frame).action

    assert np.array_equal(once(), once())


def test_noise_stream_is_a_function_of_tick_seed_and_player():
    """MODE_NOISE must be reproducible without being repetitive: the same
    (seed, player_index, tick) always gives the same torques, and no two of those
    coordinates share a stream."""
    need_checkpoint()
    from rower_soccer.skills import MODE_NOISE
    _, _, _, frame = stepped_frame(steps=1)

    def first_two(seed, player):
        c = SkillController("ant", quiet=True, seed=seed, player_index=player,
                            action_mode=MODE_NOISE)
        c.set_command("follow", (5.0, 5.0))
        return c.act(frame).action, c.act(frame).action

    a0, a1 = first_two(7, 0)
    b0, b1 = first_two(7, 0)
    assert np.array_equal(a0, b0) and np.array_equal(a1, b1)   # reproducible
    assert not np.array_equal(a0, a1)          # the stream advances with the tick
    assert not np.array_equal(a0, first_two(7, 1)[0])          # players differ
    assert not np.array_equal(a0, first_two(8, 0)[0])          # the seed matters


def test_switching_skill_leaves_no_stale_state():
    """Mid-episode switching must be clean: follow -> idle -> scripted -> follow,
    then the SAME frame must give the SAME torques as before the detour."""
    need_checkpoint()
    _, _, _, frame = stepped_frame(steps=1)
    ctrl = SkillController("ant", quiet=True, seed=3)

    ctrl.set_command("follow", (5.0, -5.0))
    before = ctrl.act(frame).action

    ctrl.set_command("idle")
    ctrl.act(frame)
    ctrl.set_command("scripted")
    ctrl.act(frame)
    ctrl.set_command("follow", (5.0, -5.0))
    after = ctrl.act(frame).action

    assert np.array_equal(before, after)


def test_retarget_is_not_a_switch():
    """Retargeting must not restart the gait's noise stream, or every mouse click
    would jolt the creature."""
    need_checkpoint()
    _, _, _, frame = stepped_frame(steps=1)
    ctrl = SkillController("ant", quiet=True, seed=3)
    ctrl.set_command("follow", (5.0, -5.0))
    ctrl.act(frame)
    ctrl.set_target((6.0, -4.0))
    assert ctrl.tick == 1
    ctrl.set_command("follow", (7.0, -3.0))     # same skill, new target
    assert ctrl.tick == 1
    ctrl.set_command("idle")                    # a real switch
    assert ctrl.tick == 0


def test_scripted_chases_the_ball():
    """The fallback aims at the ball without being told where it is, and derives
    it from the player's own observation rather than from physics."""
    need_checkpoint()
    _, src, _, frame = stepped_frame(steps=1)
    ctrl = SkillController("ant", quiet=True)
    ctrl.set_command("scripted")
    out = ctrl.act(frame)
    assert np.allclose(np.array(out.target_xy), src.ball_xy(), atol=1e-6)


def test_pool_drives_every_player():
    need_checkpoint()
    from rower_soccer.skills import SkillControllerPool
    from rower_soccer.skills.soccer import (SoccerFrameSource,
                                            make_skill_soccer_env)

    env = make_skill_soccer_env(home=("ant", "ant"), away=("ant", "ant"),
                                time_limit=1e6, random_state=0)
    src = SoccerFrameSource(env)
    pool = SkillControllerPool(["ant"] * 4, quiet=True, seed=0)
    for i in range(4):
        pool.set_command(i, "follow", (0.0, 0.0))
    ts = env.reset()
    actions = pool.actions(src.frames(ts))
    assert len(actions) == 4 and all(a.shape == (8,) for a in actions)
    # Distinct player_index => distinct noise streams => distinct torques, even
    # though all four share one set of weights and one command.
    assert not np.array_equal(actions[0], actions[1])
    env.step(actions)


@slow
def test_gate_follow_reaches_a_commanded_point():
    """WS3's gate, in miniature: commanded 4 m away, the ant closes most of the
    distance in 15 s, twice, in different directions, without an env reset."""
    need_checkpoint()
    env, src = soccer()
    ctrl = SkillController("ant", quiet=True, seed=0)
    hz = int(round(1.0 / env.task.control_timestep))
    ts = env.reset()

    for offset in [(4.0, 0.0), (-2.0, 4.0)]:
        target = src.root_xy(0) + np.asarray(offset)
        ctrl.set_command("follow", tuple(target))
        d0 = np.linalg.norm(src.root_xy(0) - target)
        for _ in range(15 * hz):
            ts = env.step([ctrl.act(src.frame(ts, 0)).action])
        d1 = np.linalg.norm(src.root_xy(0) - target)
        assert d1 < d0 - 1.0, f"offset {offset}: {d0:.2f} -> {d1:.2f} m"


# --- runner ----------------------------------------------------------------

def main(argv):
    want_slow = "--slow" in argv
    only = [a for a in argv[1:] if not a.startswith("-")]
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    if only:
        tests = [(n, f) for n, f in tests if any(o in n for o in only)]

    npass = nskip = nfail = 0
    for name, fn in tests:
        if getattr(fn, "slow", False) and not want_slow:
            print(f"  SKIP  {name} (slow; pass --slow)")
            nskip += 1
            continue
        try:
            fn()
        except Skip as e:
            print(f"  SKIP  {name} ({e})")
            nskip += 1
        except Exception:
            print(f"  FAIL  {name}")
            traceback.print_exc()
            nfail += 1
        else:
            print(f"  ok    {name}")
            npass += 1
    print(f"\n{npass} passed, {nskip} skipped, {nfail} failed")
    return 1 if nfail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
