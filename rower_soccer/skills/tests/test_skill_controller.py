"""Tests for `rower_soccer.skills`.

Run:  PYTHONPATH=<repo> MUJOCO_GL=egl pytest rower_soccer/skills/tests -q

Three layers:
  * pure — registry/layout/contract arithmetic, no sim, no checkpoint.
  * checkpoint — loading and validation. Skipped when `follow_ant_v1` is absent.
  * soccer — the WS3 gate itself, in the CPU dm_control soccer env. Slower
    (~1 min); `-m "not slow"` skips the locomotion one.

The load-bearing test is `test_obs_matches_warp_formula`: it rebuilds the drill
observation two independent ways — through `SkillController.build_obs` from the
soccer observation DICT, and directly from `physics` with the arithmetic
`warp_port/follow_env.py:_obs` used at training time — and requires them equal.
That is the whole premise of this package, and it is the thing that would fail
silently if dm_soccer ever reordered `bodies_pos`, stopped pre-scaling touch, or
the creature grew a body.
"""

import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from rower_soccer.skills import (CheckpointMismatch, PROPRIO_V1, SkillController,
                                 SkillUnavailable, UnknownSkill, contract_for,
                                 get_spec, list_skills, resolve_checkpoint)
from rower_soccer.skills.api import ObservationContractError
from rower_soccer.skills.fields import ACCEL_CLIP, ACCEL_SCALE

ANT_FOLLOW = "runs_v2/follow_ant_v1/best.pt"


def _have_ckpt():
    try:
        return os.path.exists(resolve_checkpoint(ANT_FOLLOW))
    except Exception:
        return False


needs_ckpt = pytest.mark.skipif(
    not _have_ckpt(),
    reason=f"{ANT_FOLLOW} not found (set $VC2_CHECKPOINT_ROOT)")


# --- pure ------------------------------------------------------------------

def test_ant_contract_widths():
    c = contract_for("ant")
    assert (c.n_bodies, c.n_joints, c.n_touch, c.act_dim) == (9, 8, 9, 8)
    # 3*9 + 1 + 8 + 8 + 9 + 9 + 3
    assert c.proprio_dim == 65


def test_worm_contract_widths():
    """The worm is the body every earlier checkpoint used; its 29/2 is the
    regression anchor for the width formula."""
    c = contract_for("worm")
    assert (c.n_bodies, c.n_joints, c.n_touch, c.act_dim) == (3, 2, 3, 2)
    assert c.proprio_dim == 29


def test_follow_layout_is_proprio_then_task():
    c = contract_for("ant")
    obs_dim, p_idx, t_idx = get_spec("follow").layout(c)
    assert obs_dim == 69
    assert p_idx == list(range(65))
    assert t_idx == list(range(65, 69))


def test_dribble_layout_puts_ball_ego_first():
    """dribble_env replicates dm_control's SORTED-key order, where "ball_ego"
    sorts ahead of "creature/*". The registry must encode that, not assume
    proprio-first — this is the case a naive design gets wrong."""
    c = contract_for("ant")
    obs_dim, p_idx, t_idx = get_spec("dribble").layout(c)
    assert obs_dim == 75
    assert t_idx[:6] == [0, 1, 2, 3, 4, 5]
    assert p_idx == list(range(6, 71))
    assert t_idx[6:] == [71, 72, 73, 74]


def test_proprio_block_is_the_decoder_contract():
    """Every skill riding the shared frozen decoder must present it the identical
    proprio block, or the decoder is being fed a permutation of its own input."""
    for sid in ("follow", "scripted", "dribble", "kick", "shoot"):
        assert get_spec(sid).proprio_fields() == PROPRIO_V1


def test_registry_is_configuration_not_code():
    from dataclasses import replace
    from rower_soccer.skills import SKILLS, register_skill

    spec = replace(SKILLS["dribble"], checkpoints={"ant": "/nowhere/best.pt"})
    assert not SKILLS["dribble"].is_available("ant")
    try:
        register_skill(spec, replace_existing=True)
        assert SKILLS["dribble"].is_available("ant")
        assert SKILLS["dribble"].checkpoint_for("ant") == "/nowhere/best.pt"
    finally:
        register_skill(replace(spec, checkpoints={}), replace_existing=True)


def test_unknown_and_unavailable_skills_raise():
    ctrl = SkillController("ant", quiet=True)
    with pytest.raises(UnknownSkill):
        ctrl.set_command("teleport", (0.0, 0.0))
    with pytest.raises(SkillUnavailable):
        ctrl.set_command("shoot", (0.0, 0.0))       # registered, never trained
    with pytest.raises(SkillUnavailable):
        ctrl.set_command("follow")                  # needs a target


def test_idle_needs_no_checkpoint_at_all():
    """A slot must always be fillable, even for a creature with zero weights."""
    ctrl = SkillController("worm", quiet=True)
    assert "idle" in ctrl.available_skills()
    ctrl.set_command("idle")
    out = ctrl.act(_dummy_frame())
    assert out.action.shape == (2,)
    assert not out.action.any()


def _dummy_frame():
    from rower_soccer.skills import PlayerFrame
    return PlayerFrame(obs={}, root_pos=np.zeros(3), root_mat=np.eye(3))


def test_ego_transform_matches_the_drill_env():
    """`to_ego_xy` must use the UNNORMALISED xy-projections of the body axes, as
    `follow_env._to_ego` does; normalising would be a different observation."""
    from rower_soccer.skills import to_ego_xy

    # A body pitched 60 degrees about y: its x axis projects to length 0.5 in xy.
    ang = np.deg2rad(60.0)
    mat = np.array([[np.cos(ang), 0.0, np.sin(ang)],
                    [0.0, 1.0, 0.0],
                    [-np.sin(ang), 0.0, np.cos(ang)]])
    ego = to_ego_xy(np.zeros(3), mat, (2.0, 0.0))
    assert ego == pytest.approx([2.0 * np.cos(ang), 0.0], abs=1e-6)


def test_target_clip_preserves_bearing():
    """A click at the far end of a 96 x 72 m pitch is ~10x outside the drill's
    training box; the re-aim must shorten the vector without turning it."""
    from rower_soccer.skills import PlayerFrame
    from rower_soccer.skills.fields import FieldContext, get_field

    frame = PlayerFrame(obs={}, root_pos=np.zeros(3), root_mat=np.eye(3))
    raw = get_field("target_ego").build(
        FieldContext(frame, np.array([30.0, 40.0]), 0.0))
    clipped = get_field("target_ego").build(
        FieldContext(frame, np.array([30.0, 40.0]), 10.0))
    assert np.linalg.norm(raw) == pytest.approx(50.0, abs=1e-4)
    assert np.linalg.norm(clipped) == pytest.approx(10.0, abs=1e-4)
    assert np.dot(raw / 50.0, clipped / 10.0) == pytest.approx(1.0, abs=1e-6)


# --- checkpoint ------------------------------------------------------------

@needs_ckpt
def test_checkpoint_layout_matches_derived_layout():
    """The checkpoint records the obs layout it trained on; ours must equal it."""
    import torch

    sd = torch.load(resolve_checkpoint(ANT_FOLLOW), map_location="cpu",
                    weights_only=True)
    _, p_idx, t_idx = get_spec("follow").layout(contract_for("ant"))
    assert sd["mlp_extractor"]["p_idx"].numpy().tolist() == p_idx
    assert sd["mlp_extractor"]["t_idx"].numpy().tolist() == t_idx


@needs_ckpt
def test_wrong_creature_fails_loudly():
    """The failure that has cost this project two runs: a checkpoint quietly
    loaded into the wrong body. It must raise, and name the mismatch."""
    from rower_soccer.skills import load_policy

    _, p_idx, t_idx = get_spec("follow").layout(contract_for("worm"))
    with pytest.raises(CheckpointMismatch) as e:
        load_policy(ANT_FOLLOW, proprio_indices=p_idx, task_indices=t_idx,
                    act_dim=2, device="cpu", label="worm-vs-ant")
    msg = str(e.value)
    assert "proprio" in msg and "65" in msg and "29" in msg
    assert "action width" in msg


@needs_ckpt
def test_reordered_fields_fail_loudly():
    """A skill spec whose field order drifts from the trained env is rejected —
    the widths still add up, so only the index comparison catches it."""
    from dataclasses import replace
    from rower_soccer.skills import SKILLS, load_policy

    swapped = replace(SKILLS["follow"],
                      fields=("body_height",) + tuple(
                          f for f in PROPRIO_V1 if f != "body_height")
                      + ("target_ego", "target_ego_future"))
    dim, p_idx, t_idx = swapped.layout(contract_for("ant"))
    assert dim == 69 and len(p_idx) == 65      # same widths, different order
    with pytest.raises(CheckpointMismatch) as e:
        load_policy(ANT_FOLLOW, proprio_indices=p_idx, task_indices=t_idx,
                    act_dim=8, device="cpu", label="reordered")
    assert "different order" in str(e.value)


@needs_ckpt
def test_policy_cache_shares_weights_between_players():
    from rower_soccer.skills import clear_policy_cache
    from rower_soccer.skills.policy import policy_cache_size

    clear_policy_cache()
    a = SkillController("ant", quiet=True, preload=("follow",))
    b = SkillController("ant", quiet=True, preload=("follow",))
    assert policy_cache_size() == 1
    assert a._expert("follow") is b._expert("follow")


@needs_ckpt
def test_noise_driven_checkpoint_is_detected():
    """`follow_ant_v1` trained with ent_ceil=0, so its std sits at ~1.0 against a
    [-1, 1] action range: the mean is not the policy that was scored. MODE_AUTO
    must notice."""
    from rower_soccer.skills import MODE_NOISE

    ctrl = SkillController("ant", quiet=True, preload=("follow",))
    expert = ctrl._expert("follow")
    assert expert.info.action_std > 0.9
    assert expert.noise_driven
    assert ctrl.resolved_mode("follow") == MODE_NOISE


# --- soccer env ------------------------------------------------------------

@pytest.fixture(scope="module")
def soccer():
    pytest.importorskip("dm_control")
    from rower_soccer.skills.soccer import SoccerFrameSource, make_skill_soccer_env

    env = make_skill_soccer_env(home=("ant",), time_limit=1e6, random_state=0)
    src = SoccerFrameSource(env)
    return env, src


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

    fwd, left = rot[:2, 0], rot[:2, 1]
    d = np.asarray(target_xy, dtype=np.float64) - pos[:2]
    tgt = np.array([d @ fwd, d @ left])
    return np.concatenate([bodies_ego, pos[2:3], jq, jv, sa, sg, sv, touch,
                           world_zaxis, tgt, tgt]).astype(np.float32)


def test_obs_matches_warp_formula(soccer):
    """THE test: the observation rebuilt from the soccer obs dict must equal the
    one the warp trainer would have emitted for the same physical state."""
    env, src = soccer
    walker = src.walkers[0]
    ctrl = SkillController("ant", quiet=True, target_clip=0.0)   # no re-aim
    spec = get_spec("follow")
    target = np.array([4.0, -3.0])

    ts = env.reset()
    rng = np.random.default_rng(0)
    for i in range(25):                       # a few steps, so the pose is not the spawn
        ts = env.step([rng.uniform(-1, 1, 8)])
    frame = src.frame(ts, 0)

    mine = ctrl.build_obs(spec, frame, target)
    theirs = _warp_formula_obs(env, walker, target)
    assert mine.shape == theirs.shape == (69,)
    np.testing.assert_allclose(mine, theirs, rtol=0, atol=1e-6)


def test_observation_width_mismatch_is_caught(soccer):
    """Driving an ant slot with a worm controller must fail on the first tick."""
    env, src = soccer
    ts = env.reset()
    ctrl = SkillController("worm", quiet=True)
    ctrl.set_command("idle")
    ctrl._command = ctrl._command.__class__("scripted", None)  # bypass ckpt load
    with pytest.raises(ObservationContractError) as e:
        ctrl.act(src.frame(ts, 0))
    assert "Wrong creature" in str(e.value)


@needs_ckpt
def test_repeated_act_is_bit_identical(soccer):
    """Replay determinism: same frame, same tick -> same torques, every time."""
    env, src = soccer
    ts = env.reset()
    frame = src.frame(ts, 0)
    ctrl = SkillController("ant", quiet=True, seed=7)
    ctrl.set_command("follow", (5.0, 5.0))

    a = ctrl.act(frame).action
    ctrl.reset()
    ctrl.set_command("follow", (5.0, 5.0))
    b = ctrl.act(frame).action
    np.testing.assert_array_equal(a, b)


@needs_ckpt
def test_noise_stream_is_a_function_of_tick_and_player(soccer):
    env, src = soccer
    ts = env.reset()
    frame = src.frame(ts, 0)

    def first_two(seed, player):
        c = SkillController("ant", quiet=True, seed=seed, player_index=player)
        c.set_command("follow", (5.0, 5.0))
        return c.act(frame).action, c.act(frame).action

    a0, a1 = first_two(7, 0)
    b0, b1 = first_two(7, 0)
    np.testing.assert_array_equal(a0, b0)     # reproducible
    np.testing.assert_array_equal(a1, b1)
    assert not np.array_equal(a0, a1)         # the stream advances with the tick
    c0, _ = first_two(7, 1)
    assert not np.array_equal(a0, c0)         # players do not share a stream
    d0, _ = first_two(8, 0)
    assert not np.array_equal(a0, d0)         # the seed matters


@needs_ckpt
def test_switching_skill_leaves_no_stale_state(soccer):
    """Mid-episode switching must be clean: follow -> idle -> scripted -> follow,
    then the SAME frame must give the SAME torques as before the detour."""
    env, src = soccer
    ts = env.reset()
    frame = src.frame(ts, 0)
    ctrl = SkillController("ant", quiet=True, seed=3)

    ctrl.set_command("follow", (5.0, -5.0))
    before = ctrl.act(frame).action

    ctrl.set_command("idle")
    ctrl.act(frame)
    ctrl.set_command("scripted")
    ctrl.act(frame)
    ctrl.set_command("follow", (5.0, -5.0))
    after = ctrl.act(frame).action

    np.testing.assert_array_equal(before, after)


@needs_ckpt
def test_retarget_does_not_reset_the_noise_phase(soccer):
    """Retargeting is not a switch: it must not restart the gait's noise stream,
    or every mouse click would jolt the creature."""
    env, src = soccer
    ts = env.reset()
    frame = src.frame(ts, 0)
    ctrl = SkillController("ant", quiet=True, seed=3)
    ctrl.set_command("follow", (5.0, -5.0))
    ctrl.act(frame)
    ctrl.set_target((6.0, -4.0))
    assert ctrl.tick == 1
    ctrl.set_command("follow", (7.0, -3.0))    # same skill, new target
    assert ctrl.tick == 1


@needs_ckpt
def test_scripted_chases_the_ball(soccer):
    """The fallback aims at the ball without being told where it is."""
    env, src = soccer
    ts = env.reset()
    ctrl = SkillController("ant", quiet=True)
    ctrl.set_command("scripted")
    out = ctrl.act(src.frame(ts, 0))
    np.testing.assert_allclose(np.array(out.target_xy), src.ball_xy(), atol=1e-6)


@needs_ckpt
@pytest.mark.slow
def test_gate_follow_reaches_a_commanded_point(soccer):
    """WS3's gate, in miniature: commanded to a point 8 m away, the ant closes
    most of the distance inside 10 s, and a mid-episode retarget also works."""
    env, src = soccer
    ctrl = SkillController("ant", quiet=True, seed=0)
    hz = int(round(1.0 / env.task.control_timestep))
    ts = env.reset()

    for target in [(6.0, 6.0), (-6.0, 6.0)]:
        ctrl.set_command("follow", target)
        d0 = np.linalg.norm(src.root_xy(0) - np.array(target))
        for _ in range(10 * hz):
            ts = env.step([ctrl.act(src.frame(ts, 0)).action])
        d1 = np.linalg.norm(src.root_xy(0) - np.array(target))
        assert d1 < d0 - 1.0, f"target {target}: {d0:.2f} -> {d1:.2f} m"
