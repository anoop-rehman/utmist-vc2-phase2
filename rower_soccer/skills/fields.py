"""Observation FIELDS — the atoms a skill's obs vector is assembled from.

A drill checkpoint eats one flat vector whose layout is frozen at training time.
Here that layout is spelled out as an ordered tuple of field names (see
`registry.py`); this module says, for each name, how wide it is and how to build
it from a `PlayerFrame`. Adding a skill is then a config entry — a field order
and a checkpoint path — not a new adapter.

The field set below reproduces `warp_port/follow_env.py`'s `_obs()` exactly. Two
scalings are part of the observation contract, and both are applied by
`creature.py`'s observables, so both are PASS-THROUGH here:

  * touch is divided by 10,000 (`CreatureObservables.touch_sensors`).
  * the accelerometer is divided by 100 and clipped to +/-50
    (`CreatureObservables.sensors_accelerometer`). It is the only unbounded
    input — contact spikes reach ~5,700 m/s^2.

Re-applying either would be a silent 1e-4 (touch) or 1e-2 (accel) on real
inputs, which is why `_accel` checks the range rather than transforming.

Verified against a live ant in the CPU soccer env (see tests): dm_soccer's
`bodies_pos`, `joints_pos`, `joints_vel`, `world_zaxis` and `body_height` are
bit-identical to the warp env's, in the same order — `mjcf.find_all('body')` and
MuJoCo body-id order agree (DFS pre-order), and `observable_joints` is actuator
order, which is what `SceneMeta.joint_qpos` uses.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from rower_soccer.skills.api import (ObservationContractError, ravel_obs,
                                     to_ego_xy, vec_to_ego3, world_to_ego3)

# Part of the obs contract; see module docstring.
ACCEL_SCALE = 100.0
ACCEL_CLIP = 50.0

PROPRIO = "proprio"
TASK = "task"


@dataclass(frozen=True)
class FieldSpec:
    """One named block of the observation vector.

    Attributes:
      role: `proprio` (goes to the shared decoder) or `task` (goes to the expert
        head only). This split IS the architecture; the checkpoint's `p_idx`/
        `t_idx` buffers must reproduce it exactly.
      width: (contract) -> int.
      build: (ctx) -> float32 array of that width.
      obs_key: the dm_soccer observation key it reads, if it is a straight copy.
        Used by the width cross-check; None for synthesized fields.
    """

    name: str
    role: str
    width: Callable[[object], int]
    build: Callable[[object], np.ndarray]
    obs_key: Optional[str] = None


@dataclass
class FieldContext:
    """What a field builder gets. One per tick, cheap to construct."""

    frame: object                      # PlayerFrame
    target_xy: Optional[np.ndarray]    # world target actually in force this tick
    target_clip: float                 # see registry.DEFAULT_TARGET_CLIP


_FIELDS: Dict[str, FieldSpec] = {}


def register_field(spec: FieldSpec):
    if spec.name in _FIELDS:
        raise ValueError(f"field '{spec.name}' already registered")
    _FIELDS[spec.name] = spec
    return spec


def get_field(name: str) -> FieldSpec:
    try:
        return _FIELDS[name]
    except KeyError:
        raise ObservationContractError(
            f"unknown observation field '{name}'. Registered: "
            f"{sorted(_FIELDS)}") from None


def field_width(name: str, contract) -> int:
    return int(get_field(name).width(contract))


def known_fields() -> Tuple[str, ...]:
    return tuple(sorted(_FIELDS))


# --- helpers ---------------------------------------------------------------

def _copy(key):
    def build(ctx):
        return ravel_obs(ctx.frame.obs, key).astype(np.float32)
    return build


def _effective_target(ctx) -> np.ndarray:
    if ctx.target_xy is None:
        raise ObservationContractError(
            "this skill needs a target but none is set — call "
            "SkillController.set_command(skill_id, target_xy=...) first")
    return np.asarray(ctx.target_xy, dtype=np.float64).ravel()[:2]


def _target_ego(ctx) -> np.ndarray:
    """Egocentric target, optionally re-aimed at a nearer waypoint.

    `follow_ant_v1` spawned its target 1.07-3.22 m from a creature at the origin,
    so `|target_ego|` in training lived inside that band. The pitch is 96 x 72 m:
    a human clicking the far corner would hand the expert an input ~15x anything
    it has seen. Clipping the ego vector's LENGTH (not its direction) keeps the
    input in-distribution and turns the command into "walk toward that bearing";
    the creature re-aims every tick, so the waypoint advances with it and it still
    arrives — pure pursuit. Once the real target is inside `target_clip` the clip
    stops applying and the expert gets the true, shrinking vector it needs in
    order to settle. `target_clip <= 0` disables it.

    One caveat the clip does NOT fix, because it preserves bearing by design: a
    target lying exactly on the body's forward axis is a left-right SYMMETRIC
    input to a left-right symmetric body, and a deterministic policy can answer it
    with a symmetric action and stand still forever. `follow_ant_v1/best.pt` does
    exactly that — from the drill's canonical spawn it never moves (2.94 m of a
    3 m target after 15 s), and rotating the spawn yaw by 0.3 rad fixes it (0.20
    m). Perturbing the joints does not, because the target bearing is what has to
    break. `final.pt` has no such fixed point. Real gameplay perturbs the heading
    constantly so this is rare, but it is worth knowing when a creature freezes
    while pointing straight at its target.
    """
    f = ctx.frame
    ego = to_ego_xy(f.root_pos, f.root_mat, _effective_target(ctx))
    if ctx.target_clip and ctx.target_clip > 0:
        n = float(np.linalg.norm(ego))
        if n > ctx.target_clip:
            ego = (ego * (ctx.target_clip / n)).astype(np.float32)
    return ego


# --- proprio fields (the shared decoder's input contract) ------------------

register_field(FieldSpec(
    "bodies_pos", PROPRIO, lambda c: 3 * c.n_bodies, _copy("bodies_pos"),
    obs_key="bodies_pos"))
register_field(FieldSpec(
    "body_height", PROPRIO, lambda c: 1, _copy("body_height"),
    obs_key="body_height"))
register_field(FieldSpec(
    "joints_pos", PROPRIO, lambda c: c.n_joints, _copy("joints_pos"),
    obs_key="joints_pos"))
register_field(FieldSpec(
    "joints_vel", PROPRIO, lambda c: c.n_joints, _copy("joints_vel"),
    obs_key="joints_vel"))


def _accel(ctx):
    """Pass-through — `creature.py`'s `sensors_accelerometer` observable already
    applies `/100` and `clip(+/-50)`, matching the warp envs.

    It did not always. Until WS5's fix the CPU path served the raw sensor while
    every policy was trained on the scaled one, and the resulting behaviour gap
    read convincingly as a physics sim2sim gap (WS5 measured fitness 0.284 raw vs
    0.892 scaled on `follow_ant_v1/best.pt`, 4.65 m of trajectory divergence vs
    0.047 m). Doubling the scaling here would be the same bug with the sign
    flipped, so the value is checked instead of transformed: a correctly scaled
    accelerometer is bounded by the clip, and a raw one blows past it on the first
    real contact.
    """
    a = ravel_obs(ctx.frame.obs, "sensors_accelerometer")
    if np.abs(a).max() > ACCEL_CLIP + 1e-6:
        raise ObservationContractError(
            f"sensors_accelerometer = {a} exceeds the contract's clip of "
            f"+/-{ACCEL_CLIP}, so this observation is RAW. The drills train on "
            f"raw/{ACCEL_SCALE:g} clipped to +/-{ACCEL_CLIP:g}; check that "
            "creature.py's CreatureObservables still overrides "
            "sensors_accelerometer to apply it.")
    return a.astype(np.float32)


register_field(FieldSpec(
    "sensors_accelerometer", PROPRIO, lambda c: 3, _accel,
    obs_key="sensors_accelerometer"))
register_field(FieldSpec(
    "sensors_gyro", PROPRIO, lambda c: 3, _copy("sensors_gyro"),
    obs_key="sensors_gyro"))
register_field(FieldSpec(
    "sensors_velocimeter", PROPRIO, lambda c: 3, _copy("sensors_velocimeter"),
    obs_key="sensors_velocimeter"))
# Already /10000 by creature.py's observable — pass through. See module docstring.
register_field(FieldSpec(
    "touch_sensors", PROPRIO, lambda c: c.n_touch, _copy("touch_sensors"),
    obs_key="touch_sensors"))
register_field(FieldSpec(
    "world_zaxis", PROPRIO, lambda c: 3, _copy("world_zaxis"),
    obs_key="world_zaxis"))


# --- task fields -----------------------------------------------------------

register_field(FieldSpec("target_ego", TASK, lambda c: 2, _target_ego))
# The drills' target moves; the future target is `target + vel * lookahead`. A
# commanded target is STATIC (vel = 0), so the two collapse to the same value —
# exactly the case `soccer_bridge` documented, and exactly what a stopped drill
# target looks like.
register_field(FieldSpec("target_ego_future", TASK, lambda c: 2, _target_ego))


def _target_ego3(ctx, clip=None) -> np.ndarray:
    """3-D egocentric target, as `worm_env_base._target_obs3` emits it.

    The v3 drills observe the target through `_to_ego3` on `[x, y, 0]` — a full
    root-frame rotation of the ground point, not the 2-D forward/left projection
    `_target_ego` computes. The third component is NOT zero in general: it is
    how far below the (tilted) body frame the ground target sits, and the expert
    trained on that signal. Length-clipping (pure pursuit) applies to the whole
    3-vector so the bearing — including its pitch — is preserved.
    """
    f = ctx.frame
    t = _effective_target(ctx)
    ego = world_to_ego3(f.root_pos, f.root_mat,
                        np.array([t[0], t[1], 0.0], dtype=np.float64))
    clip = ctx.target_clip if clip is None else clip
    if clip and clip > 0:
        n = float(np.linalg.norm(ego))
        if n > clip:
            ego = ego * (clip / n)
    return ego.astype(np.float32)


register_field(FieldSpec("target_ego3", TASK, lambda c: 3, _target_ego3))
# Static command => now and future coincide, as with target_ego_future above.
register_field(FieldSpec("target_ego3_future", TASK, lambda c: 3, _target_ego3))

# kick_ant_v3 trained on target_dist_range (3, 6) m, so its in-distribution
# target band tops out at 6 m, not follow's 3.2. Clipping a 5 m kick command to
# 3.2 m would misstate the commanded distance to an expert that genuinely uses
# it (segment budget and strike power both scale with it).
STRIKE_TARGET_CLIP = 6.0

register_field(FieldSpec(
    "strike_target_ego3", TASK, lambda c: 3,
    lambda ctx: _target_ego3(ctx, clip=STRIKE_TARGET_CLIP)))


def _cmd_dir_ego3(ctx) -> np.ndarray:
    """kick's command direction: the unit ball->target ground direction, rotated
    into the root frame (`_dir_ego3`).

    Training draws `cmd_dir` once per segment as unit(target - ball_at_spawn)
    and freezes it. Here it is recomputed each tick from the CURRENT ball
    position: identical while the creature lines the kick up (the ball is not
    moving), and after the strike it keeps pointing ball->target, where a
    training segment would already have closed. Degenerate case: ball within
    1 cm of the target has no direction; emit zeros rather than a random
    bearing.
    """
    f = _require_ball(ctx.frame, "the 'cmd_dir_ego3' observation field")
    t = _effective_target(ctx)
    d = np.array([t[0] - f.ball_pos[0], t[1] - f.ball_pos[1]], dtype=np.float64)
    n = float(np.linalg.norm(d))
    if n < 1e-2:
        return np.zeros(3, dtype=np.float32)
    d3 = np.array([d[0] / n, d[1] / n, 0.0], dtype=np.float64)
    return vec_to_ego3(f.root_mat, d3).astype(np.float32)


register_field(FieldSpec("cmd_dir_ego3", TASK, lambda c: 3, _cmd_dir_ego3))


# shoot_ant_v3's goal geometry, from scene.goal_geometry(pitch_scale=0.3125):
# the dm_soccer goal constants (x 42.6667, half-width 11.88, height 5.3333)
# times the pitch scale every v3 drill trained at. Hardcoded with provenance
# rather than imported, so building a game obs never imports the warp scene
# stack. If a future shoot trains at another scale, these change with it.
SHOOT_GOAL_HALF_WIDTH = 11.88 * 0.3125     # 3.7125 m
SHOOT_GOAL_HEIGHT = 5.3333 * 0.3125        # 1.6667 m


def _goal_mid_ego3(ctx) -> np.ndarray:
    """shoot's `goal_mid_ego`: the mouth centre at half crossbar height,
    egocentric, UNCLIPPED — the goal is where it is, and shrinking the vector
    would tell the expert the goal line is nearer than it is. shoot trained with
    the mouth 3.5-8 m out; commanding it from much further is extrapolation,
    which is the game's problem to avoid (press shoot near the box), not this
    field's to hide.

    The target_xy in force IS the goal mouth's ground centre: the game commands
    shoot with the opponent goal as the target, which also lets a human aim a
    shot anywhere (a "fake goal") with the same machinery.
    """
    f = ctx.frame
    t = _effective_target(ctx)
    return world_to_ego3(
        f.root_pos, f.root_mat,
        np.array([t[0], t[1], SHOOT_GOAL_HEIGHT / 2.0],
                 dtype=np.float64)).astype(np.float32)


register_field(FieldSpec("goal_mid_ego3", TASK, lambda c: 3, _goal_mid_ego3))


def _post_ego(sign):
    """The goal posts, as shoot's env emits them: the mouth endpoints at
    target_y +/- half_width, 2-D egocentric (`_to_ego`). Posts run along world
    y — both dm_soccer goals face along x, and so do the game's. NOTE: shoot
    trained only on the +x goal; when the game points a home player at the -x
    goal it must mirror the world before building obs (shoot_env.py: "training
    on one goal and mirroring at deployment is exact"). Until that lands, -x
    shots hand the expert a left/right-swapped goal frame.
    """
    def build(ctx):
        f = ctx.frame
        t = _effective_target(ctx)
        return to_ego_xy(
            f.root_pos, f.root_mat,
            np.array([t[0], t[1] + sign * SHOOT_GOAL_HALF_WIDTH],
                     dtype=np.float64)).astype(np.float32)
    return build


register_field(FieldSpec("post_left_ego", TASK, lambda c: 2, _post_ego(+1.0)))
register_field(FieldSpec("post_right_ego", TASK, lambda c: 2, _post_ego(-1.0)))


def _require_ball(frame, why):
    if frame.ball_pos is None:
        raise ObservationContractError(
            f"{why} needs the ball's WORLD position, which a dm_soccer "
            "observation does not contain in a usable frame. Build the frame "
            "with PlayerFrame(..., ball_pos=..., ball_vel=...) — "
            "skills.soccer.SoccerFrameSource does this for you. See "
            "PlayerFrame's docstring for why obs['ball_ego_position'] is not a "
            "substitute (it is expressed in MuJoCo's INERTIAL frame, the drills' "
            "ball_ego in the BODY frame; for the ant those differ by an axis "
            "permutation).")
    return frame


def _ball_ego(ctx):
    """`ball_ego` as `warp_port/dribble_env.py` emits it: ego position (3) then
    ego linear velocity (3), both in the root BODY frame (`_to_ego3` /
    `_vec_to_ego3`).

    Built from the ball's world state rather than copied from dm_soccer's
    `ball_ego_position` / `ball_ego_linear_velocity`, which are expressed in the
    inertial frame. `dribble_env.py`'s header asserts the two match; measured on
    the ant, they do not. Recomputing costs one 3x3 multiply and removes the
    dependence on MuJoCo's principal-axis ordering entirely.
    """
    f = _require_ball(ctx.frame, "the 'ball_ego' observation field")
    vel = f.ball_vel if f.ball_vel is not None else np.zeros(3)
    return np.concatenate([
        world_to_ego3(f.root_pos, f.root_mat, f.ball_pos),
        vec_to_ego3(f.root_mat, vel),
    ]).astype(np.float32)


register_field(FieldSpec("ball_ego", TASK, lambda c: 6, _ball_ego))


# --- derived world quantities (used by scripted skills, not by any obs) ----

def ball_world_xy(frame) -> np.ndarray:
    """World XY of the ball — the `scripted` chase's target."""
    return _require_ball(frame, "the 'scripted' chase skill").ball_pos[:2]
