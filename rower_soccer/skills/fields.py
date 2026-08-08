"""Observation FIELDS — the atoms a skill's obs vector is assembled from.

A drill checkpoint eats one flat vector whose layout is frozen at training time.
Here that layout is spelled out as an ordered tuple of field names (see
`registry.py`); this module says, for each name, how wide it is and how to build
it from a `PlayerFrame`. Adding a skill is then a config entry — a field order
and a checkpoint path — not a new adapter.

The field set below reproduces `warp_port/follow_env.py`'s `_obs()` exactly. Two
scalings are part of the observation contract and are easy to get wrong:

  * accelerometer is divided by 100 and clamped to +/-50. It is the only
    unbounded input (contact spikes reach ~5,700 m/s^2); the warp env applies
    this, dm_soccer does NOT, so we must.
  * touch is divided by 10,000 — and `creature.py`'s `touch_sensors` observable
    ALREADY divides by 10,000. So from a soccer observation it is passed through
    untouched. Dividing again here would be a silent 1e-4 on nine inputs.

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
    a = ravel_obs(ctx.frame.obs, "sensors_accelerometer") / ACCEL_SCALE
    return np.clip(a, -ACCEL_CLIP, ACCEL_CLIP).astype(np.float32)


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
