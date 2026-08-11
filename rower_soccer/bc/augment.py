"""Pitch-mirror augmentation — the dangerous one.

Creature soccer is symmetric about the pitch's long axis: reflect the world
through the plane ``y = 0`` and you get another perfectly legal match. That
doubles a small demo corpus for free, which matters here because we will never
have DeepMind's hours of human play. It is also the single easiest way to
poison a BC run beyond debugging, because a wrong mirror produces data that
looks completely plausible — right ranges, right correlations, right everything
— and simply teaches the policy a body it does not have.

So nothing in this module is asserted from first principles alone. Every
transform below was derived analytically AND checked against the live CPU
soccer env, and the checks are the test suite, not a comment:

  * `tests/test_mirror_physics.py::test_mirrored_rollout_matches`
    mirrors a contact-rich 4-ant state and the applied actions, steps MuJoCo
    for 10 control ticks (100 substeps) and compares against the mirror of the
    unmirrored rollout. Measured: max |dqpos| = 1.8e-15, max |dqvel| = 2.6e-14,
    while the state itself moves 1.25 m. That is the actuator map's proof.
  * `tests/test_mirror_physics.py::test_game_obs_mirror_matches_env`
    mirrors the state and compares `mirror_game_obs` against the observation
    dm_soccer computes for the mirrored state, key by key: all 47 keys, all
    four players, max error 3.6e-15.

The geometry, once
------------------
Let ``M = diag(1, -1, 1)``. Reflecting the world maps a position ``p -> Mp``,
a polar vector ``v -> Mv``, an axial vector (angular velocity, torque, hinge
axis) ``w -> -Mw``, and a body's world rotation ``R -> M R M`` (the extra M on
the right restores right-handedness — without it the "rotation" has det -1).
Every consequence follows mechanically:

  * an egocentric vector ``R.T (p - x)`` becomes ``M`` times itself;
  * an egocentric rotation ``R_a.T R_b`` becomes ``M (...) M``, i.e. the flat
    9-vector is scaled elementwise by ``s_i s_j`` with ``s = (1, -1, 1)``;
  * an egocentric ANGULAR velocity picks up the axial minus: ``-M(...)``.

Two wrinkles are specific to dm_soccer and were found the hard way:

  1. **The inertial frame.** dm_soccer builds its `ball_ego_*`, `*_ego_position`,
     `*_ego_orientation` sensors with ``objtype='body'``, which in MuJoCo means
     the body's INERTIAL frame, not its body frame (`skills/api.py` documents
     the same trap for the ball). A quantity expressed in the inertial frame
     mirrors by ``N = R_i.T M R_i`` where ``R_i`` is the body->inertial rotation
     the compiler derived from the mass distribution. For the ant
     ``iquat = (0.5, 0.5, -0.5, 0.5)``, whose matrix sends the inertial axes to
     ``(body z, -body x, -body y)``, so ``N = diag(1, 1, -1)``: the y-mirror
     shows up as a SIGN FLIP ON THE THIRD COMPONENT of every such key. That is
     the sort of thing you do not guess right; `INERTIAL_REFLECTION` carries it
     per creature and `derive_inertial_reflection` re-derives it from the
     compiled model in the tests.

  2. **The pitch corners.** `field_front_left` / `field_back_right` and the four
     goal-corner keys are egocentric views of FIXED world points, and the pairs
     dm_soccer records are DIAGONALLY opposite — the mirror maps them onto the
     other diagonal, which is not observed. There is no elementwise transform
     that can produce them. They are recomputed from the landmark's world xy
     (recovered per demo by `dataset.recover_landmarks`) and the player's root
     pose, which the game observation carries as `absolute_root_pos/mat`:

         mirrored[K] = m2 * (R2.T @ (m2 * W_K - x))     m2 = diag(1, -1)

     Mirroring a game observation therefore REQUIRES the landmarks; asking
     without them raises rather than guessing.

What is NOT mirrored: `z`
-------------------------
The latent `z` is the input to a frozen decoder that was trained with no
symmetry constraint whatsoever. `decoder(mirror(z))` is not `mirror(decoder(z))`
for any `mirror` we know, and there is no reason it should be. Mirrored samples
therefore carry `z = NaN` and `mirrored = 1`; a trainer doing latent-space BC
must mask them (`ds.arrays["mirrored"] == 0`), while action-space BC can use the
whole corpus. Silently emitting an unmirrored `z` next to a mirrored action
would be a labelling lie, which the demo schema's own header forbids.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from rower_soccer.bc.dataset import BCDataset, LANDMARK_KEYS, key_offsets

__all__ = ["BodyMirror", "body_mirror", "derive_body_mirror",
           "derive_inertial_reflection", "INERTIAL_REFLECTION",
           "mirror_action", "mirror_expert_obs", "mirror_game_obs",
           "mirror_world_pose", "mirror_dataset", "mirror_mj_state",
           "MirrorError"]

#: The world reflection this module implements: y -> -y.
M3 = np.array([1.0, -1.0, 1.0])
M2 = np.array([1.0, -1.0])

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: body -> inertial reflection, N = R_i.T M R_i. See the module docstring.
#: Verified against the compiled model by `derive_inertial_reflection`.
INERTIAL_REFLECTION: Dict[str, np.ndarray] = {
    "ant": np.diag([1.0, 1.0, -1.0]),
}


class MirrorError(RuntimeError):
    """The body (or the request) is not mirrorable. Never returns a guess."""


# --- deriving the body symmetry from the MJCF -------------------------------

def _vec(s, n, default=None):
    if s is None:
        return None if default is None else np.array(default, float)
    v = np.array([float(x) for x in str(s).split()], float)
    if v.size != n:
        raise MirrorError(f"expected {n} numbers, got {s!r}")
    return v


@dataclass(frozen=True)
class BodyMirror:
    """The y-mirror of one creature, read off its MJCF.

    All permutations are "gather" indices: ``mirrored[i] = sign[i] * x[perm[i]]``.

    Attributes:
      body_perm: over `mjcf.find_all('body')` DFS pre-order, which is what
        `creature.CreatureObservables.bodies_pos` emits and what MuJoCo's body
        ids follow (asserted in `skills/fields.py`'s header).
      act_perm / act_sign: over actuator (document) order. `observable_joints`
        is `[a.joint for a in actuators]`, so the same pair mirrors
        `joints_pos`, `joints_vel`, `prev_action` and the action itself.
      touch_perm: over the `<touch>` sensors in document order, which is the
        order `CreatureObservables.touch_sensors` reads them (seg0..segN).
    """

    creature: str
    xml_path: str
    body_names: Tuple[str, ...]
    body_perm: np.ndarray
    joint_names: Tuple[str, ...]
    act_names: Tuple[str, ...]
    act_perm: np.ndarray
    act_sign: np.ndarray
    touch_names: Tuple[str, ...]
    touch_perm: np.ndarray
    inertial: np.ndarray = field(default_factory=lambda: np.eye(3))

    @property
    def n_bodies(self):
        return len(self.body_names)

    @property
    def n_joints(self):
        return len(self.act_names)

    @property
    def n_touch(self):
        return len(self.touch_names)

    @property
    def act_dim(self):
        return len(self.act_names)

    def describe(self) -> str:
        pairs = ", ".join(f"{self.act_names[i]}<-{self.act_sign[i]:+.0f}*"
                          f"{self.act_names[self.act_perm[i]]}"
                          for i in range(self.act_dim))
        return (f"{self.creature}: bodies {self.body_perm.tolist()} "
                f"touch {self.touch_perm.tolist()}\n  actuators: {pairs}")


_MIRROR_CACHE: Dict[str, BodyMirror] = {}


def body_mirror(creature: str = "ant", xml_path: Optional[str] = None) -> BodyMirror:
    """Cached `derive_body_mirror` for a creature kind."""
    key = f"{creature}|{xml_path or ''}"
    hit = _MIRROR_CACHE.get(key)
    if hit is None:
        path = xml_path or _creature_xml(creature)
        hit = derive_body_mirror(path, creature)
        _MIRROR_CACHE[key] = hit
    return hit


def _creature_xml(kind: str) -> str:
    # Same resolution rule as skills/contract.creature_xml_path, reimplemented in
    # four lines so this module keeps importing nothing but numpy + stdlib.
    aliases = {"rower": "two_arm_rower_blueprint.xml", "worm": "three_seg_worm.xml"}
    if os.sep in kind or kind.endswith(".xml"):
        return os.path.abspath(kind)
    return os.path.join(REPO_ROOT, "creature_configs", aliases.get(kind, f"{kind}.xml"))


def derive_body_mirror(xml_path: str, creature: str = "ant") -> BodyMirror:
    """Read the creature's MJCF and work out its y-mirror, or refuse.

    The mirror is found geometrically, not by name: each body must sit at the
    reflected position of exactly one other body (or its own), and the two must
    carry the same geoms and sites under the reflection. Only then are joints
    paired — by anchor AND axis — and each hinge given a sign:

        the mirror image of a rotation about axis ``u`` is a rotation about
        ``-Mu`` (angular quantities are axial). If the partner joint declares
        ``-Mu`` the joint angle and its torque carry over unchanged; if it
        declares ``+Mu``, both flip sign.

    For the ant this yields hips negated and ankles preserved, with legs 1<->4
    and 2<->3 swapped:

        motor0_to_1 <- -motor0_to_4    motor1_to_5 <- +motor4_to_8
        motor0_to_2 <- -motor0_to_3    motor2_to_6 <- +motor3_to_7

    Raises `MirrorError` (never a guess) if the body is not mirror-symmetric,
    if any body carries a rotation attribute (the accumulation below assumes
    translation-only frames, which every creature XML in this repo satisfies),
    or if a partner's joint range/gear does not mirror.
    """
    root = ET.parse(xml_path).getroot()
    world = root.find("worldbody")
    if world is None:
        raise MirrorError(f"{xml_path}: no <worldbody>")

    bodies = []           # (name, world_pos, element, depth)

    def walk(el, origin, depth):
        for b in el.findall("body"):
            for attr in ("quat", "euler", "axisangle", "xyaxes", "zaxis"):
                if b.get(attr) is not None:
                    raise MirrorError(
                        f"{xml_path}: body '{b.get('name')}' has a {attr}; this "
                        "derivation accumulates translation-only body frames. "
                        "Extend it (rotate the local frames too) before trusting "
                        "a mirror for this creature.")
            pos = origin + _vec(b.get("pos"), 3, [0, 0, 0])
            bodies.append((b.get("name"), pos, b, depth))
            walk(b, pos, depth + 1)

    walk(world, np.zeros(3), 0)
    if not bodies:
        raise MirrorError(f"{xml_path}: no bodies")
    names = tuple(b[0] for b in bodies)
    pos = np.stack([b[1] for b in bodies])

    # --- body pairing ----------------------------------------------------
    body_perm = np.full(len(bodies), -1, int)
    for i, (name, p, _el, depth) in enumerate(bodies):
        target = p * M3
        hits = [j for j in range(len(bodies))
                if bodies[j][3] == depth and np.allclose(pos[j], target, atol=1e-9)]
        if len(hits) != 1:
            raise MirrorError(
                f"{xml_path}: body '{name}' at {p.tolist()} has {len(hits)} "
                f"candidates at its mirror image {target.tolist()}; the body is "
                "not y-symmetric (or two bodies coincide). Refusing to mirror.")
        body_perm[i] = hits[0]
    if not np.array_equal(body_perm[body_perm], np.arange(len(bodies))):
        raise MirrorError(f"{xml_path}: body pairing is not an involution")

    # --- structural check: geoms and sites must mirror too -----------------
    for i, (name, _p, el, _d) in enumerate(bodies):
        j = int(body_perm[i])
        for tag in ("geom", "site"):
            a = sorted(_shape_key(e, mirror=True) for e in el.findall(tag))
            b = sorted(_shape_key(e, mirror=False)
                       for e in bodies[j][2].findall(tag))
            if a != b:
                raise MirrorError(
                    f"{xml_path}: <{tag}>s of '{name}' do not mirror onto those "
                    f"of '{bodies[j][0]}':\n  mirrored: {a}\n  target:   {b}")

    # --- joints ------------------------------------------------------------
    jinfo = {}            # joint name -> (body index, world anchor, axis, range)
    for i, (_name, p, el, _d) in enumerate(bodies):
        for jt in el.findall("joint"):
            kind = jt.get("type", "hinge")
            if kind != "hinge":
                raise MirrorError(
                    f"{xml_path}: joint '{jt.get('name')}' is a {kind}; only "
                    "1-DOF hinges have a scalar angle whose mirror is a sign.")
            axis = _vec(jt.get("axis"), 3, [0, 0, 1])
            anchor = p + _vec(jt.get("pos"), 3, [0, 0, 0])
            rng = _vec(jt.get("range"), 2) if jt.get("range") else None
            jinfo[jt.get("name")] = (i, anchor, axis, rng)

    jpair, jsign = {}, {}
    for name, (bi, anchor, axis, rng) in jinfo.items():
        want_anchor = anchor * M3
        want_axis = -(axis * M3)                      # axial vector image
        bj = int(body_perm[bi])
        cands = [n for n, (b2, a2, u2, _r) in jinfo.items()
                 if b2 == bj and np.allclose(a2, want_anchor, atol=1e-9)
                 and (np.allclose(_unit(u2), _unit(want_axis), atol=1e-9)
                      or np.allclose(_unit(u2), -_unit(want_axis), atol=1e-9))]
        if len(cands) != 1:
            raise MirrorError(
                f"{xml_path}: joint '{name}' (anchor {anchor.tolist()}, axis "
                f"{axis.tolist()}) has {len(cands)} mirror candidates on body "
                f"'{bodies[bj][0]}'. Refusing to mirror.")
        other = cands[0]
        u2 = jinfo[other][2]
        s = 1.0 if np.allclose(_unit(u2), _unit(want_axis), atol=1e-9) else -1.0
        jpair[name], jsign[name] = other, s
        r2 = jinfo[other][3]
        if (rng is None) != (r2 is None):
            raise MirrorError(f"{xml_path}: '{name}' and '{other}' disagree on limits")
        if rng is not None:
            want = rng if s > 0 else np.array([-rng[1], -rng[0]])
            if not np.allclose(want, r2, atol=1e-9):
                raise MirrorError(
                    f"{xml_path}: joint '{name}' range {rng.tolist()} mirrors to "
                    f"{want.tolist()} but '{other}' declares {r2.tolist()}")

    # --- actuators ---------------------------------------------------------
    act_el = root.find("actuator")
    acts = list(act_el) if act_el is not None else []
    if not acts:
        raise MirrorError(f"{xml_path}: no actuators")
    act_names, act_joint = [], []
    for a in acts:
        if a.tag != "motor":
            raise MirrorError(f"{xml_path}: actuator '{a.get('name')}' is a "
                              f"<{a.tag}>; only direct-drive <motor> is handled "
                              "(a position servo would also mirror its target).")
        jn = a.get("joint")
        if jn is None or jn not in jinfo:
            raise MirrorError(f"{xml_path}: actuator '{a.get('name')}' does not "
                              "drive a known hinge")
        act_names.append(a.get("name"))
        act_joint.append(jn)
    idx_of_joint = {jn: i for i, jn in enumerate(act_joint)}
    if len(idx_of_joint) != len(act_joint):
        raise MirrorError(f"{xml_path}: two actuators share a joint")

    act_perm = np.zeros(len(acts), int)
    act_sign = np.zeros(len(acts), float)
    default_gear = _default_attr(root, "motor", "gear")
    for i, jn in enumerate(act_joint):
        other = jpair[jn]
        if other not in idx_of_joint:
            raise MirrorError(f"{xml_path}: joint '{jn}' is actuated but its "
                              f"mirror '{other}' is not")
        j = idx_of_joint[other]
        act_perm[i] = j
        act_sign[i] = jsign[jn]
        gi = acts[i].get("gear", default_gear)
        gj = acts[j].get("gear", default_gear)
        if (gi or "") != (gj or ""):
            raise MirrorError(f"{xml_path}: actuators '{act_names[i]}' and "
                              f"'{act_names[j]}' have different gears {gi} / {gj}")
    if not np.array_equal(act_perm[act_perm], np.arange(len(acts))):
        raise MirrorError(f"{xml_path}: actuator pairing is not an involution")
    if not np.allclose(act_sign[act_perm], act_sign):
        raise MirrorError(f"{xml_path}: actuator signs are not symmetric")

    # --- touch sensors ------------------------------------------------------
    sensor_el = root.find("sensor")
    touches = [] if sensor_el is None else [s for s in sensor_el if s.tag == "touch"]
    site_body = {}
    for i, (_n, _p, el, _d) in enumerate(bodies):
        for s in el.findall("site"):
            site_body[s.get("name")] = i
    touch_names = tuple(t.get("name") for t in touches)
    body_of_touch = [site_body.get(t.get("site")) for t in touches]
    if any(b is None for b in body_of_touch):
        raise MirrorError(f"{xml_path}: a <touch> sensor references an unknown site")
    touch_perm = np.zeros(len(touches), int)
    for i, bi in enumerate(body_of_touch):
        want = int(body_perm[bi])
        hits = [k for k, b in enumerate(body_of_touch) if b == want]
        if len(hits) != 1:
            raise MirrorError(f"{xml_path}: body '{bodies[want][0]}' carries "
                              f"{len(hits)} touch sensors; cannot pair")
        touch_perm[i] = hits[0]

    return BodyMirror(
        creature=creature, xml_path=os.path.abspath(xml_path),
        body_names=names, body_perm=body_perm,
        joint_names=tuple(act_joint), act_names=tuple(act_names),
        act_perm=act_perm, act_sign=act_sign,
        touch_names=touch_names, touch_perm=touch_perm,
        inertial=np.asarray(INERTIAL_REFLECTION.get(creature, np.eye(3)), float))


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n else v


def _default_attr(root, tag, attr):
    d = root.find("default")
    if d is None:
        return None
    el = d.find(tag)
    return None if el is None else el.get(attr)


def _shape_key(el, mirror: bool):
    """A geom/site's identity for the symmetry check, optionally reflected."""
    out = [el.tag, el.get("type", "sphere"), el.get("size", ""),
           el.get("density", ""), el.get("mass", "")]
    ft = _vec(el.get("fromto"), 6) if el.get("fromto") else None
    p = _vec(el.get("pos"), 3, [0, 0, 0])
    if mirror:
        p = p * M3
        if ft is not None:
            ft = np.concatenate([ft[:3] * M3, ft[3:] * M3])
    out.append(_clean(p))
    out.append(None if ft is None else _clean(ft))
    return repr(out)


def _clean(a):
    """Rounded list with no -0.0 (which `repr` distinguishes and we must not)."""
    a = np.round(np.asarray(a, float), 9) + 0.0
    return np.where(a == 0, 0.0, a).tolist()


def derive_inertial_reflection(xml_path: str, body: Optional[str] = None) -> np.ndarray:
    """``N = R_i.T M R_i`` for a creature's ROOT body, from the compiled model.

    Imports mujoco (the only thing in this module that does) because `iquat` is
    a compiler output, not an XML attribute. Used by the tests to prove the
    hardcoded `INERTIAL_REFLECTION` entry; call it yourself when adding a
    creature.
    """
    import mujoco
    m = mujoco.MjModel.from_xml_path(str(xml_path))
    bid = 1 if body is None else mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        raise MirrorError(f"{xml_path}: no body {body!r}")
    Ri = np.zeros(9)
    mujoco.mju_quat2Mat(Ri, m.body_iquat[bid])
    Ri = Ri.reshape(3, 3)
    N = Ri.T @ np.diag(M3) @ Ri
    N[np.abs(N) < 1e-12] = 0.0
    return N


# --- elementwise transforms -------------------------------------------------

def mirror_action(action, bm: BodyMirror) -> np.ndarray:
    """Mirror an actuator command. Accepts [A] or [..., A]."""
    a = np.asarray(action, dtype=np.float64)
    if a.shape[-1] != bm.act_dim:
        raise MirrorError(f"action is {a.shape[-1]} wide, {bm.creature} has "
                          f"{bm.act_dim} actuators")
    return (a[..., bm.act_perm] * bm.act_sign).astype(np.asarray(action).dtype,
                                                      copy=False)


def mirror_joints(v, bm: BodyMirror) -> np.ndarray:
    """`joints_pos` / `joints_vel` / `prev_action`: the actuator map again."""
    return mirror_action(v, bm)


def mirror_bodies_pos(v, bm: BodyMirror) -> np.ndarray:
    a = np.asarray(v)
    out = a.reshape(*a.shape[:-1], bm.n_bodies, 3)[..., bm.body_perm, :] * M3
    return out.reshape(a.shape).astype(a.dtype, copy=False)


def mirror_touch(v, bm: BodyMirror) -> np.ndarray:
    return np.asarray(v)[..., bm.touch_perm]


def mirror_world_pose(root_pos, root_mat):
    """(Mx, M R M) — a root pose reflected through the pitch's long axis."""
    x = np.asarray(root_pos, float)
    R = np.asarray(root_mat, float)
    flat = R.shape[-1] == 9
    R3 = R.reshape(*R.shape[:-1], 3, 3) if flat else R
    S = np.diag(M3)
    R3 = np.einsum("ij,...jk,kl->...il", S, R3, S)
    return x * M3, (R3.reshape(R.shape) if flat else R3)


# --- the expert (drill) observation vector ----------------------------------

class _WidthContract:
    """Just enough of `skills.contract.CreatureContract` for the field widths.

    `skills.fields`' width lambdas read `n_bodies`, `n_joints` and `n_touch` and
    nothing else. Using this instead of `contract_for()` keeps augmentation
    free of mujoco, dm_control and — importantly for a repo where the warp port
    is being edited in parallel — of `warp_port.scene`.
    """

    def __init__(self, bm: BodyMirror):
        self.n_bodies, self.n_joints, self.n_touch = bm.n_bodies, bm.n_joints, bm.n_touch
        self.act_dim = bm.act_dim
        self.kind = bm.creature


def _sign_op(s):
    s = np.asarray(s, float)
    return lambda v, bm: np.asarray(v) * s


#: field name -> (block, BodyMirror) -> mirrored block. A field NOT in here has
#: no known mirror and `mirror_expert_obs` refuses rather than passing it
#: through: a silently unmirrored field is the exact failure mode this module
#: exists to prevent.
FIELD_MIRROR = {
    # proprio
    "bodies_pos": mirror_bodies_pos,
    "body_height": lambda v, bm: np.asarray(v),
    "joints_pos": mirror_joints,
    "joints_vel": mirror_joints,
    "sensors_accelerometer": _sign_op(M3),
    "sensors_gyro": _sign_op(-M3),          # axial
    "sensors_velocimeter": _sign_op(M3),
    "touch_sensors": mirror_touch,
    "world_zaxis": _sign_op(M3),
    # task
    "target_ego": _sign_op(M2),
    "target_ego_future": _sign_op(M2),
    "target_ego3": _sign_op(M3),
    "target_ego3_future": _sign_op(M3),
    "strike_target_ego3": _sign_op(M3),
    "goal_mid_ego3": _sign_op(M3),
    "cmd_dir_ego3": _sign_op(M3),
    "ball_ego": _sign_op(np.concatenate([M3, M3])),   # ego pos (3) then ego vel (3)
}

#: Fields whose mirror is each OTHER, not themselves. Mirroring the world swaps
#: which goalpost is on the left, so `post_left_ego` must be built from
#: `post_right_ego` (and then y-negated) — an elementwise sign on each in place
#: would hand the shoot expert a goal turned inside out.
FIELD_SWAPS = (("post_left_ego", "post_right_ego"),)


def expert_field_offsets(fields: Sequence[str], bm: BodyMirror) -> Dict[str, slice]:
    """Field name -> slice into the assembled expert vector, from the registry."""
    from rower_soccer.skills.fields import field_width
    c = _WidthContract(bm)
    out, i = {}, 0
    for name in fields:
        w = int(field_width(name, c))
        out[name] = slice(i, i + w)
        i += w
    return out


def mirror_expert_obs(vec, fields: Sequence[str], bm: BodyMirror) -> np.ndarray:
    """Mirror the exact vector a drill expert consumed, field by field.

    `fields` is the field ORDER the checkpoint was trained on — take it from the
    demo's own `meta.skill_obs[skill]["fields"]` (or `dataset` layout record),
    never from the live registry, so a corpus spanning a contract change still
    mirrors each row the way that row was actually built.

    Accepts [O] or [..., O]. Raises on an unknown field or a width mismatch.
    """
    v = np.asarray(vec)
    off = expert_field_offsets(fields, bm)
    total = sum(s.stop - s.start for s in off.values())
    if v.shape[-1] != total:
        raise MirrorError(
            f"expert obs is {v.shape[-1]} wide but fields {tuple(fields)} sum to "
            f"{total} for {bm.creature}. Mirroring a vector whose layout we have "
            "wrong would be undetectable downstream.")
    out = np.array(v, copy=True)
    swap = {}
    for a, b in FIELD_SWAPS:
        swap[a], swap[b] = b, a
    for name in fields:
        if name in swap:
            other = swap[name]
            if other not in off:
                raise MirrorError(
                    f"field '{name}' mirrors onto '{other}', which this layout "
                    f"does not contain: {tuple(fields)}")
            out[..., off[name]] = np.asarray(v)[..., off[other]] * M2
            continue
        op = FIELD_MIRROR.get(name)
        if op is None:
            raise MirrorError(
                f"no mirror is defined for observation field '{name}'. Add it to "
                "bc/augment.FIELD_MIRROR (and a test) — do not let it through "
                "untransformed.")
        out[..., off[name]] = op(np.asarray(v)[..., off[name]], bm)
    return out.astype(v.dtype, copy=False)


# --- the dm_soccer game observation ----------------------------------------

def _sign9(s):
    return np.array([s[i] * s[j] for i in range(3) for j in range(3)], float)


def game_obs_ops(keys: Sequence[str], sizes: Sequence[int], bm: BodyMirror):
    """(elementwise ops, landmark keys) for one game-observation layout.

    Returns a list of `(slice, callable)` plus the list of landmark keys that
    `mirror_game_obs` must handle separately.
    """
    N = np.asarray(bm.inertial, float)
    Nd = np.diag(N).copy()
    if not np.allclose(N, np.diag(Nd)):
        raise MirrorError(
            f"{bm.creature}: the inertial reflection {N.tolist()} is not "
            "diagonal, so dm_soccer's body-frame ('objtype=body') keys mix "
            "components under the mirror. Implement the full matmul before "
            "using this creature.")
    off = key_offsets(list(keys), list(sizes))
    ops, landmarks = [], []
    for k, sl in off.items():
        n = sl.stop - sl.start
        if k in LANDMARK_KEYS:
            landmarks.append(k)
            continue
        op = _game_key_op(k, n, bm, Nd)
        if op is None:
            raise MirrorError(
                f"no mirror is defined for game observation key '{k}' ({n} wide). "
                "Add it to bc/augment._game_key_op with a test in "
                "tests/test_mirror_physics.py.")
        ops.append((sl, op))
    return ops, landmarks


def _game_key_op(k: str, n: int, bm: BodyMirror, Nd: np.ndarray):
    """The mirror of one dm_soccer observation key, or None if unknown."""
    # --- the walker's own proprioception (BODY frame; creature.py) ----------
    if k == "bodies_pos":
        return lambda v: mirror_bodies_pos(v, bm)
    if k == "touch_sensors":
        return lambda v: mirror_touch(v, bm)
    if k in ("joints_pos", "joints_vel", "prev_action"):
        return lambda v: mirror_joints(v, bm)
    if k in ("body_height",) or k.startswith("stats_"):
        return lambda v: np.asarray(v)
    if k in ("sensors_accelerometer", "sensors_velocimeter", "world_zaxis"):
        return _mul(M3)
    if k == "sensors_gyro":
        return _mul(-M3)                                  # axial
    if k == "absolute_root_pos":
        return _mul(M3)
    if k == "absolute_root_mat":
        return _mul(_sign9(M3))                           # R -> M R M
    # --- everything dm_soccer builds with objtype/reftype='body' ------------
    # ...which is the INERTIAL frame; see the module docstring.
    if k.endswith("_ego_orientation"):
        return _mul(_sign9(Nd))
    if k == "ball_ego_angular_velocity":
        return _mul(-Nd)                                  # axial, inertial frame
    if (k in ("ball_ego_position", "ball_ego_linear_velocity", "end_effectors_pos")
            or k.endswith("_ego_position") or k.endswith("_ego_linear_velocity")
            or k.endswith("_ego_end_effectors_pos") or k.endswith("_end_effectors_pos")):
        return _mul(Nd)
    # --- arena landmarks that ARE mirror-invariant world points -------------
    # The goal mouth centres sit on y = 0, so the point maps to itself and the
    # ordinary egocentric rule applies (unlike the corner keys).
    if k in ("team_goal_mid", "opponent_goal_mid"):
        return _mul(M3)
    return None


def _mul(s):
    s = np.asarray(s, float)
    return lambda v: np.asarray(v) * s


def mirror_game_obs(obs, keys: Sequence[str], sizes: Sequence[int], bm: BodyMirror,
                    landmarks: Optional[Mapping[str, np.ndarray]] = None,
                    ops=None) -> np.ndarray:
    """Mirror a dm_soccer game observation vector (or a batch of them).

    Args:
      obs: [O] or [N, O], laid out by `keys`/`sizes` (a demo's
        `meta.obs_keys` / `meta.obs_sizes`).
      landmarks: key -> world xy for each of `dataset.LANDMARK_KEYS`, as
        recovered by `dataset.recover_landmarks`. Each value may be (2,) or
        [N, 2]. REQUIRED whenever the layout contains those keys.
      ops: a precomputed `game_obs_ops(...)` result, for batching.
    """
    a = np.asarray(obs)
    single = a.ndim == 1
    x = a.reshape(1, -1) if single else a
    ops_, lm_keys = ops if ops is not None else game_obs_ops(keys, sizes, bm)
    out = np.array(x, copy=True)
    for sl, op in ops_:
        out[:, sl] = op(x[:, sl])
    if lm_keys:
        if landmarks is None:
            raise MirrorError(
                "this observation contains fixed-pitch-landmark keys "
                f"({lm_keys}) whose mirror needs their WORLD position. Pass "
                "`landmarks=` (dataset.recover_landmarks / "
                "BCDataset.landmarks_for). See the module docstring.")
        off = key_offsets(list(keys), list(sizes))
        R = x[:, off["absolute_root_mat"]].astype(np.float64).reshape(-1, 3, 3)
        pos = x[:, off["absolute_root_pos"]].astype(np.float64)[:, :2]
        R2T = np.transpose(R[:, :2, :2], (0, 2, 1))
        for k in lm_keys:
            W = np.asarray(landmarks[k], float)
            if not np.all(np.isfinite(W)):
                raise MirrorError(f"landmark '{k}' is not finite; its recovery "
                                  "failed, so this demo cannot be mirrored.")
            W = np.broadcast_to(W.reshape(-1, 2), (x.shape[0], 2))
            out[:, off[k]] = (np.einsum("nij,nj->ni", R2T, W * M2 - pos) * M2)
    return out.reshape(a.shape).astype(a.dtype, copy=False)


# --- dataset-level ----------------------------------------------------------

def mirror_dataset(ds: BCDataset, *, append: bool = True,
                   verbose: bool = False) -> BCDataset:
    """Mirror every sample of `ds`.

    Args:
      append: return the original samples followed by the mirrored ones
        (the usual "double the corpus"); False returns only the mirrored copy.

    The mirrored rows keep their provenance (same demo, tick, player, split — so
    the mirror of a validation match never leaks into training) and set
    `mirrored = 1`, `z = NaN`.
    """
    bm = body_mirror(ds.meta.get("creature", "ant"))
    keys, sizes = ds.meta["obs_keys"], ds.meta["obs_sizes"]
    ops = game_obs_ops(keys, sizes, bm)
    a = ds.arrays
    n = len(ds)
    if int(a["mirrored"].max(initial=0)) and append:
        raise MirrorError("this dataset already contains mirrored samples; "
                          "mirroring it again would double the duplicates.")

    lm_all = np.asarray(a["landmarks"])           # [n_demos, teams, K, 2]
    per_row = lm_all[a["demo"].astype(int), a["team"].astype(int)]    # [N, K, 2]
    landmarks = {k: per_row[:, i, :] for i, k in enumerate(ds.meta["landmark_keys"])}

    out = {}
    out["obs"] = mirror_game_obs(a["obs"], keys, sizes, bm, landmarks, ops=ops)
    out["action"] = mirror_action(a["action"], bm)
    out["z"] = np.full_like(a["z"], np.nan)
    out["target"] = (a["target"] * M2).astype(a["target"].dtype)
    out["aim"] = (a["aim"] * M2).astype(a["aim"].dtype)
    rp, rm = mirror_world_pose(a["root_pos"], a["root_mat"])
    out["root_pos"] = rp.astype(a["root_pos"].dtype)
    out["root_mat"] = rm.astype(a["root_mat"].dtype)
    out["ball_pos"] = (a["ball_pos"] * M3).astype(a["ball_pos"].dtype)
    out["ball_vel"] = (a["ball_vel"] * M3).astype(a["ball_vel"].dtype)
    out["mirrored"] = np.ones(n, np.int8)

    # expert obs: per layout, because the field order differs between layouts
    eo = np.array(a["expert_obs"], copy=True)
    for lay in ds.meta["layouts"]:
        fields = tuple(lay["fields"])
        dim = int(lay["obs_dim"])
        m = a["layout"] == int(lay["id"])
        if not m.any() or not fields or dim <= 0:
            continue
        eo[m, :dim] = mirror_expert_obs(a["expert_obs"][m, :dim], fields, bm)
        if verbose:
            print(f"[mirror] layout {lay['id']} ({lay['skill']}, {dim}w): "
                  f"{int(m.sum())} samples")
    out["expert_obs"] = eo

    for k, v in a.items():
        if k in out or k == "landmarks":
            continue
        out[k] = np.array(v, copy=True)

    meta = dict(ds.meta)
    if append:
        merged = {k: (v if k == "landmarks" else np.concatenate([a[k], out[k]]))
                  for k, v in a.items()}
        meta["n_samples"] = int(merged["action"].shape[0])
        meta["augmentation"] = dict(mirror="y", appended=True, n_mirrored=n)
        return BCDataset(merged, meta)
    out["landmarks"] = a["landmarks"]
    meta["n_samples"] = n
    meta["augmentation"] = dict(mirror="y", appended=False, n_mirrored=n)
    return BCDataset(out, meta)


# --- simulation-side helper (verification, and any future warp env) ---------

def mirror_mj_state(model, qpos, qvel=None, bm: Optional[BodyMirror] = None,
                    creature: str = "ant"):
    """Mirror a full MuJoCo (qpos, qvel) of a scene made of free bodies + creatures.

    Every free joint's position reflects and its quaternion ``(w, x, y, z)``
    becomes ``(w, -x, y, -z)`` (conjugation by the 180-degree y-rotation, which
    is what ``M R M`` is in quaternion form). Free-joint velocities are the easy
    case: the linear part is polar and the angular part axial, and MuJoCo's
    choice of frame for each does not matter here because ``M v = (vx, -vy, vz)``
    and ``-M w = (-wx, wy, -wz)`` hold in both the world and the body frame.
    Hinge blocks are permuted and signed by the creature's actuator map.

    Every creature in the scene is assumed to be `creature` kind, and its 8
    hinges are assumed to follow its free joint contiguously — true for the
    dm_soccer scene (`creature/`, `creature_1/`, ... each contribute
    7 + n_joints qpos in order). Verified in the tests against the real env.
    """
    import mujoco
    bm = bm or body_mirror(creature)
    nj = bm.n_joints
    q = np.array(qpos, float)
    v = None if qvel is None else np.array(qvel, float)
    out_q, out_v = q.copy(), (None if v is None else v.copy())
    free = mujoco.mjtJoint.mjJNT_FREE
    hinge = mujoco.mjtJoint.mjJNT_HINGE
    j = 0
    while j < model.njnt:
        if int(model.jnt_type[j]) != int(free):
            raise MirrorError(
                f"joint {j} is not attached to a free body; this helper only "
                "understands a scene of free bodies (ball) and free creatures.")
        adr, dof = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
        x, y, z, w, qx, qy, qz = q[adr:adr + 7]
        out_q[adr:adr + 7] = [x, -y, z, w, -qx, qy, -qz]
        if v is not None:
            lx, ly, lz, ax, ay, az = v[dof:dof + 6]
            out_v[dof:dof + 6] = [lx, -ly, lz, -ax, ay, -az]
        run = 0
        while j + 1 + run < model.njnt and int(model.jnt_type[j + 1 + run]) == int(hinge):
            run += 1
        if run == nj:                                   # a creature
            out_q[adr + 7:adr + 7 + nj] = mirror_action(q[adr + 7:adr + 7 + nj], bm)
            if v is not None:
                out_v[dof + 6:dof + 6 + nj] = mirror_action(v[dof + 6:dof + 6 + nj], bm)
        elif run != 0:
            raise MirrorError(
                f"free joint {j} is followed by {run} hinges, but "
                f"{bm.creature} has {nj}. Refusing to guess the layout.")
        j += 1 + run
    return (out_q, out_v) if qvel is not None else out_q


def main(argv=None):
    import argparse
    p = argparse.ArgumentParser(description="Show a creature's derived y-mirror.")
    p.add_argument("creature", nargs="?", default="ant")
    a = p.parse_args(argv)
    bm = body_mirror(a.creature)
    print(bm.describe())
    print("bodies:", list(bm.body_names))
    print("touch: ", list(bm.touch_names))
    print("inertial reflection:", np.diag(bm.inertial).tolist())


if __name__ == "__main__":
    main()
