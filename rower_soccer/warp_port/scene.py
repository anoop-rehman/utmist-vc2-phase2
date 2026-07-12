"""Standalone MJCF scene builder for Warp drill training.

Two scenes:

- `build_creature_scene` — creature + floor. Used by `follow`, whose targets
  are kinematic abstractions handled in torch by the env; they never touch the
  physics.
- `build_creature_ball_scene` — creature + floor + ball. Used by dribble/kick/
  shoot, where the ball IS a physics entity the creature contacts. This is the
  part PIPELINE_V2 line 91 called "a config change"; it is not.

Solver options mirror the project arena.xml where Warp supports them (elliptic
cone, dt 0.0025; noslip_iterations is not supported by mujoco_warp and is
dropped — one source of CPU/GPU behavior deltas the transfer eval watches for).

The ball is DeepMind's, and it is authoritative. `BallSpec` mirrors
`dm_control.locomotion.soccer.soccer_ball.SoccerBall` exactly (0.35 m, 0.045 kg,
condim 6, friction (0.7, 0.075, 0.075), priority 1) so the Warp drills, the CPU
drills in `rower_soccer/drills/`, and the soccer env all contact the same ball.
Do NOT rescale the ball to fit a creature: the ball is sized against the goal
and the pitch, and matching it is the creature's job (see
`tools/unity2mujoco.py --length-scale`).
"""

from dataclasses import dataclass

import mujoco
import numpy as np

_BASE_XML = """
<mujoco model="warp_drill">
  <option cone="elliptic" timestep="0.0025"/>
  <worldbody>
    <geom name="floor" type="plane" size="60 60 1" friction="1 0.5 0.5"/>
  </worldbody>
</mujoco>
"""


@dataclass
class BallSpec:
    """dm_control SoccerBall, verbatim. Defaults are the spec, not a guess."""

    radius: float = 0.35
    mass: float = 0.045
    # MuJoCo friction triple (sliding, torsional, rolling).
    friction: tuple = (0.7, 0.075, 0.075)
    # condim=6 (sliding + torsional + rolling). NOT optional, and the reason
    # DeepMind sets it: MuJoCo's default condim=3 is sliding-only and silently
    # *ignores* the 2nd and 3rd entries of `friction`, so with condim=3 a
    # rolling-friction sweep changes nothing and the ball rolls forever.
    condim: int = 6
    # priority=1 makes this geom's condim/friction win the contact mix outright
    # rather than being combined with the floor's. Without it the floor's
    # condim=3 drags the contact back down to sliding-only.
    priority: int = 1
    solref: tuple = (0.02, 1.0)
    rgba: tuple = (0.9, 0.9, 0.9, 1.0)

    @property
    def density(self):
        return self.mass / ((4.0 / 3.0) * np.pi * self.radius ** 3)


@dataclass
class SceneMeta:
    root_body: int          # body id of creature root
    body_ids: list          # creature body ids in uid order (root first)
    qpos_root: int          # start of freejoint qpos (7 numbers)
    qvel_root: int
    joint_qpos: list        # hinge qpos addresses
    joint_qvel: list
    sensor_slices: dict     # name -> (start, dim) into sensordata
    nu: int
    spawn_z: float
    # ball (None for the ball-free `follow` scene)
    ball_body: int = None
    ball_qpos: int = None   # start of ball freejoint qpos (7 numbers)
    ball_qvel: int = None   # start of ball freejoint qvel (6 numbers)
    ball_radius: float = None

    @property
    def has_ball(self):
        return self.ball_body is not None


def creature_size(creature_xml_path):
    """(total mass kg, bbox height m) of the creature alone. Sanity/reporting:
    the creature is scaled to the ball, never the other way round."""
    m = mujoco.MjModel.from_xml_path(creature_xml_path)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mass = float(m.body_mass[1:].sum())  # skip worldbody
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for g in range(m.ngeom):
        lo = np.minimum(lo, d.geom_xpos[g] - m.geom_size[g])
        hi = np.maximum(hi, d.geom_xpos[g] + m.geom_size[g])
    return mass, float((hi - lo)[2])


def build_creature_scene(creature_xml_path, prefix="c-", ball: BallSpec = None):
    """Returns (mujoco.MjModel, SceneMeta). `ball=None` -> the follow scene."""
    spec = mujoco.MjSpec.from_string(_BASE_XML)
    sub = mujoco.MjSpec.from_file(creature_xml_path)
    frame = spec.worldbody.add_frame()
    attached_root = frame.attach_body(sub.worldbody.bodies[0], prefix, "")
    attached_root.add_freejoint()

    if ball is not None:
        b = spec.worldbody.add_body(name="ball", pos=[0.0, 0.0, ball.radius])
        b.add_freejoint(name="ball_free")
        g = b.add_geom(name="ball_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                       size=[ball.radius, 0.0, 0.0])
        g.density = ball.density
        g.friction = list(ball.friction)
        g.condim = ball.condim
        g.priority = ball.priority
        g.solref = list(ball.solref)
        g.rgba = list(ball.rgba)

    model = spec.compile()

    root_name = f"{prefix}seg0"
    root_body = model.body(root_name).id
    body_ids = [model.body(i).id for i in range(model.nbody)
                if model.body(i).name.startswith(prefix)]
    ball_body = model.body("ball").id if ball is not None else None

    # Free-joint addresses. Dispatch on the joint's BODY, not just its type:
    # with a ball in the scene there are two free joints, and claiming
    # whichever comes last would silently hand the creature's root address to
    # the ball (and vice versa) -- every proprio observation would be wrong.
    root_jnt = ball_jnt = None
    joint_qpos, joint_qvel = [], []
    for j in range(model.njnt):
        jnt = model.joint(j)
        if jnt.type[0] == mujoco.mjtJoint.mjJNT_FREE:
            if int(model.jnt_bodyid[j]) == root_body:
                root_jnt = j
            elif ball_body is not None and int(model.jnt_bodyid[j]) == ball_body:
                ball_jnt = j
        elif jnt.name.startswith(prefix):
            joint_qpos.append(int(jnt.qposadr[0]))
            joint_qvel.append(int(jnt.dofadr[0]))
    if root_jnt is None:
        raise RuntimeError("creature root freejoint not found")
    if ball is not None and ball_jnt is None:
        raise RuntimeError("ball freejoint not found")

    sensor_slices = {}
    for s in range(model.nsensor):
        sen = model.sensor(s)
        name = sen.name.removeprefix(prefix)
        sensor_slices[name] = (int(sen.adr[0]), int(sen.dim[0]))

    # spawn z = root body's default z in the attached frame (converter sets
    # it so the creature rests on the floor)
    spawn_z = float(model.body(root_name).pos[2]) or float(
        mujoco.MjModel.from_xml_path(creature_xml_path).body("seg0").pos[2])

    meta = SceneMeta(
        root_body=root_body,
        body_ids=body_ids,
        qpos_root=int(model.jnt_qposadr[root_jnt]),
        qvel_root=int(model.jnt_dofadr[root_jnt]),
        joint_qpos=joint_qpos,
        joint_qvel=joint_qvel,
        sensor_slices=sensor_slices,
        nu=model.nu,
        spawn_z=spawn_z,
        ball_body=ball_body,
        ball_qpos=int(model.jnt_qposadr[ball_jnt]) if ball is not None else None,
        ball_qvel=int(model.jnt_dofadr[ball_jnt]) if ball is not None else None,
        ball_radius=ball.radius if ball is not None else None,
    )
    return model, meta


def build_creature_ball_scene(creature_xml_path, prefix="c-", ball: BallSpec = None):
    """Creature + ball. Ball defaults to dm_control's SoccerBall spec."""
    return build_creature_scene(creature_xml_path, prefix=prefix,
                                ball=ball or BallSpec())
