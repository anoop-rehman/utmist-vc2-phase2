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

# Contact time constants used for the Warp backend only; see build_creature_scene.
# CPU (arena.xml) stays at MuJoCo's 0.02. These differ ON PURPOSE.
WARP_SOLREF_TIMECONST = 0.005        # creature, ground, walls, goals  (2x dt)
# The ball needs its own, softer value: it is 45 g, and 2x dt is MuJoCo's stability
# floor, where a very light body diverges. 4x dt. See build_creature_scene.
WARP_BALL_SOLREF_TIMECONST = 0.010

# dm_soccer's 2v2 pitch, transcribed from the compiled dm_control model so the
# Warp scene and the CPU drill (rower_soccer/drills/follow.py -> Pitch) are the
# same world. Ground 96 x 72 m, four bounding walls, two goals (10 capsules each;
# posts at y = +/-11.88, crossbar at z = 5.33).
#
# At the drills' +/-10 m bounds the worm reaches NONE of this: it is here so the
# scenes do not silently diverge, and so `shoot` has a real goal to aim at.
# Measured cost of the extra 24 geoms under mujoco_warp's n-by-n broadphase:
# 268k -> 252k env-steps/s at 2048 worlds, i.e. ~6%. Worth it.
#
# Ground friction is the pitch's (1, 0.005, 1e-4), not the old floor's
# (1, 0.5, 0.5). This changes nothing: MuJoCo mixes contact friction as the
# element-wise MAX against the creature's (1, 0.5, 0.5), so the creature-ground
# contact is identical either way, and the ball wins its own contact outright via
# priority=1.
_PITCH_XML = """
    <geom name="ground" type="plane" size="48 36 0.48" friction="1 0.005 0.0001"/>
    <geom name="wall_nx" type="plane" pos="-48 0 0" zaxis="1 0 0"  size="0 0 .48"/>
    <geom name="wall_px" type="plane" pos="48 0 0"  zaxis="-1 0 0" size="0 0 .48"/>
    <geom name="wall_ny" type="plane" pos="0 -36 0" zaxis="0 1 0"  size="0 0 .48"/>
    <geom name="wall_py" type="plane" pos="0 36 0"  zaxis="0 -1 0" size="0 0 .48"/>
    <geom name="home_goal_right_post" type="capsule" size="0.4016 2.6667" pos="-42.6667 -11.8800 2.6667" quat="0.00000 1.00000 0.00000 0.00000"/>
    <geom name="home_goal_left_post" type="capsule" size="0.4016 2.6667" pos="-42.6667 11.8800 2.6667" quat="0.00000 1.00000 0.00000 0.00000"/>
    <geom name="home_goal_top_post" type="capsule" size="0.4057 11.8800" pos="-42.6667 0.0000 5.3333" quat="0.70711 0.70711 0.00000 -0.00000"/>
    <geom name="home_goal_right_base" type="capsule" size="0.4016 2.6667" pos="-45.3333 -11.8800 0.0000" quat="0.70711 0.00000 0.70711 0.00000"/>
    <geom name="home_goal_left_base" type="capsule" size="0.4016 2.6667" pos="-45.3333 11.8800 0.0000" quat="0.70711 0.00000 0.70711 0.00000"/>
    <geom name="home_goal_back_base" type="capsule" size="0.4016 11.8800" pos="-48.0000 0.0000 0.0000" quat="0.70711 0.70711 0.00000 -0.00000"/>
    <geom name="home_goal_right_support" type="capsule" size="0.3012 3.1098" pos="-46.4000 -11.8800 2.6667" quat="0.26693 -0.00000 -0.96371 0.00000"/>
    <geom name="home_goal_right_top_support" type="capsule" size="0.3042 1.0667" pos="-43.7333 -11.8800 5.3333" quat="0.70711 0.00000 -0.70711 0.00000"/>
    <geom name="home_goal_left_support" type="capsule" size="0.3012 3.1098" pos="-46.4000 11.8800 2.6667" quat="0.26693 -0.00000 -0.96371 0.00000"/>
    <geom name="home_goal_left_top_support" type="capsule" size="0.3042 1.0667" pos="-43.7333 11.8800 5.3333" quat="0.70711 0.00000 -0.70711 0.00000"/>
    <geom name="away_goal_right_post" type="capsule" size="0.4016 2.6667" pos="42.6667 11.8800 2.6667" quat="0.00000 1.00000 0.00000 0.00000"/>
    <geom name="away_goal_left_post" type="capsule" size="0.4016 2.6667" pos="42.6667 -11.8800 2.6667" quat="0.00000 1.00000 0.00000 0.00000"/>
    <geom name="away_goal_top_post" type="capsule" size="0.4057 11.8800" pos="42.6667 0.0000 5.3333" quat="0.70711 -0.70711 0.00000 0.00000"/>
    <geom name="away_goal_right_base" type="capsule" size="0.4016 2.6667" pos="45.3333 11.8800 0.0000" quat="0.70711 0.00000 -0.70711 0.00000"/>
    <geom name="away_goal_left_base" type="capsule" size="0.4016 2.6667" pos="45.3333 -11.8800 0.0000" quat="0.70711 0.00000 -0.70711 0.00000"/>
    <geom name="away_goal_back_base" type="capsule" size="0.4016 11.8800" pos="48.0000 0.0000 0.0000" quat="0.70711 -0.70711 0.00000 0.00000"/>
    <geom name="away_goal_right_support" type="capsule" size="0.3012 3.1098" pos="46.4000 11.8800 2.6667" quat="0.26693 -0.00000 0.96371 0.00000"/>
    <geom name="away_goal_right_top_support" type="capsule" size="0.3042 1.0667" pos="43.7333 11.8800 5.3333" quat="0.70711 0.00000 0.70711 0.00000"/>
    <geom name="away_goal_left_support" type="capsule" size="0.3012 3.1098" pos="46.4000 -11.8800 2.6667" quat="0.26693 -0.00000 0.96371 0.00000"/>
    <geom name="away_goal_left_top_support" type="capsule" size="0.3042 1.0667" pos="43.7333 -11.8800 5.3333" quat="0.70711 0.00000 0.70711 0.00000"/>
"""

# Goal mouth centres, for the `shoot` task obs. Home goal is at -x, away at +x.
GOAL_X = 42.6667
GOAL_HALF_WIDTH = 11.88
GOAL_HEIGHT = 5.3333

_BASE_XML = f"""
<mujoco model="warp_drill">
  <option cone="elliptic" timestep="0.0025"/>
  <worldbody>{_PITCH_XML}  </worldbody>
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


def build_creature_scene(creature_xml_path, prefix="c-", ball: BallSpec = None,
                         target_marker=False):
    """Returns (mujoco.MjModel, SceneMeta). `ball=None` -> the follow scene.

    target_marker=True appends a non-colliding red sphere on a free joint, for the
    RENDER-ONLY model (see render.py). The drills' target is a torch computation, not
    a physics entity, so it is invisible unless something draws it. It is appended
    LAST precisely so the creature's and ball's qpos addresses are unchanged -- the
    render model and the physics model must agree on those or the picture lies.

    Never put this in the physics model: it would add 7 qpos / 6 dof to every Warp
    world for something that is only ever looked at.
    """
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

    if target_marker:
        # Appended LAST so creature/ball qpos addresses stay put. contype=conaffinity=0
        # so it collides with nothing -- it is a marker, not an obstacle.
        t = spec.worldbody.add_body(name="target", pos=[0.0, 0.0, 0.5])
        t.add_freejoint(name="target_free")
        tg = t.add_geom(name="target_geom", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                        size=[0.25, 0.0, 0.0])
        tg.contype = 0
        tg.conaffinity = 0
        tg.mass = 1e-3
        tg.rgba = [1.0, 0.2, 0.2, 1.0]

    model = spec.compile()

    # ---- Warp-only contact stiffening. DO NOT "fix" this back to 0.02. -------
    # mujoco_warp resolves contacts far softer than MuJoCo CPU on byte-identical
    # parameters -- same solref/solimp/condim/friction, same timestep, same 10
    # substeps, same Newton solver, same 100 iterations. Measured with follow_v2:
    #
    #   mean floor penetration    Warp -2.28 cm     CPU -0.34 cm    (6.7x softer)
    #
    # It is NOT under-convergence: raising iterations 100 -> 500 and ls_iterations
    # 50 -> 200 moves penetration by <5%. At convergence, penetration is set by
    # constraint COMPLIANCE, not iteration count -- so solref is the only lever.
    #
    # This matters because the worm propels itself entirely by pushing against the
    # ground. Sinking 2 cm into the floor is free traction, and the policy learns
    # to farm it: stiffening contacts to CPU's level costs follow_v2 175 reward
    # (413 -> 238). That soft-contact exploit is the bulk of the residual sim2sim
    # gap -- noslip (fixed in 53971b6) was a much smaller contributor.
    #
    # 0.005 is 2*timestep, MuJoCo's documented stability floor. Verified stable:
    # 600 steps x 256 worlds of random torque, zero NaNs, max|qvel| unchanged.
    # This deliberately makes Warp's solref DIFFER from the CPU drill's 0.02; the
    # two backends need different nominal values to produce the same physics.
    #
    # THE BALL GETS ITS OWN VALUE. Both extremes were tried and both killed a run:
    #
    #   ball solref   margin    worst penetration   outcome
    #   0.005         2x dt       -5.1 cm           dribble_paper_v5: NaN at 17.7M
    #   0.02          8x dt      -20.6 cm           dribble_paper_v6: NaN at 106M
    #   0.010         4x dt       -9.9 cm           <- this
    #
    # Why the ball needs a correction at all: it carries priority=1, so ITS solref
    # governs every contact it is in -- including contact with the 22 kg creature,
    # where the forces are large. (An earlier version of this comment argued the ball
    # is too light to need stiffening. That is wrong: the contact force is set by what
    # is pressing on the ball, not by the ball's own weight.) At 0.02 in Warp the ball
    # is driven 20 cm into whatever it touches -- 57% of its own radius -- and then
    # explosively ejected. That energy injection is the divergence.
    #
    # Why it cannot simply take the creature's 0.005: that is 2*timestep, MuJoCo's
    # documented stability floor, and a 45 g body sitting exactly on the floor is
    # where it bites. Reproduced: 0.005 NaNs within a few episodes under the trained
    # policy. 0.010 keeps a 4x margin and still halves the penetration.
    #
    # This does not fully eliminate the energy injection (the ball still departs at
    # 20-30 m/s off a 0.9 m/s worm), so PPOTrainer ALSO guards against non-finite
    # states rather than trusting this to be airtight. Both are needed.
    model.geom_solref[:, 0] = WARP_SOLREF_TIMECONST
    if ball is not None:
        model.geom_solref[model.geom("ball_geom").id, 0] = WARP_BALL_SOLREF_TIMECONST

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
