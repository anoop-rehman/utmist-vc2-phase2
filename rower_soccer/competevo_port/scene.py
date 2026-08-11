"""Merged two-ant run-to-goal MJCF, rebuilt for mujoco_warp.

This is our replacement for CompetEvo's runtime XML surgery. Theirs
(`gym_compete/new_envs/utils.py:create_multiagent_xml`) parses `world_body.xml`
and two copies of `ant_body.xml` with ElementTree, prefixes every name with
`agent{i}/`, overwrites the root body pos/euler with the registered init pose,
folds each agent's `<default>` into a `<default class="agent{i}">`, and
concatenates the `<actuator>` blocks. It writes the result to disk and gymnasium
compiles it. We emit the same document from a leg table instead, so that

  * the geometry that CompetEvo's dev genome scales (`fromto` of the three leg
    capsules, their radii, the child-body `pos` that keeps links attached, and
    the motor gear) is addressable as data, not as a regex over a string -- stage
    2 writes those per world into model fields rather than re-emitting XML;
  * one compiled model serves every world (their env recompiles per episode).

Faithfulness is a TEST, not a claim: `tests/test_parity.py::test_model_matches_theirs`
compiles their checked-in merged scene and ours and asserts every mass, inertia,
joint range, damping, armature, friction, margin, gear and geom size agrees.

PHYSICS-OPTION DEVIATION (see COMPETEVO_PORT_MAP.md section 6.3)
---------------------------------------------------------------
Their `world_body.xml:4` is
    `<option integrator="RK4" timestep="0.003" solver="PGS" iterations="1000"/>`
mujoco_warp 1.16 raises `NotImplementedError: mjSOL_PGS is unsupported` at
`put_model`, so the SOLVER must change. Measured on this scene, RK4 itself IS
supported by mujoco_warp (unlike the port map's assumption), so we keep the
integrator and the timestep and change only what the backend refuses:

    solver     PGS  -> Newton     (mujoco_warp implements Newton and CG only)
    iterations 1000 -> 100        (mujoco_warp unrolls the solver loop into that
                                   many kernel launches with no early exit, so
                                   1000 is a 10x cost for iterations MuJoCo's PGS
                                   would have exited long before; Newton reaches
                                   the same tolerance in single digits here)

Everything else -- timestep 0.003, frame_skip 5, geom margin 0.01, condim 3,
friction (1, 0.5, 0.5), density 5.0, gear 150, armature 1, damping 1 -- is theirs
verbatim. Contact behaviour therefore differs between the stacks; the parity gate
in tests/ is deliberately built on hand-set states and mj_forward so it measures
the PORT, not the solver.
"""

from dataclasses import dataclass, field

import mujoco
import numpy as np

# Their registration for `run-to-goal-ants-v0` (gym_compete/__init__.py:96-108):
# agent 0 spawns at x=-1 facing +x, agent 1 at x=+1 yawed 180 deg to face -x.
INIT_POS = ((-1.0, 0.0, 0.75), (1.0, 0.0, 0.75))
INIT_EULER = ((0.0, 0.0, 0.0), (0.0, 0.0, 180.0))
# `rgb` kwarg from the same registration, applied to each agent's geom default.
AGENT_RGB = ((0.98, 0.87, 0.67), (0.98, 0.87, 0.67))

# frame_skip (multi_agent_scene.py:26) and the world timestep => control dt.
FRAME_SKIP = 5
TIMESTEP = 0.003
CONTROL_DT = FRAME_SKIP * TIMESTEP     # 0.015 s

# Goal lines: the `rightgoal`/`leftgoal` cylinders at x=+/-4 in world_body.xml.
# MultiAgentEnv.__init__:129-137 reads their compiled geom_pos[0] and hands the
# agent starting at x<0 the RIGHT goal. These rods are real, colliding geometry
# (contype/conaffinity default to 1) -- kept, because the ants can hit them.
GOAL_X = 4.0

# One ant leg, verbatim from `gym_compete/new_envs/assets/ant_body.xml:10-53`.
# (dx, dy) is the leg's planar direction; the aux capsule runs 0 -> (dx,dy), the
# mid capsule repeats it from the aux body, and the foot capsule runs to
# 2*(dx,dy). The dev genome (stage 2) scales exactly these five numbers per leg.
_LEGS = (
    # body,             (dx,  dy),  hip,     ankle,     ankle_axis, ankle_range,
    #   aux_geom,      mid_geom,             foot_geom
    ("front_left_leg",  (0.2, 0.2), "hip_1", "ankle_1", "-1 1 0", "30 70",
     "aux_1_geom", "left_leg_geom", "left_ankle_geom"),
    ("front_right_leg", (-0.2, 0.2), "hip_2", "ankle_2", "1 1 0", "-70 -30",
     "aux_2_geom", "right_leg_geom", "right_ankle_geom"),
    ("back_leg",        (-0.2, -0.2), "hip_3", "ankle_3", "-1 1 0", "-70 -30",
     "aux_3_geom", "back_leg_geom", "third_ankle_geom"),
    ("right_back_leg",  (0.2, -0.2), "hip_4", "ankle_4", "1 1 0", "30 70",
     "aux_4_geom", "rightback_leg_geom", "fourth_ankle_geom"),
)

# Their actuator block order (ant_body.xml:57-64) -- NOT leg order. The action
# vector is indexed by this, so getting it wrong silently swaps a policy's legs.
_MOTOR_JOINTS = ("hip_4", "ankle_4", "hip_1", "ankle_1",
                 "hip_2", "ankle_2", "hip_3", "ankle_3")
GEAR = 150.0

# Their per-agent body order is declaration order, which their `_set_body`
# name-filter preserves: torso, then per leg (upper, aux, foot).
TORSO_LOCAL = "torso"


def _fmt(*v):
    return " ".join(f"{x:g}" for x in v)


def _ant_body_xml(agent_id, pos, euler):
    """One agent's `<body>` subtree, already name-prefixed and class-tagged.

    Their `add_prefix(..., force_set=True)` invents `agent{i}/anon<random>` names
    for the foot bodies, which are unnamed in `ant_body.xml`. We name them
    `agent{i}/foot_{k}` -- the only textual difference from their document, and it
    is invisible to physics (nothing looks those names up; the per-agent slices
    key off the `agent{i}/` prefix, which we keep).
    """
    p = f"agent{agent_id}"
    out = [f'<body name="{p}/{TORSO_LOCAL}" pos="{_fmt(*pos)}" euler="{_fmt(*euler)}">',
           f'  <geom name="{p}/torso_geom" pos="0 0 0" size="0.25" type="sphere" class="{p}"/>',
           f'  <joint armature="0" damping="0" limited="false" margin="0.01"'
           f' name="{p}/root" pos="0 0 0" range="-30 30" type="free"/>']
    for k, (leg, (dx, dy), hip, ankle, ax, arange,
            g_aux, g_mid, g_foot) in enumerate(_LEGS, start=1):
        out += [
            f'  <body name="{p}/{leg}" pos="0 0 0">',
            f'    <geom fromto="{_fmt(0, 0, 0, dx, dy, 0)}" name="{p}/{g_aux}"'
            f' size="0.08" type="capsule" class="{p}"/>',
            f'    <body name="{p}/aux_{k}" pos="{_fmt(dx, dy, 0)}">',
            f'      <joint axis="0 0 1" name="{p}/{hip}" pos="0 0 0"'
            f' range="-30 30" type="hinge"/>',
            f'      <geom fromto="{_fmt(0, 0, 0, dx, dy, 0)}" name="{p}/{g_mid}"'
            f' size="0.08" type="capsule" class="{p}"/>',
            f'      <body name="{p}/foot_{k}" pos="{_fmt(dx, dy, 0)}">',
            f'        <joint axis="{ax}" name="{p}/{ankle}" pos="0 0 0"'
            f' range="{arange}" type="hinge"/>',
            f'        <geom fromto="{_fmt(0, 0, 0, 2 * dx, 2 * dy, 0)}"'
            f' name="{p}/{g_foot}" size="0.08" type="capsule" class="{p}"/>',
            '      </body>',
            '    </body>',
            '  </body>',
        ]
    out.append('</body>')
    return "\n".join(out)


def _agent_default_xml(agent_id, rgb):
    """Their `<default class="agent{i}">`: the ant's own `<default>` children,
    with the geom's rgba replaced by the registered colour (utils.py:79-90).
    Note what does NOT happen here for the fixed-morph ants: because the ant's
    default already declares a `<geom>`, their `color_set` flag short-circuits
    the `contype=i / conaffinity=1` branch, so BOTH agents keep contype=1
    (implicit) and conaffinity=1 -- i.e. the run-to-goal ants DO self-collide and
    DO collide with each other. The contype/conaffinity trick the port map
    describes at section 1.2 belongs to the *evo* merger (`evo_utils.py:88-89`),
    not to this one. Reproduced as-is.
    """
    p = f"agent{agent_id}"
    return f"""    <default class="{p}">
      <joint armature="1" damping="1" limited="true"/>
      <geom conaffinity="1" condim="3" density="5.0" friction="1 0.5 0.5"
            margin="0.01" rgba="{_fmt(*rgb)} 1" material="geom"/>
      <motor ctrllimited="true" ctrlrange="-.4 .4"/>
    </default>"""


def run_to_goal_xml(n_agents=2, solver="Newton", iterations=100,
                    integrator="RK4", timestep=TIMESTEP):
    """The merged scene as an MJCF string. Defaults carry the deviation
    documented in this module's docstring; pass solver="PGS", iterations=1000 to
    get their exact options for a CPU-MuJoCo cross-check."""
    bodies = "\n".join(
        _ant_body_xml(i, INIT_POS[i], INIT_EULER[i])
        for i in range(n_agents))
    defaults = "\n".join(_agent_default_xml(i, AGENT_RGB[i])
                         for i in range(n_agents))
    motors = "\n".join(
        f'    <motor ctrllimited="true" ctrlrange="-1.0 1.0"'
        f' joint="agent{i}/{j}" gear="{GEAR:g}" class="agent{i}"/>'
        for i in range(n_agents) for j in _MOTOR_JOINTS)
    return f"""<mujoco model="mutiagent_world">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="{integrator}" timestep="{timestep}" solver="{solver}" iterations="{iterations}"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
{defaults}
  </default>

  <visual>
    <headlight ambient=".1 .1 .1" diffuse=".6 .6 .6" specular="0.3 0.3 0.3"/>
    <map znear=".01"/>
    <quality shadowsize="4096"/>
    <global offwidth="1280" offheight="720"/>
  </visual>

  <asset>
    <texture builtin="gradient" height="100" rgb1=".4 .5 .6" rgb2="0 0 0" type="skybox" width="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="0 0 0" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture name="MatPlane" type="2d" builtin="checker" rgb1=".5 .5 .5" rgb2=".5 .5 .5" width="300" height="300" mark="edge" markrgb="0.1 0.1 0.1"/>
    <material name="MatPlane" texture="MatPlane" texrepeat="2 2" texuniform="true" reflectance=".2"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
  </asset>

  <worldbody>
    <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
    <geom contype="1" conaffinity="1" friction="1 .1 .1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="1 1 1 1" size="20 20 0.125" type="plane"/>
    <geom fromto="4 -5 0  4 +5 0" name="rightgoal" rgba="1 0 0 0.5" size=".03" type="cylinder"/>
    <geom fromto="-4 -5 0  -4 +5 0" name="leftgoal" rgba="1 0 0 0.5" size=".03" type="cylinder"/>
{bodies}
  </worldbody>

  <actuator>
{motors}
  </actuator>
</mujoco>
"""


@dataclass
class AgentSlices:
    """Everything `gym_compete.new_envs.agents.Agent` derives per agent, resolved
    once against the compiled model instead of after every recompile.

    `qpos`/`qvel`/`ctrl` are (start, stop) into the GLOBAL batched vectors, which
    is what makes their contiguous-slice assumption (`agent.py:206-227`) a plain
    tensor view here. `other_qpos_xy` is the complement's first two entries --
    their `get_other_qpos()[:2]`, i.e. the opponent's root x,y.
    """
    agent_id: int
    qpos: tuple
    qvel: tuple
    ctrl: tuple
    body_ids: list
    torso_body: int
    other_qpos_xy: list
    goal_x: float
    move_left: bool
    geom_ids: list = field(default_factory=list)


@dataclass
class SceneMeta:
    n_agents: int
    nq: int
    nv: int
    nu: int
    agents: list
    obs_dim: int
    act_dim: int


def _agent_slices(model, agent_id):
    prefix = f"agent{agent_id}/"
    body_ids = [i for i in range(model.nbody)
                if model.body(i).name.startswith(prefix)]
    jnt_ids = [i for i in range(model.njnt)
               if model.jnt(i).name.startswith(prefix)]
    # Their JNT_NPOS map (agent.py:17-21): free=7, ball=4, slide/hinge=1.
    npos = {mujoco.mjtJoint.mjJNT_FREE: 7, mujoco.mjtJoint.mjJNT_BALL: 4,
            mujoco.mjtJoint.mjJNT_SLIDE: 1, mujoco.mjtJoint.mjJNT_HINGE: 1}
    qadr = model.jnt_qposadr[jnt_ids]
    qpos = (int(qadr[0]), int(qadr[-1]) + npos[mujoco.mjtJoint(model.jnt_type[jnt_ids[-1]])])
    dofadr = model.jnt_dofadr[jnt_ids]
    dofnum = model.body_dofnum[body_ids]
    qvel = (int(dofadr[0]), int(dofadr[-1] + dofnum[[i for i in range(len(dofnum))
                                                    if dofnum[i] > 0][-1]]))
    act_ids = [i for i in range(model.nu)
               if model.actuator(i).name.startswith(prefix)
               or model.jnt(model.actuator_trnid[i, 0]).name.startswith(prefix)]
    ctrl = (int(act_ids[0]), int(act_ids[-1]) + 1)
    torso = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                  prefix + TORSO_LOCAL))
    # get_other_qpos()[:2]: concat(qpos[:start], qpos[stop:])[:2].
    other = [i for i in range(model.nq) if not (qpos[0] <= i < qpos[1])][:2]
    # Their goal assignment (MultiAgentEnv.__init__:133-137 + Ant.set_goal):
    # an agent starting at x>0 runs left to x=-4, otherwise right to x=+4, and
    # `move_left` sign-flips the dense forward reward.
    x0 = float(model.body_pos[torso][0])
    goal_x = -GOAL_X if x0 > 0 else GOAL_X
    return AgentSlices(agent_id, qpos, qvel, ctrl, body_ids, torso, other,
                       goal_x, x0 > 0,
                       [i for i in range(model.ngeom)
                        if model.geom(i).name.startswith(prefix)])


def build_run_to_goal_scene(n_agents=2, **xml_kwargs):
    """Compile the merged scene and resolve the per-agent index plumbing."""
    model = mujoco.MjModel.from_xml_string(run_to_goal_xml(n_agents, **xml_kwargs))
    agents = [_agent_slices(model, i) for i in range(n_agents)]
    a0 = agents[0]
    obs_dim = (a0.qpos[1] - a0.qpos[0]) + (a0.qvel[1] - a0.qvel[0]) + 2
    return model, SceneMeta(n_agents, model.nq, model.nv, model.nu, agents,
                            obs_dim, a0.ctrl[1] - a0.ctrl[0])


def their_scene_path():
    """Their checked-in merged scene, for the model-equivalence test. Note their
    env OVERWRITES this file on construction (multi_agent_env.py:114-119), so it
    is a build artifact of the last run, not a hand-maintained asset."""
    return ("/workspace/competevo/gym_compete/new_envs/assets/"
            "world_body.ant_body.ant_body.xml")
