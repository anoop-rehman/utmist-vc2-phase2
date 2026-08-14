"""N-agent (2v2) dev-ant run-to-goal scene.

`scene.dev_run_to_goal_xml` nominally takes `n_agents`, but two things in it are
hard-wired to exactly two agents and both are silent failures at four:

  1. **`INIT_POS` / `INIT_EULER` have two entries.** `n_agents=4` raises
     IndexError, which is the harmless one.
  2. **The contact bitmask is a two-agent trick.** Their evo merger
     (`competevo/evo_envs/evo_utils.py:88-89`) writes `conaffinity=i,
     contype=1-i` unconditionally. For i in {0,1} that is exactly "collide with
     the other agent and the floor, never with yourself". For i in {2,3} it is
     `contype=-1, conaffinity=2` and `contype=-2, conaffinity=3`, i.e. garbage:
     MuJoCo collides geoms when `(contype1 & conaffinity2) || (contype2 &
     conaffinity1)`, and with those values agents 2 and 3 SELF-collide and the
     pairwise table is asymmetric in a way nothing in the observation reports.

So this module keeps `_dev_ant_body_xml` (the robot is unchanged, verbatim) and
replaces the merger: a real one-bit-per-agent mask,

    agent i:   contype = 1 << i          conaffinity = ALL ^ (1 << i)
    world geoms (floor, goal rods): contype = conaffinity = ALL

which gives, for every i != j, `(1<<i) & (ALL ^ (1<<j)) = 1<<i != 0` -> collide;
for i == j, `(1<<i) & (ALL ^ (1<<i)) = 0` in both directions -> no self-collision;
and `(1<<i) & ALL != 0` -> everything collides with the floor and the goal rods.

`tests/test_team2v2.py::test_bitmask_matches_theirs_at_two_agents` asserts the
resulting COLLIDING-PAIR SET is identical to theirs at n=2 (the integers are
not, and do not need to be), and
`test_naive_bitmask_is_broken_at_four_agents` asserts their formula is broken at
n=4 -- a control that fails if the bitmask here is replaced by theirs.

Spawn geometry is the design intent of unit 2f: player 1 of each team keeps its
1v1 spawn, player 2 spawns on its OWN goal line, i.e. behind its teammate and
maximally far from the goal it is attacking.
"""

from dataclasses import dataclass

import mujoco
import numpy as np

from rower_soccer.competevo_port.scene import (_DEV_MOTOR_JOINTS, DESIGN_DIM,
                                               DEV_TORSO_LOCAL, GEAR, GOAL_X,
                                               TIMESTEP, SceneMeta,
                                               _dev_ant_body_xml, _fmt)

SPAWN_Z = 0.75

# The user's layout. Team A = agents (0, 2) attacking +x; team B = (1, 3)
# attacking -x. `_agent_slices` derives the goal from the sign of the spawn x
# (their `MultiAgentEnv.__init__:133-137`), so this table is the whole spec.
#
# `back_x` is the striker's own goal line. Spawning the COM exactly on x = 4.0
# puts the ant astride the goal-line cylinder (radius 0.03, lying on the floor
# at z=0); `probe_geometry.py` measures whether that matters.
def team_init_pose(n_agents=4, back_x=GOAL_X, front_x=1.0, back_y=0.0):
    """[(pos, euler)] for `n_agents`, in the port's agent order.

    Agent order is (A1, B1, A2, B2) so that agents 0 and 1 are BIT-FOR-BIT the
    1v1 pair: same spawn, same euler, same goal, same qpos slice offsets. That
    is deliberate -- it means a 2v2 scene truncated to its first two agents is
    the validated 1v1 scene, and `test_first_two_agents_match_1v1` checks it.
    """
    pos = [(-front_x, 0.0, SPAWN_Z), (front_x, 0.0, SPAWN_Z),
           (-back_x, back_y, SPAWN_Z), (back_x, -back_y, SPAWN_Z)]
    eul = [(0.0, 0.0, 0.0), (0.0, 0.0, 180.0),
           (0.0, 0.0, 0.0), (0.0, 0.0, 180.0)]
    return list(zip(pos[:n_agents], eul[:n_agents]))


# Team of each agent in the order above.
def team_of(n_agents=4):
    return [i % 2 for i in range(n_agents)]


def _team_agent_default_xml(agent_id, rgb, n_agents):
    """One agent's `<default>`, with the N-agent contact bitmask."""
    p = f"agent{agent_id}"
    all_bits = (1 << n_agents) - 1
    contype = 1 << agent_id
    conaff = all_bits ^ contype
    return f"""    <default class="{p}">
      <joint armature="1" damping="1" limited="true"/>
      <geom conaffinity="{conaff}" contype="{contype}" condim="3"
            density="5.0" friction="1 0.5 0.5" margin="0.01"
            rgba="{_fmt(*rgb)} 1" material="geom"/>
    </default>"""


def _their_agent_default_xml(agent_id, rgb):
    """THEIR two-agent formula, extended naively to N. Only used by the
    negative-control test, which asserts it is broken at n=4."""
    p = f"agent{agent_id}"
    return f"""    <default class="{p}">
      <joint armature="1" damping="1" limited="true"/>
      <geom conaffinity="{agent_id}" contype="{1 - agent_id}" condim="3"
            density="5.0" friction="1 0.5 0.5" margin="0.01"
            rgba="{_fmt(*rgb)} 1" material="geom"/>
    </default>"""


def dev_team_xml(n_agents=4, solver="Newton", iterations=100, integrator="RK4",
                 timestep=TIMESTEP, poses=None, naive_bitmask=False,
                 world_conaffinity=None):
    """The merged N-agent dev scene at its base design.

    `naive_bitmask=True` reproduces their 2-agent formula for every agent; it
    exists so the control test has something that is actually wrong to compare
    against.
    """
    poses = poses or team_init_pose(n_agents)
    assert len(poses) == n_agents
    all_bits = (1 << n_agents) - 1
    world = all_bits if world_conaffinity is None else world_conaffinity
    bodies = "\n".join(_dev_ant_body_xml(i, p, e)
                       for i, (p, e) in enumerate(poses))
    rgbs = _team_rgb(n_agents)
    if naive_bitmask:
        defaults = "\n".join(_their_agent_default_xml(i, rgbs[i])
                             for i in range(n_agents))
    else:
        defaults = "\n".join(_team_agent_default_xml(i, rgbs[i], n_agents)
                             for i in range(n_agents))
    motors = "\n".join(
        f'    <motor ctrllimited="true" ctrlrange="-1.0 1.0"'
        f' joint="agent{i}/{j}" gear="{GEAR:g}" name="agent{i}/{j}"'
        f' class="agent{i}"/>'
        for i in range(n_agents) for j in _DEV_MOTOR_JOINTS)
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
    <geom contype="{world}" conaffinity="{world}" friction="1 .1 .1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="1 1 1 1" size="20 20 0.125" type="plane"/>
    <geom contype="{world}" conaffinity="{world}" fromto="4 -5 0  4 +5 0" name="rightgoal" rgba="1 0 0 0.5" size=".03" type="cylinder"/>
    <geom contype="{world}" conaffinity="{world}" fromto="-4 -5 0  -4 +5 0" name="leftgoal" rgba="1 0 0 0.5" size=".03" type="cylinder"/>
{bodies}
  </worldbody>

  <actuator>
{motors}
  </actuator>
</mujoco>
"""


def _team_rgb(n_agents):
    """Team colours -- purely visual, but the renders are load-bearing here:
    a 2v2 clip in which all four ants are the same beige is unreadable."""
    a = (0.98, 0.87, 0.67)     # team A: theirs, unchanged
    b = (0.45, 0.62, 0.95)     # team B
    return [a if i % 2 == 0 else b for i in range(n_agents)]


@dataclass
class TeamSceneMeta(SceneMeta):
    team: tuple = ()               # team index per agent
    teammate: tuple = ()           # index of the (single) teammate
    opponents: tuple = ()          # indices of the two opponents, ordered
    n_others: int = 0


def _team_agent_slices(model, agent_id, order):
    """`scene._agent_slices`, but `other_qpos_xy` is a LIST of every other
    agent's root (x, y), ordered `[teammate, opp_near, opp_far]` rather than
    their `get_other_qpos()[:2]`.

    Theirs takes the first two entries of the concatenated complement, which for
    two agents is exactly "the opponent's x, y" and for four agents is "agent
    0's x, y for everybody except agent 0, which sees agent 1". That is not a
    design choice we can port; it is a two-agent idiom.
    """
    from rower_soccer.competevo_port.scene import _agent_slices
    s = _agent_slices(model, agent_id, DEV_TORSO_LOCAL)
    starts = []
    for j in order:
        pj = f"agent{j}/"
        jids = [i for i in range(model.njnt) if model.jnt(i).name.startswith(pj)]
        starts += [int(model.jnt_qposadr[jids[0]]),
                   int(model.jnt_qposadr[jids[0]]) + 1]
    s.other_qpos_xy = starts
    return s


def build_dev_team_scene(n_agents=4, back_x=GOAL_X, front_x=1.0, back_y=0.0,
                         poses=None, **xml_kwargs):
    """Compile the N-agent dev scene and resolve the per-agent plumbing.

    obs = [stage flag (1) | scale (20) | own qpos (15) | own qvel (14)
           | 2 * (n_agents - 1) other-root xy]
        = 52 at n=2 (the validated 1v1 layout, unchanged) and 56 at n=4.
    """
    poses = poses or team_init_pose(n_agents, back_x=back_x, front_x=front_x,
                                    back_y=back_y)
    model = mujoco.MjModel.from_xml_string(
        dev_team_xml(n_agents, poses=poses, **xml_kwargs))
    team = team_of(n_agents)
    # Canonical other-ordering, per agent: teammate first, then the opponents
    # nearest-spawn-first. This is what makes the observation ROLE-SYMMETRIC --
    # every agent's obs slot 0 is "my teammate", not "agent 0".
    agents, orders = [], []
    for i in range(n_agents):
        mates = [j for j in range(n_agents) if j != i and team[j] == team[i]]
        opps = [j for j in range(n_agents) if team[j] != team[i]]
        order = mates + opps
        orders.append(order)
        agents.append(_team_agent_slices(model, i, order))
    a0 = agents[0]
    sim_obs_dim = ((a0.qpos[1] - a0.qpos[0]) + (a0.qvel[1] - a0.qvel[0])
                   + 2 * (n_agents - 1))
    n_motor = a0.ctrl[1] - a0.ctrl[0]
    meta = TeamSceneMeta(n_agents, model.nq, model.nv, model.nu, agents,
                         1 + DESIGN_DIM + sim_obs_dim, DESIGN_DIM + n_motor)
    meta.sim_obs_dim, meta.n_motor = sim_obs_dim, n_motor
    meta.team = tuple(team)
    meta.teammate = tuple(o[0] for o in orders)
    meta.opponents = tuple(tuple(o[1:]) for o in orders)
    meta.n_others = n_agents - 1
    return model, meta


def colliding_pairs(model):
    """The set of (geom_i, geom_j) pairs the bitmask permits, as an agent-level
    table: `{(a, b)}` over agent indices (or -1 for a world geom). Used by the
    control test -- the integers in the XML are not the invariant, the pair set
    is."""
    def owner(g):
        nm = model.geom(g).name
        return int(nm.split("/")[0][5:]) if nm.startswith("agent") else -1
    pairs = set()
    for i in range(model.ngeom):
        for j in range(i + 1, model.ngeom):
            ct1, ca1 = int(model.geom_contype[i]), int(model.geom_conaffinity[i])
            ct2, ca2 = int(model.geom_contype[j]), int(model.geom_conaffinity[j])
            if (ct1 & ca2) or (ct2 & ca1):
                pairs.add(tuple(sorted((owner(i), owner(j)))))
    return pairs


def spawn_table(meta, model):
    """Distances the design doc quotes. Rows are agents; every number is read
    off the compiled model, not off the constants above."""
    rows = []
    xy = np.array([[float(model.body_pos[a.torso_body][0]),
                    float(model.body_pos[a.torso_body][1])]
                   for a in meta.agents])
    for i, a in enumerate(meta.agents):
        d = {"agent": i, "team": meta.team[i],
             "spawn_x": xy[i, 0], "spawn_y": xy[i, 1],
             "goal_x": a.goal_x, "move_left": a.move_left,
             "dist_to_target_line": abs(a.goal_x - xy[i, 0]),
             "dist_to_own_line": abs(-a.goal_x - xy[i, 0])}
        for j in range(meta.n_agents):
            if j != i:
                d[f"dist_to_{j}"] = float(np.linalg.norm(xy[i] - xy[j]))
        rows.append(d)
    return rows
