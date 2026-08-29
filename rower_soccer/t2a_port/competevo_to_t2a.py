"""D3 M3 E1: the CompetEvo/DeepMind ant, expressed in Transform2Act's `Robot`.

Transform2Act ships its own ant (`assets/mujoco_envs/ant.xml`): 5 bodies, one
free joint plus 4 hinges, 4 motors -- a torso with four SINGLE-segment limbs.
The creature D1 and D2 train (`competevo_port/assets/dev_ant_body.xml`) is the
DeepMind/gym ant: 13 bodies, one free joint plus 8 hinges, 8 motors -- four legs
of three links, hip and ankle actuated. PLAN_D3_M3.md section 0c: they are
different creatures, and every rung of M3 from E3 onward needs OURS inside
THEIR representation, because the soccer creature has to be the creature D1 and
D2 already train.

--------------------------------------------------------------------------
The finding that makes this cheap: the body tree needs no restructuring
--------------------------------------------------------------------------
`dev_ant_body.xml` already satisfies every structural invariant
`khrylib/robot/xml_robot.py` imposes, and it does so by accident of CompetEvo's
own naming rather than by anyone's design:

* Bodies are already named `0`, `k`, `1k`, `11k` -- which is EXACTLY what
  `Body.reindex()` generates (`str(child_index+1) + parent_name`, with the
  root's name elided). `sync_node()` therefore renames nothing.
* Joints are already `<body_name>_joint`, which is what `Joint.sync_node()`
  writes, and motors are already named after their joint.
* Every body carries at most one hinge and exactly one capsule (the root, one
  sphere), which is what `Body.get_params`/`action_to_control` assume.
* Every joint sits at its body's origin -- `Joint.__init__` ASSERTS this
  (`assert np.all(self.pos == body.pos)`).
* Each capsule's far end coincides with its single child's origin, so
  `Body.init()`'s `bone_end = mean(child bone_starts)` lands exactly on the
  capsule end and `sync_geom()` is a no-op. This is the one that would have
  silently deformed the robot had it not held: `Robot` REDRAWS every capsule
  from `bone_start`/`bone_end` on the first `set_params`, so a capsule that
  does not already end at its child's origin gets moved.

So what the converter actually does is four things, none of them a
restructuring of the creature:

1. **Give it a `<worldbody>`.** `dev_ant_body.xml` is a fragment: its
   `<worldbody>` is COMMENTED OUT, because CompetEvo's scene builder splices the
   `<body>` into a merged arena. `Robot.load_from_xml` does
   `tree.getroot().find('worldbody').find('body')` and would crash on it, and
   MuJoCo will not compile it either. The shell (floor, light, skybox, tracking
   camera) is taken from Transform2Act's own `assets/mujoco_envs/ant.xml`,
   because the TASK is theirs.

2. **Set `conaffinity="0"` on the geom default.** `dev_ant_body.xml` says
   `conaffinity="1"`, but that value is never compiled: `scene.py`'s
   `_dev_agent_default_xml` overrides it per agent with
   `conaffinity=i, contype=1-i`, so agent 0 -- the creature D1/D2 actually
   simulate -- has `contype=1, conaffinity=0`, i.e. self-collision OFF and floor
   collision ON. That is also exactly what Transform2Act's ant uses. Emitting
   the asset's literal `1` would have given our ant self-colliding legs that
   D1/D2's ant does not have. `gate_competevo_ant.py` phase A2 checks the
   converted geoms against the REAL `dev_run_to_goal_xml()` scene rather than
   against the asset, which is what caught this.

3. **Rewrite the root's placement.** Root `pos`/`euler` in `dev_ant_body.xml`
   are placeholders: `_dev_ant_body_xml(agent_id, pos, euler)` overwrites both
   with the registered init pose (`INIT_POS`/`INIT_EULER`). They are not part of
   the creature, so they are a CLI knob here, defaulting to Transform2Act's
   single-agent locomotion placement `0 0 0.75`, `0 0 0`.

4. **Canonicalise names anyway.** Steps 1-3 are all this particular asset
   needs, but the renamer is implemented generically (breadth-first over the
   body tree, T2A's `reindex` rule, joints and motors carried along) so the
   converter is not a bet on CompetEvo's names happening to match. On
   `dev_ant_body.xml` it is a no-op, and `--require-name-noop` asserts that.

--------------------------------------------------------------------------
Kept in LOCAL coordinates, deliberately
--------------------------------------------------------------------------
`xml_robot.py` reads `compiler/@coordinate` and supports both, but every XML
Transform2Act ships is `global`, and MuJoCo removed global coordinates in
2.3.3 -- which is why `xml_global_to_local.py` had to exist for the batched
port. Our source is already `local` and `Robot` never rewrites the attribute,
so every design descended from this ant compiles under modern MuJoCo directly,
with no conversion step and no `assert_no_rotation` landmine from the root's
`euler`.

--------------------------------------------------------------------------
Usage
--------------------------------------------------------------------------
    PYTHONPATH=. .venv/bin/python -m rower_soccer.t2a_port.competevo_to_t2a \
        --out /workspace/Transform2Act/assets/mujoco_envs/ant_competevo.xml

    # the byte-for-byte-comparable variant the gate uses: our shell, but the
    # source's own root placement, so a per-field model diff is elementwise.
    ... --root-pos -1 0 0.75 --root-euler 0 0 180
"""

import argparse
import os
import xml.etree.ElementTree as ET

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "competevo_port", "assets", "dev_ant_body.xml")

# Transform2Act's locomotion placement for a single ant. Their own ant.xml puts
# the root at the origin and `AntEnv.reset_state` forces qpos[2]=0.4 anyway
# (`env_specs.init_height`), so z here only sets `init_qpos`.
DEFAULT_ROOT_POS = (0.0, 0.0, 0.75)
DEFAULT_ROOT_EULER = (0.0, 0.0, 0.0)

# Verbatim from `Transform2Act/assets/mujoco_envs/ant.xml`. The task is theirs,
# so the arena is theirs: same floor size/material, same light, same skybox,
# same tracking camera. Only the robot comes from us.
_FLOOR = ('<geom conaffinity="1" condim="3" name="floor" pos="0 0 0"'
          ' rgba="0.8 0.9 0.8 1" size="200 200 .125" type="plane"'
          ' material="grid_new"/>')
_LIGHT = ('<light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3"'
          ' directional="true" exponent="1" pos="0 0 1.3"'
          ' specular=".1 .1 .1"/>')
_CAMERA = ('<camera name="track" mode="trackcom" pos="0 -3 0.3"'
           ' xyaxes="1 0 0 0 0 1"/>')

_HEADER = """<mujoco model="{model}">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <option integrator="RK4" timestep="0.01"/>
  <default>
    <joint armature="1" damping="1" limited="true"/>
    <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba="0.98 0.87 0.67 1" material="geom"/>
  </default>
  <visual>
    <headlight ambient=".1 .1 .1" diffuse=".6 .6 .6" specular="0.3 0.3 0.3"/>
    <map znear=".01"/>
    <quality shadowsize="16384"/>
  </visual>
  <worldbody>
    {light}
    {floor}
"""

_FOOTER = """  </worldbody>
  <actuator>
{motors}
  </actuator>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1=".4 .5 .6" rgb2="0 0 0" width="100" height="100"/>
    <texture builtin="flat" height="1278" mark="cross" markrgb="1 1 1" name="texgeom" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
    <texture builtin="checker" height="100" name="texplane" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d" width="100"/>
    <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
    <material name="geom" texture="texgeom" texuniform="true"/>
    <texture name="grid_new" type="2d" builtin="checker" rgb1=".1 .3 .2" rgb2=".2 .4 .3" width="1000" height="1000" mark="none" markrgb=".8 .6 .4"/>
    <material name="grid_new" texture="grid_new" texrepeat="0.2 0.2" texuniform="true" reflectance=".2"/>
  </asset>
</mujoco>
"""


class Lossy(Exception):
    """The source cannot be expressed in `Robot`'s dialect without changing the
    creature. Raised, never worked around -- a converter that quietly
    approximates is the failure mode this whole task exists to avoid."""


def _vec(s):
    return [float(v) for v in s.split()]


def _fmt(vals):
    return " ".join(f"{v:g}" for v in vals)


# ---------------------------------------------------------------------------
# parsing
# ---------------------------------------------------------------------------

def find_root_body(root):
    """The robot's root `<body>`, whether or not the file has a `<worldbody>`.

    `dev_ant_body.xml` has none -- the element is commented out, because
    CompetEvo splices the subtree into a merged arena. Anything that assumed
    `worldbody` would simply not find the robot.
    """
    wb = root.find("worldbody")
    if wb is not None:
        bodies = wb.findall("body")
    else:
        bodies = root.findall("body")
    if len(bodies) != 1:
        raise Lossy(f"expected exactly one root <body>, found {len(bodies)}")
    return bodies[0]


def _hinges(body):
    return [j for j in body.findall("joint")
            if j.get("type", "hinge") == "hinge"]


def _frees(body):
    return ([j for j in body.findall("joint") if j.get("type") == "free"]
            + body.findall("freejoint"))


def _managed_geoms(body):
    """The geoms `Robot` will own: capsules first, then spheres -- its own
    order (`Body.__init__`), which decides parameter layout."""
    return ([g for g in body.findall("geom") if g.get("type") == "capsule"]
            + [g for g in body.findall("geom") if g.get("type") == "sphere"])


# ---------------------------------------------------------------------------
# validation -- every invariant `Robot` needs, checked BEFORE emitting
# ---------------------------------------------------------------------------

def validate(body, parent=None, depth=0, problems=None):
    """Assert the source satisfies what `xml_robot.Robot` requires.

    Each check names the line in their code that would otherwise fail, or --
    worse -- silently succeed on a different robot.
    """
    problems = [] if problems is None else problems
    name = body.get("name", "<unnamed>")
    kids = body.findall("body")

    if "pos" not in body.attrib:
        problems.append(f"{name}: no pos (Body.__init__ requires it)")
    for attr in ("quat", "axisangle", "xyaxes", "zaxis"):
        if attr in body.attrib:
            problems.append(
                f"{name}: carries {attr}; Robot never reads or writes body "
                "orientation, so a rotated non-root frame would make its "
                "bone arithmetic wrong without failing")
    if depth > 0 and "euler" in body.attrib:
        problems.append(f"{name}: non-root body carries euler (same reason)")

    hinges, frees = _hinges(body), _frees(body)
    if depth == 0:
        if len(frees) != 1:
            problems.append(f"{name}: root needs exactly one free joint, "
                            f"has {len(frees)}")
        if hinges:
            problems.append(f"{name}: root carries {len(hinges)} hinge(s)")
    else:
        if frees:
            problems.append(f"{name}: non-root body carries a free joint")
        if len(hinges) > 1:
            # AntEnv.control_action_dim == 1 and action_to_control writes ONE
            # scalar per body via get_actuator_name(), which returns the first
            # actuated joint. A second hinge would be silently unactuated.
            problems.append(f"{name}: {len(hinges)} hinges; Robot's control "
                            "head emits one scalar per body")

    for j in body.findall("joint") + body.findall("freejoint"):
        p = _vec(j.get("pos", "0 0 0"))
        if any(abs(v) > 1e-12 for v in p):
            # Joint.__init__: assert(np.all(self.pos == body.pos))
            problems.append(f"{name}: joint {j.get('name')} is at {p}, not the "
                            "body origin; Joint.__init__ asserts on this")

    geoms = body.findall("geom")
    managed = _managed_geoms(body)
    if len(geoms) != len(managed):
        kinds = sorted({g.get("type") for g in geoms} - {"capsule", "sphere"})
        problems.append(f"{name}: geom type(s) {kinds} are invisible to Robot "
                        "-- it would never resize or move them")
    if len(managed) != 1:
        problems.append(f"{name}: {len(managed)} capsule/sphere geoms; Robot's "
                        "bone model is one geom per body")

    # The check that matters most: Robot REDRAWS capsules from bone_start to
    # bone_end on every set_params, and bone_end is the MEAN of the children's
    # origins. If the capsule does not already end there, the first attribute
    # transform deforms the robot -- and nothing raises.
    if managed and managed[0].get("type") == "capsule":
        ft = _vec(managed[0].get("fromto"))
        start, end = ft[:3], ft[3:]
        if any(abs(v) > 1e-12 for v in start):
            # not fatal (Robot models it as `ext_start`) but worth stating
            problems.append(f"{name}: capsule starts at {start}, not the body "
                            "origin (Robot models this as ext_start; check "
                            "it is inside geom_params.ext_start bounds)")
        if kids:
            kid_pos = [_vec(k.get("pos", "0 0 0")) for k in kids]
            mean = [sum(p[i] for p in kid_pos) / len(kid_pos) for i in range(3)]
            if any(abs(mean[i] - end[i]) > 1e-9 for i in range(3)):
                problems.append(
                    f"{name}: capsule ends at {end} but its children's mean "
                    f"origin is {mean}; Body.init/sync_geom would move the "
                    "capsule to the latter on the first attribute transform")

    for k in kids:
        validate(k, body, depth + 1, problems)
    return problems


# ---------------------------------------------------------------------------
# name canonicalisation
# ---------------------------------------------------------------------------

def canonical_names(body):
    """`{old_body_name: T2A_name}` following `Body.reindex()` exactly.

    root -> '0'; otherwise `str(index_among_siblings + 1) + parent_name`, with
    the parent's name elided when the parent is the root.
    """
    out = {}

    def walk(node, name):
        out[node.get("name")] = name
        for i, kid in enumerate(node.findall("body")):
            pname = "" if name == "0" else name
            walk(kid, f"{i + 1}{pname}")

    walk(body, "0")
    return out


def rename(body, mapping, motors):
    """Apply the mapping to bodies, their joints (`<name>_joint`, which is what
    `Joint.sync_node` writes) and the motors that drive them."""
    joint_map = {}

    def walk(node):
        new = mapping[node.get("name")]
        node.set("name", new)
        for j in list(node.findall("joint")) + list(node.findall("freejoint")):
            old_j = j.get("name")
            if old_j is not None:
                joint_map[old_j] = f"{new}_joint"
            j.set("name", f"{new}_joint")
        for kid in node.findall("body"):
            walk(kid)

    walk(body)
    for m in motors:
        m.set("joint", joint_map.get(m.get("joint"), m.get("joint")))
        m.set("name", m.get("joint"))
    return joint_map


def check_motors(body, motors):
    """Every motor drives a joint that exists, and no joint is driven twice.

    Motor ORDER is left exactly as the source has it, deliberately. It is not
    load-bearing for Transform2Act -- `AntEnv.action_to_control` looks each
    actuator up by name, and `Robot.add_child_to_body` appends new motors at the
    end of `<actuator>` anyway, so actuator order stops being body order after
    the first skeleton transform. But keeping it means `data.ctrl` indices
    coincide between the source model and the converted one, so the physics
    gate replays one recorded action array on both with no permutation, and a
    permutation is exactly the kind of silent index bug this gate exists to
    catch.
    """
    joints = set()

    def walk(node):
        for j in list(node.findall("joint")) + list(node.findall("freejoint")):
            joints.add(j.get("name"))
        for kid in node.findall("body"):
            walk(kid)

    walk(body)
    driven = [m.get("joint") for m in motors]
    missing = [j for j in driven if j not in joints]
    if missing:
        raise Lossy(f"motors drive joints not in the body tree: {missing}")
    dup = [j for j in set(driven) if driven.count(j) > 1]
    if dup:
        raise Lossy(f"joints driven by more than one motor: {dup}")
    return list(motors)


# ---------------------------------------------------------------------------
# emit
# ---------------------------------------------------------------------------

def _indent(elem, level):
    pad = "  " * level
    out = []
    attrs = " ".join(f'{k}="{v}"' for k, v in elem.attrib.items())
    kids = list(elem)
    if not kids:
        out.append(f"{pad}<{elem.tag} {attrs}/>")
        return out
    out.append(f"{pad}<{elem.tag} {attrs}>")
    for k in kids:
        out += _indent(k, level + 1)
    out.append(f"{pad}</{elem.tag}>")
    return out


def convert(src_xml_str, root_pos=DEFAULT_ROOT_POS,
            root_euler=DEFAULT_ROOT_EULER, model="ant_competevo",
            require_name_noop=False, add_camera=True):
    root = ET.fromstring(src_xml_str)
    body = find_root_body(root)
    motors = root.find("actuator").findall("motor")

    problems = validate(body)
    if problems:
        raise Lossy("source is not expressible in Robot's dialect:\n  - "
                    + "\n  - ".join(problems))

    mapping = canonical_names(body)
    if require_name_noop:
        bad = {k: v for k, v in mapping.items() if k != v}
        if bad:
            raise Lossy(f"--require-name-noop but names would change: {bad}")
    rename(body, mapping, motors)
    motors = check_motors(body, motors)

    body.set("pos", _fmt(root_pos))
    body.set("euler", _fmt(root_euler))
    if add_camera:
        body.insert(0, ET.fromstring(_CAMERA))

    head = _HEADER.format(model=model, light=_LIGHT, floor=_FLOOR)
    body_lines = "\n".join(_indent(body, 2))
    motor_lines = "\n".join("    " + ET.tostring(m, encoding="unicode").strip()
                            for m in motors)
    return head + body_lines + "\n" + _FOOTER.format(motors=motor_lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--src", default=SRC)
    p.add_argument("--out", required=True)
    p.add_argument("--root-pos", nargs=3, type=float, default=DEFAULT_ROOT_POS)
    p.add_argument("--root-euler", nargs=3, type=float,
                   default=DEFAULT_ROOT_EULER)
    p.add_argument("--model", default="ant_competevo")
    p.add_argument("--no-camera", action="store_true")
    p.add_argument("--require-name-noop", action="store_true",
                   help="fail if canonicalisation would rename anything -- "
                        "asserts the source is already in T2A's naming")
    a = p.parse_args()
    with open(a.src) as f:
        out = convert(f.read(), tuple(a.root_pos), tuple(a.root_euler),
                      a.model, a.require_name_noop, not a.no_camera)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(out)
    print(f"{a.src} -> {a.out}  ({len(out)} bytes)")


if __name__ == "__main__":
    main()
