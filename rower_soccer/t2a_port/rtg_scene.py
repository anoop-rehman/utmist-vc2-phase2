"""D3 M3 E2: emit the 1v1 run-to-goal scene for the Transform2Act stack.

The scene is CompetEvo's `run-to-goal-ants-v0` geometry with OUR DeepMind ant
on both sides, written in the XML dialect Transform2Act's `Robot` parses.

Why a generator and not a checked-in file: the merged scene is derived from
`ant_competevo.xml`, which is itself generated and gated
(`competevo_to_t2a.py` + `gate_competevo_ant.py`), so a checked-in merged file
would be a second copy that can silently drift from the gated one.

The opponent's frozen stance is NOT baked into this file. It is measured at
env construction by settling the opponent under gravity at zero torque
(`RunToGoalEnv._settle_opponent`), because MuJoCo's joint `ref` shifts the
joint COORDINATE ORIGIN rather than the pose -- baking the settled angles as
`ref` moves the limit range with them and collapses the creature (measured:
torso z 0.535 -> 0.260). `settle()` below is kept as the reporting utility
that prints the same numbers the env computes.

    cd /workspace/Transform2Act && source env-gpu.sh
    .venv-gpu/bin/python .../t2a_port/rtg_scene.py \
        --src assets/mujoco_envs/ant_competevo.xml \
        --out assets/mujoco_envs/rtg_ant.xml

What is taken from CompetEvo (`rower_soccer/competevo_port/scene.py`):

  * `INIT_POS = ((-1,0,0.75), (1,0,0.75))`, `INIT_EULER = ((0,0,0),(0,0,180))`
    -- agent 0 starts at x=-1 facing +x, agent 1 at x=+1 facing -x;
  * goal lines at x = +/-4 (`GOAL_X`), as real colliding cylinders, exactly as
    `world_body.xml` has them;
  * `<option integrator="RK4" timestep="0.003" solver="PGS" iterations="1000"/>`
    -- their world options verbatim. mujoco-py 2.1 implements PGS, so unlike
    the mujoco_warp port (`scene.py`'s docstring) nothing has to be swapped;
  * the dev merger's collision trick (`evo_utils.create_multiagent_xml_str`):
    agent 0's geoms get `contype=1 conaffinity=0`, agent 1's `contype=0
    conaffinity=1`, so neither ant self-collides and the two DO collide with
    each other. `ant_competevo.xml`'s default is already agent 0's setting.

What is NOT taken from CompetEvo: the observation (Transform2Act's is a
per-body matrix, not a flat 31-vector) and the opponent's policy (E2's
opponent is scripted -- see `design_opt/envs/run_to_goal.py`).

The FIRST `<body>` under `<worldbody>` is ours, because
`Robot.load_from_xml` (`khrylib/robot/xml_robot.py:511`) parses exactly that
one and nothing else. The opponent is a sibling after it and is therefore
invisible to `Robot`: it can never be mutated, indexed or actuated by the
design stages. That is the whole reason the opponent is expressed as XML
rather than as a second `Robot`.
"""

import argparse
import copy
import os
import sys

import numpy as np
from lxml import etree
from lxml.etree import XMLParser, parse

OPP_PREFIX = "opp_"
GOAL_X = 4.0
INIT_POS = ((-1.0, 0.0, 0.75), (1.0, 0.0, 0.75))
INIT_EULER = ((0.0, 0.0, 0.0), (0.0, 0.0, 180.0))
# world_body.xml's option line, verbatim.
OPTION = dict(integrator="RK4", timestep="0.003", solver="PGS",
              iterations="1000")
# `world_body.xml`'s two goal cylinders: fromto z 0 -> 1, radius 0.1.
GOAL_GEOMS = (("rightgoal", +GOAL_X), ("leftgoal", -GOAL_X))


def _prefix_names(node, prefix):
    """Rename every name/joint reference in a subtree. Body names are what
    `Robot.reindex` writes, so prefixing keeps the opponent's names out of
    the namespace `AntEnv.action_to_control` and `get_single_body_qposaddr`
    look ours up in."""
    for el in node.iter():
        for attr in ("name", "joint", "site", "class"):
            if attr in el.attrib and el.tag != "default":
                el.attrib[attr] = prefix + el.attrib[attr]


def build(src, opponent_src=None):
    """Return the merged tree.

    `opponent_src` (D3 M3 E4): an XML whose first `<body>` becomes the
    OPPONENT instead of a clone of ours. This is what lets the opponent be
    another lineage's evolved body. Default None reproduces E2/E3 exactly --
    the opponent is a deep copy of our own root body, byte for byte as before.

    The opponent's actuators come from `opponent_src`'s own `<actuator>` list,
    not ours: an evolved body has its own motor set and reusing ours would
    silently drop or invent motors."""
    parser = XMLParser(remove_blank_text=True)
    tree = parse(src, parser)
    root = tree.getroot()
    root.attrib["model"] = "rtg_ant"

    opt = root.find("option")
    if opt is None:
        opt = etree.SubElement(root, "option")
    for k, v in OPTION.items():
        opt.attrib[k] = v

    world = root.find("worldbody")
    ours = world.find("body")
    assert ours is not None and ours.attrib["name"] == "0", (
        "the first <body> under <worldbody> must be our ant's root -- "
        "Robot.load_from_xml parses exactly that node")
    ours.attrib["pos"] = "%g %g %g" % INIT_POS[0]
    ours.attrib["euler"] = "%g %g %g" % INIT_EULER[0]

    # ---- the opponent: agent 1, placed and yawed ------------------------
    if opponent_src is None:
        opp = copy.deepcopy(ours)
        opp_motors = [copy.deepcopy(m) for m in root.find("actuator").findall("motor")]
    else:
        otree = parse(opponent_src, XMLParser(remove_blank_text=True))
        oroot = otree.getroot()
        obody = oroot.find("worldbody").find("body")
        assert obody is not None and obody.attrib["name"] == "0", (
            "opponent_src's first <body> under <worldbody> must be its root, "
            "named '0' -- same contract as ours")
        opp = copy.deepcopy(obody)
        oact = oroot.find("actuator")
        # An evolved body is dumped as a MERGED scene, so its <actuator> list
        # already contains that scene's own opp_* motors. Take only the motors
        # belonging to its FIRST body; prefixing the stale opp_* ones would
        # give the new opponent a second, phantom motor set.
        opp_motors = ([copy.deepcopy(m) for m in oact.findall("motor")
                       if not m.attrib.get("joint", "").startswith(OPP_PREFIX)]
                      if oact is not None else [])
        assert opp_motors, "opponent_src has no non-opponent motors"
    for cam in opp.findall("camera"):
        opp.remove(cam)
    _prefix_names(opp, OPP_PREFIX)
    opp.attrib["pos"] = "%g %g %g" % INIT_POS[1]
    opp.attrib["euler"] = "%g %g %g" % INIT_EULER[1]
    # agent 1's collision mask (contype 0 / conaffinity 1): no self-collision,
    # collides with agent 0 and with the floor.
    for g in opp.iter("geom"):
        g.attrib["contype"] = "0"
        g.attrib["conaffinity"] = "1"
        g.attrib["rgba"] = "0.60 0.68 0.98 1"
    world.append(opp)

    # ---- goal lines, real colliding geometry as in world_body.xml --------
    for name, x in GOAL_GEOMS:
        etree.SubElement(world, "geom", dict(
            name=name, type="cylinder", fromto="%g 0 0 %g 0 1" % (x, x),
            size="0.1", contype="1", conaffinity="1", condim="3",
            rgba="0.9 0.2 0.2 1" if x > 0 else "0.2 0.4 0.9 1"))
    # `ant.xml` asks for a 16384px shadow map, which makes an osmesa frame of
    # this 27-body scene cost ~0.5 s. Nothing here needs shadow detail.
    q = root.find("visual/quality")
    if q is not None:
        q.attrib["shadowsize"] = "1024"
    # a fixed camera that sees the whole 8 m pitch, for the E2 videos.
    etree.SubElement(world, "camera", dict(
        name="pitch", mode="fixed", pos="0 -11 5.5",
        xyaxes="1 0 0 0 0.45 0.89"))

    # ---- the opponent's actuators ---------------------------------------
    act = root.find("actuator")
    for m2 in opp_motors:
        m2.attrib["joint"] = OPP_PREFIX + m2.attrib["joint"]
        m2.attrib["name"] = OPP_PREFIX + m2.attrib["name"]
        act.append(m2)
    return tree


def settle(xml_str, seconds=3.0):
    """Drop the opponent at zero torque and read off where it comes to rest.

    Run in mujoco-py 2.1, the engine E2 trains in, so the baked stance is a
    rest state of THAT engine and the opponent does not twitch on the first
    step of an episode."""
    import mujoco_py
    m = mujoco_py.load_model_from_xml(xml_str)
    sim = mujoco_py.MjSim(m)
    n = int(seconds / m.opt.timestep)
    for _ in range(n):
        sim.step()
    d = sim.data
    j0 = m.joint_name2id(OPP_PREFIX + "0_joint")
    adr = m.jnt_qposadr[j0]
    root_z = float(d.qpos[adr + 2])
    refs = {}
    for jid in range(m.njnt):
        nm = m.joint_id2name(jid)
        if nm and nm.startswith(OPP_PREFIX) and m.jnt_type[jid] == 3:  # hinge
            refs[nm] = float(np.rad2deg(d.qpos[m.jnt_qposadr[jid]]))
    return root_z, refs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="assets/mujoco_envs/ant_competevo.xml")
    p.add_argument("--out", default="assets/mujoco_envs/rtg_ant.xml")
    p.add_argument("--opponent-src", default=None,
                   help="D3 M3 E4: XML whose first <body> becomes the "
                        "opponent, instead of a clone of ours")
    p.add_argument("--report-settle", action="store_true",
                   help="also print the stance the env will measure")
    a = p.parse_args()

    tree = build(a.src, a.opponent_src)
    tree.write(a.out, pretty_print=True)
    print(f"[rtg_scene] -> {a.out}")
    if a.report_settle:
        z, refs = settle(etree.tostring(tree).decode("utf-8"))
        print(f"  opponent settles at root z {z:.6f}")
        for k in sorted(refs):
            print(f"    {k:16s} {refs[k]:+9.4f} deg")


if __name__ == "__main__":
    main()
