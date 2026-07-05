"""Phenotype -> MuJoCo XML emitter, matching the conventions of the original
notebook pipeline (unity2mujoco.ipynb) and the checked-in rower XML:

- Unity Y-up -> MuJoCo Z-up: sizes (x,z,y); positions 2*(child-parent) with
  the bottom-origin correction on z; joint pos = -anchor (swapped); joint
  axis swapped with signs dropped (faithful quirk of the original pipeline).
- Root body placed at z = its Unity y half-extent (rests on ground).
- Gears: the rower XML's hand-tuned gears fit gear = 2.5 * subtree_mass^1.67
  at root level; we generate that automatically (density 50, box volumes),
  scaled by --gear-scale. Gears are a free design parameter here (we train
  our own controllers), so tune for controllability, not realism.
"""

import numpy as np

DENSITY = 50.0


def _fmt(v, nd=7):
    return " ".join(f"{x:.{nd}g}" for x in v)


def _mj_size(p):
    return np.array([p.half_size[0], p.half_size[2], p.half_size[1]])


def _mj_pos(p, parent):
    if parent is None:
        return np.array([0.0, 0.0, p.half_size[1]])
    d = p.world_pos - parent.world_pos
    return np.array([2 * d[0], 2 * d[2], 2 * d[1] + p.half_size[1] - parent.half_size[1]])


def _mj_joint_pos(p):
    a = p.joint_anchor
    return np.array([-a[0], -a[2], -a[1]])


def _mj_axis(p):
    a = p.joint_axis
    return np.array([1 if a[0] else 0, 1 if a[2] else 0, 1 if a[1] else 0])


def _mass(p):
    return 8.0 * float(np.prod(p.half_size)) * DENSITY


def _subtree_mass(p):
    return _mass(p) + sum(_subtree_mass(c) for c in p.children)


def _gear(p, scale, depth):
    # Empirical fit to the hand-tuned rower gears: root-level motors follow
    # 2.5 * subtree_mass^1.67 within 2%; deeper motors carry a uniform 2.36x.
    return 2.5 * _subtree_mass(p) ** 1.67 * (2.36 if depth >= 2 else 1.0) * scale


ROOT_EXTRAS = """
      <camera name="floating" pos="-2 0 1" xyaxes="0 -1 0 .5 0 1" mode="trackcom" fovy="90" />
      <camera name="egocentric" pos=".25 0 .11" xyaxes="0 -1 0 0 0 1" fovy="90" />
"""


def emit_xml(phenotypes, model_name, gear_scale=1.0, joint_range=(-75, 75)):
    by_uid = {p.uid: p for p in phenotypes}
    root = phenotypes[0]

    def body_xml(p, indent):
        pad = " " * indent
        parent = by_uid.get(p.parent_uid) if p.parent_uid is not None else None
        lines = [f'{pad}<body name="seg{p.uid}" pos="{_fmt(_mj_pos(p, parent))}">']
        if parent is None:
            lines.append(ROOT_EXTRAS.rstrip())
        if p.joint_type == "hinge":
            lines.append(
                f'{pad}  <joint name="seg{p.parent_uid}_to_{p.uid}" range="{joint_range[0]} {joint_range[1]}" '
                f'type="hinge" axis="{_fmt(_mj_axis(p))}" pos="{_fmt(_mj_joint_pos(p))} "/>')
        euler = np.array([p.world_euler[0], p.world_euler[1], p.world_euler[2]])
        size = _mj_size(p)
        lines.append(
            f'{pad}  <geom name="seg{p.uid}_geom" type="box" pos="0 0 0" size="{_fmt(size)}" euler="{_fmt(euler)}" />')
        lines.append(
            f'{pad}  <site name="seg{p.uid}_site" type="box" pos="0 0 0" size="{_fmt(size * 1.1)}" '
            f'zaxis="0 0 1" rgba="1 1 0 0.15" />')
        for c in p.children:
            lines.append("")
            lines.append(body_xml(c, indent + 2))
        lines.append(f"{pad}</body>")
        return "\n".join(lines)

    def depth_of(p):
        d = 0
        while p.parent_uid is not None:
            p = by_uid[p.parent_uid]
            d += 1
        return d

    motors = []
    for p in phenotypes:
        if p.joint_type == "hinge":
            motors.append(
                f'    <motor name="motor{p.parent_uid}_to_{p.uid}" '
                f'joint="seg{p.parent_uid}_to_{p.uid}" gear="{_gear(p, gear_scale, depth_of(p)):.6g}" />')

    touches = [f'    <touch name="seg{p.uid}_touch" site="seg{p.uid}_site" />' for p in phenotypes]

    excludes = []
    uids = [p.uid for p in phenotypes]
    for i, a in enumerate(uids):
        for b in uids[i + 1:]:
            excludes.append(f'    <exclude body1="seg{a}" body2="seg{b}" />')

    return f"""<mujoco model="{model_name}">
    <compiler angle="degree" />
    <default>
        <motor ctrlrange="-1.0 1.0" ctrllimited="true" gear="16000" />
        <geom friction="1 0.5 0.5" solref=".02 1" solimp="0 .8 .01" material="self" density="{DENSITY:g}" />
        <joint limited="true" armature="1" damping="1" stiffness="1" solreflimit=".04 1"
            solimplimit="0 .8 .03" />
    </default>
    <asset>
        <material name="self" rgba=".8 .6 .4 1" />
    </asset>

    <worldbody>
{body_xml(root, 8)}
    </worldbody>

    <actuator>
{chr(10).join(motors)}
    </actuator>

    <sensor>
{chr(10).join(touches)}
        <velocimeter name="torso_vel" site="seg0_site" />
        <gyro name="torso_gyro" site="seg0_site" />
        <accelerometer name="torso_accel" site="seg0_site" />
    </sensor>

    <contact>
{chr(10).join(excludes)}
    </contact>
</mujoco>
"""


def main():
    import argparse

    from rower_soccer.tools.creature_format import load_creature
    from rower_soccer.tools.genotype_expand import expand

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help=".creature file")
    parser.add_argument("--out", required=True, help="output XML path")
    parser.add_argument("--name", default=None, help="model name (default: genotype name)")
    parser.add_argument("--gear-scale", type=float, default=1.0)
    args = parser.parse_args()

    genotype = load_creature(args.input)
    phenotypes = expand(genotype)
    xml = emit_xml(phenotypes, args.name or genotype["name"], args.gear_scale)
    with open(args.out, "w") as f:
        f.write(xml)
    print(f"wrote {args.out}: {len(phenotypes)} bodies, "
          f"{sum(1 for p in phenotypes if p.joint_type)} joints, "
          f"total mass {sum(_mass(p) for p in phenotypes):,.0f} kg")


if __name__ == "__main__":
    main()
