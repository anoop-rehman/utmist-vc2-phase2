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

Unity emits creatures at whatever scale the evolution ran at, which for the
3-segment worm is ~10 m and ~3980 kg. Against DeepMind's soccer env, whose ball
is fixed at 0.35 m / 0.045 kg, that is a ball:player mass ratio of 1:88,485
versus dm_soccer's own 1:489 -- a freight train kicking a balloon. Measured
consequence: the worm ejects the ball 75 m out of a +/-27 m drill. The ball,
pitch and goal are DeepMind's and stay put; the creature is what moves. Two
independent levers do that:

--mass-scale k  (PREFERRED)
    Scales mass and every force-like parameter by k, leaving all LENGTHS alone:

        density ~ k   gear ~ k   armature ~ k   damping ~ k   stiffness ~ k

    This leaves the creature's own locomotion EXACTLY invariant. F = ma with
    both force and mass scaled by k gives identical accelerations; gravity is an
    acceleration and does not care about mass; ground friction scales the same
    way (mu*N ~ m, and the force to accelerate ~ m). Same gait, same speed, same
    everything -- so drill constants need no recalibration and a trained policy
    transfers nearly intact (only the touch sensors change, since they read
    contact force, which scales with mass).

    What DOES change is the momentum ratio against the ball, whose mass is fixed
    by DeepMind. That is the one quantity that is broken, so this lever fixes it
    at zero cost to the creature. Choose k by measuring with probe_ball.py, not
    by arithmetic: the CONTACTING SEGMENT's mass governs the collision, and the
    worm's mass is spread over three segments of very different size, so hitting
    BoxHead's 22 kg total is not automatically the right target.

--length-scale s
    Froude-similar geometric rescale (the creature actually shrinks). With
    gravity and density fixed, length ~ s implies:

        mass ~ s^3    torque ~ s^4    armature ~ s^5
        (Froude time ~ sqrt(s), so damping [N.m.s/rad] ~ s^4.5, stiffness ~ s^4)

    s=0.1768 puts the worm at 1.76 m / 22.0 kg -- BoxHead's mass exactly. But it
    changes the creature: achievable speed drops 2.83 -> 1.64 m/s, so every drill
    constant must be recalibrated with probe_speed.py (the existing 2.0 m/s
    follow target cap becomes physically uncatchable, which makes the drill
    unlearnable with no error to tell you so). Prefer --mass-scale.

solref/solimp and the sim timestep are deliberately NOT scaled by either lever.
solref is a time constant, and MuJoCo normalises contact stiffness by the
effective mass, so contact behaviour is already mass-invariant.
"""

import numpy as np

DENSITY = 50.0

# Froude-similitude exponents on the length scale s (gravity, density fixed).
_E_TORQUE = 4.0      # m*g*L
_E_ARMATURE = 5.0    # m*L^2
_E_DAMPING = 4.5     # torque * time,  time ~ sqrt(s)
_E_STIFFNESS = 4.0   # torque / rad


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


ROOT_EXTRAS_TEMPLATE = """
      <camera name="floating" pos="-{d:.2f} 0 {h:.2f}" xyaxes="0 -1 0 .5 0 1" mode="trackcom" fovy="70" />
      <camera name="egocentric" pos="{ego:.2f} 0 .11" xyaxes="0 -1 0 0 0 1" fovy="90" />
"""


def _root_extras(phenotypes, s):
    # camera distance scaled to creature size so it never sits inside geometry
    max_half = max(float(np.max(p.half_size)) for p in phenotypes) * s
    return ROOT_EXTRAS_TEMPLATE.format(d=5.0 * max_half, h=2.5 * max_half,
                                       ego=max_half * 1.1)


def emit_xml(phenotypes, model_name, gear_scale=1.0, joint_range=(-75, 75),
             length_scale=1.0, mass_scale=1.0):
    by_uid = {p.uid: p for p in phenotypes}
    root = phenotypes[0]
    s = length_scale
    k = mass_scale

    def body_xml(p, indent):
        pad = " " * indent
        parent = by_uid.get(p.parent_uid) if p.parent_uid is not None else None
        lines = [f'{pad}<body name="seg{p.uid}" pos="{_fmt(_mj_pos(p, parent) * s)}">']
        if parent is None:
            lines.append(_root_extras(phenotypes, s).rstrip())
        if p.joint_type == "hinge":
            lines.append(
                f'{pad}  <joint name="seg{p.parent_uid}_to_{p.uid}" range="{joint_range[0]} {joint_range[1]}" '
                f'type="hinge" axis="{_fmt(_mj_axis(p))}" pos="{_fmt(_mj_joint_pos(p) * s)} "/>')
        euler = np.array([p.world_euler[0], p.world_euler[1], p.world_euler[2]])
        size = _mj_size(p) * s
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

    # Gears fit the ORIGINAL (unscaled) subtree masses -- that empirical fit is
    # what makes the creature controllable -- then carry the s^4 torque exponent
    # (preserves torque-to-weight, hence relative strength) and the k mass factor
    # (torque must track mass, or the creature gets relatively stronger/weaker and
    # its gait changes -- the whole point of --mass-scale is that it does not).
    torque_s = s ** _E_TORQUE * k
    motors = []
    for p in phenotypes:
        if p.joint_type == "hinge":
            g = _gear(p, gear_scale, depth_of(p)) * torque_s
            motors.append(
                f'    <motor name="motor{p.parent_uid}_to_{p.uid}" '
                f'joint="seg{p.parent_uid}_to_{p.uid}" gear="{g:.6g}" />')

    touches = [f'    <touch name="seg{p.uid}_touch" site="seg{p.uid}_site" />' for p in phenotypes]

    excludes = []
    uids = [p.uid for p in phenotypes]
    for i, a in enumerate(uids):
        for b in uids[i + 1:]:
            excludes.append(f'    <exclude body1="seg{a}" body2="seg{b}" />')

    density = DENSITY * k
    total_mass = sum(_mass(p) for p in phenotypes) * s ** 3 * k
    armature = s ** _E_ARMATURE * k
    damping = s ** _E_DAMPING * k
    stiffness = s ** _E_STIFFNESS * k
    return f"""<mujoco model="{model_name}">
    <!-- GENERATED by rower_soccer/tools/unity2mujoco.py. Do not hand-edit.
         gear_scale={gear_scale:g}  length_scale={s:g}  mass_scale={k:g}
         density={density:g}  total mass {total_mass:.3f} kg

         Regenerate with python -m rower_soccer.tools.unity2mujoco and flags:
           input=(.creature file)  out=(this file)  name={model_name}
           gear-scale={gear_scale:g}  length-scale={s:g}  mass-scale={k:g}

         gear_scale is NOT reconstructible from the XML alone. An earlier worm
         was checked in at gear_scale=0.03 with that fact recorded nowhere, so
         regenerating at the tool default of 1.0 silently produced a creature
         33x stronger, which compiles and runs and looks plausible. Hence this
         header.

         mass_scale multiplies density, gear, armature, damping and stiffness
         together, which leaves the creature's own motion exactly invariant
         (F = ma: force and mass scale alike) while changing its momentum ratio
         against the fixed-mass ball. Do not scale one without the others.

         (Flags are written as name=value, without their leading dashes: a
         double hyphen cannot appear inside an XML comment. lxml, which
         dm_control's mjcf parser uses, rejects the whole file outright, while
         MuJoCo's own parser accepts it. So a comment containing a doubled dash
         loads fine on the Warp path and blows up only on the CPU eval.) -->
    <compiler angle="degree" />
    <default>
        <motor ctrlrange="-1.0 1.0" ctrllimited="true" gear="{16000 * torque_s:.6g}" />
        <geom friction="1 0.5 0.5" solref=".02 1" solimp="0 .8 .01" material="self" density="{density:g}" />
        <joint limited="true" armature="{armature:.6g}" damping="{damping:.6g}" stiffness="{stiffness:.6g}" solreflimit=".04 1"
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
    parser.add_argument("--length-scale", type=float, default=1.0,
                        help="Froude-similar rescale of the whole creature "
                             "(it actually shrinks). Changes the creature, so "
                             "every drill constant must be recalibrated.")
    parser.add_argument("--mass-scale", type=float, default=1.0,
                        help="Scale mass and all force-like params (density, "
                             "gear, armature, damping, stiffness) by k, leaving "
                             "lengths alone. Locomotion is exactly invariant; "
                             "only the momentum ratio against the fixed-mass "
                             "ball changes. Prefer this over --length-scale.")
    args = parser.parse_args()

    genotype = load_creature(args.input)
    phenotypes = expand(genotype)
    xml = emit_xml(phenotypes, args.name or genotype["name"], args.gear_scale,
                   length_scale=args.length_scale, mass_scale=args.mass_scale)
    with open(args.out, "w") as f:
        f.write(xml)
    s, k = args.length_scale, args.mass_scale
    mass = sum(_mass(p) for p in phenotypes) * s ** 3 * k
    print(f"wrote {args.out}: {len(phenotypes)} bodies, "
          f"{sum(1 for p in phenotypes if p.joint_type)} joints, "
          f"length_scale {s:g}, mass_scale {k:g}, total mass {mass:,.3f} kg")


if __name__ == "__main__":
    main()
