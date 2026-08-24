"""Convert a Transform2Act morphology XML from global to local coordinates.

Why this has to exist. `khrylib/robot/xml_robot.py` emits
`<compiler coordinate="global">`, and **MuJoCo removed global coordinates in
2.3.3**:

    ValueError: XML Error: global coordinates no longer supported.
    To convert existing models, load and save them in MuJoCo 2.3.3 or older

So Transform2Act's models cannot be compiled by the modern bindings at all, and
the batched execution env (D3 3d step 3) cannot read them. Their own stack is
fine -- mujoco-py 2.1 against the mujoco210 binary still supports it -- which is
why this never surfaced until something outside their venv tried to load one.

The conversion is a pure translation, and that is a fact about THEIR generator
rather than a general truth: `xml_robot.py` never emits `quat` or `euler`, so
every body frame is axis-aligned with the world and going from global to local
is subtraction. **`assert_no_rotation` enforces that** rather than assuming it --
if they ever emit an orientation this must become a full frame transform, and a
silent wrong answer here would be a wrong robot, not a crash.

Under `coordinate="global"` every `pos` and `fromto` is expressed in the world
frame. Locally:

    body.pos   -= parent_body.global_pos          (root's parent is the world)
    joint.pos  -= own_body.global_pos
    geom.pos, geom.fromto -= own_body.global_pos
    site/camera.pos -= own_body.global_pos

Elements directly under `<worldbody>` are unchanged, the world frame being the
identity.
"""

import copy
import xml.etree.ElementTree as ET

# Attributes that carry a point in space and therefore need shifting. `fromto`
# is two points in one attribute.
_POINT_ATTRS = ("pos",)
_FROMTO_ATTRS = ("fromto",)
_SHIFTABLE_TAGS = ("joint", "freejoint", "geom", "site", "camera", "light")


def _vec(s):
    return [float(v) for v in s.split()]


def _fmt(v):
    return " ".join(f"{x:.9g}" for x in v)


def assert_no_rotation(root):
    """Their generator emits no orientations; this is where that is checked.

    A rotated body would make the subtraction below wrong without making it
    fail, which is the single most dangerous outcome available here.
    """
    bad = []
    for body in root.iter("body"):
        for attr in ("quat", "euler", "axisangle", "xyaxes", "zaxis"):
            if attr in body.attrib:
                bad.append((body.get("name"), attr))
    if bad:
        raise NotImplementedError(
            "global->local here assumes axis-aligned body frames and these "
            f"bodies carry an orientation: {bad}. Implement the full frame "
            "transform before trusting any output.")


def _shift_subtree(body, body_global):
    """Shift a body's own direct children into that body's frame."""
    for child in body:
        if child.tag not in _SHIFTABLE_TAGS:
            continue
        for attr in _POINT_ATTRS:
            if attr in child.attrib:
                p = _vec(child.get(attr))
                child.set(attr, _fmt([p[i] - body_global[i] for i in range(3)]))
        for attr in _FROMTO_ATTRS:
            if attr in child.attrib:
                ft = _vec(child.get(attr))
                child.set(attr, _fmt([ft[i] - body_global[i % 3]
                                      for i in range(6)]))


def _recurse(body, parent_global):
    # Under global coordinates a body's `pos` IS its world position.
    g = _vec(body.get("pos", "0 0 0"))
    _shift_subtree(body, g)
    for child in list(body):
        if child.tag == "body":
            _recurse(child, g)
    body.set("pos", _fmt([g[i] - parent_global[i] for i in range(3)]))


# MuJoCo 2.1 computed a capsule's mass as
#     rho * (pi r^2 * 2h  +  pi r^3)
# i.e. the two hemispherical caps contributed `pi r^3` instead of the correct
# `4/3 pi r^3`. Modern MuJoCo fixed it. Measured on a real Transform2Act hopper,
# `(mass/rho - V_cylinder) / V_sphere` is **exactly 0.7500** under their stack
# and **exactly 1.0000** under ours, for all 8 bodies -- so their capsules are
# 1.3-3.2% lighter than the same XML compiled today.
#
# Scaling each capsule's density by this factor reproduces their mass under a
# modern compiler. It scales the inertia tensor by the same factor, which is
# NOT automatically the old inertia: the caps' contribution is distributed
# differently from the cylinder's. `legacy_capsule_mass` therefore fixes mass
# exactly and inertia only approximately, and the gate reports both.
LEGACY_CAP_FRACTION = 0.75


def _capsule_density_scale(radius, half_len):
    v_cyl = 3.141592653589793 * radius * radius * 2.0 * half_len
    v_sph = 4.0 / 3.0 * 3.141592653589793 * radius ** 3
    return (v_cyl + LEGACY_CAP_FRACTION * v_sph) / (v_cyl + v_sph)


def _apply_legacy_capsule_mass(root, default_density=1000.0):
    n = 0
    for geom in root.iter("geom"):
        if geom.get("type") != "capsule":
            continue
        size = geom.get("size")
        ft = geom.get("fromto")
        if size is None or ft is None:
            continue
        r = float(size.split()[0])
        p = _vec(ft)
        half = 0.5 * sum((p[i] - p[i + 3]) ** 2 for i in range(3)) ** 0.5
        if half <= 0:
            continue
        d = geom.get("density")
        rho = float(d) if d is not None else default_density
        geom.set("density", f"{rho * _capsule_density_scale(r, half):.10g}")
        n += 1
    return n


def convert(xml_str, legacy_capsule_mass=False):
    """Global-coordinate morphology XML -> local-coordinate equivalent.

    `legacy_capsule_mass=True` additionally reproduces MuJoCo 2.1's capsule
    mass, which is what Transform2Act actually trained against.
    """
    root = ET.fromstring(xml_str)
    comp = root.find("compiler")
    out = copy.deepcopy(root)
    if legacy_capsule_mass:
        _apply_legacy_capsule_mass(out)
    if comp is None or comp.get("coordinate") != "global":
        return ET.tostring(out, encoding="unicode")
    assert_no_rotation(root)
    comp_out = out.find("compiler")
    del comp_out.attrib["coordinate"]

    world = out.find("worldbody")
    if world is not None:
        # Direct children of worldbody are already in the world frame.
        for body in list(world):
            if body.tag == "body":
                _recurse(body, [0.0, 0.0, 0.0])
    return ET.tostring(out, encoding="unicode")


def main():
    import argparse
    import json
    p = argparse.ArgumentParser()
    p.add_argument("--json", help="a physics_bridge_gate blob to read xml from")
    p.add_argument("--xml", help="an xml file")
    p.add_argument("--out", default="/tmp/converted.xml")
    args = p.parse_args()
    if args.json:
        src = json.load(open(args.json))["xml"]
    else:
        src = open(args.xml).read()
    got = convert(src)
    with open(args.out, "w") as f:
        f.write(got)
    print(f"wrote {args.out} ({len(got)} chars)")


if __name__ == "__main__":
    main()
