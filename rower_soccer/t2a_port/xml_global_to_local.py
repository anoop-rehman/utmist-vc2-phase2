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



# ---------------------------------------------------------------------------
# MuJoCo 2.1's capsule inertial properties, in closed form
# ---------------------------------------------------------------------------
# Recovered empirically in `legacy_capsule_fit.py` by sweeping a 49-point
# (radius x half-length) grid through THEIR compiler and fitting an independent
# monomial basis. Every coefficient came out a clean rational at ~3e-15 relative
# error, so this is the formula rather than an approximation to it:
#
#     mass  = rho*pi * (2 r^2 h  +  r^3)
#     I_ax  = rho*pi * (r^4 h  +  r^5 / 2)
#     I_tr  = rho*pi * (r^4 h  +  2/3 r^2 h^3  +  r^5 / 3  +  r^3 h^2)
#
# with r the radius and h the CYLINDER half-length. The mass agrees with
# "cylinder + 3/4 sphere"; the inertia does not reduce to any single scaling of
# the correct one, which is why the density trick above leaves it 6.9% out.
#
# A capsule is transversely isotropic, so the body-frame tensor is
#     I = I_ax * u u^T + I_tr * (Id - u u^T)
# for a unit axis u -- no quaternion needed, and `fullinertia` takes it directly.
PI = 3.141592653589793


def legacy_capsule_inertial(radius, half_len, axis, density=1000.0):
    """(mass, 3x3 body-frame inertia) as MuJoCo 2.1 would have computed them."""
    r, h = radius, half_len
    mass = density * PI * (2 * r * r * h + r ** 3)
    i_ax = density * PI * (r ** 4 * h + r ** 5 / 2.0)
    i_tr = density * PI * (r ** 4 * h + 2.0 / 3.0 * r * r * h ** 3
                           + r ** 5 / 3.0 + r ** 3 * h * h)
    n = sum(c * c for c in axis) ** 0.5
    u = [c / n for c in axis]
    I = [[0.0] * 3 for _ in range(3)]
    for a in range(3):
        for b in range(3):
            I[a][b] = (i_ax - i_tr) * u[a] * u[b] + (i_tr if a == b else 0.0)
    return mass, I


def _apply_legacy_inertial(root, default_density=1000.0):
    """Give every single-capsule body an explicit <inertial> matching 2.1.

    Explicit beats corrective: with `<inertial>` present the compiler uses it
    verbatim, so nothing depends on which MuJoCo version does the deriving.
    """
    n = 0
    for body in root.iter("body"):
        caps = [g for g in body if g.tag == "geom" and g.get("type") == "capsule"]
        others = [g for g in body if g.tag == "geom" and g.get("type") != "capsule"]
        if len(caps) != 1 or others:
            continue            # only the shape their generator actually emits
        g = caps[0]
        ft = _vec(g.get("fromto"))
        r = float(g.get("size").split()[0])
        axis = [ft[i + 3] - ft[i] for i in range(3)]
        half = 0.5 * sum(c * c for c in axis) ** 0.5
        if half <= 0:
            continue
        com = [(ft[i] + ft[i + 3]) / 2.0 for i in range(3)]
        d = g.get("density")
        rho = float(d) if d is not None else default_density
        mass, I = legacy_capsule_inertial(r, half, axis, rho)
        el = ET.Element("inertial")
        el.set("pos", _fmt(com))
        el.set("mass", f"{mass:.12g}")
        el.set("fullinertia", " ".join(
            f"{v:.12g}" for v in (I[0][0], I[1][1], I[2][2],
                                  I[0][1], I[0][2], I[1][2])))
        body.insert(0, el)
        n += 1
    return n


def _default_density(root, fallback=1000.0):
    """The `<default><geom density>` a geom without its own density inherits.

    Both legacy passes below need a density, and both used to assume MuJoCo's
    own 1000. That is right for `assets/mujoco_envs/hopper.xml`, which sets no
    density anywhere -- and wrong by a factor of 200 for
    `assets/mujoco_envs/ant.xml` and for the converted CompetEvo ant, whose
    `<default><geom>` says `density="5.0"`. Reading it makes the passes correct
    for every Transform2Act asset instead of only for the one they were written
    against.

    A class-scoped density would need per-geom resolution, so it raises rather
    than being silently ignored.
    """
    top = root.find("default")
    if top is None:
        return fallback
    for d in top.iter("default"):
        if d is top:
            continue
        g = d.find("geom")
        if g is not None and g.get("density") is not None:
            raise NotImplementedError(
                f"<default class=\"{d.get('class')}\"> sets its own geom "
                "density; per-class resolution is not implemented and assuming "
                "the top-level value would be a silently wrong mass")
    g = top.find("geom")
    if g is None or g.get("density") is None:
        return fallback
    return float(g.get("density"))


def convert(xml_str, legacy_capsule_mass=False, legacy_inertial=False):
    """Global-coordinate morphology XML -> local-coordinate equivalent.

    `legacy_capsule_mass=True` additionally reproduces MuJoCo 2.1's capsule
    mass, which is what Transform2Act actually trained against.
    """
    root = ET.fromstring(xml_str)
    rho0 = _default_density(root)
    comp = root.find("compiler")
    out = copy.deepcopy(root)
    is_global = comp is not None and comp.get("coordinate") == "global"
    if legacy_inertial:
        # Must run on LOCAL geometry, so it is applied after the shift below.
        pass
    elif legacy_capsule_mass:
        _apply_legacy_capsule_mass(out, rho0)
    # D3 M3 E1: this used to `return` here for a non-global input, which
    # silently dropped `legacy_inertial` on any XML that was ALREADY local.
    # That was harmless while every Transform2Act asset was global, and stopped
    # being harmless with `assets/mujoco_envs/ant_competevo.xml`, which is local
    # by construction (`competevo_to_t2a.py`) precisely so modern MuJoCo can
    # load its designs without this module. The coordinate shift is now
    # conditional; the legacy-inertial pass runs either way.
    if is_global:
        assert_no_rotation(root)
        comp_out = out.find("compiler")
        del comp_out.attrib["coordinate"]

        world = out.find("worldbody")
        if world is not None:
            # Direct children of worldbody are already in the world frame.
            for body in list(world):
                if body.tag == "body":
                    _recurse(body, [0.0, 0.0, 0.0])
    if legacy_inertial:
        n = _apply_legacy_inertial(out, rho0)
        # WITHOUT THIS THE <inertial> ELEMENTS ARE SILENTLY IGNORED.
        # Their compiler line says inertiafromgeom="true", which tells MuJoCo
        # to derive inertia from geoms ALWAYS, overriding an explicit
        # <inertial>. "auto" uses the explicit one when present and falls back
        # to geoms otherwise, which is exactly the wanted behaviour.
        comp_out = out.find("compiler")
        if comp_out is not None and n:
            comp_out.set("inertiafromgeom", "auto")
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
