"""Per-world model fields for a Transform2Act morphology, WITHOUT compiling.

This is the core of D3 3d step 3. The batched execution env groups worlds by
topology and compiles **one** model per group; the worlds inside a group share a
skeleton but differ in attributes (capsule radii, segment lengths, actuator
gears), so their `geom_size`, `body_mass`, `body_inertia` and friends differ per
world and have to be written into the batched model.

`competevo_port/design.py` already proves that works -- `WRITTEN_FIELDS` there
is exactly this set and `tests/test_design_parity.py` gates it to machine
epsilon. What that port does not need is a way to get the numbers without a
compile, because its morphology is one topology with a scale vector. Here the
numbers come out of their XML generator, and compiling 50,000 of them per epoch
is the thing that would sink the whole approach.

So: generate the XML per world on the CPU (their generator, cheap string work),
then read the fields off the XML **arithmetically**. The only hard part was the
inertial, and `xml_global_to_local.legacy_capsule_inertial` now has MuJoCo 2.1's
exact closed form.

    PYTHONPATH=. .venv/bin/python -m rower_soccer.t2a_port.xml_to_fields --gate

The gate compiles the same XML and compares field by field, because "computed a
number" and "computed the number MuJoCo would have" are different claims.
"""

import xml.etree.ElementTree as ET

import numpy as np

from rower_soccer.t2a_port.xml_global_to_local import (convert,
                                                       legacy_capsule_inertial)


def _v(s):
    return np.array([float(x) for x in s.split()], dtype=np.float64)


def _quat_from_z_to(axis):
    """The quaternion MuJoCo assigns a capsule: +z rotated onto `axis`.

    Capsules are declared by `fromto`, and MuJoCo stores an orientation that
    carries the geom's local +z onto that segment. Shortest-arc, with the
    antiparallel case picked explicitly rather than left to a zero cross
    product.
    """
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.linalg.norm(a)
    z = np.array([0.0, 0.0, 1.0])
    c = float(np.dot(z, a))
    if c > 1.0 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1.0 + 1e-12:
        return np.array([0.0, 1.0, 0.0, 0.0])       # 180 deg about x
    v = np.cross(z, a)
    w = 1.0 + c
    q = np.array([w, v[0], v[1], v[2]])
    return q / np.linalg.norm(q)


def fields_from_xml(xml_str, density=1000.0):
    """Compute the per-world model fields MuJoCo would derive from this XML.

    Returns a dict keyed like `competevo_port.design.WRITTEN_FIELDS`, indexed
    the way MuJoCo indexes: body 0 is the world, geoms in document order.
    """
    local = convert(xml_str)          # global coords are not readable as-is
    root = ET.fromstring(local)
    world = root.find("worldbody")

    bodies = [{"name": "world", "pos": np.zeros(3), "geoms": [], "parent": -1}]
    geoms = []

    def walk(el, parent_idx):
        idx = len(bodies)
        bodies.append({"name": el.get("name"), "parent": parent_idx,
                       "pos": _v(el.get("pos", "0 0 0")), "geoms": []})
        for child in el:
            if child.tag == "geom":
                g = {"body": idx, "type": child.get("type"),
                     "size": _v(child.get("size", "0")),
                     "fromto": (_v(child.get("fromto"))
                                if child.get("fromto") else None),
                     "pos": (_v(child.get("pos")) if child.get("pos")
                             else np.zeros(3)),
                     "density": float(child.get("density", density))}
                bodies[idx]["geoms"].append(len(geoms))
                geoms.append(g)
        for child in el:
            if child.tag == "body":
                walk(child, idx)

    # World-level geoms (the floor) come first in MuJoCo's ordering, exactly as
    # they appear in the document.
    for el in world:
        if el.tag == "geom":
            g = {"body": 0, "type": el.get("type"),
                 "size": _v(el.get("size", "0")), "fromto": None,
                 "pos": _v(el.get("pos", "0 0 0")),
                 "density": float(el.get("density", density))}
            bodies[0]["geoms"].append(len(geoms))
            geoms.append(g)
    for el in world:
        if el.tag == "body":
            walk(el, 0)

    nb, ng = len(bodies), len(geoms)
    out = {
        "body_pos": np.zeros((nb, 3)), "body_mass": np.zeros(nb),
        "body_inertia": np.zeros((nb, 3)), "body_ipos": np.zeros((nb, 3)),
        "body_subtreemass": np.zeros(nb),
        "geom_size": np.zeros((ng, 3)), "geom_pos": np.zeros((ng, 3)),
        "geom_quat": np.zeros((ng, 4)), "geom_rbound": np.zeros(ng),
        "geom_aabb": np.zeros((ng, 6)),
    }
    out["geom_quat"][:, 0] = 1.0

    for i, b in enumerate(bodies):
        out["body_pos"][i] = b["pos"]

    for gi, g in enumerate(geoms):
        if g["type"] == "capsule" and g["fromto"] is not None:
            p1, p2 = g["fromto"][:3], g["fromto"][3:]
            axis = p2 - p1
            half = 0.5 * float(np.linalg.norm(axis))
            r = float(g["size"][0])
            out["geom_size"][gi, :2] = (r, half)
            out["geom_pos"][gi] = 0.5 * (p1 + p2)
            # MuJoCo's fromto convention carries the geom's local +z onto
            # p1 - p2, not p2 - p1. Verified against the compiler on every
            # capsule of a real hopper. A capsule is symmetric under z -> -z so
            # both describe the same solid, but the STORED field has to match
            # or a batched write would silently rotate the geom.
            out["geom_quat"][gi] = _quat_from_z_to(-axis)
            out["geom_rbound"][gi] = half + r
            # MuJoCo's geom_aabb is (center, half-extent) in the GEOM frame,
            # where the capsule lies along local z.
            out["geom_aabb"][gi] = (0, 0, 0, r, r, half + r)
        elif g["type"] == "plane":
            out["geom_size"][gi] = g["size"][:3]
            out["geom_pos"][gi] = g["pos"]
            out["geom_rbound"][gi] = 0.0        # MuJoCo: planes have rbound 0
            # A plane's aabb ignores its declared size: MuJoCo treats it as
            # unbounded and stores a half-extent of 1e10 (5e9 in z). Using the
            # declared 200 x 200 x 0.125 here would be a plausible-looking
            # number that the compiler never produces.
            out["geom_aabb"][gi] = (0, 0, -5e9, 1e10, 1e10, 5e9)
        else:
            raise NotImplementedError(f"geom type {g['type']} is not ported; "
                                      f"their generator emits capsules only")

    # Inertial. Their generator gives every non-world body exactly one capsule.
    for i, b in enumerate(bodies):
        if i == 0 or not b["geoms"]:
            continue
        if len(b["geoms"]) != 1:
            raise NotImplementedError(
                f"body {b['name']} has {len(b['geoms'])} geoms; the closed "
                f"form is single-capsule only")
        g = geoms[b["geoms"][0]]
        p1, p2 = g["fromto"][:3], g["fromto"][3:]
        axis = p2 - p1
        half = 0.5 * float(np.linalg.norm(axis))
        mass, I = legacy_capsule_inertial(float(g["size"][0]), half, axis,
                                          g["density"])
        out["body_mass"][i] = mass
        out["body_ipos"][i] = 0.5 * (p1 + p2)
        # MuJoCo stores PRINCIPAL moments; for a transversely isotropic capsule
        # they are (I_tr, I_tr, I_ax) in the geom frame, and the frame is
        # body_iquat. Eigen-decomposing the tensor gets them without having to
        # reproduce MuJoCo's axis convention.
        out["body_inertia"][i] = np.sort(np.linalg.eigvalsh(np.array(I)))[::-1]

    # subtreemass: a body plus everything below it.
    for i in range(nb - 1, -1, -1):
        out["body_subtreemass"][i] += out["body_mass"][i]
        p = bodies[i]["parent"]
        if p >= 0:
            out["body_subtreemass"][p] += out["body_subtreemass"][i]
    return out


def gate(xml_str):
    """Compare every computed field against what MuJoCo actually compiles."""
    import mujoco
    m = mujoco.MjModel.from_xml_string(convert(xml_str, legacy_inertial=True))
    got = fields_from_xml(xml_str)

    results = []

    def cmp(name, ours, theirs, tol=1e-9, sort=False):
        a, b = np.asarray(ours, float), np.asarray(theirs, float)
        if sort:
            a, b = np.sort(a, axis=-1), np.sort(b, axis=-1)
        d = float(np.abs(a - b).max()) if a.shape == b.shape else float("inf")
        ok = d < tol
        results.append(ok)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name:18s} max |d| = {d:.3e}"
              + ("" if a.shape == b.shape else f"  SHAPE {a.shape} vs {b.shape}"))

    print(f"nbody={m.nbody} ngeom={m.ngeom}")
    cmp("body_pos", got["body_pos"], m.body_pos)
    cmp("body_mass", got["body_mass"], m.body_mass)
    cmp("body_ipos", got["body_ipos"], m.body_ipos)
    cmp("body_inertia", got["body_inertia"], m.body_inertia, sort=True)
    cmp("body_subtreemass", got["body_subtreemass"], m.body_subtreemass)
    cmp("geom_size", got["geom_size"], m.geom_size)
    cmp("geom_pos", got["geom_pos"], m.geom_pos)
    cmp("geom_rbound", got["geom_rbound"], m.geom_rbound)
    cmp("geom_aabb", got["geom_aabb"], m.geom_aabb)
    # A quaternion and its negation are the same rotation, so compare |dot|.
    q1 = got["geom_quat"] / np.linalg.norm(got["geom_quat"], axis=1, keepdims=True)
    q2 = np.asarray(m.geom_quat)
    q2 = q2 / np.maximum(np.linalg.norm(q2, axis=1, keepdims=True), 1e-12)
    dot = np.abs((q1 * q2).sum(1))
    d = float(np.abs(dot - 1.0).max())
    results.append(d < 1e-9)
    print(f"  [{'PASS' if d < 1e-9 else 'FAIL'}] {'geom_quat':18s} "
          f"max |1-|dot|| = {d:.3e}")

    print(f"\n{sum(results)}/{len(results)} fields match")
    return all(results)


def main():
    import argparse
    import json
    p = argparse.ArgumentParser()
    p.add_argument("--gate", action="store_true")
    p.add_argument("--json",
                   default="/tmp/claude-0/-root/453bc0de-a27f-4894-ad03-7d048158ee36/scratchpad/t2a_bridge.json")
    args = p.parse_args()
    xml = json.load(open(args.json))["xml"]
    if args.gate:
        raise SystemExit(0 if gate(xml) else 1)
    for k, v in fields_from_xml(xml).items():
        print(k, np.asarray(v).shape)


if __name__ == "__main__":
    main()
