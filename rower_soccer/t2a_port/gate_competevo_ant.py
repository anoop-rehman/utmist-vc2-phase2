"""D3 M3 E1's gate: is the converted ant the SAME CREATURE as the one D1/D2 train?

`competevo_to_t2a.py` produces a body that Transform2Act's `Robot` can parse.
That is the easy half. A converted ant that *loads* but is a slightly different
robot would train perfectly well and mean nothing -- and this project has twice
shipped an env that was numerically fine and physically wrong. So the gate is
the deliverable, and the converter is what it gates.

Five phases, three of them cross-venv because the two MuJoCo stacks cannot
share an interpreter (theirs: mujoco-py 2.1.2.14 / mujoco210; ours: mujoco
3.12).

  A  same creature   (our venv)    every MjModel array indexed by
                                   body/geom/joint/dof/actuator, compared
                                   field by field against the model D1 and D2
                                   ACTUALLY COMPILE -- `scene.dev_run_to_goal_xml`
                                   -- not against the asset file. The field
                                   list is derived from `mujoco.introspect`,
                                   not hand-maintained, because
                                   `two_stage_pipeline.differing_fields` found
                                   21 arrays differing between two designs of
                                   one topology including `body_iquat` and
                                   `geom_sameframe`, and any hand list misses
                                   some.

  B  same physics    (our venv)    identical qpos/qvel, identical recorded
                                   ctrl, 500 raw `mj_step`s on the converted
                                   model and on the real CompetEvo scene under
                                   the SAME integrator. A field A forgot to
                                   compare shows up here as divergence.

  C  Robot mutates   (their venv)  round-trip, identity attribute transform,
                                   a real attribute transform, add a limb,
                                   remove a limb -- each recompiled and
                                   stepped. Plus a full-array check that
                                   `Robot`'s parse/write cycle does not deform
                                   the ant: it REDRAWS every capsule from
                                   bone_start to bone_end, so a capsule whose
                                   end does not coincide with its child's
                                   origin gets silently moved.

  D  cross-engine    (both)        their stack records a trajectory, ours
                                   replays it. Reports the step at which they
                                   separate, following `physics_bridge_gate.py`.

  N  negative controls (our venv)  six deliberate corruptions that MUST fail.
                                   Includes the one this gate actually caught:
                                   emitting `dev_ant_body.xml`'s literal
                                   `conaffinity="1"` instead of the `0` that
                                   `scene.py` overrides it to.

Run:
    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
    cd /workspace/utmist-vc2-phase2
    PYTHONPATH=. .venv/bin/python -m rower_soccer.t2a_port.gate_competevo_ant --ours
    cd /workspace/Transform2Act && source env-gpu.sh && \
      .venv-gpu/bin/python /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/gate_competevo_ant.py --theirs
    cd /workspace/utmist-vc2-phase2 && \
      PYTHONPATH=. .venv/bin/python -m rower_soccer.t2a_port.gate_competevo_ant --cross
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SCRATCH = ("/tmp/claude-0/-root/453bc0de-a27f-4894-ad03-7d048158ee36/scratchpad")
BLOB = os.path.join(SCRATCH, "e1_cross_engine.json")

# Tolerances, stated once.
#   A: the two XMLs carry the same decimal literals through the same compiler,
#      so anything above float round-off is a real difference.
TOL_MODEL = 1e-12
#   B: same compiler, same integrator, same actions -- the only source of
#      difference is summation order, so this is tight on purpose.
TOL_TRAJ = 1e-9
#   D: two different MuJoCo builds. Divergence is expected and the useful
#      number is WHERE, not whether.
TOL_CROSS = 1e-3

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)
    return ok


# ---------------------------------------------------------------------------
# field classification, derived from MuJoCo rather than written down
# ---------------------------------------------------------------------------

# Skipped, each with a reason -- and skipped by a RULE applied to MuJoCo's own
# struct metadata, not by a hand-written list. A field is skipped only if it is
# integer-typed AND its name ends in `adr`/`id`/`num`: those are offsets into
# other tables or byte offsets into the name pool, so their values encode where
# things sit in a model that has one agent plus two goal cylinders versus one
# that has one agent and none. Comparing them compares the arena, not the
# creature -- and everything they point AT is compared. The rule is typed on
# purpose: `geom_fluid` ends in "id" but is mjtNum, so it is NOT skipped.
_SKIP_SUFFIXES = ("adr", "id", "num")
_SKIP_EXTRA = {
    "body_plugin": "no plugins in either model",
    "geom_group": "render layer, not physics",
    "geom_rgba": "colour; the scene tints agents by team",
    "geom_matid": "material index into the asset table",
}


def indexed_fields(dim):
    """Every mjModel array whose LEADING extent is `dim`, from mujoco's own
    struct metadata, split into (compared, skipped-with-reason).

    Not a hand list -- that is the whole point.
    `two_stage_pipeline.differing_fields` found 21 arrays differing between two
    designs of ONE topology, including `body_iquat`, `dof_M0` and
    `geom_sameframe`; a hand-maintained list of "fields that matter" misses
    some, and the failure mode is a physically-wrong-but-numerically-fine env.
    """
    from mujoco.introspect import structs
    keep, skip = [], {}
    for f in structs.STRUCTS["mjModel"].fields:
        ext = getattr(f, "array_extent", None)
        if not (isinstance(ext, tuple) and ext and ext[0] == dim):
            continue
        is_int = "int" in str(f.type)
        if f.name in _SKIP_EXTRA:
            skip[f.name] = _SKIP_EXTRA[f.name]
        elif is_int and f.name.endswith(_SKIP_SUFFIXES):
            skip[f.name] = "int index/address into another table"
        else:
            keep.append(f.name)
    return sorted(keep), skip


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------

def converted_xml(**kw):
    from rower_soccer.t2a_port.competevo_to_t2a import SRC, convert
    with open(SRC) as f:
        return convert(f.read(), **kw)


def scene_xml_one_agent():
    """What D1/D2 ACTUALLY compile, for one agent.

    Not the asset file: `scene._dev_agent_default_xml` overrides the asset's
    `conaffinity`, and `_dev_ant_body_xml` overrides the root pose. Gating
    against the asset would have blessed a self-colliding ant.
    """
    from rower_soccer.competevo_port import scene
    return scene.dev_run_to_goal_xml(n_agents=1)


def _names(model, objtype, n):
    import mujoco
    return [mujoco.mj_id2name(model, objtype, i) for i in range(n)]


# ---------------------------------------------------------------------------
# A -- same creature
# ---------------------------------------------------------------------------

def phase_a():
    import mujoco
    print("\nA. SAME CREATURE -- converted vs the model D1/D2 compile "
          f"(all body/geom/joint/dof/actuator arrays, tol {TOL_MODEL:g})")

    # Match the scene's registered pose for agent 0 so root placement is not a
    # spurious difference; the converter treats pos/euler as placement, which
    # `_dev_ant_body_xml(agent_id, pos, euler)` confirms they are.
    from rower_soccer.competevo_port.scene import INIT_EULER, INIT_POS
    conv = mujoco.MjModel.from_xml_string(
        converted_xml(root_pos=INIT_POS[0], root_euler=INIT_EULER[0]))
    ref = mujoco.MjModel.from_xml_string(scene_xml_one_agent())

    O = mujoco.mjtObj
    maps = {}

    # bodies / joints / actuators pair up by name, modulo the scene's
    # `agent0/` prefix.
    # `nu` is NOT the only extent name for actuator arrays: in mujoco 3.x
    # `actuator_gear` is ('nout', 6), `actuator_gainprm` is ('nactuator', ...)
    # and only `actuator_ctrlrange` is ('nu', 2). Keying on "nu" alone compared
    # two of the 31 actuator arrays and silently skipped `actuator_gear` -- the
    # gear negative control is what caught it, which is the entire reason the
    # negative controls exist.
    for dim, objtype, nc, nr in [("nbody", O.mjOBJ_BODY, conv.nbody, ref.nbody),
                                 ("njnt", O.mjOBJ_JOINT, conv.njnt, ref.njnt),
                                 ("nu", O.mjOBJ_ACTUATOR, conv.nu, ref.nu),
                                 ("nactuator", O.mjOBJ_ACTUATOR, conv.nu, ref.nu),
                                 ("nout", O.mjOBJ_ACTUATOR, conv.nu, ref.nu)]:
        cn, rn = _names(conv, objtype, nc), _names(ref, objtype, nr)
        rn_idx = {}
        for i, nm in enumerate(rn):
            if nm is None:
                continue
            rn_idx[nm] = i
            if nm.startswith("agent0/"):
                rn_idx[nm[len("agent0/"):]] = i
        want = [(i, nm) for i, nm in enumerate(cn)
                if nm is not None and nm not in ("world", "floor")]
        pairs = [(i, rn_idx[nm]) for i, nm in want if nm in rn_idx]
        maps[dim] = pairs
        check(f"A/{dim}: every converted element found in the scene",
              len(pairs) == len(want) and len(want) > 0,
              f"{len(pairs)}/{len(want)} matched")

    # Geoms are UNNAMED in Transform2Act's dialect -- `Geom.sync_node` does
    # `node.attrib.pop('name')` -- so they cannot be paired by name. Pair them
    # through their bodies instead: each robot body carries exactly one geom in
    # both models, which is itself an invariant worth asserting.
    gp, gok = [], True
    for cb, rb in maps["nbody"]:
        nc, nr = conv.body_geomnum[cb], ref.body_geomnum[rb]
        if nc != nr or nc != 1:
            gok = False
            check(f"A/ngeom: body {cb} carries one geom in both",
                  False, f"{nc} vs {nr}")
            continue
        gp.append((conv.body_geomadr[cb], ref.body_geomadr[rb]))
    maps["ngeom"] = gp
    check("A/ngeom: every robot geom paired through its body", gok and len(gp) == 13,
          f"{len(gp)} matched")

    # dof arrays follow their joint.
    jnt_pairs = dict(maps["njnt"])
    dof_pairs = []
    for ci in range(conv.nv):
        cj = int(conv.dof_jntid[ci])
        if cj not in jnt_pairs:
            continue
        rj = jnt_pairs[cj]
        off = ci - int(conv.jnt_dofadr[cj])
        dof_pairs.append((ci, int(ref.jnt_dofadr[rj]) + off))
    maps["nv"] = dof_pairs
    check("A/nv: every dof paired through its joint", len(dof_pairs) == conv.nv,
          f"{len(dof_pairs)}/{conv.nv}")

    qpos_pairs = []
    for ci in range(conv.nq):
        cj = int(np.searchsorted(conv.jnt_qposadr, ci, side="right")) - 1
        if cj not in jnt_pairs:
            continue
        off = ci - int(conv.jnt_qposadr[cj])
        qpos_pairs.append((ci, int(ref.jnt_qposadr[jnt_pairs[cj]]) + off))
    maps["nq"] = qpos_pairs
    check("A/nq: every qpos slot paired through its joint",
          len(qpos_pairs) == conv.nq, f"{len(qpos_pairs)}/{conv.nq}")

    # Counts that are not indexed by anything above: a contact exclusion, an
    # equality constraint or a tendon in one model and not the other would
    # never show up in a per-element comparison.
    for n in ("nexclude", "npair", "neq", "ntendon", "nsite", "nsensor",
              "nmocap", "nflex", "nuserdata"):
        check(f"A: {n} equal", getattr(conv, n) == getattr(ref, n),
              f"{getattr(conv, n)} vs {getattr(ref, n)}")

    skipped, compared, worst, nfail = {}, 0, [], len(FAILURES)
    for dim, pairs in maps.items():
        keep, skip = indexed_fields(dim)
        skipped.update(skip)
        ci = np.array([p[0] for p in pairs], dtype=int)
        ri = np.array([p[1] for p in pairs], dtype=int)
        for field in keep:
            va = np.asarray(getattr(conv, field))[ci].astype(np.float64)
            vb = np.asarray(getattr(ref, field))[ri].astype(np.float64)
            d = float(np.max(np.abs(va - vb))) if va.size else 0.0
            compared += 1
            worst.append((d, field))
            if d > TOL_MODEL:
                k = int(np.argmax(np.abs(va - vb).reshape(len(ci), -1).max(axis=1)))
                check(f"A: {field}", False,
                      f"max|diff| {d:.3e} at converted index {ci[k]} "
                      f"({va[k]} vs {vb[k]})")
    worst.sort(reverse=True)
    check(f"A: all {compared} indexed arrays equal", len(FAILURES) == nfail,
          f"largest residual {worst[0][0]:.3e} in {worst[0][1]}")
    print(f"      skipped {len(skipped)} arrays by rule: "
          f"{', '.join(sorted(skipped))}")

    # The headline numbers, stated explicitly rather than only implied.
    cb = [i for i, _ in maps["nbody"]]
    rb = [j for _, j in maps["nbody"]]
    check("A: total robot mass", abs(conv.body_mass[cb].sum()
                                     - ref.body_mass[rb].sum()) < TOL_MODEL,
          f"{conv.body_mass[cb].sum():.9f} kg over {len(cb)} bodies "
          f"(per-body {conv.body_mass[cb].min():.6f}-{conv.body_mass[cb].max():.6f})")
    check("A: topology", (conv.nbody - 1, conv.njnt, conv.nu) == (13, 9, 8),
          f"{conv.nbody - 1} bodies, {conv.njnt} joints "
          f"(1 free + 8 hinge), {conv.nu} motors")
    check("A: actuator gears", np.allclose(conv.actuator_gear[:, 0], 150.0),
          f"all {conv.nu} at 150")
    hinge = conv.jnt_type == mujoco.mjtJoint.mjJNT_HINGE
    rng = {tuple(np.round(r, 6)) for r in np.rad2deg(conv.jnt_range[hinge])}
    check("A: hinge ranges", rng == {(-30.0, 30.0), (30.0, 70.0)},
          f"4 hips [-30,30] deg, 4 ankles [30,70] deg")
    caps = conv.geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE
    check("A: capsule radii", np.allclose(conv.geom_size[caps, 0], 0.08),
          f"all {int(caps.sum())} at 0.08")
    halves = sorted({round(float(v), 6) for v in conv.geom_size[caps, 1]})
    check("A: capsule half-lengths", halves == [0.141421, 0.282843],
          f"{halves} (8 links at 0.2*sqrt2/2, 4 feet at twice that)")
    sph = conv.geom_type == mujoco.mjtGeom.mjGEOM_SPHERE
    check("A: torso sphere radius", np.allclose(conv.geom_size[sph, 0], 0.25),
          "0.25")
    return conv, ref


# ---------------------------------------------------------------------------
# B -- same physics, same engine
# ---------------------------------------------------------------------------

# Every geom attribute the converted file's `<default><geom>` would otherwise
# inject into a transplanted arena element. Written EXPLICITLY on each one, read
# off the reference's own compiled model, so the arena in phase B is the
# CompetEvo arena and not a hybrid. Derived from `mujoco.MjModel` rather than
# from reading the XML, for the same reason phase A derives its field list.
# (`density` is deliberately absent: it is a compile-time input that becomes
# body_mass, and every arena geom is a child of the world body, whose mass is
# fixed at 0 -- so the converted file's `density="5.0"` default cannot reach
# them. The `B: all 16 geoms identical` check would catch it if that were wrong.)
_ARENA_GEOM_ATTRS = ("margin", "gap", "friction", "contype", "conaffinity",
                     "condim", "priority", "solmix", "solref", "solimp")


def with_scene_arena(conv_xml, scene_xml):
    """The converted robot, standing in the CompetEvo arena.

    Phase A proves the two robots compile to identical arrays. Phase B has to
    prove nothing ELSE about the model changes the physics -- but running the
    converted ant on Transform2Act's floor and the CompetEvo ant on CompetEvo's
    floor makes a divergence uninterpretable, so the arenas are made identical
    and the only remaining difference is which XML the robot came from.

    Transplanting the arena ELEMENTS is not enough, and this is a real finding
    rather than a detail. Transform2Act's `ant.xml` gives its floor no `margin`,
    so the floor inherits `margin="0.01"` from the file's `<default><geom>`;
    CompetEvo's floor is outside every class and gets MuJoCo's `margin=0`. Feet
    therefore make contact **1 cm earlier** on Transform2Act's floor. Measured:
    with the floor's margin left to inherit, the two rollouts are bit-identical
    for 67 steps and then a single foot-floor contact appears in one and not the
    other, after which they separate to 0.109 in qpos. With the floor's margin
    matched, they are bit-identical (max|dqpos| exactly 0.0) for all 500 steps.

    So the arena elements are transplanted AND every geom attribute the
    converted file's default would inject is written explicitly from the
    reference's compiled model.
    """
    import xml.etree.ElementTree as ET

    import mujoco
    ref = mujoco.MjModel.from_xml_string(scene_xml)
    cr = ET.fromstring(conv_xml)
    sr = ET.fromstring(scene_xml)
    cwb, swb = cr.find("worldbody"), sr.find("worldbody")
    for el in list(cwb):
        if el.tag != "body":
            cwb.remove(el)
    for i, el in enumerate(swb):
        if el.tag == "body":
            continue
        el = ET.fromstring(ET.tostring(el))
        if el.tag == "geom":
            gid = mujoco.mj_name2id(ref, mujoco.mjtObj.mjOBJ_GEOM,
                                    el.get("name"))
            assert gid >= 0, f"arena geom {el.get('name')} not in the reference"
            el.attrib.pop("material", None)
            for a in _ARENA_GEOM_ATTRS:
                v = np.atleast_1d(getattr(ref, f"geom_{a}")[gid])
                el.set(a, " ".join(f"{x:.17g}" for x in v))
        cwb.insert(i, el)
    cr.find("option").attrib.update(sr.find("option").attrib)
    return ET.tostring(cr, encoding="unicode")


def phase_b(steps=500):
    import mujoco
    from rower_soccer.competevo_port import scene
    from rower_soccer.competevo_port.scene import INIT_EULER, INIT_POS
    print(f"\nB. SAME PHYSICS -- {steps} steps, identical state and recorded "
          f"actions, IDENTICAL ARENA (tol {TOL_TRAJ:g})")

    scene_x = scene.dev_run_to_goal_xml(n_agents=1)
    conv_x = with_scene_arena(
        converted_xml(root_pos=INIT_POS[0], root_euler=INIT_EULER[0]), scene_x)
    conv = mujoco.MjModel.from_xml_string(conv_x)
    ref = mujoco.MjModel.from_xml_string(scene_x)
    check("B: same nq/nv/nu/ngeom",
          (conv.nq, conv.nv, conv.nu, conv.ngeom)
          == (ref.nq, ref.nv, ref.nu, ref.ngeom),
          f"{conv.nq}/{conv.nv}/{conv.nu}/{conv.ngeom}")
    check("B: same integrator and timestep",
          (conv.opt.integrator, conv.opt.timestep, conv.opt.solver,
           conv.opt.iterations)
          == (ref.opt.integrator, ref.opt.timestep, ref.opt.solver,
              ref.opt.iterations),
          f"RK4, dt={conv.opt.timestep}, Newton, {conv.opt.iterations} iters "
          "-- the scene's, not Transform2Act's dt=0.01")

    # Same actuator ORDER too, or the replay would be a permutation of itself.
    O = mujoco.mjtObj
    cn = _names(conv, O.mjOBJ_ACTUATOR, conv.nu)
    rn = [n.replace("agent0/", "") for n in _names(ref, O.mjOBJ_ACTUATOR, ref.nu)]
    check("B: actuator order identical", cn == rn, f"{cn}")

    # The arena must actually BE identical, not assumed to be: every geom
    # array, arena geoms included, compared elementwise in model order.
    arena_bad = []
    for dim in ("ngeom",):
        keep, _ = indexed_fields(dim)
        for f in keep:
            va, vb = np.asarray(getattr(conv, f)), np.asarray(getattr(ref, f))
            if va.shape != vb.shape or (va.size and np.max(np.abs(
                    va.astype(float) - vb.astype(float))) > TOL_MODEL):
                arena_bad.append(f)
    check("B: all 16 geoms (arena included) identical", not arena_bad,
          "" if not arena_bad else f"differ: {arena_bad}")

    dc, dr = mujoco.MjData(conv), mujoco.MjData(ref)
    rng = np.random.default_rng(0)
    qpos0 = conv.qpos0.copy()
    qpos0[7:] += rng.uniform(-0.1, 0.1, conv.nq - 7)
    qvel0 = rng.uniform(-0.1, 0.1, conv.nv)
    ctrls = rng.uniform(-0.5, 0.5, size=(steps, conv.nu))

    for d in (dc, dr):
        d.qpos[:] = qpos0
        d.qvel[:] = qvel0
    diverge, worst, ncon_bad = None, 0.0, 0
    for t in range(steps):
        dc.ctrl[:] = ctrls[t]
        dr.ctrl[:] = ctrls[t]
        mujoco.mj_step(conv, dc)
        mujoco.mj_step(ref, dr)
        if dc.ncon != dr.ncon:
            ncon_bad += 1
        e = float(np.max(np.abs(dc.qpos - dr.qpos)))
        worst = max(worst, e)
        if diverge is None and e > TOL_TRAJ:
            diverge = t
    check("B: trajectories agree for the whole rollout", diverge is None,
          f"max|dqpos| {worst:.3e} over {steps} steps"
          + ("" if diverge is None else f"; first exceeded tol at step {diverge}"))
    check("B: same contact set at every step", ncon_bad == 0,
          f"{steps - ncon_bad}/{steps} steps with identical ncon "
          f"(final {dc.ncon})")
    check("B: the ant actually moved (the test is not vacuous)",
          np.max(np.abs(dc.qpos[:3] - qpos0[:3])) > 0.05,
          f"root moved {np.linalg.norm(dc.qpos[:3] - qpos0[:3]):.3f} m, "
          f"joints swept {np.ptp(np.abs(dc.qpos[7:])):.3f} rad")

    # Reported, not asserted: the same comparison with each robot on its OWN
    # arena, which is what E1 will actually run. It is a statement about the
    # two TASKS, not about the creature.
    conv_own = mujoco.MjModel.from_xml_string(
        converted_xml(root_pos=INIT_POS[0], root_euler=INIT_EULER[0])
        .replace('<option integrator="RK4" timestep="0.01"/>',
                 f'<option integrator="RK4" timestep="{scene.TIMESTEP}"'
                 ' solver="Newton" iterations="100"/>'))
    d2 = mujoco.MjData(conv_own)
    d2.qpos[:], d2.qvel[:] = qpos0, qvel0
    dr2 = mujoco.MjData(ref)
    dr2.qpos[:], dr2.qvel[:] = qpos0, qvel0
    sep = None
    for t in range(steps):
        d2.ctrl[:] = ctrls[t]
        dr2.ctrl[:] = ctrls[t]
        mujoco.mj_step(conv_own, d2)
        mujoco.mj_step(ref, dr2)
        if sep is None and np.max(np.abs(d2.qpos - dr2.qpos)) > TOL_TRAJ:
            sep = t
    print(f"      (for the record, NOT asserted: on Transform2Act's own floor "
          f"-- which inherits margin=0.01 where CompetEvo's floor has 0 -- the "
          f"same rollout separates at step {sep}. That is the arena, not the "
          f"robot: E1 trains on their floor, so its feet touch down 1 cm "
          f"earlier than in D1/D2.)")


# ---------------------------------------------------------------------------
# N -- negative controls
# ---------------------------------------------------------------------------

def _model_fields_equal(xml_a, xml_b):
    """True iff two XMLs compile to the same creature under the A comparison,
    restricted to the robot's own arrays (both are the T2A shell here, so the
    name maps are the identity)."""
    import mujoco
    a = mujoco.MjModel.from_xml_string(xml_a)
    b = mujoco.MjModel.from_xml_string(xml_b)
    if (a.nbody, a.ngeom, a.njnt, a.nu, a.nv) != (b.nbody, b.ngeom, b.njnt, b.nu, b.nv):
        return False
    for dim in ("nbody", "ngeom", "njnt", "nu", "nactuator",
                "nout", "nv", "nq"):
        keep, _ = indexed_fields(dim)
        for f in keep:
            va, vb = np.asarray(getattr(a, f)), np.asarray(getattr(b, f))
            if va.size and np.max(np.abs(va.astype(float) - vb.astype(float))) > TOL_MODEL:
                return False
    return True


def phase_n():
    from rower_soccer.t2a_port import competevo_to_t2a as C
    print("\nN. NEGATIVE CONTROLS -- every one of these must be rejected")
    base = converted_xml()
    src = open(C.SRC).read()

    corruptions = [
        ("one motor gear 150 -> 151",
         base.replace('joint="113_joint" gear="150"', 'joint="113_joint" gear="151"', 1)),
        ("one capsule radius 0.08 -> 0.0801",
         base.replace('fromto="0 0 0 0.4 -0.4 0" size="0.08"',
                      'fromto="0 0 0 0.4 -0.4 0" size="0.0801"', 1)),
        ("one ankle axis sign flipped",
         base.replace('axis="0.707107 -0.707107 0"', 'axis="-0.707107 0.707107 0"', 1)),
        ("one hinge range 30 70 -> 30 71",
         base.replace('range="30 70"', 'range="30 71"', 1)),
        ("the asset's literal conaffinity=1 instead of the 0 scene.py "
         "overrides it to  <-- the bug this gate caught",
         base.replace('<geom conaffinity="0" condim="3" density="5.0"',
                      '<geom conaffinity="1" condim="3" density="5.0"', 1)),
    ]
    for label, bad in corruptions:
        check(f"N: rejects {label}", not _model_fields_equal(base, bad))

    # Structural corruptions the VALIDATOR must catch before anything compiles.
    structural = [
        ("a capsule that does not end at its child's origin",
         src.replace('<geom fromto="0 0 0 0.2 0.2 0" size="0.08" type="capsule"/>',
                     '<geom fromto="0 0 0 0.25 0.25 0" size="0.08" type="capsule"/>', 1)),
        ("a joint moved off its body origin",
         src.replace('name="11_joint" pos="0 0 0"', 'name="11_joint" pos="0.01 0 0"', 1)),
        ("a second hinge on one body",
         src.replace('<joint axis="0 0 1" name="11_joint" pos="0 0 0" range="-30 30" type="hinge"/>',
                     '<joint axis="0 0 1" name="11_joint" pos="0 0 0" range="-30 30" type="hinge"/>'
                     '<joint axis="1 0 0" name="11b_joint" pos="0 0 0" range="-30 30" type="hinge"/>', 1)),
        ("a rotated non-root body frame",
         src.replace('<body name="11" pos="0.2 0.2 0">',
                     '<body name="11" pos="0.2 0.2 0" euler="0 0 20">', 1)),
    ]
    for label, bad in structural:
        try:
            C.convert(bad)
            check(f"N: validator rejects {label}", False, "converted anyway")
        except C.Lossy as e:
            check(f"N: validator rejects {label}", True,
                  str(e).splitlines()[1].strip()[:90])


# ---------------------------------------------------------------------------
# C -- Robot can mutate it (their venv, mujoco-py)
# ---------------------------------------------------------------------------

def phase_c(steps=200):
    import yaml
    sys.path.insert(0, "/workspace/Transform2Act")
    os.chdir("/workspace/Transform2Act")
    import mujoco_py
    from khrylib.robot.xml_robot import Robot

    print("\nC. ROBOT CAN MUTATE IT -- their venv, mujoco-py 2.1 / mujoco210")
    cfg = yaml.safe_load(open("khrylib/assets/ant.yml"))["robot"]
    sys.path.insert(0, REPO)
    xml = converted_xml()
    path = os.path.join(SCRATCH, "e1_conv.xml")
    open(path, "w").write(xml)

    def compile_and_step(x, n=50):
        m = mujoco_py.load_model_from_xml(x)
        sim = mujoco_py.MjSim(m)
        sim.data.qpos[2] = 0.75
        sim.forward()
        rng = np.random.default_rng(1)
        for _ in range(n):
            sim.data.ctrl[:] = rng.uniform(-0.3, 0.3, m.nu)
            sim.step()
        finite = np.all(np.isfinite(sim.data.qpos))
        return m, finite

    m0, ok0 = compile_and_step(xml)
    check("C: converted XML compiles and steps in their stack", ok0,
          f"nbody {m0.nbody - 1} nq {m0.nq} nv {m0.nv} nu {m0.nu}, "
          f"mass {m0.body_mass.sum():.9f}")

    def arrays(m):
        for nm in dir(m):
            if nm.startswith("_"):
                continue
            try:
                v = getattr(m, nm)
            except Exception:
                continue
            if isinstance(v, np.ndarray) and v.dtype.kind in "fiub":
                yield nm, v

    def all_equal(ma, mb):
        diffs = []
        ref = dict(arrays(ma))
        for nm, v in arrays(mb):
            if nm not in ref or ref[nm].shape != v.shape:
                diffs.append(nm)
                continue
            if not np.allclose(ref[nm].astype(float), v.astype(float),
                               rtol=0, atol=TOL_MODEL):
                diffs.append(nm)
        return diffs

    # C1: does Robot's parse/write cycle deform the ant? It redraws every
    # capsule from bone_start to bone_end, so this is not a formality.
    r = Robot(cfg, xml=path)
    rt = r.export_xml_string().decode()
    m1 = mujoco_py.load_model_from_xml(rt)
    d = all_equal(m0, m1)
    check("C: Robot round-trip leaves every model array unchanged",
          not d, f"{len(list(arrays(m0)))} arrays compared"
          + (f"; differ: {d}" if d else ""))

    # C2: identity attribute transform -- set_params(get_params()). This is
    # where a capsule whose end missed its child's origin would move.
    r2 = Robot(cfg, xml=path)
    r2.set_params(r2.get_params())
    m2 = mujoco_py.load_model_from_xml(r2.export_xml_string().decode())
    d = all_equal(m0, m2)
    check("C: identity attribute transform is a no-op on the compiled model",
          not d, "" if not d else f"differ: {d}")

    # C2b: the SAME identity transform through the path training actually
    # uses -- `AntEnv.set_design_params`, i.e. `pad_zeros=True`. This is not
    # the same code path as C2 (`Robot.set_params` pads nothing), and it is the
    # one that was broken: `Body.get_params` pads one zero for a jointless body
    # and `Body.set_params` did not consume it, so every field after it was
    # read one slot early. Our ant has four jointless bodies -- the leg stubs
    # between torso and hip -- and no robot Transform2Act ships has any, which
    # is why the bug was latent until now. Without the one-line fix in
    # `khrylib/robot/xml_robot.py`, this check fails: the four stub capsules
    # come back with radius 0.065 instead of 0.08 and ext_start 0.143
    # instead of 0.
    r2b = Robot(cfg, xml=path)
    p2b = []
    for b in r2b.bodies:
        v = []
        b.get_params(v, pad_zeros=True)
        p2b.append(r2b.demap_params(np.concatenate(v)))
    for params, b in zip(p2b, r2b.bodies):
        b.set_params(params, pad_zeros=True, map_params=True)
        b.sync_node()
    m2b = mujoco_py.load_model_from_xml(r2b.export_xml_string().decode())
    d = all_equal(m0, m2b)
    check("C: identity attribute transform through the TRAINING path "
          "(pad_zeros) is a no-op", not d, "" if not d else f"differ: {d}")

    # C3: a REAL attribute transform -- lengthen one foot capsule and check the
    # compiled model reflects exactly that, and nothing else moves that should
    # not. Body `111` is a foot; its bone_offset is (0.4, 0.4, 0).
    r3 = Robot(cfg, xml=path)
    b111 = [b for b in r3.bodies if b.name == "111"][0]
    before = np.linalg.norm(b111.bone_offset)
    # offset params are body-local; go through the same path AntEnv uses.
    p = []
    for b in r3.bodies:
        v = []
        b.get_params(v, pad_zeros=True)
        p.append(r3.demap_params(np.concatenate(v)))
    p = np.stack(p)
    i111 = [b.name for b in r3.bodies].index("111")
    p[i111, 0] += 0.15          # offset_x, in the normalized/sin space
    for params, b in zip(p, r3.bodies):
        b.set_params(params, pad_zeros=True, map_params=True)
        b.sync_node()
    after = np.linalg.norm(b111.bone_offset)
    m3 = mujoco_py.load_model_from_xml(r3.export_xml_string().decode())
    gi = [i for i, n in enumerate(m3.geom_bodyid)
          if m3.body_names[n] == "111"][0]
    gi0 = [i for i, n in enumerate(m0.geom_bodyid)
           if m0.body_names[n] == "111"][0]
    dlen = 2 * (m3.geom_size[gi][1] - m0.geom_size[gi0][1])
    check("C: attribute transform changes a length, and the COMPILED model "
          "shows it", abs((after - before) - dlen) < 1e-6 and abs(dlen) > 1e-3,
          f"bone_offset |.| {before:.6f} -> {after:.6f} "
          f"(+{after - before:.6f}); compiled capsule length +{dlen:.6f}")
    others = [i for i in range(m0.ngeom) if i != gi0]
    check("C: ...and no other geom changed size",
          np.allclose(m0.geom_size[others], m3.geom_size[others], atol=1e-12))
    _, ok3 = compile_and_step(r3.export_xml_string().decode())
    check("C: mutated model still steps", ok3)

    # C4: skeleton transform -- add a limb.
    r4 = Robot(cfg, xml=path)
    target = [b for b in r4.bodies if b.name == "11"][0]
    n_before, nu_before = len(r4.bodies), m0.nu
    r4.add_child_to_body(target)
    x4 = r4.export_xml_string().decode()
    m4, ok4 = compile_and_step(x4)
    check("C: skeleton ADD compiles and steps", ok4,
          f"bodies {n_before} -> {len(r4.bodies)}, nu {nu_before} -> {m4.nu}, "
          f"nq {m0.nq} -> {m4.nq}")
    check("C: the added limb is actuated", m4.nu == nu_before + 1)

    # C5: skeleton transform -- remove a limb (a depth-3 leaf, which is what
    # AntEnv.allow_remove_body permits on this body plan).
    r5 = Robot(cfg, xml=path)
    leaf = [b for b in r5.bodies if b.name == "113"][0]
    r5.remove_body(leaf)
    x5 = r5.export_xml_string().decode()
    m5, ok5 = compile_and_step(x5)
    check("C: skeleton REMOVE compiles and steps", ok5,
          f"bodies {n_before} -> {len(r5.bodies)}, nu {nu_before} -> {m5.nu}")

    # C6: the full AntEnv on our ant -- five skeleton steps, one attribute
    # step, then execution, exactly as training drives it.
    from design_opt.utils.config import Config
    cfg_all = Config("ant_competevo", tmp=True)
    from design_opt.envs.ant import AntEnv
    env = AntEnv(cfg_all, None)
    obs = env.reset()
    stages = []
    rng = np.random.default_rng(0)
    total_r = 0.0
    for t in range(cfg_all.skel_transform_nsteps + 1 + 100):
        nb = len(env.robot.bodies)
        a = np.zeros((nb, 3))
        if env.stage == "skeleton_transform":
            a[:, -1] = rng.integers(0, 3, nb)
        elif env.stage == "attribute_transform":
            a[:, 1:-1] = rng.uniform(-0.05, 0.05, (nb, 1))
        else:
            a[:, 0] = rng.uniform(-0.5, 0.5, nb)
        obs, rew, done, info = env.step(a)
        stages.append(info["stage"])
        total_r += rew
        if done:
            break
    check("C: AntEnv runs skeleton -> attribute -> execution on our ant",
          "execution" in stages and len(stages) > 10,
          f"{stages.count('skeleton_transform')} skel + "
          f"{stages.count('attribute_transform')} attr + "
          f"{stages.count('execution')} exec steps, R={total_r:.3f}, "
          f"final bodies {len(env.robot.bodies)}")

    # D-emit: record a trajectory in THEIR engine for the cross-engine check.
    m, _ = compile_and_step(xml, n=0)
    sim = mujoco_py.MjSim(m)
    rngd = np.random.default_rng(7)
    qpos0 = m.qpos0.copy()
    qpos0[7:] += rngd.uniform(-0.1, 0.1, m.nq - 7)
    qvel0 = rngd.uniform(-0.1, 0.1, m.nv)
    ctrls = rngd.uniform(-0.5, 0.5, size=(steps, m.nu))
    sim.data.qpos[:] = qpos0
    sim.data.qvel[:] = qvel0
    sim.forward()
    traj = []
    for t in range(steps):
        sim.data.ctrl[:] = ctrls[t]
        sim.step()
        traj.append(sim.data.qpos.copy().tolist())
    os.makedirs(SCRATCH, exist_ok=True)
    json.dump({"xml": xml, "qpos0": qpos0.tolist(), "qvel0": qvel0.tolist(),
               "ctrls": ctrls.tolist(), "qpos": traj,
               "body_mass": m.body_mass.tolist(),
               "body_inertia": m.body_inertia.tolist()},
              open(BLOB, "w"))
    print(f"      recorded {steps} steps of their physics -> {BLOB}")


# ---------------------------------------------------------------------------
# D -- cross-engine
# ---------------------------------------------------------------------------

def phase_d():
    """Their engine recorded the trajectory; ours replays it.

    Two different MuJoCo builds, so exact agreement is not the question. The
    question is WHAT differs, and the answer here is sharp: with MuJoCo 2.1's
    capsule inertial reapplied in closed form, the two engines agree to 1.2e-14
    for every contact-free step and part company on the step the first foot
    touches the floor. The mass difference is a known, correctable model
    difference; the rest is the contact solver, and no correction to the XML
    can remove it.
    """
    import mujoco
    from rower_soccer.t2a_port.xml_global_to_local import convert
    print(f"\nD. CROSS-ENGINE -- their mujoco210 vs our mujoco 3.12, same "
          f"recorded actions")
    b = json.load(open(BLOB))
    ctrls = np.array(b["ctrls"])
    ref = np.array(b["qpos"])
    mass_t = np.array(b["body_mass"])

    def replay(model):
        d = mujoco.MjData(model)
        d.qpos[:], d.qvel[:] = b["qpos0"], b["qvel0"]
        mujoco.mj_forward(model, d)
        errs, first_contact = [], None
        for t in range(len(ctrls)):
            d.ctrl[:] = ctrls[t]
            mujoco.mj_step(model, d)
            if first_contact is None and d.ncon > 0:
                first_contact = t
            errs.append(float(np.max(np.abs(d.qpos - ref[t]))))
        return np.array(errs), first_contact

    plain = mujoco.MjModel.from_xml_string(b["xml"])
    e_plain, fc = replay(plain)
    check("D: mass differs by exactly the known MuJoCo 2.1 capsule-cap bug",
          abs((mass_t.sum() / plain.body_mass.sum()) - 0.96468) < 5e-3,
          f"theirs {mass_t.sum():.6f} kg, ours {plain.body_mass.sum():.6f} kg, "
          f"ratio {mass_t.sum()/plain.body_mass.sum():.5f} -- 2.1 counted a "
          f"capsule's caps as 3/4 of a sphere "
          f"(xml_global_to_local.LEGACY_CAP_FRACTION)")

    corrected = mujoco.MjModel.from_xml_string(
        convert(b["xml"], legacy_inertial=True))
    e_corr, fc2 = replay(corrected)
    check("D: the legacy-inertial closed form reproduces their mass exactly",
          np.max(np.abs(corrected.body_mass - mass_t)) < 1e-9,
          f"max|dmass| {np.max(np.abs(corrected.body_mass - mass_t)):.3e} kg "
          f"over 14 bodies")
    check("D: ...and their inertia exactly",
          np.max(np.abs(corrected.body_inertia
                        - np.array(b["body_inertia"]))) < 1e-9,
          f"max|dI| "
          f"{np.max(np.abs(corrected.body_inertia - np.array(b['body_inertia']))):.3e}"
          "  (independent confirmation of legacy_capsule_fit.py's formula, "
          "which was fitted on a capsule grid, not on this robot)")

    check("D: same first-contact step in both engines", fc == fc2 and fc is not None,
          f"step {fc}")
    check(f"D: contact-free flight agrees to machine precision (steps 0-{fc - 1})",
          e_corr[:fc].max() < 1e-12,
          f"max|dqpos| {e_corr[:fc].max():.3e} corrected, "
          f"{e_plain[:fc].max():.3e} uncorrected -- a factor "
          f"{e_plain[:fc].max()/e_corr[:fc].max():.0f} from the mass alone")
    print(f"      at the FIRST FLOOR CONTACT (step {fc}) the corrected replay "
          f"jumps from {e_corr[fc-1]:.3e} to {e_corr[fc]:.3e} in one step, and "
          f"reaches {e_corr[len(ctrls)//2]:.3e} at {len(ctrls)//2} and "
          f"{e_corr[-1]:.3e} at {len(ctrls)}.")
    print(f"      NOT FIXABLE FROM THE XML: with mass and inertia matched to "
          f"1e-14 the residual is MuJoCo 2.1's contact solver against 3.12's. "
          f"E1 trains in THEIR stack, so it trains the 2.1 version of our ant: "
          f"legs {100*(1-mass_t.sum()/plain.body_mass.sum()):.1f}% lighter "
          f"than D1/D2's, and a different contact solve. Stated, not gated.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ours", action="store_true", help="phases A, B, N")
    p.add_argument("--theirs", action="store_true", help="phase C + D emit")
    p.add_argument("--cross", action="store_true", help="phase D check")
    p.add_argument("--steps", type=int, default=500)
    a = p.parse_args()
    if a.ours:
        sys.path.insert(0, REPO)
        phase_a()
        phase_b(a.steps)
        phase_n()
    if a.theirs:
        phase_c(a.steps)
    if a.cross:
        sys.path.insert(0, REPO)
        phase_d()
    print()
    if FAILURES:
        print(f"GATE FAILED: {len(FAILURES)} check(s): {FAILURES}")
        sys.exit(1)
    print("GATE PASSED")


if __name__ == "__main__":
    main()
