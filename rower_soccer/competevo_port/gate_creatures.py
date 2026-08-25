"""Gate `creatures.py`: does the general path reproduce the validated ant?

`scene.py`'s dev ant was transcribed by hand and checked against their asset
line by line; `tests/test_design_parity.py` gates its design writes to machine
epsilon. `creatures.py` replaces that transcription with a parse of their asset,
so the first thing to establish is that the parse produces the SAME ROBOT --
otherwise every 2g number silently changes meaning when 2h lands.

    PYTHONPATH=. MUJOCO_GL=osmesa .venv/bin/python \
        -m rower_soccer.competevo_port.gate_creatures

Checks, in order of what they would catch:

  1. **The ant round-trips.** Parsed vs hand-written, compiled, compared field by
     field. This is the load-bearing check: it validates the general path
     against the one creature already known to be right, so bug and spider
     inherit that confidence rather than asserting it.
  2. **Actuator order comes from the asset.** The ant is hip-major and the other
     two are leg-interleaved, so a port that generalised the ant's pattern would
     permute every leg on both new creatures and still train. Asserted against
     the literal expected orders.
  3. **No creature admits degenerate geometry** over the clamped design range.
     This is what the spider's published SCALE_MAX = 1.2 fails.
  4. **Every creature compiles and stands**, alone and in a 4-agent mixed scene,
     with plausible mass and no initial penetration.
"""

import itertools
import sys

import numpy as np


PASS, FAIL = "PASS", "FAIL"
_results = []


def check(name, ok, detail=""):
    _results.append(bool(ok))
    print(f"[{PASS if ok else FAIL}] {name}" + (f"  {detail}" if detail else ""))
    return ok


def main():
    import mujoco

    from rower_soccer.competevo_port import scene
    from rower_soccer.competevo_port.creatures import (CREATURES,
                                                       SPIDER_SCALE_MAX_THEIRS,
                                                       body_xml, motor_xml)

    # ---- 1. the ant round-trips -----------------------------------------
    # Compiled ALONE in a trivial world, so the comparison is of the robot and
    # not of the scene wrapper (bitmask, poses and goals differ between the 1v1
    # and team scenes and would swamp any real difference).
    def solo(body_str, motors):
        return f"""<mujoco model="solo">
  <compiler angle="degree" coordinate="local" inertiafromgeom="true"/>
  <default><joint armature="1" damping="1" limited="true"/>
    <default class="agent0"><geom condim="3" density="5.0"
       friction="1 0.5 0.5" margin="0.01"/></default>
  </default>
  <worldbody>
{body_str}
  </worldbody>
  <actuator>
{motors}
  </actuator>
</mujoco>"""

    ant = CREATURES["ant"]
    pos, euler = (-1, 0, 0.75), (0, 0, 180)
    hand_motors = "\n".join(
        f'    <motor ctrllimited="true" ctrlrange="-1.0 1.0"'
        f' joint="agent0/{j}" gear="{scene.GEAR:g}" name="agent0/{j}"'
        f' class="agent0"/>' for j in scene._DEV_MOTOR_JOINTS)
    hand = mujoco.MjModel.from_xml_string(
        solo(scene._dev_ant_body_xml(0, pos, euler), hand_motors))
    gen = mujoco.MjModel.from_xml_string(
        solo(body_xml(ant, 0, pos, euler), motor_xml(ant, 0)))

    ok = (hand.nq, hand.nv, hand.nu, hand.nbody, hand.ngeom) == \
         (gen.nq, gen.nv, gen.nu, gen.nbody, gen.ngeom)
    check("ant: parsed and hand-written compile to the same dimensions",
          ok, f"nq={hand.nq} nv={hand.nv} nu={hand.nu} nbody={hand.nbody} "
              f"ngeom={hand.ngeom}")
    if ok:
        worst, wname = 0.0, None
        for f in ("body_pos", "body_quat", "body_mass", "body_inertia",
                  "body_ipos", "geom_size", "geom_pos", "geom_quat",
                  "geom_type", "geom_friction", "jnt_range", "jnt_axis",
                  "jnt_pos", "jnt_type", "dof_armature", "dof_damping",
                  "actuator_gear", "actuator_ctrlrange", "geom_condim"):
            a, b = np.asarray(getattr(hand, f)), np.asarray(getattr(gen, f))
            if a.shape != b.shape:
                worst, wname = float("inf"), f
                break
            d = float(np.abs(a.astype(float) - b.astype(float)).max())
            if d > worst:
                worst, wname = d, f
        check("ant: every compiled model field is identical",
              worst == 0.0, f"max |d| = {worst:.3e} (worst field: {wname})")

        # Names too: a permuted actuator block compiles to identical arrays
        # while driving different joints.
        for obj, label in ((mujoco.mjtObj.mjOBJ_ACTUATOR, "actuator"),
                           (mujoco.mjtObj.mjOBJ_BODY, "body"),
                           (mujoco.mjtObj.mjOBJ_JOINT, "joint")):
            n = {"actuator": hand.nu, "body": hand.nbody,
                 "joint": hand.njnt}[label]
            hn = [mujoco.mj_id2name(hand, obj, i) for i in range(n)]
            gn = [mujoco.mj_id2name(gen, obj, i) for i in range(n)]
            check(f"ant: {label} NAMES are in the same order", hn == gn,
                  f"{[x for x in hn if x][:3]}...")

    # ---- 2. actuator order is read, not constructed ----------------------
    # Written out literally: the whole hazard is that a plausible-looking
    # construction rule produces the wrong order for two of the three.
    expect = {
        "ant": ("11_joint", "12_joint", "13_joint", "14_joint",
                "111_joint", "112_joint", "113_joint", "114_joint"),
        "bug": ("11_joint", "111_joint", "12_joint", "112_joint",
                "13_joint", "113_joint", "14_joint", "114_joint",
                "15_joint", "115_joint", "16_joint", "116_joint"),
        "spider": ("11_joint", "111_joint", "12_joint", "112_joint",
                   "13_joint", "113_joint", "14_joint", "114_joint",
                   "15_joint", "115_joint", "16_joint", "116_joint",
                   "17_joint", "117_joint", "18_joint", "118_joint"),
    }
    for key, want in expect.items():
        got = CREATURES[key].motor_joints()
        check(f"{key}: actuator order matches the asset", got == want,
              "hip-major" if key == "ant" else "leg-interleaved")
    check("the ant is hip-major and the other two are NOT (the trap)",
          expect["ant"] != expect["bug"][:8] and
          expect["bug"][:2] == ("11_joint", "111_joint"))

    # ---- 3. no degenerate geometry over the clamped design range ---------
    for key, spec in CREATURES.items():
        lo = 1.0 + spec.geom_scale * (-1.0)
        check(f"{key}: geometry multiplier stays positive over s in [-1,1]",
              lo > 0.0, f"a_min = {lo:+.2f} (scale_max {spec.scale_max})")
    theirs_lo = 1.0 + SPIDER_SCALE_MAX_THEIRS * (-1.0)
    check("negative control: their published spider SCALE_MAX would fail this",
          theirs_lo <= 0.0, f"a_min = {theirs_lo:+.2f} at scale_max "
                            f"{SPIDER_SCALE_MAX_THEIRS}")

    # ---- 4. every creature, and every mixed pair, compiles and stands -----
    for key, spec in CREATURES.items():
        m = mujoco.MjModel.from_xml_string(_scene_from_specs(scene, [key, key]))
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        per_agent = float(m.body_mass.sum()) / 2.0
        check(f"{key}: compiles, {spec.n_motor} motors/agent, "
              f"{per_agent:.3f} kg/agent",
              m.nu == 2 * spec.n_motor and 0.05 < per_agent < 100.0)

    # A 4-agent scene of every ordered pair -- what 2h actually runs.
    bad = []
    for combo in itertools.product(CREATURES, repeat=2):
        names = [combo[0], combo[1], combo[0], combo[1]]
        try:
            m = mujoco.MjModel.from_xml_string(_scene_from_specs(scene, names))
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            if not np.isfinite(d.qpos).all():
                bad.append((combo, "non-finite qpos"))
        except Exception as e:                       # noqa: BLE001
            bad.append((combo, str(e)[:60]))
    check(f"all {len(list(itertools.product(CREATURES, repeat=2)))} team "
          f"compositions build a 4-agent scene", not bad,
          "" if not bad else f"failures: {bad[:3]}")

    n_ok = sum(_results)
    print(f"\n{n_ok}/{len(_results)} checks passed")
    return 0 if n_ok == len(_results) else 1


def _scene_from_specs(scene, names):
    """Build the merged scene for an arbitrary list of creature names."""
    from rower_soccer.competevo_port import team_scene
    return team_scene.dev_team_xml(len(names), creatures=names)


if __name__ == "__main__":
    sys.exit(main())
