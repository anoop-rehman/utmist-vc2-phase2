"""The three dev creatures -- ant, bug, spider -- as one parameterised spec.

2h wants heterogeneous teams: each of a team's two members may be any of the
three, each still evolving its own scale genome as in 2g. `scene.py` was written
for the ant alone, with the body transcribed by hand into a Python template so
that every convention (ankle sign, actuator order, geom naming) could be checked
against their asset line by line. That was the right call for one creature and
is the wrong call for three: the spider's legs are fully 3D, its capsules are
0.04 not 0.08, its joint ranges differ per link, and its geoms carry their own
`density` overrides. Transcribing that by hand would be inventing a fourth
opportunity to ship an env that is numerically fine and visually wrong.

So this module PARSES their assets instead, applying the same transformations
`evo_utils.create_multiagent_xml_str` applies. The assets are vendored under
`assets/` rather than read from `/workspace/competevo`, so the port does not
depend on an external checkout that can be re-cloned or moved.

`gate_creatures.py` requires the parsed ant to compile to the SAME MjModel as
`scene.py`'s hand-written one, field by field. That gate is the point of the
file: it validates the general path against the one creature already known to be
right, so the two new creatures inherit that confidence instead of asserting it.

--------------------------------------------------------------------------
Three things that differ across the creatures, and only one is obvious
--------------------------------------------------------------------------
1. **Size.** 4/6/8 legs, 8/12/16 motors, 13/19/25 bodies, genome 20/30/40.
   The genome is 5 parameters per leg in all three, with the same five roles,
   which is why `genome_table` needs only the leg count.

2. **ACTUATOR ORDER, and the ant is the odd one out.** The ant's motor block is
   `11,12,13,14, 111,112,113,114` -- all hips, then all ankles. The bug and the
   spider interleave per leg: `11,111, 12,112, 13,113, ...`. This IS the action
   layout. A port that generalised the ant's pattern to the other two would
   permute every leg on both and still train perfectly happily. Read from the
   asset, never constructed.

3. **`SCALE_MAX`, and the spider's is unusable as published.** Their values are
   0.3 / 0.5 / **1.2**, and the design action is clamped to [-1, 1] (theirs and
   ours both), so the geometry multiplier `a = 1 + SCALE_MAX * s` spans:

       ant     [0.70, 1.30]
       bug     [0.50, 1.50]
       spider  [-0.20, 2.20]   <-- negative

   At `s <= -0.833` every one of the spider's 25 capsule radii goes negative and
   the body is invalid. That is 8.3% of each parameter's range, and with 40
   parameters ~97% of uniformly drawn spider genomes contain at least one. There
   is no clipping anywhere in their code, and their source reads
   `SCALE_MAX = 1.2 #0.5` -- someone raised it and left the old value in the
   comment.

   **We use 0.5 for the spider, which is a deliberate deviation from their
   source.** Beyond making the space valid it makes the three creatures
   comparable: 2h asks which creature pairing wins, and a spider free to vary
   its geometry 2.4x more than the ant would confound that question with unequal
   design freedom. `SPIDER_SCALE_MAX_THEIRS` records what we departed from.
"""

import os
import xml.etree.ElementTree as ET

ASSETS = os.path.join(os.path.dirname(__file__), "assets")

# Recorded so the deviation is greppable, not just described in prose.
SPIDER_SCALE_MAX_THEIRS = 1.2


class CreatureSpec:
    """One dev creature. Everything else in the port derives from this."""

    def __init__(self, key, asset, n_legs, scale_max, design_dim):
        self.key = key
        self.asset = asset
        self.n_legs = n_legs
        self.scale_max = scale_max
        self.design_dim = design_dim
        assert design_dim == 5 * n_legs, (
            f"{key}: their genome is 5 params per leg; {design_dim} != 5*{n_legs}")

    # -- their asset ----------------------------------------------------
    @property
    def path(self):
        return os.path.join(ASSETS, self.asset)

    def tree(self):
        return ET.parse(self.path)

    def motor_joints(self):
        """Actuator order, READ from the asset. See note 2 in the docstring."""
        root = self.tree().getroot()
        joints = [m.get("joint") for m in root.findall(".//actuator/motor")]
        assert len(joints) == 2 * self.n_legs, (
            f"{self.key}: expected {2 * self.n_legs} motors, got {len(joints)}")
        return tuple(joints)

    def motor_gears(self):
        root = self.tree().getroot()
        return tuple(float(m.get("gear")) for m in root.findall(".//actuator/motor"))

    def genome_table(self):
        """`(body_local_name, length_param, radius_param, pos_param, gear_param)`.

        Identical in shape to `scene.DEV_GENOME_TABLE`, generalised over the leg
        count. Verified against `dev_bug.set_design_params` directly rather than
        extrapolated from the ant: per leg `L` (0-based), params `5L..5L+4` are
        upper length, mid radius, mid length, foot radius, foot length, and the
        two radius params double as the gear scales for that link's motor.
        """
        rows = []
        for leg in range(self.n_legs):
            p = 5 * leg
            k = leg + 1
            rows.append((f"{k}", p + 0, None, None, None))
            rows.append((f"1{k}", p + 2, p + 1, p + 0, p + 1))
            rows.append((f"11{k}", p + 4, p + 3, p + 2, p + 3))
        return tuple(rows)

    @property
    def geom_scale(self):
        return self.scale_max

    @property
    def gear_scale(self):
        return self.scale_max * 0.5

    @property
    def n_motor(self):
        return 2 * self.n_legs

    def __repr__(self):
        return (f"CreatureSpec({self.key}, legs={self.n_legs}, "
                f"motors={self.n_motor}, genome={self.design_dim}, "
                f"scale_max={self.scale_max})")


CREATURES = {
    "ant": CreatureSpec("ant", "dev_ant_body.xml", 4, 0.3, 20),
    "bug": CreatureSpec("bug", "dev_bug_body.xml", 6, 0.5, 30),
    # 0.5, not their 1.2 -- see note 3 in the module docstring.
    "spider": CreatureSpec("spider", "dev_spider_body.xml", 8, 0.5, 40),
}


def _fmt(*v):
    return " ".join(f"{x:g}" for x in v)


def body_xml(spec, agent_id, pos, euler):
    """One agent's `<body>` subtree, prefixed and class-tagged.

    The same transformations `evo_utils.create_multiagent_xml_str` applies:
    prefix every name, overwrite the root pose, tag every geom with the agent's
    class. Their `add_prefix(force_set=True)` invents `agent{i}/anon<random>`
    for the asset's unnamed geoms; we name them `agent{i}/geom_<body>` so the
    design writer can find them. Nothing in their code reads those names.
    """
    p = f"agent{agent_id}"
    root = spec.tree().getroot()
    body = root.find("body")
    assert body is not None and body.get("name") == "0", (
        f"{spec.key}: expected a root <body name='0'>")

    def walk(el, parent_local):
        local = el.get("name") or parent_local
        if el.tag in ("body", "joint") and el.get("name") is not None:
            el.set("name", f"{p}/{el.get('name')}")
        if el.tag == "geom":
            # Unnamed in every asset; named here so `design.py` can address it.
            el.set("name", f"{p}/geom_{parent_local}")
            el.set("class", p)
        for child in list(el):
            walk(child, local)

    walk(body, "0")
    body.set("pos", _fmt(*pos))
    body.set("euler", _fmt(*euler))
    ET.indent(body, space="  ")
    return ET.tostring(body, encoding="unicode")


def motor_xml(spec, agent_id):
    """The agent's `<motor>` block, in the asset's own order."""
    p = f"agent{agent_id}"
    out = []
    for joint, gear in zip(spec.motor_joints(), spec.motor_gears()):
        out.append(f'    <motor ctrllimited="true" ctrlrange="-1.0 1.0"'
                   f' joint="{p}/{joint}" gear="{gear:g}"'
                   f' name="{p}/{joint}" class="{p}"/>')
    return "\n".join(out)


def team_composition(names):
    """`["ant", "spider", ...]` -> the specs, validated.

    Kept as a function so a composition is always spelled the same way and a
    typo is an error here rather than a KeyError three modules away.
    """
    bad = [n for n in names if n not in CREATURES]
    assert not bad, f"unknown creature(s) {bad}; have {sorted(CREATURES)}"
    return [CREATURES[n] for n in names]
