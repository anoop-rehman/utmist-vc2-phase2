"""Rescale the ant so the soccer ball sits at LEG height, not torso height.

    python -m rower_soccer.tools.scale_ant --length-scale 2.7 --out creature_configs/ant_v2.xml

Why
---
The drills use dm_control's SOCCER ball (r=0.35 m, 0.045 kg), and that size is
not negotiable: it is set by the goal (23.8 m wide) and the pitch (96 x 72 m),
so shrinking it for a drill would make the drill stop matching the game. Matching
the ball is the creature's job -- the same rule `scene.py` states and that
`unity2mujoco.py --length-scale` exists for on the rower.

Measured on the ant as first built:

    ant torso under a trained policy   0.489 m
    ball diameter                      0.700 m      -> ratio 1.43
    dm_control quadruped fetch          0.570 m torso vs 0.300 m ball -> 0.53

At 1.43 the ball spans z in [0, 0.70] while the torso sphere spans [0.24, 0.74]:
they occupy the same band, so the ant can only shove the ball with its body. At
dm_control's 0.53 the ball sits below the belly and is shepherded with the legs.

What scales, and what deliberately does not
-------------------------------------------
LENGTH scales: geom sizes, fromto endpoints, body positions, joint positions.

MASS DOES NOT follow L^3. At constant density a 2.7x ant weighs 494 kg against a
45 g ball -- a 11000:1 ratio that makes the ball inertially invisible, and any
contact launches it. dm_control's own soccer env pairs a ~50 kg humanoid with
this exact ball (~1100:1), so density is reduced to hold mass near the original
25 kg and keep the soccer-like ratio we already have (557:1). (fetch's 8:1 is
the outlier: its ball is a 14 kg medicine ball, not a football.)

GEAR DOES NOT SCALE (armature and damping do -- they are a separate knob, see
below). The first version of this file scaled gear by L too, reasoning that
longer levers need more torque. That is true only when
mass grows with volume -- and mass is held constant here, so the weight the legs
support never increased and the scaling multiplied STRENGTH-TO-WEIGHT by 2.7.
The result was an ant that flung itself 3.45 m up, nearly 2.8x its own standing
height, which read on video as low gravity. Measured under identical excitation:

    gear x1.0  ratio 16.8  max height 2.12 m  airborne 0.0%  upright 0.995
    gear x1.5  ratio 25.2  max height 2.53 m  airborne 0.0%  upright 0.989
    gear x2.0  ratio 33.6  max height 3.00 m  airborne 0.1%  upright 0.971
    gear x2.7  ratio 45.3  max height 3.45 m  airborne 0.7%  upright 0.804

x1.0 restores ant_v1's 16.8 strength-to-weight, which is the invariant that
matters. The lesson generalises past this file: a scaling rule is only valid
under the assumption it was derived from, and this one was applied to a model
built on the opposite assumption. The rower's gear_scale bug was the same class
of error pointing the other way (too little torque, 448M wasted steps), which is
why the first version cited it as justification -- and then reproduced it.

ARMATURE and DAMPING scale with L. They are a SEPARATE knob from gear, and
conflating the two is what made the first version wrong: gear sets
strength-to-weight (whether the ant launches itself), armature and damping set
joint compliance (whether it settles). With gear held at 1.0:

    arm/damp x1.0   max height 2.67 m   upright 0.967
    arm/damp x2.7   max height 1.91 m   upright 0.995
    arm/damp x5.0   max height 1.73 m   upright 0.996

Note the stability gate (random torque, no divergence) does NOT catch any of
this: the 2.7x-gear ant never diverged, it just launched. Check
strength-to-weight and uprightness explicitly.
"""

import argparse
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_IN = os.path.join(REPO, "creature_configs", "ant.xml")


def _scale_triplet(text, attr, k, n=None):
    """Multiply every number in attr="..." by k. n limits how many values."""
    def repl(m):
        vals = m.group(2).split()
        out = [f"{float(v) * k:.6g}" for v in vals[:n or len(vals)]]
        out += vals[len(out):]
        return f'{m.group(1)}="{" ".join(out)}"'
    return re.sub(rf'({attr})="([^"]+)"', repl, text)


def scale(src, length_scale, hold_mass=True, gear_scale=1.0):
    s = src
    # geometry
    s = _scale_triplet(s, "size", length_scale)
    s = _scale_triplet(s, "fromto", length_scale)
    s = _scale_triplet(s, "pos", length_scale)
    # ACTUATION IS NOT SCALED, and getting this wrong is what produced a
    # "low gravity" ant that flung itself 3.45 m into the air.
    #
    # The tempting argument is "longer levers need more torque". That is only
    # true when mass grows with volume. Here mass is HELD CONSTANT (see below),
    # so the weight the legs support never increased, and multiplying gear by L
    # multiplies strength-to-weight by L. Measured under identical excitation:
    #
    #     gear x1.0  ratio 16.8  max height 2.12 m  airborne 0.0%  upright 0.995
    #     gear x1.5  ratio 25.2  max height 2.53 m  airborne 0.0%  upright 0.989
    #     gear x2.0  ratio 33.6  max height 3.00 m  airborne 0.1%  upright 0.971
    #     gear x2.7  ratio 45.3  max height 3.45 m  airborne 0.7%  upright 0.804
    #
    # x1.0 restores ant_v1's 16.8 strength-to-weight, which is the invariant that
    # actually matters. Armature and damping stay with it: they are joint-space
    # quantities paired to gear by the invariance rule in three_seg_worm.xml's
    # header, and scaling them alone would just make the joints feel wrong.
    if gear_scale != 1.0:
        s = _scale_triplet(s, "gear", gear_scale)
    # ARMATURE and DAMPING *do* scale with length, and they are a SEPARATE knob
    # from gear -- conflating the two is what made the first version wrong.
    # gear sets strength-to-weight (whether the ant launches itself); armature
    # and damping set joint compliance (whether it settles). Measured with gear
    # held at 1.0:
    #
    #     arm/damp x1.0   max height 2.67 m   upright 0.967
    #     arm/damp x2.7   max height 1.91 m   upright 0.995
    #     arm/damp x5.0   max height 1.73 m   upright 0.996
    #
    # x2.7 (= length) already reaches 0.995; x5.0 buys almost nothing and
    # over-damped joints cost the agility the ball drills need.
    s = _scale_triplet(s, "armature", length_scale)
    s = _scale_triplet(s, "damping", length_scale)
    if hold_mass:
        # density / L^3 so total mass is unchanged; see docstring for why mass
        # must NOT follow L^3 against a 45 g ball.
        s = _scale_triplet(s, "density", 1.0 / length_scale ** 3)
    s = s.replace("<mujoco model=\"ant\">",
                  f"<mujoco model=\"ant_x{length_scale:g}\">")
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="src", default=DEFAULT_IN)
    p.add_argument("--out", required=True)
    p.add_argument("--length-scale", type=float, default=2.7)
    p.add_argument("--gear-scale", type=float, default=1.0,
                   help="multiplier on gear/armature/damping. LEAVE AT 1.0: "
                        "mass is held constant, so scaling torque with length "
                        "raises strength-to-weight and the ant launches itself "
                        "(measured: 3.45 m at 2.7x, uprightness 0.804).")
    p.add_argument("--grow-mass", action="store_true",
                   help="let mass follow L^3 (do not: see the module docstring)")
    a = p.parse_args()

    src = open(a.src).read()
    out = scale(src, a.length_scale, hold_mass=not a.grow_mass,
                gear_scale=a.gear_scale)
    note = (f"\n    <!-- GENERATED by rower_soccer.tools.scale_ant from "
            f"{os.path.basename(a.src)} at length_scale={a.length_scale:g}, "
            f"mass {'grown as L^3' if a.grow_mass else 'HELD constant'}. "
            f"Rationale in that module's docstring: the soccer ball is 0.70 m "
            f"across and cannot shrink, so the creature grows to meet it. -->")
    out = out.replace(">", ">" + note, 1)
    with open(a.out, "w") as f:
        f.write(out)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
