"""What formula did MuJoCo 2.1 use for capsule mass and inertia?

`xml_global_to_local` established that their capsules are 1-3% light because the
hemispherical caps contributed `pi r^3` instead of `4/3 pi r^3`, and that
correcting mass by density scaling leaves `body_inertia` 6.9% off. Applying the
same 0.75 factor to the caps' inertia term analytically gets within 0.7-1.4%,
so the real formula differs more specifically.

This settles it empirically instead of guessing, because their compiler is
available: sweep a grid of (radius, half-length), compile each as a one-capsule
model in THEIR stack, and record what mass and inertia come out. Then the
candidate closed forms are tested against ground truth over the whole grid
rather than against one body of one robot.

Why it is worth settling: the batched execution env needs per-world mass and
inertia as a closed form it can evaluate on the GPU. The alternative -- calling
their compiler per world -- is ~50,000 compiles per epoch and defeats the point
of batching.

    # 1. their venv, ground truth
    cd /workspace/Transform2Act && source env-gpu.sh
    .venv-gpu/bin/python .../t2a_port/legacy_capsule_fit.py --emit

    # 2. either venv, fit and test
    .venv/bin/python .../t2a_port/legacy_capsule_fit.py --fit
"""

import argparse
import itertools
import json
import os

import numpy as np

GRID = "/tmp/claude-0/-root/453bc0de-a27f-4894-ad03-7d048158ee36/scratchpad/t2a_capsule_grid.json"
PI = 3.141592653589793

_XML = """<mujoco>
  <compiler angle="degree" inertiafromgeom="true"/>
  <worldbody>
    <body name="b" pos="0 0 0">
      <freejoint/>
      <geom type="capsule" fromto="0 0 {a} 0 0 {b}" size="{r}" density="1000"/>
    </body>
  </worldbody>
</mujoco>"""


def emit(args):
    import mujoco_py
    rows = []
    radii = np.linspace(0.02, 0.12, 7)
    halves = np.linspace(0.05, 0.5, 7)
    for r, h in itertools.product(radii, halves):
        xml = _XML.format(a=-h, b=h, r=r)
        path = "/tmp/_cap.xml"
        with open(path, "w") as f:
            f.write(xml)
        m = mujoco_py.load_model_from_path(path)
        rows.append({"r": float(r), "h": float(h),
                     "mass": float(m.body_mass[1]),
                     "inertia": [float(v) for v in m.body_inertia[1]]})
    os.makedirs(os.path.dirname(GRID), exist_ok=True)
    with open(GRID, "w") as f:
        json.dump(rows, f)
    print(f"emitted {len(rows)} capsules (r x h grid) -> {GRID}")


def candidates(r, h, cap):
    """Mass and (transverse, axial) inertia for a z-aligned capsule whose caps
    carry `cap` x the correct hemisphere volume."""
    rho = 1000.0
    mc = rho * PI * r * r * 2 * h          # cylinder
    ms = rho * 4.0 / 3.0 * PI * r ** 3 * cap   # both caps together
    mass = mc + ms
    # Cylinder about its own COM.
    ax = mc * r * r / 2.0
    tr = mc * (3 * r * r + (2 * h) ** 2) / 12.0
    # Two hemispheres, treated as a sphere of mass `ms` split at +/-h. The
    # transverse term carries the standard capsule parallel-axis correction.
    ax += ms * (2.0 / 5.0) * r * r
    tr += ms * (2.0 / 5.0 * r * r + h * h + 3.0 * h * r / 4.0)
    return mass, tr, ax


def fit(args):
    rows = json.load(open(GRID))
    r = np.array([x["r"] for x in rows])
    h = np.array([x["h"] for x in rows])
    mass = np.array([x["mass"] for x in rows])
    I = np.array([x["inertia"] for x in rows])
    # MuJoCo sorts diaginertia; for a z-capsule two entries are transverse and
    # one is axial, and axial is the smallest here for every grid point.
    I_ax = I.min(axis=1)
    I_tr = np.median(I, axis=1)

    print(f"{len(rows)} capsules, r in [{r.min():.3f}, {r.max():.3f}], "
          f"h in [{h.min():.3f}, {h.max():.3f}]")

    # 1. mass: solve for the cap fraction directly.
    rho = 1000.0
    v_cyl = rho * PI * r * r * 2 * h
    v_sph = rho * 4.0 / 3.0 * PI * r ** 3
    frac = (mass - v_cyl) / v_sph
    print(f"\nMASS  cap fraction = (m - m_cyl) / m_sphere")
    print(f"  min {frac.min():.6f}  max {frac.max():.6f}  "
          f"std {frac.std():.2e}")
    print(f"  -> {'CONSTANT 0.75' if abs(frac.mean() - 0.75) < 1e-6 and frac.std() < 1e-9 else 'NOT a clean constant'}")

    # 2. inertia: does the same cap fraction explain it?
    print(f"\nINERTIA  worst relative error of the closed form, over the grid")
    for cap in (1.0, 0.75):
        _, tr, ax = candidates(r, h, cap)
        e_tr = np.abs(tr - I_tr) / I_tr
        e_ax = np.abs(ax - I_ax) / I_ax
        print(f"  cap={cap:<5}  transverse {e_tr.max():.4%}   axial {e_ax.max():.4%}")

    # 3. free the two cap fractions -- one for mass, one for each inertia term.
    #    If a single extra constant explains it, that is the formula.
    _, tr1, ax1 = candidates(r, h, 1.0)
    _, tr0, ax0 = candidates(r, h, 0.0)
    a_tr = (I_tr - tr0) / (tr1 - tr0)
    a_ax = (I_ax - ax0) / (ax1 - ax0)
    print(f"\n  implied cap fraction from TRANSVERSE inertia: "
          f"min {a_tr.min():.4f} max {a_tr.max():.4f} std {a_tr.std():.2e}")
    print(f"  implied cap fraction from AXIAL inertia:      "
          f"min {a_ax.min():.4f} max {a_ax.max():.4f} std {a_ax.std():.2e}")
    print("\n  A constant here means one scalar fixes the inertia too; spread "
          "means 2.1 used a structurally different formula and the batched env "
          "must either bake in per-world values or accept modern physics.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--emit", action="store_true")
    p.add_argument("--fit", action="store_true")
    args = p.parse_args()
    if args.emit:
        emit(args)
    elif args.fit:
        fit(args)
    else:
        raise SystemExit("pass --emit (their venv) or --fit")


if __name__ == "__main__":
    main()
