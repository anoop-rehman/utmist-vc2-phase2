"""D3 M3 E1: render dumped design XMLs into one PNG montage, so they can be
LOOKED at rather than only counted.

One ROW per XML, three views per row (top-down, three-quarter, low side), after
`--settle` seconds under gravity with zero control -- the same three views and
the same settle that `render_e1_ant.py` used to check the converter.

Runs in the REPO venv (`mujoco` 3.12 + EGL), not theirs. That is deliberate: it
also checks that a design the skeleton stage evolved still compiles under
modern MuJoCo, which is the property `D3_M3_E1_ANT_CONVERTER.md` claims for
everything descended from the converted ant.

    MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m \
        rower_soccer.t2a_port.e1_render_designs \
        --xmls a.xml b.xml --out montage.png
"""

import argparse
import os

import numpy as np

VIEWS = [("top", dict(azimuth=90, elevation=-89, distance=3.2)),
         ("3/4", dict(azimuth=135, elevation=-22, distance=3.2)),
         ("side", dict(azimuth=180, elevation=-6, distance=2.6))]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xmls", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--width", type=int, default=420)
    p.add_argument("--height", type=int, default=340)
    p.add_argument("--settle", type=float, default=2.0)
    args = p.parse_args()

    import mujoco
    import imageio
    from PIL import Image, ImageDraw

    rows = []
    for path in args.xmls:
        model = mujoco.MjModel.from_xml_path(path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for _ in range(int(args.settle / model.opt.timestep)):
            mujoco.mj_step(model, data)
        lowest = min(
            float(data.geom_xpos[g][2]
                  - (model.geom_size[g][1] * abs(
                      data.geom_xmat[g].reshape(3, 3)[2, 2])
                     + model.geom_size[g][0]
                     if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
                     else model.geom_size[g][0]))
            for g in range(model.ngeom)
            if mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) != "floor")
        tag = (f"{os.path.basename(path)[:-4]}  {model.nbody - 1}b/{model.nu}m"
               f"  mass {model.body_mass.sum():.3f}  torso z {data.qpos[2]:.3f}"
               f"  lowest {lowest:+.4f}")
        imgs = []
        with mujoco.Renderer(model, args.height, args.width) as r:
            for name, v in VIEWS:
                cam = mujoco.MjvCamera()
                mujoco.mjv_defaultCamera(cam)
                cam.azimuth, cam.elevation = v["azimuth"], v["elevation"]
                cam.distance = v["distance"]
                cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.3]
                r.update_scene(data, cam)
                im = Image.fromarray(r.render())
                d = ImageDraw.Draw(im)
                d.rectangle([0, 0, im.width, 13], fill=(0, 0, 0))
                d.text((3, 2), f"{tag}  [{name}]", fill=(255, 255, 0))
                imgs.append(np.asarray(im))
        rows.append(np.concatenate(imgs, axis=1))
        print(f"  {tag}")
    imageio.imwrite(args.out, np.concatenate(rows, axis=0))
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
