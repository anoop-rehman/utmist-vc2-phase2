"""LOOK AT IT: the converted ant beside the ant D1/D2 actually compile.

The gate says every compiled array matches and 500 steps of physics are
bit-identical. That is not the same as the robot looking right, and this
project has twice shipped an env that was numerically fine and visually wrong.
So this renders both, from the same cameras, at the same states, and writes one
PNG grid to look at.

Rows:
  1. `scene.dev_run_to_goal_xml(1)` -- the CompetEvo ant, agent 0, as D1 and D2
     compile it.
  2. `assets/mujoco_envs/ant_competevo.xml` -- the converted ant.
  3. the converted ant after a skeleton transform (one added limb) and an
     attribute transform, so the mutated designs get looked at too.

Columns: top-down, three-quarter, and a low side view that shows whether
anything is through the floor -- plus, for rows 1 and 2, the same three after
2 s of settling under gravity with zero control.

    export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log
    cd /workspace/utmist-vc2-phase2
    MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m rower_soccer.t2a_port.render_e1_ant
"""

import os

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco  # noqa: E402

OUT = "/tmp/claude-0/-root/453bc0de-a27f-4894-ad03-7d048158ee36/scratchpad"
W, H = 480, 400

VIEWS = [
    ("top-down", dict(azimuth=90, elevation=-89, distance=2.4, lookat=(0, 0, 0.2))),
    ("three-quarter", dict(azimuth=45, elevation=-22, distance=2.6, lookat=(0, 0, 0.25))),
    ("low side", dict(azimuth=0, elevation=-4, distance=2.4, lookat=(0, 0, 0.15))),
]


def shot(model, data, view, label):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.azimuth, cam.elevation = view["azimuth"], view["elevation"]
    cam.distance = view["distance"]
    cam.lookat[:] = np.array(view["lookat"]) + np.array([data.qpos[0], data.qpos[1], 0])
    with mujoco.Renderer(model, H, W) as r:
        r.update_scene(data, cam)
        return r.render()


def settle(model, seconds=2.0):
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, d)
    return d


def row(model, seconds=None):
    d = mujoco.MjData(model)
    if seconds:
        d = settle(model, seconds)
    else:
        mujoco.mj_forward(model, d)
    return [shot(model, d, v, n) for n, v in VIEWS], d


def main():
    import sys
    sys.path.insert(0, "/workspace/utmist-vc2-phase2")
    from rower_soccer.competevo_port import scene
    from rower_soccer.t2a_port.competevo_to_t2a import convert, SRC

    ref_x = scene.dev_run_to_goal_xml(n_agents=1)
    conv_x = convert(open(SRC).read(), root_pos=scene.INIT_POS[0],
                     root_euler=scene.INIT_EULER[0])

    models = [("CompetEvo (D1/D2)", mujoco.MjModel.from_xml_string(ref_x)),
              ("converted (T2A)", mujoco.MjModel.from_xml_string(conv_x))]

    # A mutated design, so the thing training will actually produce gets looked
    # at too. Built through the same code path AntEnv uses.
    import yaml
    sys.path.insert(0, "/workspace/Transform2Act")
    mutated = None
    try:
        from khrylib.robot.xml_robot import Robot
        cfg = yaml.safe_load(
            open("/workspace/Transform2Act/khrylib/assets/ant.yml"))["robot"]
        path = os.path.join(OUT, "e1_render_src.xml")
        open(path, "w").write(convert(open(SRC).read()))
        r = Robot(cfg, xml=path)
        r.add_child_to_body([b for b in r.bodies if b.name == "11"][0])
        p = []
        for b in r.bodies:
            v = []
            b.get_params(v, pad_zeros=True)
            p.append(r.demap_params(np.concatenate(v)))
        p = np.stack(p)
        p[:, 0] += 0.20                      # lengthen every bone offset in x
        for params, b in zip(p, r.bodies):
            b.set_params(params, pad_zeros=True, map_params=True)
            b.sync_node()
        mx = r.export_xml_string().decode()
        open(os.path.join(OUT, "e1_render_mutated.xml"), "w").write(mx)
        models.append(("+1 limb, longer bones",
                       mujoco.MjModel.from_xml_string(mx)))
    except Exception as e:                                  # noqa: BLE001
        print(f"mutated row skipped: {type(e).__name__}: {e}")

    rows, labels = [], []
    for name, m in models:
        imgs, d = row(m)
        rows.append(imgs)
        labels.append(f"{name}  (t=0, z={d.qpos[2]:.3f})")
    for name, m in models[:2]:
        imgs, d = row(m, seconds=2.0)
        rows.append(imgs)
        labels.append(f"{name}  settled 2 s, z={d.qpos[2]:.3f}, "
                      f"lowest geom z={_lowest(m, d):.4f}")

    grid = np.concatenate([np.concatenate(r, axis=1) for r in rows], axis=0)
    try:
        from PIL import Image, ImageDraw
        im = Image.fromarray(grid)
        dr = ImageDraw.Draw(im)
        for i, lab in enumerate(labels):
            dr.rectangle([0, i * H, 3 * W, i * H + 16], fill=(0, 0, 0))
            dr.text((4, i * H + 3), lab + "     |     "
                    + "   |   ".join(n for n, _ in VIEWS), fill=(255, 255, 0))
        im.save(os.path.join(OUT, "e1_ant_render.png"))
    except ImportError:
        import imageio
        imageio.imwrite(os.path.join(OUT, "e1_ant_render.png"), grid)
    print(f"wrote {OUT}/e1_ant_render.png  {grid.shape}")
    for lab in labels:
        print("  " + lab)


def _lowest(model, data):
    """Lowest point of any robot geom -- a foot through the floor shows here."""
    zs = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] == 0:
            continue
        z = data.geom_xpos[g][2]
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            half = model.geom_size[g][1]
            ax = data.geom_xmat[g].reshape(3, 3)[:, 2]
            z = min(z + half * ax[2], z - half * ax[2])
        zs.append(z - model.geom_size[g][0])
    return min(zs)


if __name__ == "__main__":
    main()
