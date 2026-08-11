"""Picture of what stage 2 actually bought: per-world MORPHOLOGY.

Stage 0/1 render one fixed ant racing another fixed ant, so nothing in their
video can show whether the design head does anything. This renders the thing
that is new -- the genome -> body map -- by applying a design to a render-only
MjModel through the SAME DesignWriter the batched env uses, then taking a
picture. If the writer were wrong, these ants would be wrong, which is the
point: the image is a check, not an illustration.

    python -m rower_soccer.competevo_port.render_designs --out designs.png

Physics is never involved: mj_forward places geoms from qpos and nothing is
stepped. Masses and inertias are still written (via mj_setConst) so that what
is drawn is exactly the body that would be simulated.
"""

import argparse
import os

import mujoco
import numpy as np
import torch

from rower_soccer.competevo_port.design import (CONST_FIELDS, DESIGN_DIM,
                                                WRITTEN_FIELDS, DesignWriter,
                                                build_design_spec)
from rower_soccer.competevo_port.scene import build_dev_scene


def _render_model(n_agents=2):
    """A single MjModel plus a writer that edits it in place.

    The batched writer wants `[nworld, ...]` arrays; a lone MjModel has
    `[...]`. Wrapping each field in a length-1 torch view that ALIASES the
    numpy array gives the writer the shape it expects and makes its writes land
    in the model -- no copy back, so there is no path where the render diverges
    from what was written.
    """
    model, meta = build_dev_scene(n_agents=n_agents)
    spec = build_design_spec(model, meta, device="cpu", dtype=torch.float64)
    arrays = {}
    for name in tuple(WRITTEN_FIELDS) + tuple(CONST_FIELDS):
        arr = getattr(model, name)
        arrays[name] = torch.from_numpy(np.asarray(arr)).unsqueeze(0)
    writer = DesignWriter(spec, arrays, model=model, exact_constants=True)
    return model, meta, writer, arrays


def apply_design(model, writer, arrays, scale):
    """Write one genome `[n_agents, DESIGN_DIM]` into `model`."""
    idx = torch.zeros(1, dtype=torch.long)
    writer.write(idx, torch.as_tensor(scale, dtype=torch.float64).unsqueeze(0))
    # The torch views alias the numpy buffers, so the model already has the new
    # geometry and masses. mj_setConst derives the rest (invweight etc.) --
    # skipping it leaves dof_invweight0 up to 46% wrong.
    for name, t in arrays.items():
        getattr(model, name)[:] = t[0].numpy()
    mujoco.mj_setConst(model, mujoco.MjData(model))


def shot(model, renderer, cam, qpos=None):
    data = mujoco.MjData(model)
    data.qpos[:] = model.qpos0 if qpos is None else qpos
    mujoco.mj_forward(model, data)
    renderer.update_scene(data, camera=cam)
    return renderer.render()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="designs.png")
    p.add_argument("--policy", default=None,
                   help="dev policy .pt; sample designs from its design head "
                        "instead of uniformly")
    p.add_argument("--n", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--width", type=int, default=420)
    p.add_argument("--height", type=int, default=420)
    args = p.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    model, meta, writer, arrays = _render_model()
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    cam = mujoco.MjvCamera()
    cam.distance, cam.elevation, cam.azimuth = 5.0, -20.0, 90.0

    # The EXTREMES bracket the genome space, so they say what the design head
    # can reach; N random draws only say where U(-1,1) happens to land, which
    # for a 20-dim box concentrates hard near the middle and makes a wide
    # parameterisation look narrow.
    ones = torch.ones(meta.n_agents, DESIGN_DIM, dtype=torch.float64)
    designs = [-ones, torch.zeros_like(ones), ones]
    labels = ["genome = -1 (min)", "genome = 0 (base)", "genome = +1 (max)"]
    g = torch.Generator().manual_seed(args.seed)
    for i in range(args.n):
        designs.append(torch.rand(meta.n_agents, DESIGN_DIM, generator=g,
                                  dtype=torch.float64) * 2 - 1)
        labels.append(f"U(-1,1) draw {i + 1}")

    tiles, masses = [], []
    for d, lab in zip(designs, labels):
        apply_design(model, writer, arrays, d)
        # Report the agent-0 subtree mass so the picture has a number attached:
        # "these look different" is weaker than "these weigh different
        # amounts", and mass is what the writer's parity gate measures.
        masses.append(float(np.asarray(model.body_mass)[
            list(meta.agents[0].body_ids)].sum()))
        data = mujoco.MjData(model)
        data.qpos[:] = model.qpos0
        mujoco.mj_forward(model, data)
        cam.lookat[:] = data.xpos[meta.agents[0].torso_body]
        tiles.append(shot(model, renderer, cam))
        print(f"{lab:20s} agent-0 mass {masses[-1]:.4f} kg", flush=True)

    import imageio
    grid = np.concatenate(tiles, axis=1)
    imageio.imwrite(args.out, grid)
    print(f"\nwrote {args.out}  ({grid.shape[1]}x{grid.shape[0]})")
    print(f"mass spread over {len(masses)} designs: "
          f"{min(masses):.3f} - {max(masses):.3f} kg "
          f"({max(masses) / min(masses):.2f}x)")


if __name__ == "__main__":
    main()
