"""Measure the creature's achievable locomotion speed, to calibrate the drills.

PIPELINE_V2 (line 74) says the follow drill's target speed range must be
"calibrated to worm's achievable speed (measure first with random policy)".
A random policy is the wrong instrument: random torques on a 2-DoF worm mostly
wiggle it in place, so it measures the noise floor, not the ceiling. If the
target moves faster than the creature possibly can, the drill is unlearnable no
matter how good the policy is -- and we would not find that out for a billion
steps.

So this sweeps open-loop travelling-wave gaits, which is the cheapest honest
upper bound available without training:

    ctrl_j(t) = amp * sin(2*pi*freq*t + j*phase)

over a grid of (freq, amp, phase), one gait per Warp world, and reports the net
displacement speed of the best. A trained policy should meet or beat this; a
random policy is reported alongside purely as the floor.

Run:  python -m rower_soccer.warp_port.probe_speed
"""

import argparse

import mujoco
import numpy as np
import torch
import warp as wp

import mujoco_warp as mjw

from rower_soccer.warp_port.scene import build_creature_scene, creature_size

CONTROL_DT = 0.025
SUBSTEPS = 10


def _rollout(wm, wd, qpos, ctrl, xpos, meta, n, steps, action_fn, settle=40):
    """Run `steps` control steps; return net XY speed (m/s) per world."""
    rb = meta.root_body
    # let the creature topple and settle before timing it -- it spawns as an
    # upright stack, and the fall alone displaces the root by ~a body length.
    for _ in range(settle):
        ctrl.zero_()
        for _ in range(SUBSTEPS):
            mjw.step(wm, wd)
    wp.synchronize_device()
    start = xpos[:, rb, :2].clone()
    for t in range(steps):
        ctrl.copy_(action_fn(t))
        for _ in range(SUBSTEPS):
            mjw.step(wm, wd)
    wp.synchronize_device()
    disp = torch.linalg.norm(xpos[:, rb, :2] - start, dim=-1)
    return disp / (steps * CONTROL_DT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="creature_configs/three_seg_worm.xml")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dev = "cuda"
    c_mass, c_height = creature_size(args.xml)

    # gait grid: one world per (freq, amp, phase) combination
    freqs = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0])   # Hz
    amps = np.array([0.25, 0.5, 0.75, 1.0])
    phases = np.array([0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4,
                       np.pi, 5 * np.pi / 4, 3 * np.pi / 2, 7 * np.pi / 4])
    grid = np.array([(f, a, ph) for f in freqs for a in amps for ph in phases])
    n = len(grid)

    model, meta = build_creature_scene(args.xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    wm = mjw.put_model(model)
    wd = mjw.put_data(model, data, nworld=n, nconmax=64, njmax=512)
    qpos, ctrl = wp.to_torch(wd.qpos), wp.to_torch(wd.ctrl)
    xpos = wp.to_torch(wd.xpos)

    print(f"creature : {args.xml}")
    print(f"           {c_mass:.2f} kg   {c_height:.2f} m   nu={meta.nu}")
    print(f"gait grid: {len(freqs)} freqs x {len(amps)} amps x {len(phases)} "
          f"phases = {n} worlds, {args.seconds:.0f}s each\n")

    def reset():
        qpos.zero_()
        qr = meta.qpos_root
        qpos[:, qr + 2] = meta.spawn_z
        qpos[:, qr + 3] = 1.0
        wp.to_torch(wd.qvel).zero_()
        mjw.forward(wm, wd)
        wp.synchronize_device()

    steps = int(round(args.seconds / CONTROL_DT))
    g = torch.as_tensor(grid, device=dev, dtype=torch.float32)
    f_t, a_t, ph_t = g[:, 0], g[:, 1], g[:, 2]
    j = torch.arange(meta.nu, device=dev, dtype=torch.float32)

    def gait(t):
        # [n, nu]: joint j of world w driven at that world's (freq, amp, phase)
        arg = 2 * np.pi * f_t[:, None] * (t * CONTROL_DT) + j[None, :] * ph_t[:, None]
        return (a_t[:, None] * torch.sin(arg)).clamp(-1, 1)

    reset()
    speed = _rollout(wm, wd, qpos, ctrl, xpos, meta, n, steps, gait)

    order = torch.argsort(speed, descending=True)
    print("[gait sweep] best open-loop travelling-wave gaits")
    print(f"  {'freq':>5} {'amp':>5} {'phase':>6}   speed (m/s)   body-len/s")
    for i in order[:8].tolist():
        f, a, ph = grid[i]
        v = float(speed[i])
        print(f"  {f:5.2f} {a:5.2f} {np.degrees(ph):5.0f}deg   {v:8.3f}    "
              f"{v / c_height:8.2f}")
    v_max = float(speed.max())
    print(f"\n  achievable speed (best gait) : {v_max:.3f} m/s")

    # random-policy floor, for reference
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    reset()
    rnd = _rollout(wm, wd, qpos, ctrl, xpos, meta, n, steps,
                   lambda t: torch.rand(n, meta.nu, generator=gen, device=dev) * 2 - 1)
    print(f"  random-policy floor          : {float(rnd.mean()):.3f} m/s "
          f"(max {float(rnd.max()):.3f})")

    print("\n[calibration] suggested follow-drill params")
    print(f"  target_speed_range : (0.10, {0.8 * v_max:.2f})  "
          f"# up to 80% of achievable, so the target is catchable")
    print(f"  spawn distance     : ({1.0 * c_height:.2f}, {3.0 * c_height:.2f})  "
          f"# 1-3 body lengths")
    print(f"  reward_coef        : 0.5   # the paper's value; the worm is now at "
          f"the paper's scale")


if __name__ == "__main__":
    main()
