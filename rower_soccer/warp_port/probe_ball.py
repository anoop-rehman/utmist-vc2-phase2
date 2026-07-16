"""Empirical check that the creature can actually move the ball.

Everything in the dribble/kick/shoot plan is wasted if the body cannot control
the ball, and ball mass/radius/friction relative to the worm is a real tuning
risk rather than a formality (the three-segment worm is ~3980 kg and ~10 m).
This probe answers, on the Warp path that training actually uses:

  1. Does the ball settle at rest on the floor (z == radius, no jitter)?
  2. Does a random-action creature contact and displace the ball at all?
  3. When it does, is the ball speed in a controllable band -- not glued to the
     floor (unpushable) and not launched across the pitch (a marble)?
  4. Does a rolling ball come to rest in a sane time? MuJoCo's default rolling
     friction (1e-4) lets a sphere roll essentially forever, which would let the
     drill keep paying out for a touch made ten seconds ago.

Run:  python -m rower_soccer.warp_port.probe_ball
      python -m rower_soccer.warp_port.probe_ball --radius 0.9 --mass 40
"""

import argparse

import mujoco
import numpy as np
import torch
import warp as wp

import mujoco_warp as mjw

from rower_soccer.warp_port.scene import (BallSpec, build_creature_ball_scene,
                                          creature_size)

CONTROL_DT = 0.025
SUBSTEPS = 10


def build(xml, spec, n, device, nconmax=64, njmax=512):
    model, meta = build_creature_ball_scene(xml, ball=spec)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    wm = mjw.put_model(model)
    # njmax/nconmax must be set explicitly. put_data auto-sizes them from the
    # *initial* MjData, where nothing is in contact yet, so the buffers come out
    # far too small once the creature lands and the condim-6 ball (6 constraint
    # rows per contact, vs 3) starts touching things. On overflow mujoco_warp
    # drops constraints and the sim diverges to NaN rather than erroring out.
    wd = mjw.put_data(model, data, nworld=n, nconmax=nconmax, njmax=njmax)
    t = lambda a: wp.to_torch(a)  # noqa: E731
    return meta, wm, wd, t(wd.qpos), t(wd.qvel), t(wd.ctrl), t(wd.xpos), t(wd.sensordata)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--xml", default="creature_configs/three_seg_worm.xml")
    p.add_argument("--worlds", type=int, default=256)
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--radius", type=float, default=None)
    p.add_argument("--mass", type=float, default=None)
    p.add_argument("--roll-friction", type=float, default=None)
    p.add_argument("--ball-dist", type=float, default=3.0,
                   help="ball spawn distance from creature root (m)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dev = "cuda"
    c_mass, c_height = creature_size(args.xml)
    spec = BallSpec()
    if args.radius is not None:
        spec.radius = args.radius
    if args.mass is not None:
        spec.mass = args.mass
    if args.roll_friction is not None:
        spec.friction = (spec.friction[0], spec.friction[1], args.roll_friction)

    print(f"creature : mass {c_mass:8.1f} kg   height {c_height:5.2f} m")
    print(f"ball     : mass {spec.mass:8.2f} kg   radius {spec.radius:5.3f} m  "
          f"density {spec.density:.1f}  friction {spec.friction}")
    print(f"           ball:creature mass ratio 1 : {c_mass / spec.mass:.0f}   "
          f"(dm_soccer BoxHead = 1:489)\n")

    meta, wm, wd, qpos, qvel, ctrl, xpos, sensordata = build(
        args.xml, spec, args.worlds, dev)
    n = args.worlds
    gen = torch.Generator(device=dev).manual_seed(args.seed)
    steps = int(round(args.seconds / CONTROL_DT))
    bq, bv, r = meta.ball_qpos, meta.ball_qvel, spec.radius

    # ---- reset: creature at origin w/ random yaw, ball at --ball-dist away ---
    qpos.zero_()
    qvel.zero_()
    qr = meta.qpos_root
    yaw = torch.rand(n, generator=gen, device=dev) * (2 * np.pi)
    qpos[:, qr + 2] = meta.spawn_z
    qpos[:, qr + 3] = torch.cos(yaw / 2)
    qpos[:, qr + 6] = torch.sin(yaw / 2)
    ang = torch.rand(n, generator=gen, device=dev) * (2 * np.pi)
    qpos[:, bq + 0] = args.ball_dist * torch.cos(ang)
    qpos[:, bq + 1] = args.ball_dist * torch.sin(ang)
    qpos[:, bq + 2] = r
    qpos[:, bq + 3] = 1.0  # unit quat
    mjw.forward(wm, wd)
    wp.synchronize_device()

    ball0 = qpos[:, bq:bq + 2].clone()
    touch_slices = [meta.sensor_slices[f"seg{i}_touch"] for i in range(3)]

    # ---- settle check: no actions, ball must come to rest at z == r ---------
    # Park the ball far away first. The creature topples on reset (it spawns as
    # an upright stack), and at --ball-dist it can reach the ball while falling,
    # which contaminates an "untouched ball" measurement with real contacts.
    qpos[:, bq + 0] = 30.0
    qpos[:, bq + 1] = 0.0
    qpos[:, bq + 2] = r
    qvel[:, bv:bv + 6] = 0.0
    mjw.forward(wm, wd)
    for _ in range(int(1.0 / CONTROL_DT)):
        ctrl.zero_()
        for _ in range(SUBSTEPS):
            mjw.step(wm, wd)
    wp.synchronize_device()
    z = qpos[:, bq + 2]
    v_idle = torch.linalg.norm(qvel[:, bv:bv + 3], dim=-1)
    print("[settle]  1s, zero action, ball parked 30 m from the creature")
    print(f"  ball z      : {z.mean():.4f} m  (expect {r:.4f})   "
          f"drift {abs(float(z.mean()) - r):.5f}")
    print(f"  ball speed  : mean {v_idle.mean():.5f}  max {v_idle.max():.5f} m/s "
          f" (expect ~0)")
    settle_ok = abs(float(z.mean()) - r) < 0.02 * r and float(v_idle.max()) < 0.05
    print(f"  -> {'OK' if settle_ok else 'FAIL: ball does not rest on the floor'}\n")

    # re-place the ball (the settle phase may have nudged it)
    qpos[:, bq + 0] = args.ball_dist * torch.cos(ang)
    qpos[:, bq + 1] = args.ball_dist * torch.sin(ang)
    qpos[:, bq + 2] = r
    qvel[:, bv:bv + 6] = 0.0
    mjw.forward(wm, wd)
    wp.synchronize_device()
    ball0 = qpos[:, bq:bq + 2].clone()

    # ---- random-action rollout: does the creature contact & move the ball? --
    max_speed = torch.zeros(n, device=dev)
    touched = torch.zeros(n, dtype=torch.bool, device=dev)
    last_touch_step = torch.full((n,), -1, dtype=torch.long, device=dev)
    for t in range(steps):
        ctrl.copy_(torch.rand(n, meta.nu, generator=gen, device=dev) * 2 - 1)
        for _ in range(SUBSTEPS):
            mjw.step(wm, wd)
        wp.synchronize_device()
        spd = torch.linalg.norm(qvel[:, bv:bv + 2], dim=-1)
        max_speed = torch.maximum(max_speed, spd)
        moving = spd > 0.1
        touched |= moving
        last_touch_step = torch.where(moving, torch.full_like(last_touch_step, t),
                                      last_touch_step)

    disp = torch.linalg.norm(qpos[:, bq:bq + 2] - ball0, dim=-1)
    tch = torch.cat([sensordata[:, s:s + d] for s, d in touch_slices], -1).sum(-1)
    frac = float(touched.float().mean())
    print(f"[contact] {args.seconds:.0f}s random actions, {n} worlds, "
          f"ball spawned {args.ball_dist} m away")
    print(f"  worlds that moved the ball : {frac * 100:5.1f}%")
    print(f"  ball displacement          : mean {disp.mean():6.2f} m   "
          f"p50 {disp.median():6.2f}   max {disp.max():6.2f}")
    print(f"  ball peak speed            : mean {max_speed.mean():6.2f} m/s  "
          f"max {max_speed.max():6.2f}")
    print(f"  touch sensor (final step)  : mean {tch.mean():.1f}")

    if frac < 0.05:
        print("  -> FAIL: creature essentially never moves the ball "
              "(unreachable, too heavy, or no contact)")
    elif float(max_speed.max()) > 50.0:
        print("  -> WARN: ball gets launched; too light for the body's torques")
    else:
        print("  -> OK: creature contacts and displaces the ball")
    print()

    # ---- roll-out decay: kick the ball, time until it stops -----------------
    qvel[:, bv:bv + 6] = 0.0
    qvel[:, bv + 0] = 5.0  # 5 m/s along +x
    mjw.forward(wm, wd)
    wp.synchronize_device()
    stop_step = torch.full((n,), -1, dtype=torch.long, device=dev)
    for t in range(int(20.0 / CONTROL_DT)):
        ctrl.zero_()
        for _ in range(SUBSTEPS):
            mjw.step(wm, wd)
        wp.synchronize_device()
        spd = torch.linalg.norm(qvel[:, bv:bv + 2], dim=-1)
        newly = (spd < 0.1) & (stop_step < 0)
        stop_step = torch.where(newly, torch.full_like(stop_step, t), stop_step)
        if bool((stop_step >= 0).all()):
            break
    stopped = stop_step >= 0
    print("[roll]    ball launched at 5 m/s, no actions, 20 s budget")
    if bool(stopped.any()):
        secs = stop_step[stopped].float() * CONTROL_DT
        print(f"  stopped in <20s            : {float(stopped.float().mean()) * 100:.0f}% "
              f"of worlds")
        print(f"  time to rest               : mean {secs.mean():.2f} s   "
              f"max {secs.max():.2f} s")
    else:
        print("  stopped in <20s            :   0% of worlds")
    if not bool(stopped.all()):
        print("  -> WARN: ball rolls > 20 s; raise rolling friction "
              "(--roll-friction) or the drill rewards stale touches")
    else:
        print("  -> OK: ball comes to rest")


if __name__ == "__main__":
    main()
