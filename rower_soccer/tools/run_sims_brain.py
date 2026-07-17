"""Run the TWO_ARM_ROWER under its evolved Unity brain in CPU MuJoCo.

Produces videos/sims_brain_rower.mp4 and runs_v2/sims_brain_rower_traj.npz
(the "evolution mocap" motion prior: per-step joint trajectories + effector
commands).

Physics reconciliation (Unity-fidelity choices, with receipts)
==============================================================
The creature was evolved in uvc2's FloorEnv (user-confirmed): flat floor,
gravity, NO fluid -- FloorEnv has no FluidManager, so FluidDrag.FixedUpdate
early-returns and applies nothing.

* Masses: the Segment prefab's Rigidbody says 997, but Segment.Initialize()
  (run on every pool Get, and in Awake) calls RestoreState() with a default
  RigidbodyState whose mass is 1, and SetupSegment then does
  rb.mass *= dimX*dimY*dimZ (dims incl. cumulative scale).  Actual Unity
  mass is therefore 1 * volume  (density 1 kg/m^3) -- sub-kilogram limbs
  driven by 400 N*m motors: torque authority is enormous and the gait is
  dominated by joint-limit slams and ground contact, which is exactly what
  KSS floor evolution selected on.

* Scale: creature_configs/two_arm_rower_blueprint.xml (the task-mandated
  geometry) is 2x Unity lengths -- the original conversion pipeline used
  Unity FULL box extents as MuJoCo HALF extents (and doubled positions
  consistently).  We keep the blueprint and instead make the 2x world an
  EXACT dynamical similarity of Unity's at 1:1 time: lengths x2, masses x1
  (geom density 1/8 so 8*prod(half)*rho = 1*prod(unity_dims)), all
  accelerations x2 => gravity = 2*9.81 = 19.62, forces x(m*L/T^2) = x2,
  torques x(m*L^2/T^2) = x4 => motor force cap 400 -> 1600 N*m.  Friction is
  dimensionless and carries over.  Angles, angular rates and timings then
  match Unity 1:1; world positions are exactly 2x.

* Floor: FloorEnv's floor quad has no PhysicMaterial => PhysX default
  friction 0.6 (static == dynamic, combine=average) -> slide friction 0.6
  on floor and segments.  Segments spawn with the root's bottom face 1.08 m
  above the floor (SpawnTransform y=-5.34 vs floor y=-6.42) and drop in:
  we offset the free joint by 2*1.08 = 2.16 m.

* Segment-segment collisions: OFF in Unity -- the solid segment collider
  lives on layer 6 and the physics layer-collision matrix clears the (6,6)
  bit.  The blueprint XML's exclude-all-pairs matches exactly.

* Rigidbody.angularDrag = 0.05 (drag = 0): applied as an explicit per-body
  torque tau = -0.05 * I_world * omega (PhysX's per-step velocity scaling
  ~= e^{-0.05 t}; negligible per step, not over 30 s).

* PhysX maxAngularVelocity = 7 rad/s (Unity default) clamps every body's
  world spin each step.  With sub-kilogram limbs on 400 N*m motors this cap
  is a first-order part of Unity's dynamics (without it limbs spin at
  hundreds of rad/s during limit slams).  Emulated as a per-body braking
  torque tau = -I_world * (omega - 7 * omega_hat) / dt_sub applied whenever
  |omega| > 7, i.e. a dead-beat pull-back to the cap surface.  (Directly
  clamping qvel post-step is NOT viable: velocity edits behind the contact
  solver's back produce inconsistent contact states and launch impulses.)

* Motors: Unity HingeJoint velocity motor (PhysX velocity constraint with
  force cap).  Emulated as a per-substep dead-beat velocity servo:
  tau = clamp(beta * M_jj * (omega_target - qvel_j) / dt_sub, +-1600) with
  beta = 0.5 and M_jj the joint's diagonal inertia from mj_fullM.  We do NOT
  use MuJoCo <velocity> actuators: at PhysX-constraint-like gains,
  implicitfast's actuator derivative (which ignores forcerange clamping)
  would inject large artificial damping whenever the motor saturates.

* Joint limits +-75 deg (Segment.AttachHingeJoint) -- already in the XML.

* Hinge sign convention: the conversion pipeline dropped joint-axis signs
  (uid4, uid5 have Unity axis (0,0,-1) and uid8 (0,-1,0), all emitted as
  positive MuJoCo axes).  Unity axes map to MuJoCo by (x,y,z)->(x,z,y),
  which also converts the left-handed rotation sense to right-handed with
  sign preserved, so a dropped axis sign flips that joint's sense: we
  multiply those joints' velocity targets by -1.  A residual global-sign
  ambiguity of PhysX's motor convention remains (--global-sign to flip).

* Brain cadence: dt = 1/30 (TimeManager.asset, NOT Unity's default 0.02),
  FeedForward twice per FixedUpdate, commands held (ZOH) over the physics
  substeps -- as in Unity, where the motor target set in FixedUpdate
  persists through the PhysX step.  Oscillator phase uses absolute
  Time.time; evolution saw arbitrary spawn times, we canonically use t0=0
  (--t0 to change).  The brain is fully open-loop for this creature: its
  only sensor inputs are contact sensors, which ALWAYS read -1 in Unity
  due to the Segment(-50)/Creature(0) execution-order bug (see
  sims_brain.py), and photosensors, which read 0 (no light source), and
  nothing consumes them anyway.

Usage:
    MUJOCO_GL=egl .venv/bin/python -m rower_soccer.tools.run_sims_brain
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

import mujoco

from rower_soccer.tools.creature_format import load_creature
from rower_soccer.tools.genotype_expand import expand
from rower_soccer.tools.sims_brain import SimsBrain, UNITY_DT, spawn_instances

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CREATURE = os.path.join(REPO, "creature_configs", "TWO_ARM_ROWER.creature")
BLUEPRINT = os.path.join(REPO, "creature_configs", "two_arm_rower_blueprint.xml")
SCENE_OUT = os.path.join(REPO, "runs_v2", "sims_brain_scene.xml")
TRAJ_OUT = os.path.join(REPO, "runs_v2", "sims_brain_rower_traj.npz")
VIDEO_OUT = os.path.join(REPO, "videos", "sims_brain_rower.mp4")

SIM_SECONDS = 30.0
SUBSTEPS = 20                      # MuJoCo substeps per Unity control step
DT_CTRL = float(UNITY_DT)          # 1/30 s
DT_SUB = DT_CTRL / SUBSTEPS

LENGTH_RATIO = 2.0                 # blueprint lengths / Unity lengths
UNITY_DENSITY = 1.0                # Segment mass = 1 * volume (see docstring)
GRAVITY = 9.81 * LENGTH_RATIO      # accelerations x2 for 1:1-time similarity
TORQUE_CAP = 400.0 * LENGTH_RATIO ** 2  # Unity 400 N*m x (m L^2) scaling
FRICTION = 0.6                     # PhysX default material, combine=average
SPAWN_DROP = 1.08 * LENGTH_RATIO   # SpawnTransform vs floor height, x2
ANGULAR_DRAG = 0.05                # Rigidbody.angularDrag
SERVO_BETA = 0.5                   # dead-beat velocity servo gain fraction

FPS = 50                           # render every 6th substep: 300/6 = 50 Hz
RENDER_EVERY = int(round(1.0 / (FPS * DT_SUB)))


def build_scene():
    """Blueprint XML -> Unity-faithful FloorEnv scene (see module docstring)."""
    tree = ET.parse(BLUEPRINT)
    root = tree.getroot()

    # Unity-equivalent masses: 8 * prod(half) * rho == 1 * prod(unity_dims)
    d_geom = root.find("./default/geom")
    d_geom.set("density", str(UNITY_DENSITY / 8.0))
    d_geom.set("friction", f"{FRICTION} 0.005 0.0001")
    # Unity hinge: no armature/damping/stiffness (angularDrag applied manually)
    d_joint = root.find("./default/joint")
    d_joint.set("armature", "0")
    d_joint.set("damping", "0")
    d_joint.set("stiffness", "0")
    # PhysX hinge limits are near-rigid velocity-level constraints; the
    # blueprint's soft .04 lets 1600 N*m on sub-kg limbs overshoot by ~40 deg
    d_joint.set("solreflimit", ".02 1")
    d_joint.set("solimplimit", "0 .9 .01")

    # torques applied manually via qfrc_applied -> drop the hand-tuned motors
    for act in root.findall("actuator"):
        root.remove(act)

    opt = ET.SubElement(root, "option")
    opt.set("timestep", repr(DT_SUB))
    opt.set("gravity", f"0 0 -{GRAVITY}")
    opt.set("integrator", "implicitfast")

    worldbody = root.find("worldbody")
    seg0 = worldbody.find("./body[@name='seg0']")
    seg0.insert(0, ET.Element("freejoint", {"name": "root"}))
    # pull the tracking camera back: the creature spans ~8 m
    cam = seg0.find("./camera[@name='floating']")
    cam.set("pos", "-16 0 7")
    cam.set("xyaxes", "0 -1 0 .4 0 1")

    asset = root.find("asset")
    ET.SubElement(asset, "texture", {
        "name": "grid", "type": "2d", "builtin": "checker",
        "rgb1": ".3 .35 .3", "rgb2": ".42 .48 .42",
        "width": "512", "height": "512"})
    ET.SubElement(asset, "material", {
        "name": "grid", "texture": "grid", "texrepeat": "24 24",
        "reflectance": "0.05"})
    ET.SubElement(asset, "texture", {
        "name": "sky", "type": "skybox", "builtin": "gradient",
        "rgb1": ".55 .7 .9", "rgb2": ".2 .3 .5",
        "width": "256", "height": "256"})
    ET.SubElement(worldbody, "geom", {
        "name": "floor", "type": "plane", "pos": "0 0 0",
        "size": "500 500 1", "material": "grid",
        "friction": f"{FRICTION} 0.005 0.0001"})
    ET.SubElement(worldbody, "light", {
        "pos": "0 0 40", "dir": "0 0 -1", "directional": "true",
        "diffuse": ".85 .85 .85", "specular": ".2 .2 .2"})

    os.makedirs(os.path.dirname(SCENE_OUT), exist_ok=True)
    tree.write(SCENE_OUT)
    return SCENE_OUT


MAX_ANGVEL = 7.0  # PhysX Rigidbody.maxAngularVelocity default


def apply_angular_drag(model, data, body_ids):
    """PhysX Rigidbody.angularDrag=0.05 as tau = -c * I_world * omega, plus
    the maxAngularVelocity=7 cap as a dead-beat braking torque (see module
    docstring)."""
    vel6 = np.zeros(6)
    for b in body_ids:
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY,
                                 b, vel6, 0)
        w = vel6[:3]
        Ri = data.ximat[b].reshape(3, 3)

        def i_world(v):
            return Ri @ (model.body_inertia[b] * (Ri.T @ v))

        tau = -ANGULAR_DRAG * i_world(w)
        n = np.linalg.norm(w)
        if n > MAX_ANGVEL:
            excess = w * (1.0 - MAX_ANGVEL / n)
            tau -= i_world(excess) / DT_SUB
        mujoco.mj_applyFT(model, data, np.zeros(3), tau, data.xipos[b], b,
                          data.qfrc_applied)


def main(seconds=SIM_SECONDS, video=True, global_sign=1.0, t0=0.0):
    genotype = load_creature(CREATURE)
    phenotypes = expand(genotype)
    instances = spawn_instances(genotype)
    # cross-check: brain spawn order == geometry expansion order
    assert [(p.uid, p.parent_uid, p.type_id) for p in phenotypes] == \
           [(i.uid, None if i.parent is None else i.parent.uid, i.type_id)
            for i in instances], "spawn DFS mismatch vs genotype_expand"

    brain = SimsBrain(genotype)  # contact bug (-1), no photosensor light

    scene = build_scene()
    model = mujoco.MjModel.from_xml_path(scene)
    data = mujoco.MjData(model)
    data.qpos[2] += SPAWN_DROP           # SpawnTransform is above the floor

    # joints: name seg{parent}_to_{uid}; Unity axis sign dropped by pipeline
    joint_uids, dof_of, sign_of = [], {}, {}
    for p in phenotypes:
        if p.joint_type != "hinge":
            continue
        jname = f"seg{p.parent_uid}_to_{p.uid}"
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        assert jid >= 0, jname
        joint_uids.append(p.uid)
        dof_of[p.uid] = model.jnt_dofadr[jid]
        sign_of[p.uid] = float(np.sign(p.joint_axis[np.argmax(
            np.abs(p.joint_axis))]))
    body_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"seg{u}")
                for u in range(len(phenotypes))]
    hinge_qpos_adr = {u: model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, f"seg{p}_to_{u}")]
        for u, p in ((pp.uid, pp.parent_uid) for pp in phenotypes
                     if pp.joint_type == "hinge")}

    print(f"model: nq={model.nq} nv={model.nv} "
          f"mass={model.body_mass[body_ids].sum():.3f} kg (Unity-equal), "
          f"gravity={GRAVITY:.2f}, torque cap={TORQUE_CAP:.0f}")
    print("joint signs (unity axis):", {u: sign_of[u] for u in joint_uids})

    n_steps = int(round(seconds / DT_CTRL))
    fullM = np.zeros((model.nv, model.nv))
    mujoco.mj_forward(model, data)  # populate qM before the first servo read

    renderer = writer = None
    scene_option = None
    if video:
        renderer = mujoco.Renderer(model, height=480, width=640)
        scene_option = mujoco.MjvOption()
        scene_option.sitegroup[:] = 0        # hide site boxes/rangefinders
        import imageio
        os.makedirs(os.path.dirname(VIDEO_OUT), exist_ok=True)
        writer = imageio.get_writer(VIDEO_OUT, fps=FPS, codec="libx264",
                                    quality=8, pixelformat="yuv420p",
                                    macro_block_size=1)

    log = {k: [] for k in ("t", "qpos", "qvel", "cmd_deg", "target_radps",
                           "torque", "H", "joint_deg")}
    max_body_spin = 0.0
    vel6 = np.zeros(6)
    substep_count = 0
    for step in range(n_steps):
        t = t0 + step * DT_CTRL
        cmds = brain.fixed_update(t)                     # {uid: deg/s, Unity sign}
        targets = {u: global_sign * sign_of[u] * math_radians(cmds[u])
                   for u in joint_uids}

        applied = np.zeros(len(joint_uids))
        for _ in range(SUBSTEPS):
            data.qfrc_applied[:] = 0.0
            apply_angular_drag(model, data, body_ids)
            mujoco.mj_fullM(model, data, fullM)
            for i, u in enumerate(joint_uids):
                dof = dof_of[u]
                m_jj = fullM[dof, dof]
                tau = SERVO_BETA * m_jj * (targets[u] - data.qvel[dof]) / DT_SUB
                tau = float(np.clip(tau, -TORQUE_CAP, TORQUE_CAP))
                data.qfrc_applied[dof] += tau
                applied[i] = tau
            mujoco.mj_step(model, data)
            if writer is not None and substep_count % RENDER_EVERY == 0:
                renderer.update_scene(data, camera="floating",
                                      scene_option=scene_option)
                writer.append_data(renderer.render())
            substep_count += 1

        for b in body_ids:  # PhysX maxAngularVelocity deviation statistic
            mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY,
                                     b, vel6, 0)
            max_body_spin = max(max_body_spin, float(np.linalg.norm(vel6[:3])))

        log["t"].append(t)
        log["qpos"].append(data.qpos.copy())
        log["qvel"].append(data.qvel.copy())
        log["cmd_deg"].append([cmds[u] for u in joint_uids])
        log["target_radps"].append([targets[u] for u in joint_uids])
        log["torque"].append(applied.copy())
        log["H"].append(float(brain.neurons[0].out))
        log["joint_deg"].append([np.degrees(data.qpos[hinge_qpos_adr[u]])
                                 for u in joint_uids])

        if np.any(~np.isfinite(data.qpos)) or np.any(~np.isfinite(data.qvel)):
            raise RuntimeError(f"NaN in state at t={t:.2f}")
        if step % 150 == 0:
            jd = log["joint_deg"][-1]
            print(f"t={t:5.1f}s  root={data.qpos[0]:6.2f},{data.qpos[1]:6.2f},"
                  f"{data.qpos[2]:6.2f}  H={log['H'][-1]:6.1f}  "
                  f"joints(deg)=" + " ".join(f"{x:6.1f}" for x in jd))

    if writer is not None:
        writer.close()
    arr = {k: np.asarray(v) for k, v in log.items()}
    arr["joint_uids"] = np.asarray(joint_uids)
    arr["joint_names"] = np.asarray(
        [f"seg{p.parent_uid}_to_{p.uid}" for p in phenotypes
         if p.joint_type == "hinge"])
    arr["unity_axis_sign"] = np.asarray([sign_of[u] for u in joint_uids])
    arr["dt"] = np.asarray(DT_CTRL)
    arr["length_ratio_vs_unity"] = np.asarray(LENGTH_RATIO)
    np.savez_compressed(TRAJ_OUT, **arr)
    print(f"\nwrote {TRAJ_OUT}" + (f" and {VIDEO_OUT}" if video else ""))
    d = arr["qpos"][-1][:3] - arr["qpos"][0][:3]
    print(f"net root displacement over {seconds:.0f}s: "
          f"dx={d[0]:+.2f} dy={d[1]:+.2f} dz={d[2]:+.2f} m "
          f"(WalkingFitness locomotion plane = MuJoCo x/y)")
    print(f"max body |omega| seen: {max_body_spin:.1f} rad/s "
          f"(PhysX would clamp at 7)")
    jr = np.ptp(arr["joint_deg"], axis=0)
    print("joint angle ranges (deg):",
          {int(u): round(float(r), 1) for u, r in zip(joint_uids, jr)})


def math_radians(deg):
    return deg * np.pi / 180.0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=SIM_SECONDS)
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--global-sign", type=float, default=1.0,
                    help="flip all hinge velocity targets (PhysX motor sign "
                         "convention calibration)")
    ap.add_argument("--t0", type=float, default=0.0,
                    help="Unity Time.time at spawn (phase of oscillators)")
    a = ap.parse_args()
    main(seconds=a.seconds, video=not a.no_video, global_sign=a.global_sign,
         t0=a.t0)
