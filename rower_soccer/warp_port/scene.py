"""Standalone MJCF scene builder for Warp drill training.

The scene contains only physics-relevant entities (creature + floor). Drill
targets are kinematic abstractions handled in torch by the env — they never
touch the physics. Solver options mirror the project arena.xml where Warp
supports them (elliptic cone, dt 0.0025; noslip_iterations is not supported
by mujoco_warp and is dropped — one source of CPU/GPU behavior deltas the
transfer eval watches for).
"""

from dataclasses import dataclass

import mujoco
import numpy as np

_BASE_XML = """
<mujoco model="warp_drill">
  <option cone="elliptic" timestep="0.0025"/>
  <worldbody>
    <geom name="floor" type="plane" size="60 60 1" friction="1 0.5 0.5"/>
  </worldbody>
</mujoco>
"""


@dataclass
class SceneMeta:
    root_body: int          # body id of creature root
    body_ids: list          # creature body ids in uid order (root first)
    qpos_root: int          # start of freejoint qpos (7 numbers)
    qvel_root: int
    joint_qpos: list        # hinge qpos addresses
    joint_qvel: list
    sensor_slices: dict     # name -> (start, dim) into sensordata
    nu: int
    spawn_z: float


def build_creature_scene(creature_xml_path, prefix="c-"):
    """Returns (mujoco.MjModel, SceneMeta)."""
    spec = mujoco.MjSpec.from_string(_BASE_XML)
    sub = mujoco.MjSpec.from_file(creature_xml_path)
    frame = spec.worldbody.add_frame()
    attached_root = frame.attach_body(sub.worldbody.bodies[0], prefix, "")
    attached_root.add_freejoint()
    model = spec.compile()

    root_name = f"{prefix}seg0"
    root_body = model.body(root_name).id
    body_ids = [model.body(i).id for i in range(model.nbody)
                if model.body(i).name.startswith(prefix)]

    # freejoint address
    root_jnt = None
    joint_qpos, joint_qvel = [], []
    for j in range(model.njnt):
        jnt = model.joint(j)
        if jnt.type[0] == mujoco.mjtJoint.mjJNT_FREE:
            root_jnt = j
        elif jnt.name.startswith(prefix):
            joint_qpos.append(int(jnt.qposadr[0]))
            joint_qvel.append(int(jnt.dofadr[0]))

    sensor_slices = {}
    for s in range(model.nsensor):
        sen = model.sensor(s)
        name = sen.name.removeprefix(prefix)
        sensor_slices[name] = (int(sen.adr[0]), int(sen.dim[0]))

    # spawn z = root body's default z in the attached frame (converter sets
    # it so the creature rests on the floor)
    spawn_z = float(model.body(root_name).pos[2]) or float(
        mujoco.MjModel.from_xml_path(creature_xml_path).body("seg0").pos[2])

    meta = SceneMeta(
        root_body=root_body,
        body_ids=body_ids,
        qpos_root=int(model.jnt_qposadr[root_jnt]),
        qvel_root=int(model.jnt_dofadr[root_jnt]),
        joint_qpos=joint_qpos,
        joint_qvel=joint_qvel,
        sensor_slices=sensor_slices,
        nu=model.nu,
        spawn_z=spawn_z,
    )
    return model, meta
