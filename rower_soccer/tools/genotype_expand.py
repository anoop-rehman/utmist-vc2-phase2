"""Genotype -> phenotype expansion for UTMIST .creature files.

Reproduces the Unity-side expansion semantics, reverse-engineered from the
TWO_ARM_ROWER pair (.creature genotype vs two_arm_rower_blueprint.json
phenotype) — all 9 bodies match numerically to float precision:

- Phenotype root is genotype segment id 1 (segment id 0 is a ghost node).
- Children are instantiated depth-first, in stored connection order.
- Per-path recursion: a connection to segment type T is expanded only if the
  number of T-instances on the ancestry path (including the new one) does not
  exceed T.recursiveLimit.
- Sizes: child half-extents (Unity axes) = segment dims * cumulative scale.
- Anchor: normalized parent coords, y measured from parent bottom face:
  anchor' = (ax, ay + 0.5, az); world_pos(child) = world_pos(parent) +
  R_world(parent) @ (anchor' * parent_half_extents).
- Rotation: cumulative euler (degrees), conn euler composed onto parent.
- Reflection: a connection with reflected=True mirrors across the parent's
  local YZ plane; the parity propagates down the subtree (anchor x negated,
  joint axis negated, euler y/z negated).
- Joint axis (Unity frame): jointType 2 -> (0, parity, 0);
  jointType 3 -> (0, 0, parity). jointType 0 -> no joint (root).
  terminalOnly connections expand only where the chain would otherwise end.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation as R

JOINT_AXES = {2: np.array([0.0, 1.0, 0.0]), 3: np.array([0.0, 0.0, 1.0])}


@dataclass
class Phenotype:
    uid: int
    parent_uid: int | None
    type_id: int
    world_pos: np.ndarray          # Unity frame, JSON "Position"/"LocalPosition"
    world_euler: np.ndarray        # degrees, JSON "Rotation"
    half_size: np.ndarray          # Unity frame, JSON "Size"
    color: tuple
    joint_type: str | None         # "hinge" or None (root)
    joint_anchor: np.ndarray | None  # normalized, JSON "JointAnchorPos"
    joint_axis: np.ndarray | None
    children: list = field(default_factory=list)


def _rot(euler_deg):
    return R.from_euler("xyz", euler_deg, degrees=True)


def expand(genotype):
    """Returns list[Phenotype] in DFS/uid order (root first)."""
    segments = {s["id"]: s for s in genotype["segments"]}
    root_seg = segments[1]
    out = []
    counter = [0]

    def instantiate(seg, parent, cum_scale, parity, path_counts,
                    anchor=None, conn_euler=None):
        uid = counter[0]
        counter[0] += 1
        dims = np.array([seg["dimensionX"], seg["dimensionY"], seg["dimensionZ"]])
        half = dims * cum_scale
        color = (seg["r"], seg["g"], seg["b"])
        if parent is None:
            node = Phenotype(uid, None, seg["id"], np.zeros(3), np.zeros(3),
                             half, color, None, None, None)
        else:
            anchor_eff = np.array([parity * anchor[0], anchor[1] + 0.5, anchor[2]])
            euler_eff = np.array([conn_euler[0], parity * conn_euler[1],
                                  parity * conn_euler[2]])
            parent_rot = _rot(parent.world_euler)
            pos = parent.world_pos + parent_rot.apply(anchor_eff * parent.half_size)
            euler = (parent_rot * _rot(euler_eff)).as_euler("xyz", degrees=True)
            axis = JOINT_AXES[seg["jointType"]] * parity
            node = Phenotype(uid, parent.uid, seg["id"], pos, euler, half, color,
                             "hinge", anchor_eff, axis)
            parent.children.append(node)
        out.append(node)

        # expand connections (depth-first, stored order)
        conns = seg["connections"]
        expandable = []
        for c in conns:
            dest = segments[c["destination"]]
            new_count = path_counts.get(dest["id"], 0) + 1
            allowed = new_count <= dest["recursiveLimit"]
            expandable.append((c, dest, allowed))
        chain_continues = any(a and not c["terminalOnly"] for c, d, a in expandable)
        for c, dest, allowed in expandable:
            if not allowed:
                continue
            if c["terminalOnly"] and chain_continues:
                continue
            child_parity = -parity if c["reflected"] else parity
            child_counts = dict(path_counts)
            child_counts[dest["id"]] = child_counts.get(dest["id"], 0) + 1
            instantiate(
                dest, node, cum_scale * c["scale"], child_parity, child_counts,
                anchor=np.array([c["anchorX"], c["anchorY"], c["anchorZ"]]),
                conn_euler=np.array([c["eulerX"], c["eulerY"], c["eulerZ"]]),
            )

    instantiate(root_seg, None, 1.0, 1.0, {1: 1})
    return out


if __name__ == "__main__":
    import sys

    from rower_soccer.tools.creature_format import load_creature

    g = load_creature(sys.argv[1])
    for p in expand(g):
        print(f"uid={p.uid} parent={p.parent_uid} type={p.type_id} "
              f"pos=({p.world_pos[0]:.5f},{p.world_pos[1]:.5f},{p.world_pos[2]:.5f}) "
              f"half=({p.half_size[0]:.5f},{p.half_size[1]:.5f},{p.half_size[2]:.5f}) "
              f"euler=({p.world_euler[0]:.4f},{p.world_euler[1]:.4f},{p.world_euler[2]:.4f}) "
              + (f"anchor=({p.joint_anchor[0]:.5f},{p.joint_anchor[1]:.5f},{p.joint_anchor[2]:.5f}) "
                 f"axis={p.joint_axis.astype(int).tolist()}" if p.joint_axis is not None else "ROOT"))
