"""Per-creature observation contract, derived from the creature XML itself.

Everything a skill's observation layout depends on is a property of the body:
how many bodies, how many actuated joints, how many touch sensors. Rather than
hardcode `65` for the ant and `29` for the worm, `contract_for()` compiles the
creature's MJCF through the SAME `build_creature_scene()` the warp trainer used
and reads the widths off the compiled model. A checkpoint is then validated
against these numbers, so pointing an ant checkpoint at a worm (or at an ant XML
that has quietly grown a leg) fails on load with a readable message instead of
producing a plausible-looking policy that controls nothing.

This module imports mujoco and numpy only — no dm_control, no torch.
"""

import os
import re
import threading
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import numpy as np

from rower_soccer.skills.api import ObservationContractError

# Repo root = two levels up from rower_soccer/skills/.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Creature kind -> XML basename, for the kinds whose file is not `<kind>.xml`.
# `ant.xml` needs no entry; it is resolved by the default rule.
_XML_ALIASES = {
    "rower": "two_arm_rower_blueprint.xml",
    "worm": "three_seg_worm.xml",
}


def creature_xml_path(kind: str) -> str:
    """Absolute path to a creature kind's MJCF. Accepts a path directly."""
    if os.path.sep in kind or kind.endswith(".xml"):
        return os.path.abspath(kind)
    name = _XML_ALIASES.get(kind, f"{kind}.xml")
    path = os.path.join(REPO_ROOT, "creature_configs", name)
    if not os.path.exists(path):
        raise ObservationContractError(
            f"no MJCF for creature kind '{kind}' (looked for {path}). Known "
            f"kinds: {sorted(_XML_ALIASES) + ['ant']}, or pass an explicit path.")
    return path


@dataclass(frozen=True)
class CreatureContract:
    """Observation/action widths for one creature, read off its compiled model.

    `proprio_dim` reproduces `warp_port/follow_env.py`'s formula
    `3*nbody + 1 + nu + nu + 9 + n_touch + 3` — the shared decoder's entire input
    contract. Ant: 27+1+8+8+9+9+3 = 65. Worm: 9+1+2+2+9+3+3 = 29.
    """

    kind: str
    xml_path: str
    n_bodies: int
    n_joints: int          # == number of actuators; joints_pos and joints_vel each
    n_touch: int
    act_dim: int
    body_names: Tuple[str, ...]
    joint_names: Tuple[str, ...]

    @property
    def proprio_dim(self) -> int:
        return (3 * self.n_bodies + 1 + 2 * self.n_joints + 9 + self.n_touch + 3)

    def describe(self) -> str:
        return (f"{self.kind}: bodies={self.n_bodies} joints={self.n_joints} "
                f"touch={self.n_touch} -> proprio={self.proprio_dim} "
                f"act={self.act_dim}")


_CACHE: Dict[str, CreatureContract] = {}
_LOCK = threading.Lock()


def contract_for(kind: str) -> CreatureContract:
    """Compile (once, cached) the creature and read its observation widths."""
    with _LOCK:
        hit = _CACHE.get(kind)
        if hit is not None:
            return hit

    xml = creature_xml_path(kind)
    # Lazy: pulls mujoco and compiles the pitch scene. `scene.py` imports only
    # mujoco/numpy at module level, so this stays warp-free and CPU-safe.
    from rower_soccer.warp_port.scene import build_creature_scene, touch_slices

    model, meta = build_creature_scene(xml)
    body_names = tuple(model.body(i).name for i in meta.body_ids)
    joint_names = tuple(
        model.joint(j).name for j in range(model.njnt)
        if int(model.joint(j).qposadr[0]) in set(meta.joint_qpos))
    c = CreatureContract(
        kind=kind,
        xml_path=xml,
        n_bodies=len(meta.body_ids),
        n_joints=len(meta.joint_qpos),
        n_touch=len(touch_slices(meta)),
        act_dim=int(meta.nu),
        body_names=body_names,
        joint_names=joint_names,
    )
    if c.n_joints != c.act_dim:
        # The drill obs uses `nu` for both joints_pos and joints_vel widths, which
        # only holds for a body whose every actuated joint is 1-DOF. Ball joints
        # would break it silently.
        raise ObservationContractError(
            f"{kind}: {c.n_joints} observable joint DOFs but {c.act_dim} actuators. "
            "The drill proprio contract assumes one 1-DOF joint per actuator.")
    with _LOCK:
        _CACHE[kind] = c
    return c


def clear_contract_cache():
    with _LOCK:
        _CACHE.clear()


# --- ordering assumptions, asserted rather than assumed ---------------------

_SEG = re.compile(r"seg(\d+)")


def check_soccer_obs_widths(contract: CreatureContract,
                            obs: Mapping[str, np.ndarray],
                            required) -> None:
    """Verify a live dm_soccer observation dict agrees with `contract`.

    Catches the failure this project has hit twice: a controller built for one
    body driven against another. The widths are cheap and total — if
    `bodies_pos` is 27 wide the walker really does have 9 bodies.
    """
    from rower_soccer.skills.fields import field_width, get_field

    problems = []
    for name in required:
        spec = get_field(name)
        if spec.obs_key is None:
            continue
        want = field_width(name, contract)
        if spec.obs_key not in obs:
            problems.append(f"  missing '{spec.obs_key}' (needed for field {name})")
            continue
        got = int(np.asarray(obs[spec.obs_key]).size)
        if got != want:
            problems.append(
                f"  '{spec.obs_key}': observation is {got} wide, "
                f"{contract.kind} contract says {want}")
    if problems:
        raise ObservationContractError(
            "the live observation does not match the creature contract "
            f"({contract.describe()}):\n" + "\n".join(problems) +
            "\nWrong creature in this player slot?")
