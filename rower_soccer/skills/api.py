"""Public data contract for `rower_soccer.skills` — the WS3/WS4 interface.

This module holds ONLY plain data types and pure geometry. It imports numpy and
nothing else, so WS4 (the play server) and, next sprint, the BC/self-play code
can build against it without pulling in torch, mujoco, or dm_control.

The one-paragraph version of the whole package
----------------------------------------------
A trained drill expert is a function of a *drill* observation vector — a fixed,
ordered concatenation of proprioception and a task block, in the exact order the
warp drill env emitted during training. The soccer env produces a *different*
observation (a dict, different key names, no task block, sensors unscaled). A
`SkillController` reconstructs the drill vector from a soccer observation, runs
the skill's expert head to a latent `z`, and pushes `z` through the shared frozen
decoder to joint torques. `soccer_bridge.py` proved this trick for the worm/
dm_control drill; this package generalises it to any creature, any skill, and
the warp-trained checkpoints the ant sprint actually uses.

What a caller must supply per player per tick: a `PlayerFrame`.

    PlayerFrame(obs, root_pos, root_mat)

`obs` is one player's dm_soccer observation dict (the `i`-th entry of
`timestep.observation`). `root_pos`/`root_mat` are the player root body's world
pose, which dm_soccer does NOT put in the observation (`creature.py`'s
`proprioception` deliberately drops `absolute_root_pos`/`absolute_root_mat`
because the low-level decoder must never see global position or heading). The
pose is needed only to turn a *world* target — where the human clicked — into the
egocentric target the expert was trained on. `skills.soccer.SoccerFrameSource`
reads it straight off `physics`; `PlayerFrame.from_physics` does the same for a
single walker.

**WS4, record `root_pos` and `root_mat` in the demo file.** They are 12 floats
per player per tick and they are the only thing in a `PlayerFrame` that is not
already in the observation. Without them a recorded demo cannot be replayed
through a SkillController.
"""

from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "PlayerFrame", "SkillCommand", "SkillOutput",
    "to_ego_xy", "ego3_to_world", "world_to_ego3", "vec_to_ego3",
    "SkillError", "UnknownSkill", "SkillUnavailable", "CheckpointMismatch",
    "ObservationContractError",
]


# --- errors ----------------------------------------------------------------
# All loudly-failing paths raise one of these. Nothing in this package returns a
# zero vector or a random policy when something does not line up: two runs in
# this project were lost to a silent body/shape mismatch that "worked".

class SkillError(RuntimeError):
    """Base class for every failure this package raises."""


class UnknownSkill(SkillError):
    """`skill_id` is not in the registry."""


class SkillUnavailable(SkillError):
    """A registered skill has no usable checkpoint for this creature yet."""


class CheckpointMismatch(SkillError):
    """A checkpoint's dimensions/obs layout disagree with the live creature."""


class ObservationContractError(SkillError):
    """The supplied observation is missing a field, or a field is the wrong width."""


# --- data ------------------------------------------------------------------

@dataclass(frozen=True)
class PlayerFrame:
    """Everything the controller needs about ONE player at ONE tick.

    Attributes:
      obs: the player's dm_soccer observation dict. Values may carry dm_control's
        leading singleton buffer dim (`(1, 27)`); every reader here ravels first,
        so both `strip_singleton_obs_buffer_dim` settings work.
      root_pos: (3,) world position of the walker's root body (`seg0`).
      root_mat: (3, 3) world<-body rotation of the root body, row-major, i.e.
        `world_vec = root_mat @ body_vec`. Accepts a flat (9,) too. This is
        MuJoCo's `xmat` (the BODY frame), not `ximat` (the inertial frame) —
        see `ball_pos` below for why the distinction bites.
      ball_pos: (3,) WORLD position of the ball, optional. Required by any skill
        that observes the ball (`dribble`/`kick`/`shoot`) or targets it
        (`scripted`).

        Why not read it out of `obs["ball_ego_position"]`? Because dm_soccer
        builds that sensor with `objtype='body', reftype='body'`, which in MuJoCo
        means the INERTIAL frames, while the warp drill envs compute their
        `ball_ego` in the BODY frame (`_to_ego3` uses `xmat`). For the ant those
        two frames differ by a full axis permutation — measured `|ximat - xmat|`
        = 1.09, because MuJoCo orders principal axes of inertia and the ant's
        torso is nearly symmetric, so the ordering is arbitrary. Feeding the game
        observation straight into a drill-trained expert would hand it a permuted
        ball vector. Taking the world position and applying the drill's own
        transform sidesteps the whole question.
      ball_vel: (3,) WORLD linear velocity of the ball, optional; same reasoning.
    """

    obs: Mapping[str, np.ndarray]
    root_pos: np.ndarray
    root_mat: np.ndarray
    ball_pos: Optional[np.ndarray] = None
    ball_vel: Optional[np.ndarray] = None

    def __post_init__(self):
        object.__setattr__(
            self, "root_pos",
            np.asarray(self.root_pos, dtype=np.float64).ravel()[:3])
        object.__setattr__(
            self, "root_mat",
            np.asarray(self.root_mat, dtype=np.float64).ravel().reshape(3, 3))
        for name in ("ball_pos", "ball_vel"):
            v = getattr(self, name)
            if v is not None:
                object.__setattr__(
                    self, name, np.asarray(v, dtype=np.float64).ravel()[:3])

    @property
    def root_xy(self) -> np.ndarray:
        return self.root_pos[:2]

    @classmethod
    def from_physics(cls, obs, physics, walker) -> "PlayerFrame":
        """Build a frame for `walker` (a `creature.Creature`) from live physics."""
        b = physics.bind(walker.root_body)
        return cls(obs=obs, root_pos=np.array(b.xpos), root_mat=np.array(b.xmat))


@dataclass(frozen=True)
class SkillCommand:
    """What a human (or, next sprint, a high-level policy) asks for.

    `target_xy` is a WORLD point on the pitch in metres. Skills that derive their
    own target (`scripted` chases the ball) ignore it and may be given None.
    """

    skill_id: str
    target_xy: Optional[Tuple[float, float]] = None

    def with_target(self, target_xy) -> "SkillCommand":
        return SkillCommand(self.skill_id, _as_xy(target_xy))


@dataclass(frozen=True)
class SkillOutput:
    """One tick of controller output. This is also the demo-recording record.

    Attributes:
      action: (act_dim,) float32 joint torques in [-1, 1], ready for
        `env.step([...])`.
      z: (z_dim,) float32 latent motor intention, or None for skills with no
        expert head (`idle`). PIPELINE_V2's BC stage trains pi(z | game_obs), so
        this is the label WS4's demo file must store.
      skill_id: the skill that produced the action.
      target_xy: the world target actually used this tick (for `scripted` this is
        the ball, not the commanded point).
      obs_vector: (obs_dim,) float32 — the exact reconstructed drill observation
        fed to the expert. Recorded so a replay can be checked bit-for-bit
        without re-deriving it.
    """

    action: np.ndarray
    z: Optional[np.ndarray]
    skill_id: str
    target_xy: Optional[Tuple[float, float]]
    obs_vector: np.ndarray = field(repr=False)


# --- geometry --------------------------------------------------------------
# These reproduce the warp drill envs bit-for-bit. Do not "fix" them.

def to_ego_xy(root_pos, root_mat, world_xy) -> np.ndarray:
    """Project a world XY point into the root frame, exactly as the drill env's
    `_to_ego` does: dot with the UNNORMALISED xy-projections of the body's x
    (forward) and y (left) axes.

    The axes are deliberately not renormalised. `rot[:2, 0]` shortens as the body
    pitches, and the policy trained against that shortening; normalising here
    would hand it an observation it never saw. (`soccer_bridge._to_ego` and
    `warp_port/follow_env._to_ego` are the two references.)
    """
    root_pos = np.asarray(root_pos, dtype=np.float64).ravel()
    root_mat = np.asarray(root_mat, dtype=np.float64).reshape(3, 3)
    fwd, left = root_mat[:2, 0], root_mat[:2, 1]
    d = np.asarray(world_xy, dtype=np.float64).ravel()[:2] - root_pos[:2]
    return np.array([float(d @ fwd), float(d @ left)], dtype=np.float32)


def world_to_ego3(root_pos, root_mat, world_xyz) -> np.ndarray:
    """Full 3-D `R^T (x - p)` — the transform MuJoCo's `framepos` sensor with a
    reference body applies, and the one behind dm_soccer's `*_ego_position`."""
    root_pos = np.asarray(root_pos, dtype=np.float64).ravel()[:3]
    root_mat = np.asarray(root_mat, dtype=np.float64).reshape(3, 3)
    return (root_mat.T @ (np.asarray(world_xyz, dtype=np.float64).ravel()[:3]
                          - root_pos)).astype(np.float32)


def vec_to_ego3(root_mat, world_vec) -> np.ndarray:
    """Rotate a world-frame VECTOR (no translation) into the root body frame —
    `warp_port/worm_env_base._vec_to_ego3`, used for the ball's velocity."""
    root_mat = np.asarray(root_mat, dtype=np.float64).reshape(3, 3)
    return (root_mat.T @ np.asarray(world_vec, dtype=np.float64).ravel()[:3]
            ).astype(np.float32)


def ego3_to_world(root_pos, root_mat, ego_xyz) -> np.ndarray:
    """Inverse of `world_to_ego3`: turn an egocentric observation (e.g.
    `ball_ego_position`) back into a world point, so it can be used as a target."""
    root_pos = np.asarray(root_pos, dtype=np.float64).ravel()[:3]
    root_mat = np.asarray(root_mat, dtype=np.float64).reshape(3, 3)
    return root_pos + root_mat @ np.asarray(ego_xyz, dtype=np.float64).ravel()[:3]


# --- small helpers ---------------------------------------------------------

def _as_xy(xy) -> Optional[Tuple[float, float]]:
    if xy is None:
        return None
    a = np.asarray(xy, dtype=np.float64).ravel()
    if a.size < 2:
        raise ValueError(f"target_xy must have 2 components, got {a.size}")
    return (float(a[0]), float(a[1]))


def ravel_obs(obs: Mapping[str, np.ndarray], key: str) -> np.ndarray:
    """Fetch `key` from an observation dict as a flat float64 array, raising a
    contract error (not a KeyError) when it is absent."""
    try:
        v = obs[key]
    except KeyError:
        raise ObservationContractError(
            f"observation is missing '{key}'. Available keys: "
            f"{sorted(obs)}") from None
    return np.asarray(v, dtype=np.float64).ravel()


def keys_present(obs: Mapping[str, np.ndarray], keys: Sequence[str]):
    return [k for k in keys if k not in obs]
