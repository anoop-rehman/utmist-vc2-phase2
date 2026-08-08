"""Checkpoint loading, validation, and caching for skill experts.

The network is `warp_port.ppo.ActorCritic` — the same class the drills trained,
so there is no re-implementation to drift out of sync. What is new here is the
loading *guard*.

Why the guard is the point of this file
---------------------------------------
This project has twice lost a run to a checkpoint that loaded "successfully" into
the wrong body: `load_pretrained` skips shape-mismatched tensors in silence, so a
rower run once trained around a randomly initialised frozen decoder while every
curve looked healthy. At play time the same class of bug is worse — nothing
crashes, the creature simply twitches.

So `load_policy` refuses anything it cannot prove is the right checkpoint:

  * `p_idx` / `t_idx` — the extractor's own record of which columns of the
    observation are proprio and which are task — must equal, element for element,
    the indices derived from the skill's field order and this creature's
    contract. This catches a wrong body, a wrong skill, a reordered obs, and a
    checkpoint trained on a different obs layout, all at once.
  * every weight shape must match the constructed module (`strict=True`), so a
    different z_dim or hidden width is rejected rather than partially loaded.
  * the action width must match the live env's action spec.

Caching is keyed on (realpath, mtime, size, device, layout), so four players
sharing one checkpoint load it once, and re-exporting a checkpoint mid-session
invalidates the entry instead of serving stale weights.
"""

import os
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from rower_soccer.skills.api import CheckpointMismatch
from rower_soccer.skills.contract import REPO_ROOT

__all__ = ["LatentPolicy", "load_policy", "resolve_checkpoint",
           "clear_policy_cache", "policy_cache_size"]

_CACHE = {}
_LOCK = threading.RLock()


# --- path resolution -------------------------------------------------------

def _worktree_base(path: str) -> Optional[str]:
    """If `path` is inside `<checkout>/.claude/worktrees/<name>/`, return
    `<checkout>`. Training writes checkpoints to the main checkout's gitignored
    `runs_v2/`, which an agent worktree does not have; this makes a relative
    checkpoint path resolve there anyway instead of failing confusingly."""
    marker = os.path.join(".claude", "worktrees")
    i = path.find(marker)
    return path[:i].rstrip(os.sep) if i > 0 else None


def resolve_checkpoint(path: str) -> str:
    """Absolute path to a checkpoint. Search order for a RELATIVE path:

      1. `$VC2_CHECKPOINT_ROOT`
      2. the repo root
      3. the base checkout, when running inside a `.claude/worktrees/` worktree
      4. the current working directory

    `gs://` URIs are fetched with `gcloud storage cp` into
    `$VC2_CHECKPOINT_CACHE` (default `~/.cache/vc2-checkpoints`) and cached on
    disk, so `gs://vc2-2026-checkpoints/follow_ant_v1/best.pt` works directly.
    """
    if path.startswith("gs://"):
        return _fetch_gcs(path)
    if os.path.isabs(path):
        return path

    roots = []
    env_root = os.environ.get("VC2_CHECKPOINT_ROOT")
    if env_root:
        roots.append(env_root)
    roots.append(REPO_ROOT)
    base = _worktree_base(REPO_ROOT)
    if base:
        roots.append(base)
    roots.append(os.getcwd())

    tried = []
    for r in roots:
        cand = os.path.join(r, path)
        tried.append(cand)
        if os.path.exists(cand):
            return cand
    raise CheckpointMismatch(
        f"checkpoint '{path}' not found. Tried:\n  " + "\n  ".join(tried) +
        "\nSet $VC2_CHECKPOINT_ROOT, or pass an absolute path / gs:// URI.")


def _fetch_gcs(uri: str) -> str:
    cache = os.environ.get(
        "VC2_CHECKPOINT_CACHE",
        os.path.join(os.path.expanduser("~"), ".cache", "vc2-checkpoints"))
    local = os.path.join(cache, uri[len("gs://"):])
    if os.path.exists(local):
        return local
    os.makedirs(os.path.dirname(local), exist_ok=True)
    gcloud = shutil.which("gcloud")
    if gcloud is None:
        raise CheckpointMismatch(
            f"cannot fetch {uri}: gcloud is not on PATH. Download it manually "
            f"to {local}, or pass a local path.")
    with tempfile.NamedTemporaryFile(dir=os.path.dirname(local), delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run([gcloud, "storage", "cp", uri, tmp_path],
                       check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        os.unlink(tmp_path)
        raise CheckpointMismatch(
            f"gcloud storage cp {uri} failed: "
            f"{e.stderr.decode('utf-8', 'replace')[-400:]}") from None
    os.replace(tmp_path, local)   # atomic: a killed fetch never leaves a partial
    return local


# --- the policy ------------------------------------------------------------

#: Mean action std above which a checkpoint is treated as "noise-driven": its
#: exploration noise is the same size as its whole action range, so the
#: distribution MEAN is not the behaviour the run was scored on. See
#: `LatentExpert.noise_driven` and MODE_AUTO in controller.py.
NOISE_DRIVEN_STD = 0.5


@dataclass(frozen=True)
class PolicyInfo:
    path: str
    obs_dim: int
    act_dim: int
    z_dim: int
    proprio_dim: int
    task_dim: int
    state_dependent_std: bool
    #: mean of the global action std, or None for a state-dependent head (whose
    #: std is only known per observation).
    action_std: Optional[float] = None

    def describe(self) -> str:
        std = "state-dependent" if self.action_std is None else f"{self.action_std:.2f}"
        return (f"obs={self.obs_dim} (proprio={self.proprio_dim} "
                f"task={self.task_dim}) act={self.act_dim} z={self.z_dim} "
                f"std={std}")


class LatentExpert:
    """A loaded, frozen, eval-mode expert head + shared decoder.

    Immutable and therefore safe to share between players — all per-player state
    lives in `SkillController`. `reset()` exists so a future recurrent expert
    (the paper's expert has a small LSTM) can clear its hidden state on a skill
    switch without any caller changing; today it is a no-op, and `has_state`
    tells `SkillController` whether an instance may be shared.
    """

    has_state = False

    def __init__(self, ac, info: PolicyInfo, device: str):
        self._ac = ac
        self.info = info
        self._device = device

    @property
    def noise_driven(self) -> bool:
        """True when this checkpoint's exploration noise is as large as its whole
        action range, i.e. the distribution MEAN is not the behaviour the run was
        scored on.

        `follow_ant_v1` is the motivating case: it trained with `ent_ceil = 0`,
        so `log_std` sat at its ceiling and finished at std ~= 1.0 on all eight
        joints, with actions clamped to [-1, 1]. PPO scores the SAMPLED policy,
        so its 0.997 fitness is the fitness of `clamp(mean + N(0, 1))`. Measured
        on CPU MuJoCo over 15 s to a point 3 m away: sampled reaches it (fitness
        0.944-0.996), the mean crouches and never moves (0.23). The gait is the
        noise. See `controller.MODE_AUTO`.
        """
        return (self.info.action_std is not None
                and self.info.action_std >= NOISE_DRIVEN_STD)

    # -- inference ---------------------------------------------------------
    def act(self, obs_vector, mode: str = "mean", noise=None):
        """(action, z) for one observation vector.

        Args:
          mode: `"mean"` uses the action distribution's mean — bit-exact for any
            caller. `"noise"` adds `std * noise`, where `noise` is supplied by the
            caller, so reproducibility is the CALLER's to guarantee (the
            controller derives it from a seed and a tick index, which keeps a
            recorded demo replayable). Nothing here ever touches a global RNG.
          noise: (act_dim,) array, required when `mode="noise"`.
        """
        import torch

        obs = torch.as_tensor(np.asarray(obs_vector, dtype=np.float32),
                              device=self._device).reshape(1, -1)
        with torch.no_grad():
            dist = self._ac.dist(obs)
            act = dist.mean
            if mode == "noise":
                if noise is None:
                    raise ValueError("mode='noise' requires a noise vector")
                eps = torch.as_tensor(np.asarray(noise, dtype=np.float32),
                                      device=act.device).reshape(act.shape)
                act = act + dist.stddev * eps
            elif mode != "mean":
                raise ValueError(f"unknown action mode '{mode}'")
            z = self._ac.z(obs)
        # Training clamped actions to [-1, 1] before applying them
        # (`PPOTrainer.collect`), and the soccer action spec is [-1, 1]; clamp so
        # deployment applies exactly the torque training did.
        act = act.clamp(-1.0, 1.0)
        return (act.squeeze(0).cpu().numpy().astype(np.float32),
                z.squeeze(0).cpu().numpy().astype(np.float32))

    def reset(self):
        """Clear any per-episode/per-skill internal state. No-op today."""

    def __repr__(self):
        return f"<LatentExpert {self.info.describe()} from {self.info.path}>"


# --- loading + validation --------------------------------------------------

def _flatten(sd):
    """One flat ActorCritic state_dict from either checkpoint layout."""
    from rower_soccer.warp_port.ppo import _flatten_checkpoint
    return _flatten_checkpoint(sd)


def load_policy(path: str,
                *,
                proprio_indices: Sequence[int],
                task_indices: Sequence[int],
                act_dim: int,
                device: str = "cpu",
                label: str = "") -> LatentExpert:
    """Load and validate a drill checkpoint. Cached; see module docstring."""
    real = resolve_checkpoint(path)
    st = os.stat(real)
    p_idx = np.asarray(list(proprio_indices), dtype=np.int64)
    t_idx = np.asarray(list(task_indices), dtype=np.int64)
    key = (os.path.realpath(real), st.st_mtime_ns, st.st_size, device,
           p_idx.tobytes(), t_idx.tobytes(), int(act_dim))
    with _LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            return hit

    expert = _load_uncached(real, p_idx, t_idx, act_dim, device, label or path)
    with _LOCK:
        _CACHE[key] = expert
    return expert


def _load_uncached(real, p_idx, t_idx, act_dim, device, label):
    import torch
    from rower_soccer.warp_port.ppo import ActorCritic

    raw = torch.load(real, map_location="cpu", weights_only=True)
    if "plain_state_dict" in raw:
        raise CheckpointMismatch(
            f"{label}: {real} is a SimpleActorCritic export (no latent "
            "bottleneck, no shared decoder). SkillController only runs "
            "expert->z->decoder checkpoints.")
    try:
        flat = _flatten(raw)
    except (KeyError, TypeError) as e:
        raise CheckpointMismatch(
            f"{label}: {real} is not a recognised checkpoint "
            f"(expected keys from warp_port.ppo.export_sb3_compatible or a "
            f"resume checkpoint with 'ac'); {e}") from None

    problems = []

    # 1. the obs layout the checkpoint itself recorded
    ck_p = flat.get("mlp_extractor.p_idx")
    ck_t = flat.get("mlp_extractor.t_idx")
    if ck_p is None or ck_t is None:
        problems.append("  checkpoint has no p_idx/t_idx buffers — cannot verify "
                        "its observation layout, refusing to guess")
    else:
        ck_p = ck_p.cpu().numpy().astype(np.int64)
        ck_t = ck_t.cpu().numpy().astype(np.int64)
        if ck_p.shape != p_idx.shape or not np.array_equal(ck_p, p_idx):
            problems.append(_layout_diff("proprio", ck_p, p_idx))
        if ck_t.shape != t_idx.shape or not np.array_equal(ck_t, t_idx):
            problems.append(_layout_diff("task", ck_t, t_idx))

    # 2. action width
    ck_act = int(flat["action_net.weight"].shape[0])
    if ck_act != int(act_dim):
        problems.append(
            f"  action width: checkpoint {ck_act}, live creature {int(act_dim)}")

    if problems:
        raise CheckpointMismatch(
            f"{label}: {real} does not match this creature/skill.\n"
            + "\n".join(problems) +
            "\nA checkpoint is dimensioned by the body AND the observation "
            "layout it trained on. Check the creature kind and the skill's "
            "`fields` tuple in rower_soccer/skills/registry.py.")

    z_dim = int(flat["mlp_extractor.z_proj.weight"].shape[0])
    state_dependent = any(k.startswith("log_std_net") for k in flat)

    ac = ActorCritic(obs_dim=len(p_idx) + len(t_idx), act_dim=ck_act,
                     proprio_indices=p_idx.tolist(), task_indices=t_idx.tolist(),
                     z_dim=z_dim, state_dependent_std=state_dependent)
    try:
        ac.load_state_dict(flat, strict=True)
    except RuntimeError as e:
        raise CheckpointMismatch(
            f"{label}: {real} has weights that do not fit the architecture "
            f"(z_dim={z_dim}, state_dependent_std={state_dependent}):\n{e}"
        ) from None
    ac.to(device).eval()
    for p in ac.parameters():
        p.requires_grad_(False)

    action_std = None
    if not state_dependent:
        action_std = float(flat["log_std"].exp().mean())
    info = PolicyInfo(path=real, obs_dim=len(p_idx) + len(t_idx), act_dim=ck_act,
                      z_dim=z_dim, proprio_dim=len(p_idx), task_dim=len(t_idx),
                      state_dependent_std=state_dependent, action_std=action_std)
    return LatentExpert(ac, info, device)


def _layout_diff(which, ck, want) -> str:
    if ck.shape != want.shape:
        return (f"  {which} block: checkpoint is {ck.size} wide, this creature's "
                f"layout is {want.size} wide")
    bad = int(np.argmax(ck != want))
    return (f"  {which} block: same width ({ck.size}) but different positions — "
            f"first difference at slot {bad}: checkpoint column {ck[bad]}, "
            f"derived column {want[bad]}. The observation fields are in a "
            "different order than the checkpoint trained on.")


def clear_policy_cache():
    with _LOCK:
        _CACHE.clear()


def policy_cache_size() -> int:
    with _LOCK:
        return len(_CACHE)
