"""The BC policy: game observation -> z -> the FROZEN shared decoder -> action.

    from rower_soccer.bc.model import BCConfig, BCPolicy
    pol = BCPolicy(BCConfig(obs_keys=ds.meta["obs_keys"], obs_sizes=ds.meta["obs_sizes"]))

Why this architecture (measured, not assumed)
---------------------------------------------
The demo corpus records, per tick, the game observation, the latent `z` the drill
expert emitted, and the action that was applied. Three facts about the current
corpus decide the design, and all three are checked by `tests/test_model.py`
against the real checkpoints:

 1. **Every v3 drill shares one decoder, byte for byte.** `follow_ant_final_frozen`,
    `dribble_ant_v3`, `kick_ant_v3` and `shoot_ant_v3` hold `decoder.*` and
    `action_net.*` identical to `runs_v2/_decoder_ant_final.pt` (max abs diff
    0.0). Only `log_std` differs, and the demos were all recorded in MODE_MEAN,
    which never touches it.

 2. **The game observation contains the decoder's whole input.** Columns
    `bodies_pos | body_height | joints_pos | joints_vel | sensors_accelerometer |
    sensors_gyro | sensors_velocimeter | touch_sensors | world_zaxis` of the
    186-wide dm_soccer observation are bit-identical to the first 65 entries of
    the recorded expert vector (measured: max diff 0.0 on all 34,261 v3 rows).

 3. Therefore ``clamp(action_net(decoder([proprio, z])))`` **reproduces the
    recorded action exactly** — measured max abs error 1.6e-6 over the same
    34,261 rows, i.e. float32 storage noise.

So behaviour cloning here is not "fit a policy to actions". The motor controller
is already correct and already known; the ONLY unknown is the map
``game observation -> z``. That is a 186 -> 16 regression whose target is exactly
achievable, instead of a 186 -> 8 regression that has to re-learn locomotion.
Freezing the decoder is not a convenience, it is what makes the small corpus
tractable — and it makes the BC prior arithmetically compatible with the drill
checkpoints, so unit 1f's KL anchor compares two policies that differ only in
their expert head.

Which loss
----------
`loss="action"` (default) trains the expert head by backpropagating the ACTION
error through the frozen decoder. `loss="latent"` regresses z directly.
`loss="both"` sums them. Action is the default because:

  * it optimises the quantity that is actually applied — an error in a z
    direction the decoder ignores costs nothing, and one it amplifies costs a
    lot, which plain z-MSE cannot know;
  * it is defined on mirror-augmented rows, whose `z` is NaN by design;
  * fact 3 above means it has the same global optimum as the z loss on
    unmirrored rows, so nothing is given up.

`loss="latent"` is one config flag away and is the better probe when you want to
know whether the head has found the expert's own latent code rather than some
other preimage of the same action.

What this policy structurally cannot learn
------------------------------------------
Worth knowing before reading a val number. In the recorded demos the action is a
function of the expert observation, which contains the HUMAN'S COMMANDED TARGET
and the skill they picked. Given those, the mapping is exact (fact 3). The BC
policy is given the game observation, which does not contain them: a human's
click is not a physical quantity anyone can see. So the residual this network
cannot remove is precisely the part of the demonstrator's intent that the world
state does not imply, and MSE resolves that ambiguity by averaging the modes —
"go left" and "go right" become "stand still". Measured on the first real run:
the policy's per-actuator std is below the demos' on all eight joints and it
saturates 18% of commands against the corpus's 47%, i.e. visibly timid, which is
what mode-averaging looks like at the actuator.

The remedy is not a bigger head. It is the OTHER half of the plan's step 3 — a
high-level policy `game obs -> (skill, target)` — with this network as the
low-level half it drives. `dataset.py` already carries `skill` and `target` as
per-sample columns for exactly that. Until that exists, the honest description
of this checkpoint is "a prior over plausible motor output", not "a player".

Observation normalisation, and why it is not a runtime step
-----------------------------------------------------------
The game observation mixes touch sensors (~1e-4) with pitch coordinates (~15 m),
so the encoders need their input whitened or the small features never move. But
the decoder must keep receiving RAW proprio — it was trained on raw proprio and
it is frozen. So normalisation is applied to the ENCODER inputs only
(`_NormLatentExtractor`), and at export time it is folded exactly into the first
layer of `proprio_enc`, `task_enc` and `critic`:

    W (x - m)/s + b  ==  (W/s) x + (b - (W/s) m)

`export()` therefore writes a plain `warp_port.ppo.ActorCritic` state dict that
eats the raw observation, with no normalisation wrapper to forget at deploy time
— the failure mode this project has already paid for twice. `test_model.py`
checks fold-equivalence to 1e-5 on random observations.

Checkpoint format
-----------------
`export()` writes exactly what `warp_port.ppo.export_sb3_compatible` writes
(`mlp_extractor` / `action_net` / `value_net` / `log_std`), plus one extra
`bc_meta` JSON string. `skills.policy.load_policy` loads it as-is, provided the
caller passes the `proprio_indices` / `task_indices` this module derived — see
`load_policy_kwargs()`. What it is NOT is a registered `SkillSpec`: the BC
policy's task block is the whole game observation (ball, teammates, opponents,
goal corners, stats), and `skills/fields.py` has no builders for those keys. A
BC prior is also not a skill — it takes no target. Wiring it into
`SkillController` would mean teaching `fields.py` the game observation, which is
unit 1f's business, not this one's.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from rower_soccer.models.latent_policy import LatentExtractor
from rower_soccer.skills.registry import PROPRIO_V1
from rower_soccer.warp_port.ppo import ActorCritic, SimpleActorCritic

__all__ = ["BCConfig", "BCPolicy", "BCRunner", "obs_layout", "split_indices",
           "load_frozen_decoder", "load_bc_checkpoint", "load_policy_kwargs",
           "SCHEMA_NAME", "SCHEMA_VERSION", "DEFAULT_DECODER", "LOSSES", "ARCHS"]

SCHEMA_NAME = "rower_soccer.bc.model"
SCHEMA_VERSION = 1

#: The shared low-level controller every v3 ant drill holds (verified identical
#: in `tests/test_model.py::test_registry_checkpoints_share_the_frozen_decoder`).
DEFAULT_DECODER = "runs_v2/_decoder_ant_final.pt"

LOSSES = ("action", "latent", "both")
ARCHS = ("latent", "plain")

#: Below this an observation column is treated as constant and left unscaled —
#: dividing by a std that is really float32 noise turns a dead feature into a
#: loud one.
MIN_STD = 1e-3


# --- observation layout ----------------------------------------------------

def obs_layout(obs_keys: Sequence[str], obs_sizes: Sequence[int],
               drop_keys: Sequence[str] = ()) -> Tuple[List[str], List[int], np.ndarray]:
    """(kept keys, kept sizes, column indices into the dataset's obs vector).

    `drop_keys` exists for one experiment in particular: `prev_action` is in the
    dm_soccer observation, and a BC policy that can see the previous action can
    score beautifully on held-out agreement by copying it forward while learning
    nothing about the task (causal confusion, Ortega et al. 2021 / de Haan 2019).
    Dropping it is a one-flag ablation; `eval.py` reports both.
    """
    drop = set(drop_keys)
    unknown = drop - set(obs_keys)
    if unknown:
        raise ValueError(f"drop_keys names keys the observation does not have: "
                         f"{sorted(unknown)}")
    keys, sizes, cols, i = [], [], [], 0
    for k, n in zip(obs_keys, obs_sizes):
        n = int(n)
        if k not in drop:
            keys.append(k)
            sizes.append(n)
            cols.append(np.arange(i, i + n))
        i += n
    if not keys:
        raise ValueError("drop_keys dropped every observation key")
    return keys, sizes, np.concatenate(cols).astype(np.int64)


def split_indices(keys: Sequence[str], sizes: Sequence[int],
                  proprio_fields: Sequence[str] = PROPRIO_V1
                  ) -> Tuple[List[int], List[int]]:
    """(proprio_indices, task_indices) into the assembled BC observation.

    Proprio comes out in `PROPRIO_V1` order — NOT the observation's own key
    order — because that tuple IS the frozen decoder's input contract, and the
    decoder receives `obs.index_select(-1, p_idx)`. Everything else is task, in
    observation order. Task here is the whole rest of the game observation: the
    ball, both opponents, the teammate, the goal corners, the stats block.
    """
    off, i = {}, 0
    for k, n in zip(keys, sizes):
        off[k] = (i, i + int(n))
        i += int(n)
    missing = [f for f in proprio_fields if f not in off]
    if missing:
        raise ValueError(
            f"the observation is missing proprio fields {missing}; the frozen "
            f"decoder cannot be fed. Keys present: {sorted(off)}")
    p_idx: List[int] = []
    for f in proprio_fields:
        a, b = off[f]
        p_idx.extend(range(a, b))
    taken = set(p_idx)
    t_idx = [c for c in range(i) if c not in taken]
    return p_idx, t_idx


# --- config ----------------------------------------------------------------

@dataclass
class BCConfig:
    """Everything that changes the network, and nothing that changes training."""

    obs_keys: List[str] = field(default_factory=list)
    obs_sizes: List[int] = field(default_factory=list)
    drop_keys: List[str] = field(default_factory=list)
    act_dim: int = 8
    z_dim: int = 16
    arch: str = "latent"
    loss: str = "action"
    #: weight on the z term when `loss="both"` (the action term has weight 1).
    z_loss_weight: float = 1.0
    freeze_decoder: bool = True
    decoder_path: str = DEFAULT_DECODER
    proprio_fields: List[str] = field(default_factory=lambda: list(PROPRIO_V1))
    enc_hidden: int = 128
    expert_hidden: int = 256
    dec_hidden: int = 256
    plain_hidden: int = 256

    def __post_init__(self):
        if self.arch not in ARCHS:
            raise ValueError(f"arch must be one of {ARCHS}, got {self.arch!r}")
        if self.loss not in LOSSES:
            raise ValueError(f"loss must be one of {LOSSES}, got {self.loss!r}")
        if self.arch == "plain":
            if self.loss != "action":
                raise ValueError("arch='plain' has no latent bottleneck, so only "
                                 "loss='action' is defined for it")
            if self.freeze_decoder:
                # Nothing to freeze; say so rather than pretending.
                self.freeze_decoder = False
        self.obs_keys = list(self.obs_keys)
        self.obs_sizes = [int(s) for s in self.obs_sizes]
        self.drop_keys = list(self.drop_keys)
        self.proprio_fields = list(self.proprio_fields)

    # -- derived -----------------------------------------------------------
    def layout(self):
        """(keys, sizes, source columns, p_idx, t_idx) for this config."""
        keys, sizes, cols = obs_layout(self.obs_keys, self.obs_sizes, self.drop_keys)
        p_idx, t_idx = split_indices(keys, sizes, self.proprio_fields)
        return keys, sizes, cols, p_idx, t_idx

    @property
    def obs_dim(self) -> int:
        keys, sizes, _ = obs_layout(self.obs_keys, self.obs_sizes, self.drop_keys)
        return int(sum(sizes))

    @classmethod
    def from_dataset(cls, ds, **kw) -> "BCConfig":
        return cls(obs_keys=list(ds.meta["obs_keys"]),
                   obs_sizes=list(ds.meta["obs_sizes"]),
                   act_dim=int(ds.meta["act_dim"]),
                   z_dim=int(ds.meta["z_dim"]), **kw)


# --- the normalised extractor ----------------------------------------------

class _NormLatentExtractor(LatentExtractor):
    """`LatentExtractor` with whitening on the ENCODER inputs only.

    The decoder keeps its raw proprio (it is frozen and was trained on raw
    proprio); the critic is whitened because it is trained here from scratch.
    `fold_into` removes this module entirely by absorbing the affine into the
    first Linear of each encoder — see the module docstring.
    """

    def __init__(self, proprio_indices, task_indices, z_dim=16, **kw):
        super().__init__(proprio_indices, task_indices, z_dim=z_dim, **kw)
        p, t = len(proprio_indices), len(task_indices)
        self.register_buffer("p_mean", torch.zeros(p))
        self.register_buffer("p_scale", torch.ones(p))
        self.register_buffer("t_mean", torch.zeros(t))
        self.register_buffer("t_scale", torch.ones(t))

    def _norm(self, prop, task):
        return ((prop - self.p_mean) / self.p_scale,
                (task - self.t_mean) / self.t_scale)

    def z(self, obs):
        prop, task = self.split(obs)
        np_, nt_ = self._norm(prop, task)
        h = self.expert(torch.cat([self.proprio_enc(np_), self.task_enc(nt_)], -1))
        return self.z_proj(h)

    def forward_actor(self, obs):
        prop, _ = self.split(obs)
        return self.decoder(torch.cat([prop, self.z(obs)], -1))

    def forward_critic(self, obs):
        prop, task = self.split(obs)
        np_, nt_ = self._norm(prop, task)
        return self.critic(torch.cat([np_, nt_], -1))

    # -- export ------------------------------------------------------------
    @torch.no_grad()
    def folded_state_dict(self) -> Dict[str, torch.Tensor]:
        """This extractor's weights as a plain `LatentExtractor` state dict."""
        sd = {k: v.detach().clone() for k, v in self.state_dict().items()
              if k not in ("p_mean", "p_scale", "t_mean", "t_scale")}
        _fold_linear(sd, "proprio_enc.0", self.p_mean, self.p_scale)
        _fold_linear(sd, "task_enc.0", self.t_mean, self.t_scale)
        _fold_linear(sd, "critic.0",
                     torch.cat([self.p_mean, self.t_mean]),
                     torch.cat([self.p_scale, self.t_scale]))
        return sd


def _fold_linear(sd: Dict[str, torch.Tensor], prefix: str,
                 mean: torch.Tensor, scale: torch.Tensor):
    """In place: make `sd[prefix]` consume raw x instead of (x - mean)/scale."""
    w = sd[f"{prefix}.weight"]
    b = sd[f"{prefix}.bias"]
    w_new = w / scale.to(w.device, w.dtype)
    sd[f"{prefix}.weight"] = w_new
    sd[f"{prefix}.bias"] = b - w_new @ mean.to(w.device, w.dtype)


class _NormActorCritic(ActorCritic):
    """`ActorCritic` whose extractor whitens the encoder inputs."""

    def __init__(self, obs_dim, act_dim, proprio_indices, task_indices, **kw):
        super().__init__(obs_dim, act_dim, proprio_indices, task_indices, **kw)
        z_dim = int(self.mlp_extractor.z_proj.out_features)
        self.mlp_extractor = _NormLatentExtractor(
            proprio_indices=proprio_indices, task_indices=task_indices, z_dim=z_dim,
            enc_hidden=self.mlp_extractor.proprio_enc[0].out_features,
            expert_hidden=self.mlp_extractor.expert[0].out_features,
            dec_hidden=self.mlp_extractor.latent_dim_pi)


class _NormSimpleActorCritic(SimpleActorCritic):
    """Plain-MLP control arm: no bottleneck, no decoder, whitened input."""

    def __init__(self, obs_dim, act_dim, hidden=256, **kw):
        super().__init__(obs_dim, act_dim, hidden=hidden, **kw)
        self.register_buffer("o_mean", torch.zeros(obs_dim))
        self.register_buffer("o_scale", torch.ones(obs_dim))

    def dist(self, obs):
        x = (ActorCritic._clean(obs) - self.o_mean) / self.o_scale
        return torch.distributions.Normal(self.action_net(self.pi(x)),
                                          self.log_std.exp())

    def value(self, obs):
        x = (ActorCritic._clean(obs) - self.o_mean) / self.o_scale
        return self.value_net(self.vf(x)).squeeze(-1)

    @torch.no_grad()
    def folded_state_dict(self):
        sd = {k: v.detach().clone() for k, v in self.state_dict().items()
              if k not in ("o_mean", "o_scale")}
        _fold_linear(sd, "pi.0", self.o_mean, self.o_scale)
        _fold_linear(sd, "vf.0", self.o_mean, self.o_scale)
        return sd


# --- the frozen decoder ----------------------------------------------------

def load_frozen_decoder(path: str = DEFAULT_DECODER, device: str = "cpu"):
    """(decoder state dict, action_net state dict, log_std) from a checkpoint.

    Accepts anything `warp_port.ppo._flatten_checkpoint` accepts — an
    `export_sb3_compatible` export, a resume checkpoint, or the published
    `_decoder_ant_final.pt`. The path is resolved with `skills.policy`'s search
    order, so a relative path works from a worktree and a `gs://` URI works at
    all.
    """
    from rower_soccer.skills.policy import resolve_checkpoint
    from rower_soccer.warp_port.ppo import _flatten_checkpoint

    real = resolve_checkpoint(path)
    flat = _flatten_checkpoint(torch.load(real, map_location=device,
                                          weights_only=True))
    dec = {k[len("mlp_extractor.decoder."):]: v for k, v in flat.items()
           if k.startswith("mlp_extractor.decoder.")}
    act = {k[len("action_net."):]: v for k, v in flat.items()
           if k.startswith("action_net.")}
    if not dec or not act:
        raise ValueError(f"{real}: no decoder/action_net weights in this "
                         f"checkpoint (keys: {sorted(flat)[:8]}...)")
    return dec, act, flat.get("log_std"), real


# --- the policy ------------------------------------------------------------

class BCPolicy(nn.Module):
    """The trainable BC policy. Owns an `ActorCritic`-shaped network."""

    def __init__(self, cfg: BCConfig, device: str = "cpu"):
        super().__init__()
        self.cfg = cfg
        keys, sizes, cols, p_idx, t_idx = cfg.layout()
        self.obs_keys, self.obs_sizes = keys, sizes
        self.register_buffer("src_cols", torch.as_tensor(cols, dtype=torch.long))
        self._cols_are_identity = bool(np.array_equal(cols, np.arange(len(cols))))
        self.p_idx, self.t_idx = p_idx, t_idx
        obs_dim = int(sum(sizes))
        self.obs_dim = obs_dim
        self.decoder_source = None

        if cfg.arch == "latent":
            self.ac = _NormActorCritic(
                obs_dim=obs_dim, act_dim=cfg.act_dim,
                proprio_indices=p_idx, task_indices=t_idx, z_dim=cfg.z_dim)
            if cfg.decoder_path:
                dec, act, log_std, real = load_frozen_decoder(cfg.decoder_path)
                self.ac.mlp_extractor.decoder.load_state_dict(dec)
                self.ac.action_net.load_state_dict(act)
                if log_std is not None:
                    with torch.no_grad():
                        self.ac.log_std.copy_(log_std)
                self.decoder_source = real
            elif cfg.freeze_decoder:
                raise ValueError("freeze_decoder=True with no decoder_path would "
                                 "freeze random weights")
            if cfg.freeze_decoder:
                for p in self.ac.mlp_extractor.decoder.parameters():
                    p.requires_grad_(False)
                for p in self.ac.action_net.parameters():
                    p.requires_grad_(False)
            self.ac.log_std.requires_grad_(False)
        else:
            self.ac = _NormSimpleActorCritic(obs_dim=obs_dim, act_dim=cfg.act_dim,
                                             hidden=cfg.plain_hidden)
            self.ac.log_std.requires_grad_(False)
        self.to(device)

    # -- introspection -----------------------------------------------------
    @property
    def frozen_parameter_names(self) -> Tuple[str, ...]:
        return tuple(n for n, p in self.ac.named_parameters() if not p.requires_grad)

    def trainable_parameters(self):
        return [p for p in self.ac.parameters() if p.requires_grad]

    def n_trainable(self) -> int:
        return int(sum(p.numel() for p in self.trainable_parameters()))

    # -- data plumbing -----------------------------------------------------
    def take_obs(self, obs: torch.Tensor) -> torch.Tensor:
        """Dataset observation [.., O_full] -> the BC observation [.., obs_dim]."""
        if self._cols_are_identity and obs.shape[-1] == self.obs_dim:
            return obs
        return obs.index_select(-1, self.src_cols.to(obs.device))

    @torch.no_grad()
    def set_normalization(self, obs: torch.Tensor, eps: float = MIN_STD):
        """Fit the encoder whitening on `obs` (already in BC layout)."""
        obs = obs.to(next(self.parameters()).device, torch.float32)
        mean = obs.mean(0)
        std = obs.std(0)
        std = torch.where(std < eps, torch.ones_like(std), std)
        if self.cfg.arch == "plain":
            self.ac.o_mean.copy_(mean)
            self.ac.o_scale.copy_(std)
            return
        ex = self.ac.mlp_extractor
        ex.p_mean.copy_(mean[ex.p_idx])
        ex.p_scale.copy_(std[ex.p_idx])
        ex.t_mean.copy_(mean[ex.t_idx])
        ex.t_scale.copy_(std[ex.t_idx])

    # -- inference ---------------------------------------------------------
    def forward(self, obs: torch.Tensor):
        """(unclamped action mean, z or None) for a BC-layout observation batch."""
        action = self.ac.dist(obs).mean
        z = self.ac.z(obs) if self.cfg.arch == "latent" else None
        return action, z

    @torch.no_grad()
    def act(self, obs: torch.Tensor) -> torch.Tensor:
        """The deployed action: the mean, clamped to the actuator range."""
        return self.ac.dist(obs).mean.clamp(-1.0, 1.0)

    # -- loss --------------------------------------------------------------
    def losses(self, obs, action, z=None, z_mask=None, weight=None) -> Dict[str, torch.Tensor]:
        """All terms, plus `total`. `obs` is already in BC layout.

        The action term compares the UNCLAMPED prediction with the recorded
        (already clamped) action. Its optimum for a saturated target is exactly
        +/-1, so nothing is lost, and unlike clamping the prediction it leaves a
        gradient everywhere — a prediction stuck at +5 against a -1 target would
        otherwise never move.
        """
        pred, z_pred = self(obs)
        per = (pred - action) ** 2
        if weight is not None:
            w = weight.reshape(-1, 1)
            act_loss = (per * w).sum() / (w.sum() * per.shape[1]).clamp(min=1e-8)
        else:
            act_loss = per.mean()
        out = {"action": act_loss}
        if z_pred is not None and z is not None:
            m = torch.ones(z.shape[0], dtype=torch.bool, device=z.device) \
                if z_mask is None else z_mask
            if m.any():
                zp = ((z_pred[m] - z[m]) ** 2).mean()
            else:
                zp = act_loss.new_zeros(())
            out["latent"] = zp
            out["latent_frac"] = m.float().mean().detach()
        if self.cfg.loss == "action":
            out["total"] = out["action"]
        elif self.cfg.loss == "latent":
            if "latent" not in out:
                raise ValueError("loss='latent' needs z targets")
            out["total"] = out["latent"]
        else:
            out["total"] = out["action"] + self.cfg.z_loss_weight * out["latent"]
        return out

    # -- export ------------------------------------------------------------
    @torch.no_grad()
    def calibrate_log_std(self, obs, action, floor: float = 1e-3):
        """Set `log_std` to the per-actuator RMS residual on `obs`/`action`.

        BC fits a mean; the Gaussian head's std is then free. Setting it to the
        residual makes the exported checkpoint's SAMPLED policy an honest model
        of the demonstrator's spread, which is what unit 1f's KL anchor will
        actually be measured against. Costs one forward pass.
        """
        pred = self.ac.dist(obs).mean
        rms = ((pred - action) ** 2).mean(0).sqrt().clamp(min=floor)
        self.ac.log_std.copy_(rms.log())
        return rms

    @torch.no_grad()
    def folded_state_dict(self) -> Dict[str, torch.Tensor]:
        """Weights as a plain (un-normalised) module, eating the raw observation."""
        return self.ac.mlp_extractor.folded_state_dict() if self.cfg.arch == "latent" \
            else self.ac.folded_state_dict()

    def meta(self, extra: Optional[dict] = None) -> dict:
        m = dict(schema=SCHEMA_NAME, version=SCHEMA_VERSION,
                 config=asdict(self.cfg), obs_keys=list(self.obs_keys),
                 obs_sizes=list(self.obs_sizes),
                 src_cols=[int(c) for c in self.src_cols.cpu().numpy()],
                 proprio_indices=list(self.p_idx), task_indices=list(self.t_idx),
                 obs_dim=int(self.obs_dim),
                 decoder_source=self.decoder_source,
                 frozen=list(self.frozen_parameter_names),
                 n_trainable=self.n_trainable(),
                 critic_trained=False,
                 normalization="folded into the first encoder layer")
        if extra:
            m.update(extra)
        return m

    @torch.no_grad()
    def export(self, path: str, extra: Optional[dict] = None) -> str:
        """Write an `export_sb3_compatible`-shaped checkpoint + `bc_meta`.

        Atomic (tmp + rename), so a killed trainer never leaves a half file that
        loads as garbage.
        """
        path = str(path)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        if self.cfg.arch == "plain":
            out = {"plain_state_dict": self.folded_state_dict(),
                   "bc_meta": json.dumps(self.meta(extra), sort_keys=True)}
        else:
            sd = self.folded_state_dict()
            out = {
                "mlp_extractor": sd,
                "action_net": {k[len("action_net."):]: v.detach().cpu()
                               for k, v in self.ac.named_parameters()
                               if k.startswith("action_net.")},
                "value_net": {k[len("value_net."):]: v.detach().cpu()
                              for k, v in self.ac.named_parameters()
                              if k.startswith("value_net.")},
                "log_std": self.ac.log_std.detach().cpu(),
                "bc_meta": json.dumps(self.meta(extra), sort_keys=True),
            }
            out["mlp_extractor"] = {k: v.detach().cpu() for k, v in sd.items()}
        tmp = path + ".tmp"
        torch.save(out, tmp)
        os.replace(tmp, path)
        return path


# --- loading for evaluation / deployment -----------------------------------

def load_bc_checkpoint(path: str, device: str = "cpu"):
    """(module, meta dict) for a checkpoint written by `BCPolicy.export`.

    The module is a plain `warp_port.ppo.ActorCritic` (or `SimpleActorCritic`)
    eating the RAW observation — no normaliser to remember.
    """
    raw = torch.load(str(path), map_location=device, weights_only=True)
    if "bc_meta" not in raw:
        raise ValueError(f"{path}: not a BC checkpoint (no 'bc_meta'); it may be "
                         "a drill checkpoint, which has no observation layout "
                         "recorded in a form this loader can use")
    meta = json.loads(str(raw["bc_meta"]))
    if meta.get("schema") != SCHEMA_NAME:
        raise ValueError(f"{path}: bc_meta schema {meta.get('schema')!r}")
    if int(meta.get("version", 0)) != SCHEMA_VERSION:
        raise ValueError(f"{path}: bc checkpoint v{meta.get('version')}, this "
                         f"reader is v{SCHEMA_VERSION}")
    cfg = BCConfig(**meta["config"])
    if cfg.arch == "plain":
        ac = SimpleActorCritic(obs_dim=int(meta["obs_dim"]), act_dim=cfg.act_dim,
                               hidden=cfg.plain_hidden)
        ac.load_state_dict(raw["plain_state_dict"])
    else:
        from rower_soccer.warp_port.ppo import _flatten_checkpoint
        ac = ActorCritic(obs_dim=int(meta["obs_dim"]), act_dim=cfg.act_dim,
                         proprio_indices=meta["proprio_indices"],
                         task_indices=meta["task_indices"], z_dim=cfg.z_dim)
        ac.load_state_dict(_flatten_checkpoint(raw), strict=True)
    ac.to(device).eval()
    for p in ac.parameters():
        p.requires_grad_(False)
    return ac, meta


def load_policy_kwargs(meta: dict) -> dict:
    """The kwargs `skills.policy.load_policy` needs for a BC checkpoint.

        from rower_soccer.skills.policy import load_policy
        _, meta = load_bc_checkpoint(path)
        expert = load_policy(path, **load_policy_kwargs(meta))

    `load_policy` derives nothing itself — the caller supplies the observation
    layout and it verifies the checkpoint's own `p_idx`/`t_idx` against it. That
    check is exactly what makes this safe: the layout in `bc_meta` and the one
    baked into the weights have to agree or nothing loads.
    """
    return dict(proprio_indices=list(meta["proprio_indices"]),
                task_indices=list(meta["task_indices"]),
                act_dim=int(meta["config"]["act_dim"]))


class BCRunner:
    """Deploy-side wrapper: a dm_soccer observation dict in, an action out.

    Holds the key order the policy was trained on, so a live env whose
    observation dict is in a different order (or has extra keys) is assembled
    correctly rather than silently mis-sliced.
    """

    def __init__(self, path: str, device: str = "cpu"):
        self.ac, self.meta = load_bc_checkpoint(path, device)
        self.device = device
        self.path = str(path)
        self.obs_keys = list(self.meta["obs_keys"])
        self.obs_sizes = [int(s) for s in self.meta["obs_sizes"]]
        self.obs_dim = int(self.meta["obs_dim"])
        self._checked = False

    def obs_vector(self, obs_dict) -> np.ndarray:
        missing = [k for k in self.obs_keys if k not in obs_dict]
        if missing:
            raise ValueError(
                f"{self.path}: the live observation is missing keys the policy "
                f"trained on: {missing[:6]}")
        v = np.concatenate([np.asarray(obs_dict[k], np.float32).ravel()
                            for k in self.obs_keys])
        if not self._checked:
            if v.size != self.obs_dim:
                raise ValueError(
                    f"{self.path}: assembled observation is {v.size} wide, the "
                    f"policy wants {self.obs_dim}. Per-key sizes differ from "
                    "training — a different creature or a different dm_soccer.")
            self._checked = True
        return v

    def action(self, obs_dict) -> np.ndarray:
        v = torch.as_tensor(self.obs_vector(obs_dict), device=self.device)[None, :]
        with torch.no_grad():
            a = self.ac.dist(v).mean.clamp(-1.0, 1.0)
        return a[0].cpu().numpy().astype(np.float32)

    def z(self, obs_dict) -> Optional[np.ndarray]:
        if not hasattr(self.ac, "z"):
            return None
        v = torch.as_tensor(self.obs_vector(obs_dict), device=self.device)[None, :]
        with torch.no_grad():
            return self.ac.z(v)[0].cpu().numpy().astype(np.float32)

    def reset(self):
        """No per-episode state today; here so the game loop never has to know."""

    def __repr__(self):
        return (f"<BCRunner {os.path.basename(self.path)} obs={self.obs_dim} "
                f"act={self.meta['config']['act_dim']} "
                f"arch={self.meta['config']['arch']}>")
