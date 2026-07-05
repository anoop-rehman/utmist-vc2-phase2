"""Latent-bottleneck actor-critic for drill training (SB3 custom policy).

Actor path (mirrors the NPMP/DeepMind-2021 structure, scaled down):

    proprio ──► proprio_enc (2x128 ELU) ─┐
                                          ├─► expert MLP (256) ─► z (deterministic, dim d)
    task ─────► task_enc  (2x128 ELU) ───┘                          │
                                                                    ▼
    proprio ────────────────────────────────► decoder MLP (3x256 ELU) ─► action head

- z is a deterministic bottleneck during joint training (stochasticity lives
  in the Gaussian action head), which keeps PPO log-probs exact. z is exposed
  via `get_z(obs)` for later BC recording.
- The decoder (+ action head) is the "low-level controller": input = raw
  proprio + z only — it never sees task observations, which is what makes it
  shareable across tasks and reusable by the football agent.
- The expert = encoders + expert MLP + z projection: per-task weights.
"""

import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy


class LatentExtractor(nn.Module):
    def __init__(self, proprio_indices, task_indices, z_dim=16,
                 enc_hidden=128, expert_hidden=256, dec_hidden=256):
        super().__init__()
        self.register_buffer("p_idx", torch.as_tensor(proprio_indices))
        self.register_buffer("t_idx", torch.as_tensor(task_indices))
        p, t = len(proprio_indices), len(task_indices)

        def mlp(i, hs, act=nn.ELU):
            layers, last = [], i
            for h in hs:
                layers += [nn.Linear(last, h), act()]
                last = h
            return nn.Sequential(*layers)

        # expert (per-task weights)
        self.proprio_enc = mlp(p, [enc_hidden, enc_hidden])
        self.task_enc = mlp(t, [enc_hidden, enc_hidden])
        self.expert = mlp(2 * enc_hidden, [expert_hidden])
        self.z_proj = nn.Linear(expert_hidden, z_dim)

        # low-level controller (shared-across-tasks weights)
        self.decoder = mlp(p + z_dim, [dec_hidden, dec_hidden, dec_hidden])

        # critic
        self.critic = mlp(p + t, [256, 256])

        self.latent_dim_pi = dec_hidden
        self.latent_dim_vf = 256

    def split(self, obs):
        return obs.index_select(-1, self.p_idx), obs.index_select(-1, self.t_idx)

    def z(self, obs):
        prop, task = self.split(obs)
        h = self.expert(torch.cat([self.proprio_enc(prop), self.task_enc(task)], -1))
        return self.z_proj(h)

    def forward_actor(self, obs):
        prop, _ = self.split(obs)
        z = self.z(obs)
        return self.decoder(torch.cat([prop, z], -1))

    def forward_critic(self, obs):
        prop, task = self.split(obs)
        return self.critic(torch.cat([prop, task], -1))

    def forward(self, obs):
        return self.forward_actor(obs), self.forward_critic(obs)


class LatentActorCriticPolicy(ActorCriticPolicy):
    """SB3 ActorCriticPolicy with the latent-bottleneck extractor.

    Pass `extractor_kwargs=dict(proprio_indices=..., task_indices=..., z_dim=...)`
    through policy_kwargs.
    """

    def __init__(self, *args, extractor_kwargs=None, **kwargs):
        self._extractor_kwargs = extractor_kwargs or {}
        kwargs["share_features_extractor"] = True
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = LatentExtractor(**self._extractor_kwargs)

    @torch.no_grad()
    def get_z(self, obs):
        """Motor intention for BC recording; obs: tensor [B, obs_dim]."""
        obs = self.obs_to_tensor(obs)[0] if not torch.is_tensor(obs) else obs
        return self.mlp_extractor.z(self.extract_features(obs))

    def decoder_state_dict(self):
        """The shareable low-level controller weights (decoder + action head)."""
        return {"decoder": self.mlp_extractor.decoder.state_dict(),
                "action_net": self.action_net.state_dict(),
                "log_std": self.log_std.detach().cpu()}

    def load_decoder_state_dict(self, sd):
        self.mlp_extractor.decoder.load_state_dict(sd["decoder"])
        self.action_net.load_state_dict(sd["action_net"])
        with torch.no_grad():
            self.log_std.copy_(sd["log_std"].to(self.log_std.device))
