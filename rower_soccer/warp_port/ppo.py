"""Compact GPU PPO for Warp-batched envs.

Reuses rower_soccer.models.latent_policy.LatentExtractor so weights are
interchangeable with the SB3 CPU path (same module names; export helpers
below). Everything (rollout buffer, GAE, updates) stays on-GPU.

Exploration option: AR(1)-correlated action noise in z-space is emulated by
an auxiliary smoothness penalty plus an entropy floor; the paper's NPMP prior
motivates temporally-correlated exploration for gait discovery.
"""

import numpy as np
import torch
import torch.nn as nn

from rower_soccer.models.latent_policy import LatentExtractor


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, proprio_indices, task_indices,
                 z_dim=16, log_std_init=0.0):
        super().__init__()
        self.mlp_extractor = LatentExtractor(
            proprio_indices=proprio_indices, task_indices=task_indices, z_dim=z_dim)
        self.action_net = nn.Linear(self.mlp_extractor.latent_dim_pi, act_dim)
        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        self.log_std = nn.Parameter(torch.ones(act_dim) * log_std_init)

    def dist(self, obs):
        lat = self.mlp_extractor.forward_actor(obs)
        mean = self.action_net(lat)
        return torch.distributions.Normal(mean, self.log_std.exp())

    def value(self, obs):
        return self.value_net(self.mlp_extractor.forward_critic(obs)).squeeze(-1)

    @torch.no_grad()
    def act(self, obs):
        d = self.dist(obs)
        a = d.sample()
        return a, d.log_prob(a).sum(-1), self.value(obs)

    def z(self, obs):
        return self.mlp_extractor.z(obs)


class PPOTrainer:
    def __init__(self, env, ac: ActorCritic, lr=3e-4, rollout_len=64,
                 minibatches=8, epochs=4, gamma=0.99, gae_lambda=0.95,
                 clip=0.2, ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5,
                 z_smooth_coef=0.0, ent_floor=None, ent_ceil=None, device="cuda"):
        self.env, self.ac = env, ac.to(device)
        self.opt = torch.optim.Adam(ac.parameters(), lr=lr)
        self.T, self.N = rollout_len, env.n
        self.minibatches, self.epochs = minibatches, epochs
        self.gamma, self.lam, self.clip = gamma, gae_lambda, clip
        self.ent_coef, self.vf_coef = ent_coef, vf_coef
        self.max_grad_norm = max_grad_norm
        self.z_smooth_coef = z_smooth_coef
        self.ent_floor = ent_floor  # min log_std, e.g. -1.5
        # max log_std. Actions are clamped to [-1, 1] in collect(), so a std
        # much above ~1 samples almost entirely into the clamp and explores at
        # random. Without a ceiling log_std drifts up unboundedly.
        self.ent_ceil = ent_ceil
        self.device = device
        d = device
        self.obs_buf = torch.zeros(self.T, self.N, env.obs_dim, device=d)
        self.act_buf = torch.zeros(self.T, self.N, env.act_dim, device=d)
        self.logp_buf = torch.zeros(self.T, self.N, device=d)
        self.rew_buf = torch.zeros(self.T, self.N, device=d)
        self.val_buf = torch.zeros(self.T, self.N, device=d)
        self.done_buf = torch.zeros(self.T, self.N, device=d)
        self._obs = env.reset()
        self.total_steps = 0

    def collect(self):
        ep_returns = []
        for t in range(self.T):
            a, logp, v = self.ac.act(self._obs)
            obs2, rew, done = self.env.step(a.clamp(-1, 1))
            self.obs_buf[t] = self._obs
            self.act_buf[t] = a
            self.logp_buf[t] = logp
            self.rew_buf[t] = rew
            self.val_buf[t] = v
            self.done_buf[t] = float(done)
            if done:  # world-synchronized episode end
                ep_returns.append(float(rew.mean()))
                obs2 = self.env.reset()
            self._obs = obs2
        self.total_steps += self.T * self.N
        with torch.no_grad():
            last_val = self.ac.value(self._obs)
        # GAE (time-limit truncation: bootstrap through resets)
        adv = torch.zeros_like(self.rew_buf)
        gae = torch.zeros(self.N, device=self.device)
        for t in reversed(range(self.T)):
            nonterminal = 1.0  # truncation, not failure -> always bootstrap
            next_v = last_val if t == self.T - 1 else self.val_buf[t + 1]
            delta = self.rew_buf[t] + self.gamma * next_v * nonterminal - self.val_buf[t]
            gae = delta + self.gamma * self.lam * nonterminal * gae
            # reset advantage accumulation across episode boundary
            if self.done_buf[t, 0] > 0:
                gae = delta
            adv[t] = gae
        ret = adv + self.val_buf
        return adv, ret

    def update(self, adv, ret):
        B = self.T * self.N
        obs = self.obs_buf.reshape(B, -1)
        act = self.act_buf.reshape(B, -1)
        logp_old = self.logp_buf.reshape(B)
        adv_f = adv.reshape(B)
        ret_f = ret.reshape(B)
        adv_f = (adv_f - adv_f.mean()) / (adv_f.std() + 1e-8)
        idx = torch.randperm(B, device=self.device)
        mb = B // self.minibatches
        stats = {}
        for _ in range(self.epochs):
            for i in range(self.minibatches):
                j = idx[i * mb:(i + 1) * mb]
                d = self.ac.dist(obs[j])
                logp = d.log_prob(act[j]).sum(-1)
                ratio = (logp - logp_old[j]).exp()
                pg = -torch.min(
                    ratio * adv_f[j],
                    ratio.clamp(1 - self.clip, 1 + self.clip) * adv_f[j]).mean()
                v = self.ac.value(obs[j])
                vloss = ((v - ret_f[j]) ** 2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pg + self.vf_coef * vloss - self.ent_coef * ent
                if self.z_smooth_coef > 0:
                    z = self.ac.z(obs[j])
                    loss = loss + self.z_smooth_coef * (z ** 2).mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
                self.opt.step()
                if self.ent_floor is not None or self.ent_ceil is not None:
                    with torch.no_grad():
                        self.ac.log_std.clamp_(min=self.ent_floor, max=self.ent_ceil)
                stats = {"pg": float(pg), "vf": float(vloss), "ent": float(ent),
                         "std": float(self.ac.log_std.exp().mean())}
        return stats

    def train_iter(self):
        adv, ret = self.collect()
        stats = self.update(adv, ret)
        stats["ep_rew_env_mean"] = float(self.rew_buf.mean() * self.env.episode_steps)
        return stats


def save_checkpoint(trainer: "PPOTrainer", path):
    """Atomic full checkpoint (model + optimizer + step count). Overwrites:
    writes to <path>.tmp then os.replace, so a kill mid-write never corrupts
    the previous checkpoint and disk usage stays at one file."""
    import os
    tmp = path + ".tmp"
    torch.save({
        "ac": trainer.ac.state_dict(),
        "opt": trainer.opt.state_dict(),
        "total_steps": trainer.total_steps,
    }, tmp)
    os.replace(tmp, path)


def load_checkpoint(trainer: "PPOTrainer", path):
    sd = torch.load(path, map_location=trainer.device, weights_only=True)
    trainer.ac.load_state_dict(sd["ac"])
    trainer.opt.load_state_dict(sd["opt"])
    trainer.total_steps = int(sd["total_steps"])
    return trainer.total_steps


def export_sb3_compatible(ac: ActorCritic, path):
    """Save weights loadable into LatentActorCriticPolicy (CPU eval path)."""
    torch.save({
        "mlp_extractor": ac.mlp_extractor.state_dict(),
        "action_net": ac.action_net.state_dict(),
        "value_net": ac.value_net.state_dict(),
        "log_std": ac.log_std.detach().cpu(),
    }, path)


def load_into_sb3_policy(policy, path):
    sd = torch.load(path, map_location=policy.device, weights_only=True)
    policy.mlp_extractor.load_state_dict(sd["mlp_extractor"])
    policy.action_net.load_state_dict(sd["action_net"])
    policy.value_net.load_state_dict(sd["value_net"])
    with torch.no_grad():
        policy.log_std.copy_(sd["log_std"].to(policy.device))
