"""Their `DevPolicy` / `DevValue`, vectorized: two heads behind one stage mask.

`custom/models/dev_actor.py` is written for a Python list of per-sample
observations: `forward` loops over the batch, buckets each sample by its stage
flag, runs the scale head on one bucket and the control head on the other, and
`get_log_prob` scatters the two log-prob columns back into one. That structure
is exactly a batched masked computation, which is what this is -- both heads run
on the whole batch and the mask picks the answer. Their loop and this are the
same function; theirs just pays Python for it.

The two heads, from `config/run-to-goal-devants-v0.yaml`:

  scale head    RunningNorm(20) -> MLP [64, 64] tanh -> Linear(20)
                output weights x1.0 (NOT x0.1 -- the control head is the one
                that gets damped), log_std init 0, and the distribution is built
                with **std / 5** (dev_actor.py:91), so the design is sampled with
                sigma 0.2 and then clamped to [-1, 1]. Input is ONLY the current
                scale vector: the design policy never sees the sim state.
  control head  RunningNorm(31) -> MLP [64, 128, 64] tanh -> Linear(8)
                output weights x0.1, log_std init 0. Input is ONLY sim_obs
                (`use_entire_obs: false`).
  critic        RunningNorm(52) -> MLP [64, 64, 64] tanh -> Linear(1), on the
                FULL observation including the stage flag and the design.

The design action is stored as step 0 of the trajectory with reward 0 and
mask 1; that is how the genome gets PPO credit, through GAE bootstrapping from
the rewards of the episode it produced (port map section 4.2).
"""

import torch
import torch.nn as nn

from rower_soccer.competevo_port.ppo import (RunningNorm, SelfPlayPPO, _mlp)

# `termination_epoch: 1000` in the DEV config -- five times the fixed-morph
# ants' 200. Their epoch is `min_batch_size = 50,000` agent-steps.
DEV_CURRICULUM_STEPS = 1000 * 50_000

# `DevPolicy.forward` (dev_actor.py:91): the scale distribution's std is the
# exponentiated log_std DIVIDED BY 5. With log_std init 0 that is sigma = 0.2,
# so a fresh design policy explores a fifth of the genome box, not all of it.
SCALE_STD_DIVISOR = 5.0


class DevActorCritic(nn.Module):
    """Stage-masked two-head actor + full-observation critic.

    Observation layout (`DevAnt._get_obs`): `[flag(1) | scale(20) | sim(31)]`.
    Action layout (`DevPolicy.select_action`): `[design(20) | motor(8)]`, with
    the INACTIVE block zeroed -- their env ignores it and their `get_log_prob`
    never reads it, so zeros keep the buffer unambiguous.
    """

    def __init__(self, design_dim=20, sim_obs_dim=31, n_motor=8,
                 scale_hidden=(64, 64), control_hidden=(64, 128, 64),
                 critic_hidden=(64, 64, 64), log_std_init=0.0):
        super().__init__()
        self.design_dim, self.sim_obs_dim, self.n_motor = (design_dim,
                                                           sim_obs_dim, n_motor)
        self.obs_dim = 1 + design_dim + sim_obs_dim
        self.act_dim = design_dim + n_motor

        self.scale_norm = RunningNorm(design_dim)
        self.scale_mlp, d = _mlp(design_dim, scale_hidden)
        self.scale_mean = nn.Linear(d, design_dim)
        self.scale_log_std = nn.Parameter(torch.ones(design_dim) * log_std_init)

        self.control_norm = RunningNorm(sim_obs_dim)
        self.control_mlp, d = _mlp(sim_obs_dim, control_hidden)
        self.control_mean = nn.Linear(d, n_motor)
        self.control_log_std = nn.Parameter(torch.ones(n_motor) * log_std_init)

        self.vf_norm = RunningNorm(self.obs_dim)
        self.vf, d = _mlp(self.obs_dim, critic_hidden)
        self.value_net = nn.Linear(d, 1)

        # Their `init_fc_weights` per head, with the asymmetry intact:
        # dev_actor.py:29-30 scales the scale head's weights by 1 and
        # dev_actor.py:50-51 scales the control head's by 0.1; the critic head is
        # 0.1 (dev_critic.py:29-30). Biases zeroed everywhere.
        for head, w in ((self.scale_mean, 1.0), (self.control_mean, 0.1),
                        (self.value_net, 0.1)):
            head.weight.data.mul_(w)
            head.bias.data.mul_(0.0)

    @staticmethod
    def _clean(obs):
        return torch.nan_to_num(obs, nan=0.0, posinf=1e3,
                                neginf=-1e3).clamp(-1e3, 1e3)

    def split(self, obs):
        obs = self._clean(obs)
        return (obs[..., 0], obs[..., 1:1 + self.design_dim],
                obs[..., 1 + self.design_dim:])

    def dists(self, obs):
        _, scale, sim = self.split(obs)
        s_mean = self.scale_mean(self.scale_mlp(self.scale_norm(scale)))
        c_mean = self.control_mean(self.control_mlp(self.control_norm(sim)))
        return (torch.distributions.Normal(
                    s_mean, self.scale_log_std.exp() / SCALE_STD_DIVISOR),
                torch.distributions.Normal(c_mean, self.control_log_std.exp()))

    def value(self, obs):
        return self.value_net(self.vf(self.vf_norm(self._clean(obs)))).squeeze(-1)

    def _assemble(self, is_design, a_scale, a_ctrl):
        """`select_action`: the design block for design-stage rows, the motor
        block for the rest, zeros in the other half."""
        m = is_design.unsqueeze(-1).to(a_scale.dtype)
        return torch.cat([a_scale * m, a_ctrl * (1.0 - m)], dim=-1)

    def log_prob(self, obs, action):
        """One column, per their `get_log_prob` scatter: the active head's
        log-probability of the block that head produced."""
        flag, _, _ = self.split(obs)
        is_design = flag < 0.5
        d_s, d_c = self.dists(obs)
        lp_s = d_s.log_prob(action[..., :self.design_dim]).sum(-1)
        lp_c = d_c.log_prob(action[..., self.design_dim:]).sum(-1)
        return torch.where(is_design, lp_s, lp_c)

    def entropy(self, obs):
        flag, _, _ = self.split(obs)
        d_s, d_c = self.dists(obs)
        return torch.where(flag < 0.5, d_s.entropy().sum(-1),
                           d_c.entropy().sum(-1))

    def mean_action(self, obs):
        flag, _, _ = self.split(obs)
        d_s, d_c = self.dists(obs)
        return self._assemble(flag < 0.5, d_s.mean.clamp(-1.0, 1.0), d_c.mean)

    @torch.no_grad()
    def act(self, obs):
        flag, _, _ = self.split(obs)
        is_design = flag < 0.5
        d_s, d_c = self.dists(obs)
        # `select_action` clamps the sampled design to [-1, 1] and their learner
        # then recomputes the old log-prob FROM THE STORED (clamped) action, so
        # the log-prob reported here is of the clamped sample, not the raw one.
        a_s = d_s.sample().clamp(-1.0, 1.0)
        a_c = d_c.sample()
        logp = torch.where(is_design, d_s.log_prob(a_s).sum(-1),
                           d_c.log_prob(a_c).sum(-1))
        return self._assemble(is_design, a_s, a_c), logp, self.value(obs)


class DevSelfPlayPPO(SelfPlayPPO):
    """`SelfPlayPPO` with the two-head log-probability. Everything else -- GAE,
    their mask semantics, globally standardized advantages, the two optimizers,
    the exploration curriculum -- is unchanged, because none of it depends on
    which head produced the action."""

    def __init__(self, env, ac, curriculum_steps=DEV_CURRICULUM_STEPS, **kw):
        super().__init__(env, ac, curriculum_steps=curriculum_steps, **kw)

    def _param_groups(self):
        pi = [p for n, p in self.ac.named_parameters()
              if n.startswith(("scale_", "control_"))]
        vf = [p for n, p in self.ac.named_parameters()
              if n.startswith(("vf", "value_net"))]
        return pi, vf

    def _logp_entropy(self, obs, act):
        return self.ac.log_prob(obs, act), self.ac.entropy(obs)
