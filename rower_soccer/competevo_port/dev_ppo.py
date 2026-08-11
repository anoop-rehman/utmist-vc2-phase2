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
    def act(self, obs, noise=None):
        """`noise=(eps_scale, eps_ctrl)` replaces the internal draw with
        `mean + std * eps`, which is what `Normal.sample()` is. It exists so a
        test can drive this path and the batched `StackedDevActors` path with
        the SAME randomness and compare the actions; production passes None."""
        flag, _, _ = self.split(obs)
        is_design = flag < 0.5
        d_s, d_c = self.dists(obs)
        # `select_action` clamps the sampled design to [-1, 1] and their learner
        # then recomputes the old log-prob FROM THE STORED (clamped) action, so
        # the log-prob reported here is of the clamped sample, not the raw one.
        if noise is None:
            a_s, a_c = d_s.sample(), d_c.sample()
        else:
            a_s = d_s.mean + d_s.stddev * noise[0]
            a_c = d_c.mean + d_c.stddev * noise[1]
        a_s = a_s.clamp(-1.0, 1.0)
        logp = torch.where(is_design, d_s.log_prob(a_s).sum(-1),
                           d_c.log_prob(a_c).sum(-1))
        return self._assemble(is_design, a_s, a_c), logp, self.value(obs)


def _tower_layers(mlp, head):
    """The `[Linear, Tanh] * k + Linear` chain `_mlp(...) + head` really is.

    Asserted rather than assumed: `StackedDevActors` reimplements this chain
    with stacked weights, and it can only be equivalent if the chain is what it
    thinks it is."""
    lins = []
    for i, m in enumerate(mlp):
        if i % 2 == 0:
            assert isinstance(m, nn.Linear), f"layer {i} is {type(m).__name__}"
            lins.append(m)
        else:
            assert isinstance(m, nn.Tanh), f"layer {i} is {type(m).__name__}"
    assert len(mlp) % 2 == 0, "the mlp does not end in an activation"
    return lins + [head]


class StackedDevActors(nn.Module):
    """`n_groups x n_slots` copies of `DevActorCritic`'s ACTION path, with their
    weights stacked so all of them evaluate in ONE batched forward.

    Why this exists: stage 3 runs `blocks` opponent networks per side, and the
    stage-3 profile measured those forwards at 11.0 s of a 28.4 s iteration --
    640 `act` calls against 64, at ~17 ms each for a 38k-parameter MLP. That is
    kernel-launch and host-sync overhead, not arithmetic: the FLOPs are
    negligible at any batch size this port uses. The opponents all share an
    architecture and differ only in weights, so the whole set is one
    broadcasting `matmul` per layer over a leading `[groups, slots]` axis.

    Three things it does NOT do, each deliberate:

      * **no critic.** `CoEvoPPO` throws the opponent's value away
        (`_opponent_actions` takes `[0]`), so a third of the per-slot forward
        was dead work. Skipping it cannot change an action: the value net
        consumes no randomness and, with the module in `eval()`, its
        `RunningNorm` does not move either.
      * **no log-prob.** Opponent transitions are never trained on.
      * **no `.item()`.** `RunningNorm.forward` branches on `self.n.item()`,
        which is a device->host sync. Each `DevActorCritic.act` runs three
        normalizers, so `2 x blocks = 8` opponent forwards were 24 stalls per
        env step. Here the same branch is a `torch.where` on the device, which
        is the identical function of the same inputs (`n == 0` -> the raw
        observation, else the whitened one) with no sync.

    Weights are buffers, not parameters: nothing here is ever optimized, and
    `CoEvoPPO` overwrites them from `opp_nets` at the top of every rollout.
    """

    def __init__(self, template, n_groups, n_slots):
        super().__init__()
        self.n_groups, self.n_slots = int(n_groups), int(n_slots)
        S = self.n_groups * self.n_slots
        self.n_stacked = S
        self.design_dim = template.design_dim
        self.sim_obs_dim = template.sim_obs_dim
        self.n_motor = template.n_motor
        self.obs_dim = template.obs_dim
        self.act_dim = template.act_dim

        self._depth = {}
        for tower, mlp, head, norm in (
                ("scale", template.scale_mlp, template.scale_mean,
                 template.scale_norm),
                ("control", template.control_mlp, template.control_mean,
                 template.control_norm)):
            lins = _tower_layers(mlp, head)
            self._depth[tower] = len(lins)
            for i, lin in enumerate(lins):
                self.register_buffer(f"{tower}_w{i}",
                                     torch.zeros(S, *lin.weight.shape))
                self.register_buffer(f"{tower}_b{i}",
                                     torch.zeros(S, lin.bias.numel()))
            assert norm.demean and norm.destd, \
                "the stacked normalizer assumes RunningNorm(demean, destd)"
            self.register_buffer(f"{tower}_n", torch.zeros(S, 1))
            self.register_buffer(f"{tower}_mean",
                                 torch.zeros(S, norm.mean.numel()))
            self.register_buffer(f"{tower}_var",
                                 torch.ones(S, norm.var.numel()))
        self.clip = template.scale_norm.clip
        assert self.clip == template.control_norm.clip
        self.register_buffer("scale_log_std", torch.zeros(S, self.design_dim))
        self.register_buffer("control_log_std", torch.zeros(S, self.n_motor))

    # -- weight loading ------------------------------------------------------
    @torch.no_grad()
    def sync_from(self, nets):
        """`nets[g][k]` -> stacked row `g * n_slots + k`. Cheap enough to run
        every rollout (a few hundred tiny copies once per iteration), which is
        what keeps the stack from ever being stale."""
        assert len(nets) == self.n_groups
        for g, group in enumerate(nets):
            assert len(group) == self.n_slots
            for k, net in enumerate(group):
                s = g * self.n_slots + k
                for tower, mlp, head, norm, log_std in (
                        ("scale", net.scale_mlp, net.scale_mean,
                         net.scale_norm, net.scale_log_std),
                        ("control", net.control_mlp, net.control_mean,
                         net.control_norm, net.control_log_std)):
                    for i, lin in enumerate(_tower_layers(mlp, head)):
                        getattr(self, f"{tower}_w{i}")[s].copy_(lin.weight)
                        getattr(self, f"{tower}_b{i}")[s].copy_(lin.bias)
                    getattr(self, f"{tower}_n")[s].copy_(norm.n)
                    getattr(self, f"{tower}_mean")[s].copy_(norm.mean)
                    getattr(self, f"{tower}_var")[s].copy_(norm.var)
                    getattr(self, f"{tower}_log_std")[s].copy_(log_std)

    # -- forward -------------------------------------------------------------
    def _norm(self, x, tower):
        """`RunningNorm.forward` for every slot at once, with its `n == 0`
        identity branch as a `where` instead of a host sync."""
        G, K = self.n_groups, self.n_slots
        mean = getattr(self, f"{tower}_mean").view(G, K, 1, -1)
        var = getattr(self, f"{tower}_var").view(G, K, 1, -1)
        n = getattr(self, f"{tower}_n").view(G, K, 1, 1)
        y = x - mean
        y = y / (var.sqrt() + 1e-8)
        if self.clip:
            y = y.clamp(-self.clip, self.clip)
        return torch.where(n > 0, y, x)

    def _tower(self, x, tower):
        G, K = self.n_groups, self.n_slots
        for i in range(self._depth[tower]):
            w = getattr(self, f"{tower}_w{i}")
            b = getattr(self, f"{tower}_b{i}")
            x = torch.matmul(x, w.view(G, K, *w.shape[1:]).transpose(-1, -2))
            x = x + b.view(G, K, 1, -1)
            if i < self._depth[tower] - 1:
                x = torch.tanh(x)
        return x

    def slot_dists(self, obs):
        """`obs` `[G, M, obs_dim]` -> every slot's action distribution:
        means `[G, K, M, d]`, stds `[G, K, d]`. This is the tensor the
        equivalence gate compares against the per-slot modules."""
        G, K = self.n_groups, self.n_slots
        assert obs.shape[0] == G and obs.shape[-1] == self.obs_dim
        obs = DevActorCritic._clean(obs)
        d = self.design_dim
        scale = obs[:, None, :, 1:1 + d]                  # [G, 1, M, d]
        sim = obs[:, None, :, 1 + d:]
        s_mean = self._tower(self._norm(scale, "scale"), "scale")
        c_mean = self._tower(self._norm(sim, "control"), "control")
        s_std = (self.scale_log_std.exp() / SCALE_STD_DIVISOR).view(G, K, d)
        c_std = self.control_log_std.exp().view(G, K, self.n_motor)
        return s_mean, s_std, c_mean, c_std

    @torch.no_grad()
    def act(self, obs, slots, noise=None):
        """Actions for `[G, M, obs_dim]` observations, row `(g, m)` taken from
        slot `slots[g, m]` of group `g`.

        The gather happens on the DISTRIBUTION, not on a sample: row `m` is
        drawn once from `N(mean[slot], std[slot])` instead of drawing all `K`
        and discarding `K - 1`. Same distribution, `K`x fewer normal variates.
        `noise=(eps_scale, eps_ctrl)`, shaped like the returned blocks, replaces
        the draw with `mean + std * eps` -- the gate's seam."""
        G, K = self.n_groups, self.n_slots
        M = obs.shape[1]
        s_mean, s_std, c_mean, c_std = self.slot_dists(obs)
        si = slots.view(G, 1, M, 1)
        s_mean = s_mean.gather(1, si.expand(G, 1, M, self.design_dim))[:, 0]
        c_mean = c_mean.gather(1, si.expand(G, 1, M, self.n_motor))[:, 0]
        sj = slots.view(G, M, 1)
        s_std = s_std.gather(1, sj.expand(G, M, self.design_dim))
        c_std = c_std.gather(1, sj.expand(G, M, self.n_motor))
        if noise is None:
            a_s = torch.normal(s_mean, s_std)
            a_c = torch.normal(c_mean, c_std)
        else:
            a_s = s_mean + s_std * noise[0]
            a_c = c_mean + c_std * noise[1]
        a_s = a_s.clamp(-1.0, 1.0)
        is_design = DevActorCritic._clean(obs)[..., 0] < 0.5
        m = is_design.unsqueeze(-1).to(a_s.dtype)
        return torch.cat([a_s * m, a_c * (1.0 - m)], dim=-1)


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
