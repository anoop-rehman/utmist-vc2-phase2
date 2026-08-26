"""2h Option A: one independent actor-critic per (side, SLOT).

2f/2g run ONE net per side for both teammates, telling them apart with a 2-dim
role one-hot appended to the design head's input (§15: front-vs-back genome SMD
0.110 -> 0.833 once that head can see the role). That works only because the two
teammates are the same creature and therefore the same observation and action
width.

Option A drops the shared net: each slot gets its own. Role stops being an input
and becomes the identity of the network, which is strictly more expressive than
a one-hot -- every weight can specialise, not just the design head's first
layer. It is also the only option that needs no masking, no padding semantics
and no cross-creature weight sharing, so an ant slot and a spider slot can never
end up sharing a weight that means different things in the two bodies.

What it costs, stated plainly: **the 2g measurement does not transfer.** There
is no shared design head left to make role-visible, so "does the design head
seeing the role produce specialisation" is not a question you can ask of this
architecture -- specialisation is unconditionally available. The homogeneous
arms of the 2h sweep are therefore NOT a reproduction of 2g; they are a
different architecture on the same task, which is exactly why they need running
as their own baseline before the heterogeneous arms mean anything.

--------------------------------------------------------------------------
The interface this has to satisfy
--------------------------------------------------------------------------
`CoEvoPPO.collect` drives a whole side in one call --
`lr.ac.act(obs[w][:, mine])` with `mine` a lane pair like [0, 2] -- and
`DevSelfPlayPPO` additionally needs `value`, `log_prob`, `entropy`,
`mean_action`, and a `named_parameters()` whose names let `_param_groups`
split policy from value parameters. This class provides all of them by
dispatching over the lane axis, so nothing in the trainer changes.

Per-slot observation widths are supported (`obs_cols`) for the heterogeneous
case, where each net gathers only the columns that are real for its creature;
on a homogeneous team those are the identity and the gather is a no-op.
"""

import torch
import torch.nn as nn

from rower_soccer.competevo_port.team_policy import TeamActorCritic


class SlotTeamActorCritic(nn.Module):
    """`nn.ModuleList` of per-slot `TeamActorCritic`s, dispatched by lane.

    `obs` everywhere is `[..., L, D]` with L the number of slots this side
    owns (2 in 2v2) and D the env's padded observation width. Slot `l` reads
    `obs[..., l, obs_cols[l]]`.
    """

    def __init__(self, n_agents=4, design_dims=(20, 20), n_motors=(8, 8),
                 obs_cols=None, act_dim=None, **kw):
        super().__init__()
        assert len(design_dims) == len(n_motors), "one width pair per slot"
        self.n_slots = len(design_dims)
        self.design_dims = tuple(design_dims)
        self.n_motors = tuple(n_motors)
        # role_in_design is meaningless here and is refused rather than
        # silently ignored: with one net per slot the role is the network, and
        # accepting the flag would imply a 2g-style comparison this
        # architecture cannot support.
        assert not kw.pop("role_in_design", False), (
            "role_in_design has no meaning under Option A -- the role IS the "
            "net. Use TeamActorCritic for the 2g architecture.")
        self.nets = nn.ModuleList([
            TeamActorCritic(n_agents=n_agents, design_dim=d, n_motor=m, **kw)
            for d, m in zip(design_dims, n_motors)])
        # Padded action width, shared across slots so the env's [n, A, act_dim]
        # tensor stays rectangular. Each slot writes its own leading columns of
        # the design block and of the motor block.
        self.max_design = max(design_dims)
        self.act_dim = act_dim or (self.max_design + max(n_motors))
        if obs_cols is None:
            obs_cols = [None] * self.n_slots
        for i, c in enumerate(obs_cols):
            self.register_buffer(f"cols_{i}",
                                 None if c is None
                                 else torch.as_tensor(c, dtype=torch.long),
                                 persistent=False)

    # -- attributes `TeamPolicyObsEnv` reads off the policy ----------------
    # It builds the POLICY observation from the SCENE observation, which means
    # it needs the widths and the per-agent `expand_obs`. Delegated to slot 0,
    # which is correct whenever the slots are shape-identical (every
    # homogeneous composition). A mixed composition needs a per-slot wrapper;
    # `expand_obs` asserts rather than quietly using slot 0's widths.
    @property
    def obs_dim(self):
        return self.nets[0].obs_dim if hasattr(self.nets[0], "obs_dim") else (
            1 + self.design_dims[0] + self.nets[0].env_sim_dim + 2)

    @property
    def design_dim(self):
        return self.design_dims[0]

    @property
    def env_sim_dim(self):
        return self.nets[0].env_sim_dim

    @property
    def scale_log_std(self):
        """Concatenated across slots, so the logged `design_std` is the mean
        over BOTH nets rather than slot 0's alone -- under Option A the two
        slots are free to diverge in exploration scale, and that divergence is
        one of the things worth watching."""
        return torch.cat([n.scale_log_std.reshape(-1) for n in self.nets])

    @property
    def control_log_std(self):
        return torch.cat([n.control_log_std.reshape(-1) for n in self.nets])

    def expand_obs(self, obs, agent_idx):
        assert len(set(self.design_dims)) == 1 and len(set(self.n_motors)) == 1, (
            "expand_obs via the shared wrapper assumes shape-identical slots; "
            "a mixed composition must expand per slot")
        return self.nets[0].expand_obs(obs, agent_idx)

    def _slot_obs(self, obs, l):
        c = getattr(self, f"cols_{l}")
        o = obs[..., l, :]
        return o if c is None else o.index_select(-1, c.to(o.device))

    def _pad_action(self, a, l):
        """A slot's `[..., d_l + m_l]` action into the shared `act_dim` layout
        `[design(max_design) | motor(max_motor)]`."""
        if a.shape[-1] == self.act_dim:
            return a
        d = self.design_dims[l]
        out = torch.zeros(*a.shape[:-1], self.act_dim, device=a.device,
                          dtype=a.dtype)
        out[..., :d] = a[..., :d]
        out[..., self.max_design:self.max_design + self.n_motors[l]] = a[..., d:]
        return out

    def _unpad_action(self, a, l):
        d = self.design_dims[l]
        m = self.n_motors[l]
        if a.shape[-1] == d + m:
            return a
        return torch.cat([a[..., :d],
                          a[..., self.max_design:self.max_design + m]], -1)

    # -- the trainer's interface, dispatched over lanes --------------------
    def act(self, obs, noise=None):
        """`noise` is the per-lane `(eps_scale, eps_ctrl)` the reference
        opponent path passes so a test can drive two implementations with the
        same randomness. Sliced per slot rather than dropped: forwarding the
        whole tensor would give slot 1 slot 0's noise, which reads as
        agreement in exactly the equivalence test the argument exists for."""
        outs = [self.nets[l].act(self._slot_obs(obs, l),
                                 noise=None if noise is None
                                 else tuple(n[..., l, :] for n in noise))
                for l in range(self.n_slots)]
        a = torch.stack([self._pad_action(o[0], l)
                         for l, o in enumerate(outs)], dim=-2)
        logp = torch.stack([o[1] for o in outs], dim=-1)
        v = torch.stack([o[2] for o in outs], dim=-1)
        return a, logp, v

    def value(self, obs):
        return torch.stack([self.nets[l].value(self._slot_obs(obs, l))
                            for l in range(self.n_slots)], dim=-1)

    def log_prob(self, obs, action):
        return torch.stack(
            [self.nets[l].log_prob(self._slot_obs(obs, l),
                                   self._unpad_action(action[..., l, :], l))
             for l in range(self.n_slots)], dim=-1)

    def entropy(self, obs):
        return torch.stack([self.nets[l].entropy(self._slot_obs(obs, l))
                            for l in range(self.n_slots)], dim=-1)

    # -- FLAT dispatch, for the update -------------------------------------
    # `PPOTrainer.update` flattens [T, N, A, D] to [B, D] and shuffles, so the
    # lane axis is gone and each row carries its slot as a separate index.
    # Masked per slot rather than looped per row: two masked forwards, not B.
    def _flat(self, fn, obs, slots, *rest):
        out = None
        for l in range(self.n_slots):
            m = slots == l
            if not bool(m.any()):
                continue
            args = [r[m] for r in rest]
            y = fn(self.nets[l], self._slot_obs(obs[m].unsqueeze(-2), 0)
                   if False else obs[m], l, *args)
            if out is None:
                out = torch.zeros(obs.shape[0], *y.shape[1:], device=y.device,
                                  dtype=y.dtype)
            out[m] = y
        return out

    def log_prob_flat(self, obs, action, slots):
        return self._flat(
            lambda net, o, l, a: net.log_prob(o, self._unpad_action(a, l)),
            obs, slots, action)

    def entropy_flat(self, obs, slots):
        return self._flat(lambda net, o, l: net.entropy(o), obs, slots)

    def value_flat(self, obs, slots):
        return self._flat(lambda net, o, l: net.value(o), obs, slots)

    def mean_action(self, obs):
        return torch.stack(
            [self._pad_action(self.nets[l].mean_action(self._slot_obs(obs, l)), l)
             for l in range(self.n_slots)], dim=-2)


def from_env(env, lanes, **kw):
    """Build the policy for one side of `env`, given that side's lane indices.

    Reads the widths off the env rather than taking them as arguments, so a
    mismatch between the scene and the policy is impossible by construction
    (it was possible before: `TeamActorCritic`'s defaults are 20/8, the ant's,
    and a spider scene would have built an ant-shaped net that still ran).

    **Homogeneous compositions only, deliberately.** `env.obs_cols` indexes the
    RAW scene observation, but a policy is handed the EXPANDED one --
    `TeamPolicyObsEnv` runs `expand_obs` first, which permutes the others block
    and appends the role one-hot. Passing the raw columns through was silently
    dropping the role and handing the control head a 35-wide sim block where it
    wanted 37. The mixed path needs its column map recomputed in EXPANDED
    space; until that exists this refuses rather than building a net that runs
    and is wrong.
    """
    if not getattr(env, "homogeneous_design", True) or \
            len(set(env.n_motors)) != 1:
        raise NotImplementedError(
            "SlotTeamActorCritic.from_env supports homogeneous compositions "
            "only: a mixed team needs obs_cols expressed in EXPANDED "
            "observation space (post expand_obs), which is not built yet")
    return SlotTeamActorCritic(
        n_agents=env.n_agents,
        design_dims=[env.design_dims[l] for l in lanes],
        n_motors=[env.n_motors[l] for l in lanes],
        obs_cols=None, act_dim=env.act_dim, **kw)
