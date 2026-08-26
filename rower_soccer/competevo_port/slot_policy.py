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

--------------------------------------------------------------------------
2h: the column map has to live in EXPANDED space, not scene space
--------------------------------------------------------------------------
`env.obs_cols[i]` indexes the RAW scene observation. A policy is never handed
that: `TeamPolicyObsEnv._expand` runs `TeamActorCritic.expand_obs` first, which
PERMUTES the others block (scene `[teammate, opp_near, opp_far]` -> policy
`[opp_near, teammate, opp_far]`) and APPENDS the 2-dim role one-hot. Passing
the raw columns through therefore dropped the role and handed the control head
a 35-wide sim block where it wanted 37 -- which is why `from_env` refused the
mixed case rather than building a net that runs and is wrong.

`MixedPolicyObsEnv` below is the fix, and it inverts the layering. Instead of
expanding per agent with one net's widths (impossible: a mixed team's agents
have different `design_dim` and different own-state widths, so there is no
single `expand_obs`), it expands the PADDED scene observation as a whole:

    expanded = [flag(1) | scale(max_design) | qpos(max_q) | qvel(max_v)
                | others PERMUTED (2 * n_others) | role(2)]

which is exactly the env's own padded layout with the others block permuted in
place and two columns appended. The permutation is a gather inside a block
whose width does not depend on the creature, so it is well defined for a mixed
team; the padding stays where `dev_env` put it. `expanded_obs_cols` then maps
each slot to its real columns of THAT tensor, and the gathered result is
`[flag | scale_i | qpos_i | qvel_i | others_permuted | role]` -- precisely what
`TeamActorCritic(design_dim=d_i, own_dim=q_i+v_i)` consumes.

THE ROLE ONE-HOT IS KEPT for mixed teams even though under Option A it is
constant per net (the role IS the net, so it carries no information). Keeping
it means the mixed and homogeneous paths build the identical `TeamActorCritic`
shape -- `sim_obs_dim = own + 2*n_others + ROLE_DIM` in both -- and it costs
two dead input columns. Dropping it would have made the two paths' nets
different shapes for no gain.

Homogeneous compositions do NOT go through any of this: `train_team_selfplay`
keeps wrapping them in `TeamPolicyObsEnv` with `obs_cols=None`, which is the
2f/2g path untouched.
"""

import torch
import torch.nn as nn

from rower_soccer.competevo_port.team_policy import (OWN_DIM, ROLE_DIM,
                                                     TeamActorCritic,
                                                     others_permutation,
                                                     role_onehot)


class SlotTeamActorCritic(nn.Module):
    """`nn.ModuleList` of per-slot `TeamActorCritic`s, dispatched by lane.

    `obs` everywhere is `[..., L, D]` with L the number of slots this side
    owns (2 in 2v2) and D the env's padded observation width. Slot `l` reads
    `obs[..., l, obs_cols[l]]`.
    """

    def __init__(self, n_agents=4, design_dims=(20, 20), n_motors=(8, 8),
                 own_dims=None, obs_cols=None, act_dim=None, max_design=None,
                 **kw):
        super().__init__()
        assert len(design_dims) == len(n_motors), "one width pair per slot"
        self.n_slots = len(design_dims)
        self.design_dims = tuple(design_dims)
        self.n_motors = tuple(n_motors)
        # Per-slot own-state width (qpos + qvel). Defaults to the ant's, which
        # is what every homogeneous 2f/2g caller means.
        self.own_dims = tuple(own_dims or [OWN_DIM] * self.n_slots)
        assert len(self.own_dims) == self.n_slots
        # role_in_design is meaningless here and is refused rather than
        # silently ignored: with one net per slot the role is the network, and
        # accepting the flag would imply a 2g-style comparison this
        # architecture cannot support.
        assert not kw.pop("role_in_design", False), (
            "role_in_design has no meaning under Option A -- the role IS the "
            "net. Use TeamActorCritic for the 2g architecture.")
        self.nets = nn.ModuleList([
            TeamActorCritic(n_agents=n_agents, design_dim=d, n_motor=m,
                            own_dim=o, **kw)
            for d, m, o in zip(design_dims, n_motors, self.own_dims)])
        # Padded action width, shared across slots so the env's [n, A, act_dim]
        # tensor stays rectangular. Each slot writes its own leading columns of
        # the design block and of the motor block.
        #
        # `max_design` is the ENV's design block width, NOT this side's. They
        # differ as soon as the two sides carry different creatures (an
        # [ant, spider] side against a [bug, bug] side: 40 vs 30), and taking
        # the side's own maximum would start the motor block 10 columns early
        # and drive the wrong actuators with a perfectly rectangular tensor.
        self.max_design = int(max_design or max(design_dims))
        assert self.max_design >= max(design_dims)
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
        assert (len(set(self.design_dims)) == 1
                and len(set(self.n_motors)) == 1
                and len(set(self.own_dims)) == 1), (
            "expand_obs via the shared wrapper assumes shape-identical slots; "
            "a mixed composition must use MixedPolicyObsEnv, which expands the "
            "padded scene observation once and lets each slot gather its own "
            "columns of it")
        return self.nets[0].expand_obs(obs, agent_idx)

    def _take(self, o, l):
        """Slot `l`'s columns of a `[..., D]` observation."""
        c = getattr(self, f"cols_{l}")
        return o if c is None else o.index_select(-1, c.to(o.device))

    def _slot_obs(self, obs, l):
        return self._take(obs[..., l, :], l)

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
            # `_take`, not the raw rows: the lane axis is already gone here, so
            # `_slot_obs` does not apply, but the per-slot column gather still
            # does. Without it a mixed slot's net was handed the full padded
            # width in the UPDATE while `collect` handed it the gathered one --
            # a shape error at best and, at equal widths, silently different
            # inputs between the rollout and the ratio it is trained on.
            y = fn(self.nets[l], self._take(obs[m], l), l, *args)
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


def env_own_dims(env):
    """Each agent's REAL own-state width (qpos + qvel), read off `dev_env`'s
    padding masks rather than from a creature table -- the masks are what the
    observation is actually built with."""
    return [int(env.qpos_mask[i].sum().item())
            + int(env.qvel_mask[i].sum().item())
            for i in range(env.n_agents)]


def env_is_mixed(env):
    """True when the agents differ in any width a policy has to match."""
    return (len(set(env.design_dims)) != 1 or len(set(env.n_motors)) != 1
            or len(set(env_own_dims(env))) != 1)


def _raw_blocks(env):
    """`(others_offset, others_width)` of `dev_env`'s padded observation."""
    base_o = (1 + env.design_dim + env.qpos_idx.shape[1]
              + env.qvel_idx.shape[1])
    n_other = env.other_xy_idx.shape[1]
    assert base_o + n_other == env.obs_dim, (base_o, n_other, env.obs_dim)
    return base_o, n_other


def expansion_permutation(env):
    """`[env.obs_dim]` gather that turns the padded SCENE observation into the
    padded EXPANDED one, minus the appended role.

    Identity everywhere except inside the others block, which is reordered
    `[teammate, opp_near, opp_far] -> [opp_near, teammate, opp_far]`. That is
    `team_policy.others_permutation`, applied to the whole tensor at once
    instead of per agent -- legal precisely because the others block is
    `2 * (n_agents - 1)` wide for every creature.
    """
    base_o, n_other = _raw_blocks(env)
    cols = list(range(base_o))
    for o in others_permutation(env.n_agents):
        cols += [base_o + 2 * o, base_o + 2 * o + 1]
    assert len(cols) == base_o + n_other
    return torch.as_tensor(cols, dtype=torch.long, device=env.device)


def expanded_obs_cols(env):
    """`cols[i]` = the columns of the PADDED EXPANDED observation that are real
    for agent `i`, including the two role columns at the end.

    Built from `env.obs_cols` (which is scene-space) rather than re-derived, so
    the two cannot drift: the own-state part of a scene column map is unchanged
    by the expansion (the permutation only touches the others block, and every
    agent takes that block whole), and the role columns are appended.
    """
    base_o, n_other = _raw_blocks(env)
    others = list(range(base_o, base_o + n_other))
    role = [env.obs_dim + k for k in range(ROLE_DIM)]
    out = []
    for c in env.obs_cols:
        c = c.tolist()
        own = [j for j in c if j < base_o]
        assert c[len(own):] == others, (
            "env.obs_cols does not end in the whole others block; the "
            "expansion's in-place permutation assumes it does")
        out.append(torch.as_tensor(own + others + role, dtype=torch.long,
                                   device=env.device))
    return out


class MixedPolicyObsEnv:
    """`train_team_smoke.TeamPolicyObsEnv` for a MIXED composition.

    Same contract -- delegate everything, present the policy's observation --
    but the expansion is driven by the ENV rather than by one actor-critic,
    because on a mixed team no single `expand_obs` is correct for all four
    agents (see this module's docstring). The output is the env's padded
    observation with the others block permuted and a per-agent role one-hot
    appended, and each slot gathers its own columns of that via
    `expanded_obs_cols`.

    Deliberately a separate class: the homogeneous path keeps using
    `TeamPolicyObsEnv` unchanged, so 2f/2g runs are not perturbed by 2h.
    """

    def __init__(self, env):
        self._env = env
        self.obs_dim = env.obs_dim + ROLE_DIM
        self._perm = expansion_permutation(env)
        self._roles = torch.stack(
            [role_onehot(i, env.n_agents, device=env.device, dtype=env.dtype)
             for i in range(env.n_agents)])

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _expand(self, obs):
        """`[n, A, obs_dim]` -> `[n, A, obs_dim + ROLE_DIM]`."""
        o = obs.index_select(-1, self._perm.to(obs.device))
        role = self._roles.to(device=obs.device, dtype=obs.dtype)
        role = role.unsqueeze(0).expand(obs.shape[0], -1, -1)
        return torch.cat([o, role], dim=-1)

    def reset(self):
        return self._expand(self._env.reset())

    def step(self, action):
        obs, rew, done, info = self._env.step(action)
        return self._expand(obs), rew, done, info


def wrap_env(env, ac):
    """The right observation wrapper for this composition.

    One call site in the trainer instead of an `if` there, so the choice is
    made from the SCENE (which knows whether it is mixed) rather than from a
    command-line flag that could disagree with it.
    """
    if env_is_mixed(env):
        return MixedPolicyObsEnv(env)
    from rower_soccer.competevo_port.train_team_smoke import TeamPolicyObsEnv
    return TeamPolicyObsEnv(env, ac)


def from_env(env, lanes, **kw):
    """Build the policy for one side of `env`, given that side's lane indices.

    Reads the widths off the env rather than taking them as arguments, so a
    mismatch between the scene and the policy is impossible by construction
    (it was possible before: `TeamActorCritic`'s defaults are 20/8, the ant's,
    and a spider scene would have built an ant-shaped net that still ran).

    Heterogeneous compositions are supported as of 2h. Each slot gets its own
    `design_dim`, `n_motor` and own-state width, plus a column map into the
    PADDED EXPANDED observation `MixedPolicyObsEnv` produces. A homogeneous
    composition still gets `obs_cols=None` -- the whole observation, no gather
    -- which is byte-for-byte the 2f/2g path.
    """
    own = env_own_dims(env)
    cols = None if not env_is_mixed(env) else [
        c for i, c in enumerate(expanded_obs_cols(env)) if i in set(lanes)]
    # `lanes` is ascending (team_lanes is built by a filter over range), so the
    # comprehension above is in lane order; asserted rather than assumed.
    assert list(lanes) == sorted(lanes), f"lanes must be ascending: {lanes}"
    return SlotTeamActorCritic(
        n_agents=env.n_agents,
        design_dims=[env.design_dims[l] for l in lanes],
        n_motors=[env.n_motors[l] for l in lanes],
        own_dims=[own[l] for l in lanes],
        obs_cols=cols, act_dim=env.act_dim, max_design=env.design_dim, **kw)
