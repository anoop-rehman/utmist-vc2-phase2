"""2f step 3: `DevActorCritic` widened from 1v1 to a team, with a regression.

The design doc's step 3 asks for the observation and network plumbing plus one
specific check: *a 1v1 net loaded into the widened one, with the teammate and
role channels zeroed, must reproduce its 1v1 behaviour to fp32 on the same
states*. Without it a silent input permutation is indistinguishable from "2v2
is hard" -- and this project has shipped two envs that were numerically fine
and semantically wrong, so the check is the point of the file, not a courtesy.

--------------------------------------------------------------------------
The one design decision here: the near opponent keeps the 1v1 slot
--------------------------------------------------------------------------
`team_scene._team_agent_slices` orders an agent's view of the others as

    [teammate, opp_near, opp_far]

which is the natural SCENE order -- teammate first, then opponents by spawn
distance. Feeding that straight into a widened 1v1 net would drop the
TEAMMATE's (x, y) into the two columns the 1v1 net trained to read as the
OPPONENT's. Nothing would crash; the policy would just be wrong in a way that
looks like a hard task.

So the policy reorders to

    [opp_near, teammate, opp_far]

and the whole widening becomes "copy the leading columns, zero the trailing
ones" -- for the control head AND the critic, because with opp_near first the
leading 52 columns of a 2v2 observation are exactly a 1v1 observation.

That is worth more than passing a test. `runs/competevo_port/m2e_fixed` is a
1v1 pair at 83.9% goal rate, and this ordering makes it a legal warm start for
2f rather than a thing to throw away. `widen_from_1v1` is that warm start.

The reorder lives here rather than in the env on purpose: the env's ordering is
a fact about the scene that the design doc's measurements were taken against,
and moving it would silently invalidate them. This is a policy-side view.

--------------------------------------------------------------------------
Role
--------------------------------------------------------------------------
One net per team drives both teammates (doc §5), so something has to tell the
two apart or they are the same agent twice. `team_init_pose` spawns agents 0
and 1 at the 1v1 position and agents 2 and 3 on their own goal line, so role is
a static function of agent index: `[front, back]` as a 2-dim one-hot. It is
appended AFTER the sim block, so it lands in the zeroed region and the 1v1 net
starts life ignoring it exactly.
"""

import torch
import torch.nn as nn

from rower_soccer.competevo_port.dev_ppo import DevActorCritic

# `sim_obs` is [own qpos | own qvel | 2 per other] (dev_env.sim_obs). These are
# THE ANT's widths and remain the default, so every 2f/2g caller is unchanged;
# 2h passes `own_dim` explicitly, because a bug is 21+20 = 41 and a spider
# 27+26 = 53.
OWN_QPOS_DIM = 15
OWN_QVEL_DIM = 14
OWN_DIM = OWN_QPOS_DIM + OWN_QVEL_DIM
ROLE_DIM = 2                      # [front, back]
N_FRONT = 2                       # agents 0, 1 spawn at the 1v1 position


def others_permutation(n_agents):
    """Scene order `[teammate, opp_near, opp_far]` -> policy order
    `[opp_near, teammate, opp_far]`, as indices into the OTHERS block.

    Written for the general case but only exercised at 2 and 4; at 2 there is
    one other and it is the opponent, so this is the identity and a widened
    2-agent net is column-compatible with the 1v1 net trivially.
    """
    n_others = n_agents - 1
    if n_others <= 1:
        return list(range(n_others))
    # teammate at 0, opponents at 1.., in the scene's near-to-far order.
    return [1, 0] + list(range(2, n_others))


def role_onehot(agent_idx, n_agents, device=None, dtype=torch.float32):
    """`[ROLE_DIM]`. Front for the two agents that keep the 1v1 spawn."""
    r = torch.zeros(ROLE_DIM, device=device, dtype=dtype)
    r[0 if agent_idx < N_FRONT else 1] = 1.0
    return r


class TeamActorCritic(DevActorCritic):
    """`DevActorCritic` over a team observation.

    Consumes `[..., 1 + design_dim + sim_dim + ROLE_DIM]` -- 58 at four agents
    -- which `expand_obs` builds from the env's `[..., 56]`. Everything
    inherited (the stage mask, the two heads, `log_prob`, `entropy`) works
    unchanged, because the widening is entirely in the input width.
    """

    def __init__(self, n_agents=4, design_dim=20, n_motor=8, own_dim=OWN_DIM,
                 role_in_design=False, scale_hidden=(64, 64), **kw):
        n_others = n_agents - 1
        self.n_agents = n_agents
        # 2h: the own-state block is the creature's, not the ant's. Defaults to
        # the ant so the 2f/2g construction is character-for-character the one
        # `gate_team_policy` measured.
        self.own_dim = int(own_dim)
        self.role_in_design = bool(role_in_design)
        self.env_sim_dim = self.own_dim + 2 * n_others
        super().__init__(design_dim=design_dim,
                         sim_obs_dim=self.env_sim_dim + ROLE_DIM,
                         n_motor=n_motor, scale_hidden=scale_hidden, **kw)
        if self.role_in_design:
            # THE 2g EXPERIMENT, in two lines. `self.scale` is already
            # per-agent, so the env can carry two different bodies on a team --
            # what stops it is that the design head sees ONLY the scale vector
            # (design doc section 5: "the design head cannot see the world at
            # all"). Both teammates therefore run the same function of the same
            # random draw and converge on the same body, which is exactly what
            # was measured: front-vs-back SMD 0.052, masses 0.974 vs 0.973 kg.
            #
            # Widening the design head's INPUT by the role one-hot is the
            # smallest change that makes morphological specialisation
            # expressible. It does not make it happen -- that is the question.
            import torch.nn as _nn
            from rower_soccer.competevo_port.ppo import RunningNorm, _mlp
            self.scale_norm = RunningNorm(design_dim + ROLE_DIM)
            self.scale_mlp, _d = _mlp(design_dim + ROLE_DIM, scale_hidden)
            assert isinstance(self.scale_mean, _nn.Linear)
        perm = others_permutation(n_agents)
        # Column indices into the env's sim block, in policy order.
        cols = list(range(self.own_dim))
        for o in perm:
            cols += [self.own_dim + 2 * o, self.own_dim + 2 * o + 1]
        self.register_buffer("sim_perm", torch.tensor(cols, dtype=torch.long),
                             persistent=False)
        roles = torch.stack([role_onehot(i, n_agents) for i in range(n_agents)])
        self.register_buffer("roles", roles, persistent=False)

    def dists(self, obs):
        """As `DevActorCritic.dists`, but the design head may also see the role.

        Only the design head changes; the control head and critic already see
        the role because it is part of the observation they consume.
        """
        if not self.role_in_design:
            return super().dists(obs)
        import torch.distributions as D
        from rower_soccer.competevo_port.dev_ppo import SCALE_STD_DIVISOR
        obs = self._clean(obs)
        scale = obs[..., 1:1 + self.design_dim]
        sim = obs[..., 1 + self.design_dim:]
        role = obs[..., -ROLE_DIM:]
        s_in = torch.cat([scale, role], dim=-1)
        s_mean = self.scale_mean(self.scale_mlp(self.scale_norm(s_in)))
        c_mean = self.control_mean(self.control_mlp(self.control_norm(sim)))
        return (D.Normal(s_mean, self.scale_log_std.exp() / SCALE_STD_DIVISOR),
                D.Normal(c_mean, self.control_log_std.exp()))

    def expand_obs(self, obs, agent_idx):
        """Env observation `[..., 56]` for one agent -> policy input `[..., 58]`.

        Pure and explicit: the permutation is a gather against a registered
        index buffer, so the mapping is one readable line and the gate can
        break it on demand.
        """
        assert obs.shape[-1] == 1 + self.design_dim + self.env_sim_dim, (
            f"expected env obs width "
            f"{1 + self.design_dim + self.env_sim_dim}, got {obs.shape[-1]}")
        head = obs[..., :1 + self.design_dim]
        sim = obs[..., 1 + self.design_dim:]
        sim = sim.index_select(-1, self.sim_perm.to(sim.device))
        role = self.roles[agent_idx].to(device=sim.device, dtype=sim.dtype)
        role = role.expand(*sim.shape[:-1], ROLE_DIM)
        return torch.cat([head, sim, role], dim=-1)


def _widen_linear(dst, src):
    """Copy `src`'s weights into `dst`'s LEADING columns; zero the rest.

    Correct only because of the reorder documented at the top of this file --
    with `opp_near` first, `src`'s columns are a prefix of `dst`'s.
    """
    assert dst.out_features == src.out_features
    assert dst.in_features >= src.in_features
    with torch.no_grad():
        dst.weight.zero_()
        dst.weight[:, :src.in_features].copy_(src.weight)
        dst.bias.copy_(src.bias)


def _widen_norm(dst, src):
    """Leading dims keep their statistics; new dims get mean 0, var 1.

    A new dim therefore passes through as itself (clipped at +/-5), which does
    not matter because every weight column reading it is zero. Copying `n` is
    what makes the 1v1 path bit-identical rather than merely close: at n = 0
    `RunningNorm.forward` returns its input unchanged, so a zeroed `n` would
    silently disable normalisation on the columns that DO matter.
    """
    with torch.no_grad():
        dst.mean.zero_()
        dst.var.fill_(1.0)
        k = src.mean.numel()
        dst.mean[:k].copy_(src.mean)
        dst.var[:k].copy_(src.var)
        dst.n.copy_(src.n)


def widen_from_1v1(ac, n_agents=4, **kw):
    """A `TeamActorCritic` that reproduces `ac` on 1v1-shaped states.

    The genuinely useful direction, not just the tested one: it warm-starts 2f
    from the validated 1v1 pair instead of from noise.
    """
    assert isinstance(ac, DevActorCritic)
    out = TeamActorCritic(n_agents=n_agents, design_dim=ac.design_dim,
                          n_motor=ac.n_motor, **kw)
    # The design head is untouched -- it reads only the scale vector, whose
    # width does not change with the number of agents (doc §5: the design head
    # cannot see the world at all).
    if getattr(out, "role_in_design", False):
        # Same leading-columns-copy as the control head: the role bit is
        # appended after the scale vector, so a 1v1 net's weights are a prefix
        # and the new columns start at zero -- the widened net begins life
        # ignoring the role exactly.
        _widen_norm(out.scale_norm, ac.scale_norm)
        _widen_linear(out.scale_mlp[0], ac.scale_mlp[0])
        for d, s_ in zip(list(out.scale_mlp)[1:], list(ac.scale_mlp)[1:]):
            if isinstance(d, nn.Linear):
                d.load_state_dict(s_.state_dict())
    else:
        out.scale_norm.load_state_dict(ac.scale_norm.state_dict())
        out.scale_mlp.load_state_dict(ac.scale_mlp.state_dict())
    out.scale_mean.load_state_dict(ac.scale_mean.state_dict())
    with torch.no_grad():
        out.scale_log_std.copy_(ac.scale_log_std)
        out.control_log_std.copy_(ac.control_log_std)

    _widen_norm(out.control_norm, ac.control_norm)
    _widen_linear(out.control_mlp[0], ac.control_mlp[0])
    for d, s in zip(list(out.control_mlp)[1:], list(ac.control_mlp)[1:]):
        if isinstance(d, nn.Linear):
            d.load_state_dict(s.state_dict())
    out.control_mean.load_state_dict(ac.control_mean.state_dict())

    _widen_norm(out.vf_norm, ac.vf_norm)
    _widen_linear(out.vf[0], ac.vf[0])
    for d, s in zip(list(out.vf)[1:], list(ac.vf)[1:]):
        if isinstance(d, nn.Linear):
            d.load_state_dict(s.state_dict())
    out.value_net.load_state_dict(ac.value_net.state_dict())
    return out
