"""D3 unit 3d, step 1: their policy, dense, in our tensors -- gated against theirs.

The port map's plan is to run the execution stage on batched GPU physics with
worlds GROUPED BY TOPOLOGY (section 11, measured: 21 distinct topologies at
worst, 2 once trained). Inside a group every world has the same graph, so the
natural representation is dense `[G, N, F]` node features with one shared
`[N, N]` adjacency -- not the ragged concatenation their `batch_data` builds.

This is that policy. Parameter names are identical to theirs, so their
checkpoint loads with `strict=True` and any structural drift is a load error
rather than a silent difference. `gate.py`'s check is the point of the file:
same observations in, same actions out, to fp64 round-off.

What is NOT ported here, deliberately:

* Their `get_log_prob` cumsum-and-difference reduction. Section 5 measured it as
  the only reason the codebase needs float64; the dense form has each graph's
  nodes on their own axis, so a per-graph sum is `.sum(1)` -- no cancellation,
  and fp32 becomes viable. Porting the numerically worse version to reproduce it
  bit-for-bit would be reproducing a bug.
* Training (superseded). `RunningNorm` was eval-only here: padded rows would
  poison the running statistics. The
  dense form has no padded rows at all (grouping by topology guarantees it), so
  `RunningNorm.update` is their update over a flattened node axis. Sampling,
  per-graph log-probs and the critic were added for step 5/6 and are gated in
  `gate_dense_policy.py`.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningNorm(nn.Module):
    """Their `khrylib/rl/core/running_norm.py`, eval path only.

    Their buffers are `n, mean, var, std` and `std` is stored rather than
    derived, so it is a buffer here too -- otherwise their checkpoint would not
    load strictly and the mismatch would be silent.
    """

    def __init__(self, dim, demean=True, destd=True, clip=5.0):
        super().__init__()
        self.demean, self.destd, self.clip = demean, destd, clip
        self.register_buffer("n", torch.tensor(0, dtype=torch.long))
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.zeros(dim))
        self.register_buffer("std", torch.zeros(dim))

    def update(self, x, node_mask=None):
        """Their `update`, over a FLATTENED node axis.

        Their statistics are per-node: `x` reaches their RunningNorm as
        `[total_nodes, F]`. Here it arrives as `[G, N, F]`, and flattening is
        exactly their input as long as every row is a REAL node.

        `node_mask [G, N]` is how that stays true when the caller pads graphs
        of different sizes into one block -- which the PPO update does, because
        one padded forward per stage beats 29 tiny ones by 10x. Padded rows are
        dropped here rather than being normalised away later: a zero row is not
        a neutral sample, it drags the running mean toward zero and the running
        variance up, and nothing downstream would ever report it. This is the
        exact hazard this file's docstring warned about.
        """
        f = x.reshape(-1, x.shape[-1])
        if node_mask is not None:
            f = f[node_mask.reshape(-1).bool()]
        var_x, mean_x = torch.var_mean(f, dim=0, unbiased=False)
        m = f.shape[0]
        w = self.n.to(f.dtype) / (m + self.n).to(f.dtype)
        self.var[:] = (w * self.var + (1 - w) * var_x
                       + w * (1 - w) * (mean_x - self.mean).pow(2))
        self.mean[:] = w * self.mean + (1 - w) * mean_x
        self.std[:] = torch.sqrt(self.var)
        self.n += m

    def forward(self, x, node_mask=None):
        if self.training:
            with torch.no_grad():
                self.update(x, node_mask)
        if self.n <= 0:
            return x
        if self.demean:
            x = x - self.mean
        if self.destd:
            x = x / (self.std + 1e-8)
        if self.clip:
            x = torch.clamp(x, -self.clip, self.clip)
        return x


class _GraphConv(nn.Module):
    """`torch_geometric.nn.GraphConv` as two dense matmuls.

    PyG computes `lin_l(sum_{j in N(i)} x_j) + lin_r(x_i)` with the bias on
    `lin_l` only. Verified against PyG to 0.0 max error in fp32 by
    `gnn_playground.py`.
    """

    def __init__(self, in_dim, out_dim, bias=True):
        super().__init__()
        self.lin_l = nn.Linear(in_dim, out_dim, bias=bias)
        self.lin_r = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, adj):
        return self.lin_l(torch.matmul(adj, x)) + self.lin_r(x)


class DenseGNN(nn.Module):
    """`GNNSimple` with the same parameter names."""

    def __init__(self, in_dim, cfg):
        super().__init__()
        hdims = cfg["hdims"]
        self.residual = cfg.get("residual", False)
        self.cat_input = cfg.get("cat_input", False)
        self.num_layer_update = cfg.get("num_layer_update", 1)
        act = cfg.get("act", "relu")
        self.act = {"relu": torch.relu, "tanh": torch.tanh}.get(act, torch.sigmoid)
        self.in_fc = nn.Linear(in_dim, hdims[0])
        self.gconv_layers = nn.ModuleList([
            _GraphConv(hdims[0] if i == 0 else hdims[i - 1], hdims[i],
                       bias=cfg["bias"])
            for i in range(len(hdims))])
        self.out_dim = hdims[-1] + (in_dim if self.cat_input else 0)

    def forward(self, x, adj):
        x_init = x
        x = self.in_fc(x)
        for conv in self.gconv_layers:
            for _ in range(self.num_layer_update):
                x_in = x
                x = self.act(conv(x, adj))
                if self.residual and x.shape[-1] == x_in.shape[-1]:
                    x = x + x_in
        if self.cat_input:
            x = torch.cat([x, x_init], dim=-1)
        return x


class IndexLinear(nn.Module):
    """Per-body-type weights, selected by `body_index`.

    The loop over distinct indices is THEIR implementation and is kept on
    purpose: section 11 measured 7-12 distinct indices in a real batch, and at
    that count the loop beats a batched gather by ~6x (the gather materialises
    an `[n, out, in]` weight tensor). It is not a shortcut, it is the measured
    faster branch.
    """

    def __init__(self, in_dim, out_dim, max_index=256, zero_init=False):
        super().__init__()
        self.out_dim = out_dim
        self.W = nn.Parameter(torch.zeros(max_index, out_dim, in_dim))
        self.b = nn.Parameter(torch.zeros(max_index, out_dim))
        # `torch.zeros` is the ALLOCATION, not the initialisation. Their
        # `jsmlp.py:14-15` allocates zeros and then calls `reset_parameters()`
        # two lines later unless `zero_init`, which no hopper cfg sets. The
        # port dropped that call, and a zero `IndexLinear` is not merely a bad
        # init -- it is a DEAD one: with W = 0 the layer's output is `b`, its
        # input gradient is `W^T delta = 0`, and `dL/dW = delta a^T` is zero
        # too because the previous layer's activation `a = tanh(0)` is zero.
        # So every parameter of the stack except the last layer's bias gets an
        # exactly-zero gradient forever, the GNN feeding it gets one as well,
        # and the policy collapses to a per-`body_index` constant. Measured on
        # `runs/t2a_port/port_s1`: 48 of its 53 trainable policy tensors are
        # bit-identical between epoch 100 and epoch 1000. See D3_HANDOFF.md,
        # "Update 2026-08-28 (second)".
        if not zero_init:
            self.reset_parameters()

    def reset_parameters(self):
        """`jsmlp.py:26-30`, verbatim. `_calculate_fan_in_and_fan_out` on a
        3-D `[max_index, out, in]` tensor reads `size(1)` as the input map and
        `shape[2:]` as the receptive field, so their `fan_in` is
        `out_dim * in_dim` rather than `in_dim`. That is what their runs used;
        it is reproduced, not corrected."""
        from torch.nn import init
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x, ind):
        flat_x = x.reshape(-1, x.shape[-1])
        flat_i = ind.reshape(-1)
        out = torch.zeros(flat_x.shape[0], self.out_dim, device=x.device,
                          dtype=x.dtype)
        for i in flat_i.unique():
            sel = flat_i == i
            out[sel] = torch.addmm(self.b[i], flat_x[sel], self.W[i].t())
        return out.reshape(*x.shape[:-1], self.out_dim)


class JSMLP(nn.Module):
    def __init__(self, in_dim, hdims, linear_dim, max_index=256,
                 activation="tanh", rescale_linear=False, zero_init=False):
        super().__init__()
        self.activation = {"tanh": torch.tanh, "relu": torch.relu}[activation]
        self.affine_layers = nn.ModuleList()
        cur = in_dim
        for h in hdims:
            self.affine_layers.append(IndexLinear(cur, h, max_index, zero_init))
            cur = h
        self.linear = IndexLinear(cur, linear_dim, max_index, zero_init)
        # `jsmlp.py:54-56`. Every hopper cfg sets `rescale_linear: true` on all
        # three index MLPs, so the head starts small and the head bias starts
        # at zero -- the standard "small last layer" policy init. It is part of
        # the initialisation, so it only ever shows up in a run that trains
        # from scratch, which is why loading their checkpoint hid its absence.
        if rescale_linear:
            self.linear.W.data.mul_(0.1)
            self.linear.b.data.mul_(0.0)
        self.out_dim = linear_dim

    def forward(self, x, ind):
        for layer in self.affine_layers:
            x = self.activation(layer(x, ind))
        return self.linear(x, ind)


class _Tower(nn.Module):
    """One stage: norm -> gnn -> jsmlp. Named so their keys land unchanged."""

    def __init__(self, in_dim, gnn_cfg, imlp_cfg, out_dim, norm=True):
        super().__init__()
        self.norm = RunningNorm(in_dim) if norm else None
        self.gnn = DenseGNN(in_dim, gnn_cfg)
        self.ind_mlp = JSMLP(self.gnn.out_dim, imlp_cfg["hdims"], out_dim,
                             imlp_cfg.get("max_index", 256),
                             imlp_cfg.get("htype", "tanh"),
                             imlp_cfg.get("rescale_linear", False),
                             imlp_cfg.get("zero_init", False))

    def forward(self, x, adj, ind, node_mask=None):
        if self.norm is not None:
            x = self.norm(x, node_mask)
        return self.ind_mlp(self.gnn(x, adj), ind)


class DenseTransform2ActPolicy(nn.Module):
    """Their `Transform2ActPolicy` for the configs D3 actually runs.

    Restricted on purpose to the branch `hopper_gpu` uses -- a GNN and an index
    MLP per stage, no `pre_mlp`, no plain `mlp`, no non-indexed head. Their
    class builds any of eight combinations from the yaml; supporting the seven
    unused ones would be untested code that looks tested.
    """

    def __init__(self, cfg, attr_fixed_dim, sim_obs_dim, attr_design_dim,
                 skel_action_dim, control_action_dim):
        super().__init__()
        for key in ("skel_pre_mlp", "skel_mlp", "attr_pre_mlp", "attr_mlp",
                    "control_pre_mlp", "control_mlp"):
            assert key not in cfg, f"{key} is not ported; see the class docstring"
        self.attr_fixed_dim = attr_fixed_dim
        self.attr_design_dim = attr_design_dim
        self.control_action_dim = control_action_dim
        self.attr_action_dim = attr_design_dim
        self.action_dim = control_action_dim + attr_design_dim + 1

        design_in = attr_fixed_dim + attr_design_dim
        control_in = attr_fixed_dim + sim_obs_dim + attr_design_dim
        self.skel = _Tower(design_in, cfg["skel_gnn_specs"],
                           cfg["skel_index_mlp"], skel_action_dim)
        self.attr = _Tower(design_in, cfg["attr_gnn_specs"],
                           cfg["attr_index_mlp"], attr_design_dim,
                           norm=cfg.get("attr_norm", True))
        self.control = _Tower(control_in, cfg["control_gnn_specs"],
                              cfg["control_index_mlp"], control_action_dim)
        self.attr_action_log_std = nn.Parameter(
            torch.ones(1, attr_design_dim) * cfg["attr_log_std"])
        self.control_action_log_std = nn.Parameter(
            torch.ones(1, control_action_dim) * cfg["control_log_std"])

    # Their names are flat (`skel_norm`, `skel_gnn`, `skel_ind_mlp`); ours are
    # nested under a tower. Translating on load keeps their checkpoint the
    # source of truth without forcing a flat, three-times-repeated __init__.
    _PREFIX = {"skel": "skel", "attr": "attr", "control": "control"}

    def load_their_state_dict(self, sd, strict=True):
        out = {}
        for k, v in sd.items():
            for stage in ("skel", "attr", "control"):
                if k.startswith(f"{stage}_norm."):
                    out[f"{stage}.norm." + k.split(".", 1)[1]] = v
                    break
                if k.startswith(f"{stage}_gnn."):
                    out[f"{stage}.gnn." + k.split(".", 1)[1]] = v
                    break
                if k.startswith(f"{stage}_ind_mlp."):
                    out[f"{stage}.ind_mlp." + k.split(".", 1)[1]] = v
                    break
            else:
                out[k] = v
        return self.load_state_dict(out, strict=strict)

    def design_input(self, obs):
        """The design towers see `[attr_fixed | attr_design]` only -- they are
        blind to `sim_obs`, which sits between the two in the observation."""
        return torch.cat([obs[..., :self.attr_fixed_dim],
                          obs[..., -self.attr_design_dim:]], dim=-1)

    @torch.no_grad()
    def mean_action(self, stage, obs, adj, ind):
        """`obs [G, N, F]`, `adj [G, N, N]` (adj[g,i,j]=1 when j sends to i),
        `ind [G, N]`. Returns `[G, N, action_dim]` with the inactive slices
        zeroed, exactly as their `select_action` assembles it."""
        G, N = obs.shape[:2]
        action = torch.zeros(G, N, self.action_dim, device=obs.device,
                             dtype=obs.dtype)
        if stage == "execution":
            action[..., :self.control_action_dim] = self.control(obs, adj, ind)
        elif stage == "attr_trans":
            x = self.design_input(obs)
            action[..., self.control_action_dim:-1] = self.attr(x, adj, ind)
        elif stage == "skel_trans":
            x = self.design_input(obs)
            logits = self.skel(x, adj, ind)
            action[..., -1] = logits.argmax(-1).to(obs.dtype)
        else:
            raise ValueError(stage)
        return action

    # ------------------------------------------------------------------
    # Sampling and log-probs (added for 3d step 5/6: the training path)
    # ------------------------------------------------------------------
    # Their reduction to a PER-GRAPH log-prob is a cumsum over the whole
    # concatenated batch differenced at the graph boundaries
    # (`transform2act_policy.py:238-241`). PORT_MAP section 5 measured that as
    # the sole reason their codebase runs in float64: fp64 cumsum-diff errs
    # 1.7e-10, fp32 cumsum-diff errs 1.3e-1. Here each graph owns an axis, so
    # the same quantity is `.sum(1)` -- no cancellation, and fp32 is fine.
    def _stage_head(self, stage, obs, adj, ind, node_mask=None):
        if stage == "execution":
            return self.control(obs, adj, ind, node_mask)
        x = self.design_input(obs)
        return (self.attr if stage == "attr_trans" else self.skel)(
            x, adj, ind, node_mask)

    def _gauss(self, stage, head):
        log_std = (self.control_action_log_std if stage == "execution"
                   else self.attr_action_log_std)
        return head, log_std.expand_as(head).exp()

    def slice_for(self, stage):
        """Where a stage's action lives in the `[.., action_dim]` row."""
        if stage == "execution":
            return slice(0, self.control_action_dim)
        if stage == "attr_trans":
            return slice(self.control_action_dim, self.action_dim - 1)
        return slice(self.action_dim - 1, self.action_dim)

    def act(self, stage, obs, adj, ind, mean_action=False, generator=None):
        """Returns `(action [G, N, action_dim], log_prob [G])`.

        The whole `action_dim` row is returned with the other stages' slices
        left at zero, exactly as their `select_action` assembles it -- their
        env asserts those zeros (`hopper.py:146`).
        """
        G, N = obs.shape[:2]
        action = torch.zeros(G, N, self.action_dim, device=obs.device,
                             dtype=obs.dtype)
        head = self._stage_head(stage, obs, adj, ind)
        sl = self.slice_for(stage)
        if stage == "skel_trans":
            logp_all = torch.log_softmax(head, dim=-1)
            if mean_action:
                a = head.argmax(-1)
            else:
                probs = logp_all.exp().reshape(-1, head.shape[-1])
                a = torch.multinomial(probs, 1, generator=generator
                                      ).reshape(G, N)
            lp = logp_all.gather(-1, a.unsqueeze(-1)).squeeze(-1).sum(1)
            action[..., -1] = a.to(obs.dtype)
        else:
            mean, std = self._gauss(stage, head)
            if mean_action:
                a = mean
            else:
                # `generator` must live on the same device as `mean`;
                # torch refuses to mix, and silently falling back to the
                # default generator would make a seeded run unreproducible.
                eps = torch.randn(mean.shape, generator=generator,
                                  device=mean.device, dtype=mean.dtype)
                a = mean + std * eps
            lp = _normal_log_prob(a, mean, std).sum(-1).sum(-1)
            action[..., sl] = a
        return action, lp

    def log_prob(self, stage, obs, adj, ind, action, node_mask=None):
        """`action` is the full `[G, N, action_dim]` row; the stage's slice is
        read out of it. Differentiable -- this is the PPO ratio's numerator.

        `node_mask [G, N]` zeroes padded nodes BEFORE the per-graph sum. A
        padded node's log-prob is not small, it is whatever the network
        happens to emit for a zero row, so leaving it in would corrupt every
        graph in a padded block by a different amount.

        Padded nodes cannot leak through the graph itself: their adjacency row
        and column are zero, so `matmul(adj, x)` never mixes them into a real
        node, and `IndexLinear` is per-row.
        """
        head = self._stage_head(stage, obs, adj, ind, node_mask)
        if stage == "skel_trans":
            a = action[..., -1].long()
            lp = torch.log_softmax(head, dim=-1).gather(
                -1, a.unsqueeze(-1)).squeeze(-1)
        else:
            mean, std = self._gauss(stage, head)
            a = action[..., self.slice_for(stage)]
            lp = _normal_log_prob(a, mean, std).sum(-1)
        if node_mask is not None:
            lp = lp * node_mask.to(lp.dtype)
        return lp.sum(1)


def _normal_log_prob(value, mean, std):
    """`torch.distributions.Normal.log_prob`, written out so the dtype and the
    device follow the inputs and nothing constructs a distribution object per
    call inside the PPO inner loop."""
    var = std ** 2
    return (-((value - mean) ** 2) / (2 * var) - std.log()
            - math.log(math.sqrt(2 * math.pi)))


class DenseTransform2ActValue(nn.Module):
    """Their `Transform2ActValue`, dense.

    Two details their critic gets right that a port would plausibly get wrong,
    both from `transform2act_critic.py`:

    * the value of a graph is read off its **first node**, not pooled
      (`critic.py:78-81`);
    * the one-hot **stage flag** is concatenated to the observation
      (`design_flag_in_state`/`onehot_design_flag`, both set for hopper), which
      the critic needs because design steps pay 0 reward and execution steps do
      not.

    Parameter names match theirs (`norm`, `gnn`, `mlp`, `value_head`), so their
    checkpoint loads with `strict=True`.
    """

    STAGES = ("skel_trans", "attr_trans", "execution")

    def __init__(self, cfg, state_dim):
        super().__init__()
        self.design_flag_in_state = cfg.get("design_flag_in_state", False)
        self.onehot_design_flag = cfg.get("onehot_design_flag", False)
        assert self.design_flag_in_state and self.onehot_design_flag, (
            "only the flag combination hopper uses is ported")
        assert "pre_mlp" not in cfg, "pre_mlp is not ported"
        self.state_dim = state_dim + 3
        self.norm = RunningNorm(self.state_dim)
        self.gnn = DenseGNN(self.state_dim, cfg["gnn_specs"])
        cur = self.gnn.out_dim
        self.mlp = _MLP(cur, cfg["mlp"], cfg.get("htype", "tanh"))
        self.value_head = nn.Linear(self.mlp.out_dim, 1)
        # `transform2act_critic.py:36` -> `tools.init_fc_weights`: the value
        # head is rescaled at init exactly as the policy's linear head is.
        self.value_head.weight.data.mul_(0.1)
        self.value_head.bias.data.mul_(0.0)

    def forward(self, stage, obs, adj, node_mask=None):
        """`obs [G, N, F]` -> `[G, 1]`. The value is read off node 0, which is
        the root and is therefore never a padded row."""
        G, N = obs.shape[:2]
        flag = torch.zeros(G, N, 3, device=obs.device, dtype=obs.dtype)
        flag[..., self.STAGES.index(stage)] = 1.0
        x = self.norm(torch.cat([obs, flag], dim=-1), node_mask)
        x = self.mlp(self.gnn(x, adj))
        return self.value_head(x)[:, 0]


class _MLP(nn.Module):
    """`khrylib/models/mlp.py`, with their parameter names."""

    def __init__(self, in_dim, hdims, activation="tanh"):
        super().__init__()
        self.activation = {"tanh": torch.tanh, "relu": torch.relu,
                           "sigmoid": torch.sigmoid}[activation]
        self.out_dim = hdims[-1]
        self.affine_layers = nn.ModuleList()
        cur = in_dim
        for h in hdims:
            self.affine_layers.append(nn.Linear(cur, h))
            cur = h

    def forward(self, x):
        for layer in self.affine_layers:
            x = self.activation(layer(x))
        return x
