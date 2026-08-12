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
* Training. `RunningNorm` here is eval-only: padded rows would poison the
  running statistics, so a training port must mask them before the update. The
  assert in `RunningNorm.forward` makes that a crash rather than a slow drift.
"""

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

    def forward(self, x):
        assert not self.training, (
            "eval-only: padded nodes would enter the running statistics. A "
            "training port must update from real nodes only.")
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

    def __init__(self, in_dim, out_dim, max_index=256):
        super().__init__()
        self.out_dim = out_dim
        self.W = nn.Parameter(torch.zeros(max_index, out_dim, in_dim))
        self.b = nn.Parameter(torch.zeros(max_index, out_dim))

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
                 activation="tanh"):
        super().__init__()
        self.activation = {"tanh": torch.tanh, "relu": torch.relu}[activation]
        self.affine_layers = nn.ModuleList()
        cur = in_dim
        for h in hdims:
            self.affine_layers.append(IndexLinear(cur, h, max_index))
            cur = h
        self.linear = IndexLinear(cur, linear_dim, max_index)
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
                             imlp_cfg.get("htype", "tanh"))

    def forward(self, x, adj, ind):
        if self.norm is not None:
            x = self.norm(x)
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
