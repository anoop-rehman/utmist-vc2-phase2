"""D3 unit 3c -- their nets, our tensors, no simulator.

Three questions block the Transform2Act GPU port, and all three are answerable
without a physics engine. Prose cannot answer them; this can:

1. **Can a dense masked-adjacency GraphConv replace PyG's?** Their `GNNSimple`
   is `torch_geometric.nn.GraphConv` over a ragged concatenated batch. A GPU
   port over a fixed-size superset morphology wants dense `[W, N, F]` tensors
   and a per-world adjacency mask instead. If the dense form is not bit-close
   to theirs, every downstream number is off and no gate would catch it.

2. **Can `IndexLinear` be batched?** Their per-body-type weights are applied by
   a PYTHON LOOP over `ind.unique()` -- one `addmm` per distinct body index per
   layer per forward. On a 50,000-node batch that is the shape of thing that
   dominates a GPU port's update time.

3. **Is float64 actually required?** The whole codebase runs in float64. The
   suspicious part is the per-graph log-prob: they sum node log-probs with
   `cumsum` and then DIFFERENCE consecutive graph boundaries, which is
   catastrophic cancellation by construction -- the cumsum over a 50k-row batch
   grows to ~1e5 while the quantity wanted is ~1e0. A segment sum (`index_add`)
   computes the same thing without ever forming the large partial sums. If that
   is the only reason for float64, fp32 buys ~2x memory and bandwidth.

Run with THEIR venv, because PyG is the reference being checked against:

    cd /workspace/Transform2Act && source env-gpu.sh
    .venv-gpu/bin/python /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/gnn_playground.py
"""

import math
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init

try:
    from torch_geometric.nn import GraphConv
except ImportError:  # pragma: no cover
    print("needs torch_geometric -- run with Transform2Act's .venv-gpu")
    sys.exit(1)


DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# helpers: ragged graph batches, the way their `batch_data` builds them
# --------------------------------------------------------------------------

def random_tree_edges(n, rng):
    """A rooted tree, which is what `robot.get_gnn_edges()` returns -- both
    directions per edge, as PyG wants."""
    edges = []
    for child in range(1, n):
        parent = int(rng.integers(0, child))
        edges.append([parent, child])
        edges.append([child, parent])
    if not edges:  # a one-node graph has no edges at all
        return np.zeros((2, 0), dtype=np.int64)
    return np.array(edges, dtype=np.int64).T


def make_batch(n_graphs, min_n, max_n, feat, rng, dtype):
    """Returns both representations of the SAME batch: their ragged
    concatenation, and the dense padded form a superset-model port would use."""
    sizes = rng.integers(min_n, max_n + 1, size=n_graphs)
    xs, edge_blocks, offset = [], [], 0
    for n in sizes:
        xs.append(torch.randn(n, feat, dtype=dtype))
        edge_blocks.append(torch.from_numpy(random_tree_edges(n, rng)) + offset)
        offset += n
    x_flat = torch.cat(xs).to(DEV)
    edges = torch.cat(edge_blocks, dim=1).to(DEV)

    N = int(max(sizes))
    x_dense = torch.zeros(n_graphs, N, feat, dtype=dtype, device=DEV)
    # A[g, i, j] = 1 when a message flows j -> i, matching PyG's convention
    # that `propagate` aggregates x_j at index edge_index[1].
    adj = torch.zeros(n_graphs, N, N, dtype=dtype, device=DEV)
    node_mask = torch.zeros(n_graphs, N, dtype=torch.bool, device=DEV)
    pos = 0
    for g, n in enumerate(sizes):
        x_dense[g, :n] = x_flat[pos:pos + n]
        node_mask[g, :n] = True
        block = edge_blocks[g] - pos
        adj[g, block[1], block[0]] = 1.0
        pos += n
    return x_flat, edges, x_dense, adj, node_mask, sizes


# --------------------------------------------------------------------------
# 1. dense GraphConv against PyG
# --------------------------------------------------------------------------

def dense_graph_conv(x_dense, adj, lin_l_w, lin_l_b, lin_r_w):
    """PyG's GraphConv is `lin_l(sum_{j in N(i)} x_j) + lin_r(x_i)`, with the
    bias living on lin_l only. As dense tensors that is one bmm and two matmuls
    -- no gather, no scatter, no variable-length anything."""
    aggr = torch.bmm(adj, x_dense)
    return aggr @ lin_l_w.t() + lin_l_b + x_dense @ lin_r_w.t()


def check_dense_conv(dtype):
    rng = np.random.default_rng(0)
    F_IN, F_OUT = 17, 23
    x_flat, edges, x_dense, adj, mask, sizes = make_batch(
        64, 3, 12, F_IN, rng, dtype)

    conv = GraphConv(F_IN, F_OUT, aggr="add", bias=True).to(DEV).to(dtype)
    with torch.no_grad():
        ref = conv(x_flat, edge_index=edges)
        got = dense_graph_conv(x_dense, adj, conv.lin_l.weight,
                               conv.lin_l.bias, conv.lin_r.weight)
    # Compare only real nodes: padding rows are not defined by the reference.
    got_flat = got[mask]
    err = (got_flat - ref).abs().max().item()
    scale = ref.abs().max().item()
    print(f"  dense GraphConv vs PyG [{dtype}]: max abs err {err:.3e} "
          f"(values up to {scale:.2f})  -> {'MATCH' if err < 1e-5 * max(scale, 1) else 'MISMATCH'}")
    return err


# --------------------------------------------------------------------------
# 2. IndexLinear: their loop vs one batched gather+bmm
# --------------------------------------------------------------------------

class TheirIndexLinear(nn.Module):
    """Copied from design_opt/models/jsmlp.py so the comparison is against the
    real thing rather than my memory of it."""

    def __init__(self, input_dim, out_dim, max_index=256):
        super().__init__()
        self.out_dim = out_dim
        self.W = nn.Parameter(torch.zeros(max_index, out_dim, input_dim))
        self.b = nn.Parameter(torch.zeros(max_index, out_dim))
        init.kaiming_uniform_(self.W, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.W)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x, ind):
        uni_ind = ind.unique()
        out = torch.zeros((x.shape[0], self.out_dim), device=x.device,
                          dtype=x.dtype)
        for ind_i in uni_ind:
            W = self.W[ind_i]
            b = self.b[ind_i]
            x_ind = ind == ind_i
            out[x_ind] = torch.addmm(b, x[x_ind], W.t())
        return out


def batched_index_linear(x, ind, W, b):
    """Same map, no loop: gather each row's matrix and do one batched matvec.

    Costs `len(x) * out_dim * in_dim` of gathered weight memory, which is the
    trade -- at 50k nodes and 128x128 that is 3.3 GB in fp32, so this is the
    right shape only when the per-row matrices are small or the batch is
    chunked. The loop is O(unique indices) kernels; this is 1.
    """
    Wg = W[ind]                        # [n, out, in]
    bg = b[ind]                        # [n, out]
    return torch.baddbmm(bg.unsqueeze(1), x.unsqueeze(1), Wg.transpose(1, 2)).squeeze(1)


def check_index_linear(dtype, n_nodes=50_000, n_types=40, dim=128, quiet=False):
    torch.manual_seed(0)
    lin = TheirIndexLinear(dim, dim).to(DEV).to(dtype)
    x = torch.randn(n_nodes, dim, dtype=dtype, device=DEV)
    ind = torch.randint(0, n_types, (n_nodes,), device=DEV)

    with torch.no_grad():
        ref = lin(x, ind)
        got = batched_index_linear(x, ind, lin.W, lin.b)
    err = (got - ref).abs().max().item()
    if not quiet:
        print(f"  batched IndexLinear vs loop [{dtype}]: max abs err {err:.3e} "
              f"-> {'MATCH' if err < 1e-4 else 'MISMATCH'}")

    def timeit(fn, iters=20):
        for _ in range(3):
            fn()
        torch.cuda.synchronize() if DEV.type == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize() if DEV.type == "cuda" else None
        return (time.perf_counter() - t0) / iters * 1e3

    with torch.no_grad():
        t_loop = timeit(lambda: lin(x, ind))
        t_batch = timeit(lambda: batched_index_linear(x, ind, lin.W, lin.b))
    print(f"    {n_nodes:>6,} nodes x {n_types:>3} body types, {dim}x{dim}: "
          f"loop {t_loop:6.2f} ms  batched {t_batch:6.2f} ms  "
          f"speedup {t_loop / t_batch:5.2f}x")
    return err


# --------------------------------------------------------------------------
# 3. the per-graph log-prob reduction: is float64 load-bearing?
# --------------------------------------------------------------------------

def cumsum_diff_sum(values, num_nodes_cum):
    """Their reduction, verbatim in shape: cumulative-sum the whole batch, read
    it at each graph boundary, then difference consecutive boundaries."""
    c = torch.cumsum(values, dim=0)
    c = c[num_nodes_cum - 1]
    return torch.cat([c[[0]], c[1:] - c[:-1]])


def segment_sum(values, sizes):
    """The same quantity, never forming a partial sum bigger than one graph."""
    seg = torch.repeat_interleave(
        torch.arange(len(sizes), device=values.device), sizes)
    out = torch.zeros(len(sizes), values.shape[-1], device=values.device,
                      dtype=values.dtype)
    return out.index_add_(0, seg, values)


def check_logprob_precision(n_graphs=4000, min_n=3, max_n=12):
    rng = np.random.default_rng(1)
    sizes_np = rng.integers(min_n, max_n + 1, size=n_graphs)
    sizes = torch.from_numpy(sizes_np).to(DEV)
    total = int(sizes_np.sum())
    # Node log-probs of a DiagGaussian over ~8 dims sit around -10 with a heavy
    # left tail; that magnitude is what makes the running cumsum large.
    vals64 = (torch.randn(total, 1, dtype=torch.float64, device=DEV) * 4.0
              - 10.0)
    cum = torch.from_numpy(np.cumsum(sizes_np)).to(DEV)

    ref = segment_sum(vals64, sizes)                       # fp64 segment sum
    variants = {
        "fp64 cumsum-diff (theirs)": cumsum_diff_sum(vals64, cum),
        "fp32 cumsum-diff": cumsum_diff_sum(vals64.float(), cum).double(),
        "fp32 segment-sum": segment_sum(vals64.float(), sizes).double(),
    }
    print(f"  per-graph log-prob over {n_graphs:,} graphs "
          f"({total:,} nodes, running cumsum reaches "
          f"{abs(float(vals64.sum())):.3e}):")
    for name, got in variants.items():
        err = (got - ref).abs().max().item()
        rel = err / ref.abs().mean().item()
        print(f"    {name:28s} max abs err {err:.3e}   "
              f"({100 * rel:.4f}% of a typical value)")


def main():
    print(f"device: {DEV}   torch {torch.__version__}")
    print("\n[1] dense masked-adjacency GraphConv vs torch_geometric")
    check_dense_conv(torch.float64)
    check_dense_conv(torch.float32)

    print("\n[2] IndexLinear (JSMLP's per-body-type weights)")
    check_index_linear(torch.float32)
    # One point would not settle it: the loop costs one kernel per DISTINCT
    # body index, so its penalty grows with the number of body types, while the
    # batched form pays a gathered-weight tensor that grows with node count.
    # Sweep both axes before calling this a win or a non-win.
    print("    -- sweep --")
    for n_nodes in (5_000, 50_000):
        for n_types in (8, 40, 256):
            check_index_linear(torch.float32, n_nodes=n_nodes,
                               n_types=n_types, quiet=True)

    print("\n[3] is float64 load-bearing, or is it just the reduction?")
    check_logprob_precision()


if __name__ == "__main__":
    main()
