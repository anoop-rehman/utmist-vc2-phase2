"""Gate: does the dense policy produce THEIR actions on THEIR observations?

Runs their env with their trained checkpoint, captures real observations at each
of the three stages, and feeds each one through both their `Transform2ActPolicy`
and our `DenseTransform2ActPolicy` loaded from the same weights.

Real observations, not random tensors, because the failure this is guarding
against is a layout mistake -- the design towers slice
`[attr_fixed | ... | attr_design]` out of the middle of the observation, the
adjacency has a direction, and `body_index` selects which weight bank each node
uses. Random inputs of the right shape would pass all three while being wrong.

    cd /workspace/Transform2Act && source env-gpu.sh
    .venv-gpu/bin/python /workspace/utmist-vc2-phase2/rower_soccer/t2a_port/gate_dense_policy.py
"""

import argparse
import glob
import os
import re
import sys

sys.path.append("/workspace/Transform2Act")
sys.path.append("/workspace/utmist-vc2-phase2")
os.chdir("/workspace/Transform2Act")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from design_opt.agents.transform2act_agent import Transform2ActAgent  # noqa: E402
from design_opt.utils.config import Config  # noqa: E402
from rower_soccer.t2a_port.dense_policy import (  # noqa: E402
    DenseTransform2ActPolicy, DenseTransform2ActValue)

STAGES = ["skel_trans", "attr_trans", "execution"]
_results = []


def check(name, ok, detail=""):
    _results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")


def tensorfy(np_list):
    if isinstance(np_list[0], list):
        return [[torch.tensor(x) for x in y] for y in np_list]
    return [torch.tensor(y) for y in np_list]


def collect_states(agent, n_episodes):
    """One state per stage per episode, in their list format."""
    env, out = agent.env, []
    for _ in range(n_episodes):
        state = env.reset()
        for _ in range(agent.cfg.skel_transform_nsteps + 6):
            out.append((STAGES[int(state[2].item())], state))
            with torch.no_grad():
                a = agent.policy_net.select_action(
                    tensorfy([state]), True).numpy().astype(np.float64)
            state, _, done, _ = env.step(a)
            if done:
                break
    return out


def to_dense(state):
    """Their `[obs, edges, stage, num_nodes, body_ind]` -> dense tensors.

    `edges` is `[2, E]` with PyG's convention that `propagate` aggregates the
    source `edges[0]` at the target `edges[1]`, so the dense adjacency is
    `adj[target, source]`. That orientation is written the right way round here,
    but it turns out not to be a hazard: `robot.get_gnn_edges()` emits BOTH
    directions of every tree edge, so the adjacency is symmetric and a port that
    transposed it would be numerically identical. Measured, not assumed -- the
    gate below asserts the symmetry rather than leaving it as folklore.
    """
    obs, edges, _, num_nodes, body_ind = state
    n = int(num_nodes[0])
    x = torch.as_tensor(obs, dtype=torch.float64).reshape(1, n, -1)
    adj = torch.zeros(1, n, n, dtype=torch.float64)
    e = torch.as_tensor(np.asarray(edges), dtype=torch.long)
    if e.numel():
        adj[0, e[1], e[0]] = 1.0
    ind = torch.as_tensor(np.asarray(body_ind), dtype=torch.long).reshape(1, n)
    return x, adj, ind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="hopper_gpu")
    ap.add_argument("--checkpoint", default="latest",
                    help="an epoch number, 'best', or 'latest' (the default): "
                         "the highest epoch_*.p present")
    args = ap.parse_args()

    torch.set_default_dtype(torch.float64)
    cfg = Config(args.cfg, tmp=False)
    # Was hardcoded to 400. That checkpoint died with the pod, and hardcoding
    # an epoch means the gate is unrunnable for the first 400 epochs of every
    # re-run -- exactly when you most want to know the port still matches.
    ckpt = args.checkpoint
    if ckpt == "latest":
        eps = sorted(int(m.group(1)) for m in
                     (re.search(r"epoch_(\d+)\.p$", f) for f in
                      glob.glob(os.path.join(cfg.model_dir, "epoch_*.p")))
                     if m)
        if not eps:
            raise SystemExit(
                f"no epoch_*.p under {cfg.model_dir} -- train first, or pass "
                f"--checkpoint best if best.p exists")
        ckpt = eps[-1]
        print(f"checkpoint: latest = epoch {ckpt} (of {len(eps)} present)")
    elif ckpt != "best":
        ckpt = int(ckpt)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    agent = Transform2ActAgent(cfg=cfg, dtype=torch.float64,
                               device=torch.device("cpu"), seed=cfg.seed,
                               num_threads=1, training=False, checkpoint=ckpt)
    theirs = agent.policy_net
    theirs.eval()

    ours = DenseTransform2ActPolicy(
        cfg.policy_specs, agent.env.attr_fixed_dim, agent.env.sim_obs_dim,
        agent.env.attr_design_dim, agent.env.skel_num_action,
        agent.env.control_action_dim).double()
    ours.eval()

    sd = {k: v for k, v in theirs.state_dict().items()}
    missing, unexpected = ours.load_their_state_dict(sd, strict=True)
    check("their checkpoint loads strictly into the dense policy",
          not missing and not unexpected,
          f"{len(sd)} tensors, 0 missing, 0 unexpected")

    states = collect_states(agent, 6)
    seen = {s: 0 for s in STAGES}
    worst = {s: 0.0 for s in STAGES}
    for stage, state in states:
        x, adj, ind = to_dense(state)
        with torch.no_grad():
            ref = theirs.select_action(tensorfy([state]), True)
            got = ours.mean_action(stage, x, adj, ind)[0]
        # Their skeleton branch takes an argmax over logits; a tie would make
        # the comparison depend on tie-breaking rather than on the maths, so
        # compare the continuous slices exactly and the discrete one for equality.
        cont = (got[:, :-1] - ref[:, :-1]).abs().max().item()
        disc = (got[:, -1] - ref[:, -1]).abs().max().item()
        err = max(cont, disc if stage == "skel_trans" else 0.0)
        worst[stage] = max(worst[stage], err)
        seen[stage] += 1

    for stage in STAGES:
        if not seen[stage]:
            check(f"{stage}: observations were collected", False,
                  "0 states seen -- the gate did not exercise this stage")
            continue
        check(f"{stage}: dense actions match theirs", worst[stage] < 1e-9,
              f"{seen[stage]} real observations, max abs diff {worst[stage]:.2e}")

    # A gate that cannot fail is not a gate. The first control tried here was
    # "transposing the adjacency must change the answer" -- it does not, because
    # their edges are undirected, so that control could never have failed for a
    # wrong reason OR a right one. Record the symmetry as its own check and use
    # two controls that do bite.
    asym = sum(1 for _, st in states
               if (lambda a: (a - a.transpose(1, 2)).abs().max().item() > 0)(
                   to_dense(st)[1]))
    check("their graph is undirected, so adjacency ORIENTATION cannot be got "
          "wrong", asym == 0,
          f"0 of {len(states)} adjacencies are asymmetric "
          "(get_gnn_edges emits both directions)")

    # Compare the CONTINUOUS head output, not the assembled action. The
    # skeleton head ends in an argmax, and an argmax absorbs a perturbation
    # whenever the winning logit stays the winner -- the first version of this
    # control read 42/60 for exactly that reason, which measures how decisive
    # the logits are rather than whether the perturbation reached them.
    def head_out(stage, x, adj, ind):
        if stage == "execution":
            return ours.control(x, adj, ind)
        return (ours.skel if stage == "skel_trans" else ours.attr)(
            ours.design_input(x), adj, ind)

    for label, mutate in (
            ("dropping every edge", lambda x, a, i: (x, torch.zeros_like(a), i)),
            ("rolling body_index by one",
             lambda x, a, i: (x, a, torch.roll(i, 1, dims=-1)))):
        broke = tried = 0
        for stage, state in states[:60]:
            x, adj, ind = to_dense(state)
            if float(adj.sum()) == 0 or int(ind.numel()) < 2:
                continue
            tried += 1
            with torch.no_grad():
                ref = head_out(stage, x, adj, ind)
                alt = head_out(stage, *mutate(x, adj, ind))
            if (ref - alt).abs().max().item() > 1e-9:
                broke += 1
        check(f"negative control: {label} changes the head output",
              tried > 0 and broke == tried,
              f"{broke}/{tried} sampled states changed")

    # Everything above ran one graph at a time, which is not what this policy
    # exists for: the port groups worlds by topology and evaluates the group as
    # one [G, N, F] batch. A broadcasting mistake would be invisible at G=1 and
    # wrong for every real batch, so stack same-topology states and require the
    # per-graph answers to be unchanged.
    by_shape = {}
    for stage, state in states:
        x, adj, ind = to_dense(state)
        key = (stage, x.shape[1], tuple(ind[0].tolist()),
               tuple(adj[0].reshape(-1).tolist()))
        by_shape.setdefault(key, []).append((x, adj, ind))
    groups = [v for v in by_shape.values() if len(v) > 1]
    worst_batch, checked, biggest = 0.0, 0, 0
    for grp in groups:
        xs = torch.cat([g[0] for g in grp])
        adj = torch.cat([g[1] for g in grp])
        ind = torch.cat([g[2] for g in grp])
        stage = next(st for (st, *_), v in by_shape.items() if v is grp)
        with torch.no_grad():
            batched = ours.mean_action(stage, xs, adj, ind)
            singles = torch.cat([ours.mean_action(stage, *g) for g in grp])
        worst_batch = max(worst_batch, (batched - singles).abs().max().item())
        checked += len(grp)
        biggest = max(biggest, len(grp))
    check("a G-graph batch gives the same answers as G single graphs",
          checked > 0 and worst_batch < 1e-12,
          f"{len(groups)} same-topology groups, {checked} graphs, largest G="
          f"{biggest}, max abs diff {worst_batch:.2e}")

    # ---- mixed topologies, one dense batch ------------------------------
    # The trainer buckets PPO minibatches by (stage, node count), not by
    # topology, and relies on `adj`/`ind` being per-ROW. That is a different
    # claim from the same-topology batch check above -- there ONE adjacency was
    # broadcast over the batch, here every row carries its own -- and the
    # update's speed depends on it, so it gets its own check.
    #
    # A trained checkpoint emits one or two topologies, so waiting for two
    # DIFFERENT real graphs of the same size to turn up would leave this check
    # silently unexercised. Node permutations of a real graph are genuinely
    # different graphs of the same size, so they are used to build the mixed
    # batch deliberately.
    worst_mixed, n_mixed, n_bucket = 0.0, 0, 0
    rng = np.random.RandomState(0)
    by_stage_n = {}
    for stage, state in states:
        x, adj, ind = to_dense(state)
        if x.shape[1] < 3:
            continue
        p_ = rng.permutation(x.shape[1])
        pt = torch.as_tensor(p_, dtype=torch.long)
        perm = (x[:, pt], adj[:, pt][:, :, pt], ind[:, pt])
        by_stage_n.setdefault((stage, x.shape[1]), []).extend(
            [(x, adj, ind), perm])
    for (stage, _), grp in by_stage_n.items():
        graphs = {tuple(g[2][0].tolist()) + tuple(g[1].reshape(-1).tolist())
                  for g in grp}
        if len(graphs) < 2:
            continue
        n_bucket += 1
        xs = torch.cat([g[0] for g in grp])
        adj = torch.cat([g[1] for g in grp])
        ind = torch.cat([g[2] for g in grp])
        with torch.no_grad():
            batched = head_out(stage, xs, adj, ind)
            singles = torch.cat([head_out(stage, *g) for g in grp])
        worst_mixed = max(worst_mixed, (batched - singles).abs().max().item())
        n_mixed += len(grp)
    check("DIFFERENT topologies of the same size batch together",
          n_mixed > 0 and worst_mixed < 1e-12,
          f"{n_bucket} (stage, n) buckets carrying >1 distinct graph, "
          f"{n_mixed} graphs, max abs diff {worst_mixed:.2e}")

    # ---- the training path: per-graph log-probs and the critic -----------
    # Added for 3d steps 5/6. `mean_action` alone is enough to SAMPLE with, and
    # nothing above would notice if the log-prob or the value were wrong -- but
    # PPO is exactly those two quantities, so they get the same treatment: their
    # network, their weights, their observations.
    worst_lp = {s: 0.0 for s in STAGES}
    n_lp = {s: 0 for s in STAGES}
    gen = torch.Generator().manual_seed(0)
    for stage, state in states:
        x, adj, ind = to_dense(state)
        with torch.no_grad():
            # A SAMPLED action, not the mean -- a log-prob evaluated only at the
            # mode would miss any error in the std or in the quadratic term.
            act, lp_ours = ours.act(stage, x, adj, ind, generator=gen)
            lp_theirs = theirs.get_log_prob(tensorfy([state]), [act[0]])
        d = abs(float(lp_ours[0]) - float(lp_theirs.reshape(-1)[0]))
        worst_lp[stage] = max(worst_lp[stage], d)
        n_lp[stage] += 1
        # log_prob() must agree with what act() reported for the same action.
        with torch.no_grad():
            d2 = abs(float(ours.log_prob(stage, x, adj, ind, act)[0])
                     - float(lp_ours[0]))
        assert d2 < 1e-12, f"act/log_prob disagree by {d2}"
    for stage in STAGES:
        check(f"{stage}: per-graph log-prob matches their cumsum reduction",
              n_lp[stage] > 0 and worst_lp[stage] < 1e-9,
              f"{n_lp[stage]} sampled actions, max abs diff "
              f"{worst_lp[stage]:.2e}")

    theirs_v = agent.value_net
    theirs_v.eval()
    ours_v = DenseTransform2ActValue(cfg.value_specs, agent.state_dim).double()
    ours_v.load_state_dict(theirs_v.state_dict(), strict=True)
    ours_v.eval()
    worst_v = 0.0
    for stage, state in states:
        x, adj, ind = to_dense(state)
        with torch.no_grad():
            ref = theirs_v(tensorfy([state]))
            got = ours_v(stage, x, adj)
        worst_v = max(worst_v, abs(float(ref.reshape(-1)[0]) - float(got[0])))
    check("dense critic matches theirs", worst_v < 1e-9,
          f"{len(states)} real observations, max abs diff {worst_v:.2e}")

    # Negative control for the critic: their value is read off the FIRST node,
    # not pooled. Averaging instead must move the answer, or the check above
    # would pass for a critic that pooled.
    moved = tried = 0
    for stage, state in states[:60]:
        x, adj, ind = to_dense(state)
        if x.shape[1] < 2:
            continue
        tried += 1
        with torch.no_grad():
            G, N = x.shape[:2]
            flag = torch.zeros(G, N, 3, dtype=x.dtype)
            flag[..., STAGES.index(stage)] = 1.0
            h = ours_v.mlp(ours_v.gnn(ours_v.norm(torch.cat([x, flag], -1)), adj))
            nodes = ours_v.value_head(h)
            if abs(float(nodes[0, 0]) - float(nodes.mean())) > 1e-9:
                moved += 1
    check("negative control: the critic reads node 0, not a pool",
          tried > 0 and moved == tried,
          f"{moved}/{tried} states where pooling would differ")

    # ---- padding: graphs of different sizes in one block -----------------
    # The PPO update pads every stage's graphs to a common node count so one
    # forward serves the whole minibatch (29 buckets -> 3, and 382 ms per
    # gradient step -> ~30). Padding is the hazard `dense_policy.py`'s
    # docstring names, so it gets three checks: the answers must be unchanged,
    # the mask must be load-bearing, and the running statistics must not see a
    # zero row.
    import torch.nn.functional as _F
    pad_worst, pad_n = 0.0, 0
    ctrl_moved, ctrl_tried = 0, 0
    for stage, state in states:
        x, adj, ind = to_dense(state)
        n = x.shape[1]
        if n >= 10:
            continue
        pad = 10 - n
        xp = _F.pad(x, (0, 0, 0, pad))
        ap = _F.pad(adj, (0, pad, 0, pad))
        ip = _F.pad(ind, (0, pad))
        m = torch.zeros(1, 10, dtype=torch.bool)
        m[:, :n] = True
        gen2 = torch.Generator().manual_seed(1)
        with torch.no_grad():
            act, _ = ours.act(stage, x, adj, ind, generator=gen2)
            ref = ours.log_prob(stage, x, adj, ind, act)
            actp = _F.pad(act, (0, 0, 0, pad))
            got = ours.log_prob(stage, xp, ap, ip, actp, m)
            bad = ours.log_prob(stage, xp, ap, ip, actp, None)
        pad_worst = max(pad_worst, abs(float(ref[0]) - float(got[0])))
        pad_n += 1
        ctrl_tried += 1
        ctrl_moved += abs(float(bad[0]) - float(ref[0])) > 1e-6
    check("padded graphs give the same per-graph log-prob", 
          pad_n > 0 and pad_worst < 1e-12,
          f"{pad_n} graphs padded to 10 nodes, max abs diff {pad_worst:.2e}")
    check("  control: dropping the node mask changes the log-prob",
          ctrl_tried > 0 and ctrl_moved == ctrl_tried,
          f"{ctrl_moved}/{ctrl_tried} graphs")

    # RunningNorm must not take a padded row as a sample. Compare the running
    # statistics after a masked update on a padded block against an update on
    # the unpadded rows, and against the unmasked (wrong) version.
    from rower_soccer.t2a_port.dense_policy import RunningNorm
    xs = torch.randn(7, 5, 4, dtype=torch.float64)
    mk = torch.zeros(7, 5, dtype=torch.bool)
    mk[:, :3] = True
    a_, b_, c_ = RunningNorm(4), RunningNorm(4), RunningNorm(4)
    a_.train(); b_.train(); c_.train()
    a_(xs, mk)                       # padded + masked
    b_(xs[:, :3])                    # the real rows, unpadded
    c_(xs)                           # padded, mask ignored
    d_ok = float((a_.mean - b_.mean).abs().max()) < 1e-12 and \
        float((a_.std - b_.std).abs().max()) < 1e-12
    d_bad = float((c_.mean - b_.mean).abs().max())
    check("RunningNorm's statistics ignore padded rows", d_ok,
          f"masked-vs-unpadded mean/std diff < 1e-12; n {int(a_.n)} vs "
          f"{int(b_.n)}")
    check("  control: an unmasked update DOES corrupt them", d_bad > 1e-6,
          f"mean moves by {d_bad:.2e}")

    n_fail = sum(1 for _, ok in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
