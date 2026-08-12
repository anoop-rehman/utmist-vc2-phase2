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

import os
import sys

sys.path.append("/workspace/Transform2Act")
sys.path.append("/workspace/utmist-vc2-phase2")
os.chdir("/workspace/Transform2Act")

import numpy as np  # noqa: E402
import torch  # noqa: E402

from design_opt.agents.transform2act_agent import Transform2ActAgent  # noqa: E402
from design_opt.utils.config import Config  # noqa: E402
from rower_soccer.t2a_port.dense_policy import DenseTransform2ActPolicy  # noqa: E402

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
    torch.set_default_dtype(torch.float64)
    cfg = Config("hopper_gpu", tmp=False)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    agent = Transform2ActAgent(cfg=cfg, dtype=torch.float64,
                               device=torch.device("cpu"), seed=cfg.seed,
                               num_threads=1, training=False, checkpoint=400)
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

    n_fail = sum(1 for _, ok in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} passed")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
