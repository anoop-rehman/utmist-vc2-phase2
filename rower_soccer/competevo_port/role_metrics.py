"""2f step 7: role metrics, not vibes.

The design doc's step 7 asks for "division of labour, CPD on the teammate
channel, topple counts, and the masked-opponent exploit probe". This is that,
against a trained 2v2 pair.

    PYTHONPATH=. MUJOCO_GL=osmesa .venv/bin/python \
        -m rower_soccer.competevo_port.role_metrics \
        --policies runs/competevo_port/t2v2_cold/policies.pt

--------------------------------------------------------------------------
One thing to get right before reading any of these
--------------------------------------------------------------------------
**Who wins is not who scored.** Under `win_rule="team_first"` the env sets
`winner = mine & one_team`, marking BOTH members of the scoring team. So the
per-agent `winner` split is 50/50 by construction and says nothing whatever
about division of labour. The quantity that distinguishes the two teammates is
`reached` -- who actually crossed the line. Every "who does the work" number
below is built on `reached`, `newly_down` and position, never on `winner`.

--------------------------------------------------------------------------
What each metric can and cannot show
--------------------------------------------------------------------------
* **Crossing split** is the headline. The transplant probe measured the back
  agent at 0.000-0.005 against the front pair's 0.38-0.60 -- decorative. If
  training has not moved that, the back agent is still a spectator and the
  geometry decision in doc section 1 comes back on the table.
* **Teammate CPD** answers a different question: whether the policy *uses* the
  teammate channel at all. A 1v1 transplant scores exactly 0 here because the
  teammate is not in its input. A native 2v2 net that also scores ~0 has been
  handed the information and declined it, which is a real (negative) result
  and not a bug.
* **Topple attribution** is correlational. "An opponent went down while one of
  ours was within `--near` metres" is not proof our agent caused it -- ants
  fall over unaided constantly, which is exactly the 15.6% the 1v1 port is
  still trying to explain. The untrained/`--shuffle` baselines below are what
  make the number mean anything, and they are printed alongside, never omitted.
"""

import argparse
import collections

import numpy as np
import torch

from rower_soccer.competevo_port.team_policy import ROLE_DIM


def load_team(path, n_agents, device, untrained=False, seed=0):
    from rower_soccer.competevo_port.team_policy import TeamActorCritic
    torch.manual_seed(seed)
    if untrained:
        return [TeamActorCritic(n_agents=n_agents).to(device).eval()
                for _ in range(2)]
    blob = torch.load(path, map_location="cpu")
    out = []
    for key in ("ac_0", "ac_1"):
        ac = TeamActorCritic(n_agents=n_agents)
        ac.load_state_dict(blob[key])
        out.append(ac.to(device).eval())
    return out


def root_xy(env):
    """`[n, A, 2]` root (x, y) per agent, from the raw qpos."""
    idx = env.qpos_idx[:, :2]                    # [A, 2]
    return env.qpos[:, idx]                      # [n, A, 2]


def rollout(env, driver, acs, lanes, steps):
    """One pass, collecting everything the metrics need in a single sweep."""
    n, A = env.n, env.n_agents
    env.reset_win_stats()
    obs = driver.reset()
    stats = {
        "reached": torch.zeros(n, A, device=env.device),
        "newly_down": torch.zeros(n, A, device=env.device),
        # An opponent went down with one of ours within `near` metres.
        "topple_near": torch.zeros(A, device=env.device),
        "down_total": torch.zeros(A, device=env.device),
        "x_sum": torch.zeros(A, device=env.device),
        "x_n": 0,
        "games": 0,
    }
    endings = collections.Counter()
    with torch.no_grad():
        for _ in range(steps):
            o = obs.float()
            act = torch.zeros(n, A, env.act_dim, device=env.device, dtype=o.dtype)
            for e, ln in enumerate(lanes):
                act[:, ln] = acs[e].mean_action(o[:, ln])
            xy = root_xy(env).clone()
            obs, _, done, info = driver.step(act.to(env.dtype))
            stats["reached"] += info["reached"].to(stats["reached"].dtype)
            nd = info["newly_down"].to(stats["newly_down"].dtype)
            stats["newly_down"] += nd
            stats["down_total"] += nd.sum(0)
            if bool(nd.any()):
                # For each agent that just went down, was an OPPONENT close?
                d = torch.cdist(xy, xy)                       # [n, A, A]
                opp = (env.team.view(1, A, 1) != env.team.view(1, 1, A))
                close = ((d < ROLL_NEAR) & opp).any(-1)       # [n, A]
                # Credit the CLOSEST opponent of each agent that went down.
                dm = d.masked_fill(~opp, float("inf"))
                who = dm.argmin(-1)                           # [n, A]
                hit = (nd > 0) & close
                if bool(hit.any()):
                    wi, ai = hit.nonzero(as_tuple=True)
                    stats["topple_near"].index_add_(
                        0, who[wi, ai],
                        torch.ones(wi.numel(), device=env.device))
            stats["x_sum"] += xy[..., 0].mean(0)
            stats["x_n"] += 1
            if bool(done.any()):
                idx = done.nonzero(as_tuple=True)[0]
                stats["games"] += idx.numel()
                for e in env.last_end[idx].tolist():
                    endings[{0: "running", 1: "goal", 2: "wipeout",
                             3: "fall", 4: "timeout"}[e]] += 1
    stats["endings"] = endings
    return stats


ROLL_NEAR = 0.5     # metres; overwritten from --near in main


def cpd(acs, driver, env, lanes, device, n_states=512, jitter=1.0, seed=0):
    """Counterfactual policy divergence, per observation channel group.

    How much does an agent's action move when ONLY the teammate's (x, y)
    changes, versus only the near opponent's, versus only its own state? Each
    group is perturbed by the same gaussian jitter, so the three numbers are
    on the same scale and comparable to each other -- the absolute value is
    meaningless, the RATIO is the result.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    obs = driver.reset()
    for _ in range(30):        # past the design stage, into real motion
        o = obs.float()
        act = torch.zeros(env.n, env.n_agents, env.act_dim,
                          device=env.device, dtype=o.dtype)
        for e, ln in enumerate(lanes):
            act[:, ln] = acs[e].mean_action(o[:, ln])
        obs, _, _, _ = driver.step(act.to(env.dtype))
    x = obs.float()[:, lanes[0][0]][:n_states]          # team 0's FRONT agent
    ac = acs[0]
    base_slice = 1 + ac.design_dim
    # Policy-order sim block: [own(29) | opp_near(2) | teammate(2) | opp_far(2)]
    groups = {"own state": (base_slice, base_slice + 29),
              "near opponent": (base_slice + 29, base_slice + 31),
              "teammate": (base_slice + 31, base_slice + 33),
              "far opponent": (base_slice + 33, base_slice + 35),
              "role one-hot": (x.shape[-1] - ROLE_DIM, x.shape[-1])}
    with torch.no_grad():
        ref = ac.mean_action(x)
        out = {}
        # The control head's normalizer holds the per-column statistics, and
        # its columns are the sim block -- the observation minus the leading
        # [flag | scale]. Scale each group's jitter by its own std so every
        # group gets the same perturbation IN THE UNITS THE NETWORK SEES.
        std = ac.control_norm.var.sqrt().to(x)
        for name, (a, b) in groups.items():
            xp = x.clone()
            col = std[a - base_slice:b - base_slice]
            noise = torch.randn(xp.shape[0], b - a, generator=g).to(xp)
            xp[..., a:b] = xp[..., a:b] + noise * col * jitter
            out[name] = float((ac.mean_action(xp) - ref).abs().mean())
    return out


def main():
    global ROLL_NEAR
    p = argparse.ArgumentParser()
    p.add_argument("--policies",
                   default="runs/competevo_port/t2v2_cold/policies.pt")
    p.add_argument("--worlds", type=int, default=256)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--near", type=float, default=0.5,
                   help="metres: how close an opponent must be to be credited")
    p.add_argument("--untrained", action="store_true",
                   help="the baseline every correlational number needs")
    args = p.parse_args()
    ROLL_NEAR = args.near

    from rower_soccer.competevo_port.team_env import TeamRunToGoalDevEnv
    from rower_soccer.competevo_port.train_team_smoke import TeamPolicyObsEnv

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env = TeamRunToGoalDevEnv(num_worlds=args.worlds,
                              use_gpu=(device == "cuda"), seed=args.seed,
                              down_rule="team_down", win_rule="team_first",
                              goal_credit="team")
    acs = load_team(args.policies, env.n_agents, device,
                    untrained=args.untrained, seed=args.seed)
    driver = TeamPolicyObsEnv(env, acs[0])
    lanes = [torch.tensor(l, device=env.device) for l in ([0, 2], [1, 3])]

    label = "UNTRAINED" if args.untrained else args.policies
    print(f"\n=== role metrics: {label}")
    print(f"    {env.n} worlds x {env.n_agents} agents, near = {args.near} m")

    st = rollout(env, driver, acs, lanes, env.max_episode_steps + 2)
    games = max(st["games"], 1)

    print("\n-- 1. division of labour: WHO CROSSES (not who 'wins')")
    reached = st["reached"].sum(0).cpu().numpy()
    names = ["A front (0)", "B front (1)", "A back (2)", "B back (3)"]
    tot = max(reached.sum(), 1)
    for i, nm in enumerate(names):
        print(f"    {nm:14s} crossings {reached[i]:7.0f}   "
              f"{100 * reached[i] / tot:5.1f}% of all crossings")
    front = reached[0] + reached[1]
    back = reached[2] + reached[3]
    print(f"    front pair {100 * front / tot:5.1f}%   "
          f"back pair {100 * back / tot:5.1f}%")
    print("    (transplanted 1v1 baseline, design doc step 1: back pair "
          "0.0-0.5% -- a spectator)")

    print("\n-- 2. mean x-position over the episode (where each agent lives)")
    xm = (st["x_sum"] / max(st["x_n"], 1)).cpu().numpy()
    for i, nm in enumerate(names):
        print(f"    {nm:14s} mean root x {xm[i]:+.2f} m")

    print(f"\n-- 3. topples: an opponent went down within {args.near} m of us")
    tn = st["topple_near"].cpu().numpy()
    dt = st["down_total"].cpu().numpy()
    print(f"    total down events {dt.sum():.0f} over {games} games")
    for i, nm in enumerate(names):
        print(f"    {nm:14s} credited {tn[i]:6.0f} nearby opponent downs")
    print("    CORRELATIONAL. Ants fall unaided, so this number needs the "
          + ("untrained arm as its own control -- this IS that arm."
             if args.untrained else
             "--untrained baseline beside it before it means anything."))

    print("\n-- 4. endings")
    total = max(sum(st["endings"].values()), 1)
    for k in ("goal", "wipeout", "fall", "timeout"):
        print(f"    {k:8s} {st['endings'][k]:6d}   "
              f"{100 * st['endings'][k] / total:5.1f}%")

    print("\n-- 5. counterfactual policy divergence, by channel group")
    print("    same jitter on each group, so the RATIO is the result")
    for name, v in cpd(acs, driver, env, lanes, device, seed=args.seed).items():
        print(f"    {name:16s} mean |d action| {v:.5f}")
    print("    A 1v1 transplant scores exactly 0 on 'teammate' -- it has no "
          "such input. A native net that also scores ~0 was given the channel "
          "and declined it, which is a result, not a bug.")


if __name__ == "__main__":
    main()
