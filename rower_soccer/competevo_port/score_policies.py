"""Score a saved 1v1 or 2v2 pair on the SAME protocol, so the two compare.

Design doc §8 step 6 wants the 2v2 run judged "against the 1v1 run as the
control on every shared quantity ... train return per step, eval length,
ending histogram". The per-iteration eval in the two trainers does not support
that: the 1v1 one logs `eval_win` / `eval_len` and no endings, the 2v2 one logs
endings. Comparing them would be comparing two protocols.

So this is one scorer, run after the fact against a checkpoint, producing the
same table for either. It is also the protocol M2E §10 used for its headline
(384 mean-action games), so its 1v1 numbers are comparable to that table too.

    PYTHONPATH=. MUJOCO_GL=osmesa .venv/bin/python \
        -m rower_soccer.competevo_port.score_policies \
        --policies runs/competevo_port/m2e_fixed/policies.pt --kind 1v1
    ... --policies runs/competevo_port/t2v2_cold/policies.pt --kind 2v2

Mean actions, no sampling: `mean_action` takes the non-sampling branch, which
is what their eval does and what makes the number a property of the policy
rather than of one draw.
"""

import argparse
import collections

import numpy as np
import torch


def build(kind, worlds, seed, device):
    from rower_soccer.competevo_port.dev_env import RunToGoalDevEnv
    if kind == "1v1":
        env = RunToGoalDevEnv(num_worlds=worlds, use_gpu=(device == "cuda"),
                              seed=seed)
        # One net per AGENT at 1v1; the lanes are the agents.
        return env, env, [[0], [1]]
    from rower_soccer.competevo_port.team_env import TeamRunToGoalDevEnv
    env = TeamRunToGoalDevEnv(num_worlds=worlds, use_gpu=(device == "cuda"),
                              seed=seed, down_rule="team_down",
                              win_rule="team_first", goal_credit="team")
    return env, None, [[0, 2], [1, 3]]


def load(kind, path, n_agents, device):
    from rower_soccer.competevo_port.dev_ppo import DevActorCritic
    blob = torch.load(path, map_location="cpu")
    if kind == "1v1":
        out = []
        for key in ("ac_0", "ac_1"):
            ac = DevActorCritic()
            ac.load_state_dict(blob[key])
            out.append(ac.to(device).eval())
        return out
    from rower_soccer.competevo_port.team_policy import TeamActorCritic
    out = []
    for key in ("ac_0", "ac_1"):
        ac = TeamActorCritic(n_agents=n_agents)
        ac.load_state_dict(blob[key])
        out.append(ac.to(device).eval())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policies", required=True)
    p.add_argument("--kind", choices=("1v1", "2v2"), required=True)
    p.add_argument("--worlds", type=int, default=384)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--episodes", type=int, default=1,
                   help="episodes per world; games = worlds x episodes")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    env, _, lanes = build(args.kind, args.worlds, args.seed, device)
    acs = load(args.kind, args.policies, env.n_agents, device)
    lanes = [torch.tensor(l, device=env.device) for l in lanes]

    driver = env
    if args.kind == "2v2":
        from rower_soccer.competevo_port.train_team_smoke import TeamPolicyObsEnv
        driver = TeamPolicyObsEnv(env, acs[0])

    env.reset_win_stats()
    obs = driver.reset()
    endings, lens = collections.Counter(), []
    both = 0
    budget = args.episodes * (env.max_episode_steps + 2) + 8
    with torch.no_grad():
        for _ in range(budget):
            o = obs.float()
            act = torch.zeros(env.n, env.n_agents, env.act_dim,
                              device=env.device, dtype=o.dtype)
            for e, ln in enumerate(lanes):
                act[:, ln] = acs[e].mean_action(o[:, ln])
            obs, _, done, info = driver.step(act.to(env.dtype))
            if bool(done.any()):
                idx = done.nonzero(as_tuple=True)[0]
                won = info["winner"][idx].any(-1)
                fell = info["fell"][idx].any(-1)
                trunc = info["truncated"][idx]
                both += int((won & fell).sum())
                # Precedence is the env's own: a crossed goal line ends the
                # game whatever the torso height is doing.
                for w, f, t in zip(won.tolist(), fell.tolist(), trunc.tolist()):
                    endings["goal" if w else "fell" if f
                            else "timeout" if t else "other"] += 1
                lens.extend(env.last_len[idx].float().cpu().tolist())

    total = sum(endings.values()) or 1
    per_agent = np.atleast_1d(env.win_rate())
    print(f"\n{args.kind}  {args.policies}")
    print(f"  {total} games over {env.n} worlds x {env.n_agents} agents, "
          f"mean actions")
    for k in ("goal", "fell", "timeout", "other"):
        print(f"    {k:8s} {endings[k]:6d}   {100.0 * endings[k] / total:5.1f}%")
    print(f"    mean episode length {np.mean(lens) if lens else 0:.1f} "
          f"of {env.max_episode_steps}")
    print(f"    win rate per agent  {[round(float(x), 3) for x in per_agent]}")
    if args.kind == "2v2":
        teams = env.team.tolist()
        pt = [round(float(sum(per_agent[i] for i in range(env.n_agents)
                              if teams[i] == e)), 3) for e in range(2)]
        print(f"    win rate per TEAM   {pt}   (sum {sum(pt):.3f})")
    else:
        print(f"    win rate summed     {float(per_agent.sum()):.3f}")
    # A goal and a fall in the same episode means the two categories are not
    # cleanly exclusive; if it is ever large the histogram needs a rethink.
    print(f"    (episodes where a goal and a fall coincided: {both})")


if __name__ == "__main__":
    main()
