"""2f step 4: the single-learner 2v2 smoke.

One `TeamActorCritic` drives all four ants, both teams, no opponent ring, a
handful of iterations. That is deliberately not a legitimate training setup --
a net optimising both sides of a zero-sum game is its own opponent -- and it is
the right shape for a smoke, whose question is "does the machinery run at four
agents", not "does it learn".

    PYTHONPATH=. MUJOCO_GL=osmesa .venv/bin/python \
        -m rower_soccer.competevo_port.train_team_smoke --iters 20

The checks are the design doc's (§8 step 4), and each one prints its number
rather than a verdict alone, because "no NaNs" is only reassuring next to the
value that could have been NaN.

--------------------------------------------------------------------------
Why there is no new PPO code here
--------------------------------------------------------------------------
`SelfPlayPPO.collect` already flattens `[n, A, ...]` over agents and drives one
net across every lane -- it is agent-count agnostic as written. The one thing
that does not fit is width: the env emits 56 and the team policy consumes 58
(reordered others + role one-hot). So the adaptation is an env WRAPPER that
presents the policy's observation, and the trainer is used unmodified.

Writing a fourth copy of the rollout loop to accommodate two extra columns
would have been the larger change and the one more likely to drift from the
validated one.
"""

import argparse
import json
import os
import time

import numpy as np
import torch


class TeamPolicyObsEnv:
    """`TeamRunToGoalDevEnv` presenting the POLICY's observation instead of the
    scene's.

    Delegates everything it does not override, so the trainer, the evaluator
    and the probes see the env they expect. `obs_dim` is the one attribute that
    must NOT be delegated -- the buffers are allocated from it.
    """

    def __init__(self, env, ac):
        self._env, self._ac = env, ac
        self.obs_dim = ac.obs_dim
        assert env.obs_dim == 1 + ac.design_dim + ac.env_sim_dim, (
            f"env obs {env.obs_dim} does not match the policy's expected "
            f"{1 + ac.design_dim + ac.env_sim_dim}")

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _expand(self, obs):
        """`[n, A, 56]` -> `[n, A, 58]`, each lane with ITS OWN role one-hot.

        The per-agent loop is the point: a single `expand_obs` call over the
        whole tensor would give every agent agent-0's role, which is exactly
        the class of silent bug `gate_team_policy` exists to catch.
        """
        return torch.stack(
            [self._ac.expand_obs(obs[:, a], a) for a in range(self._env.n_agents)],
            dim=1)

    def reset(self):
        return self._expand(self._env.reset())

    def step(self, action):
        obs, rew, done, info = self._env.step(action)
        return self._expand(obs), rew, done, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, default=512)
    p.add_argument("--rollout", type=int, default=64)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--minibatch", type=int, default=2048)
    p.add_argument("--policy-lr", type=float, default=5e-5)
    p.add_argument("--value-lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--down-rule", default="team_down")
    p.add_argument("--win-rule", default="team_first")
    p.add_argument("--goal-credit", default="team")
    p.add_argument("--warm-start", default=None,
                   help="a 1v1 policies.pt; ac_0 is widened into the team net. "
                        "The whole reason team_policy orders opp_near first.")
    p.add_argument("--out", default="runs/competevo_port/team_smoke")
    args = p.parse_args()

    from rower_soccer.competevo_port.dev_ppo import (DevActorCritic,
                                                     DevSelfPlayPPO)
    from rower_soccer.competevo_port.team_env import TeamRunToGoalDevEnv
    from rower_soccer.competevo_port.team_policy import (TeamActorCritic,
                                                         widen_from_1v1)

    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    env = TeamRunToGoalDevEnv(num_worlds=args.worlds, use_gpu=(device == "cuda"),
                              seed=args.seed, down_rule=args.down_rule,
                              win_rule=args.win_rule,
                              goal_credit=args.goal_credit)
    if args.warm_start:
        blob = torch.load(args.warm_start, map_location="cpu")
        src = DevActorCritic()
        src.load_state_dict(blob["ac_0"])
        ac = widen_from_1v1(src, n_agents=env.n_agents)
        print(f"warm start: ac_0 of {args.warm_start}, widened to "
              f"{env.n_agents} agents")
    else:
        ac = TeamActorCritic(n_agents=env.n_agents)
    ac = ac.to(device)

    trainer = DevSelfPlayPPO(TeamPolicyObsEnv(env, ac), ac,
                             rollout_len=args.rollout, epochs=args.epochs,
                             minibatch_size=args.minibatch,
                             policy_lr=args.policy_lr, value_lr=args.value_lr,
                             device=device)

    # The design head is the thing most likely to be silently inert: it reads
    # only the scale vector, so nothing about four agents touches it, and a
    # plumbing mistake that froze it would look like a healthy run.
    scale_w0 = ac.scale_mean.weight.detach().clone()
    scale_prev = scale_w0.clone()

    print(f"worlds {env.n} x {env.n_agents} agents, rollout {args.rollout}, "
          f"{args.iters} iters, obs {env.obs_dim} -> {ac.obs_dim}")
    rows, t0 = [], time.time()
    for it in range(args.iters):
        gae = trainer.collect()
        stats = trainer.update(*gae)
        alpha = trainer.alpha()
        row = {
            "iter": it,
            "sec": time.time() - t0,
            "steps": trainer.total_steps,
            "fwd_per_step": trainer.ep_fwd,
            "alpha": None if alpha is None else float(alpha),
            "diverged": env.n_diverged,
            # Cumulative drift AND this iteration's step. Cumulative alone
            # passes for a head that moved once and then froze, which is the
            # failure a plumbing bug would actually produce.
            "design_drift": float((ac.scale_mean.weight - scale_w0)
                                  .abs().max()),
            "design_step": float((ac.scale_mean.weight - scale_prev)
                                 .abs().max()),
            "design_std": float(ac.scale_log_std.exp().mean()),
            **{k: (float(v) if np.isscalar(v) or torch.is_tensor(v) else v)
               for k, v in stats.items()},
        }
        scale_prev = ac.scale_mean.weight.detach().clone()
        rows.append(row)
        with open(os.path.join(args.out, "log.json"), "w") as f:
            json.dump({"args": vars(args), "iters": rows}, f, indent=1)
        print(f"  it {it:3d}  fwd/step {row['fwd_per_step']:+.4f}  "
              f"alpha {row['alpha']}  diverged {row['diverged']}  "
              f"pi {stats.get('pi_loss', float('nan')):+.5f}  "
              f"vf {stats.get('vf_loss', float('nan')):.1f}  "
              f"design d{row['design_drift']:.2e}/s{row['design_step']:.2e}")

    print("\n=== smoke checks (design doc section 8, step 4)")
    ok = True

    def check(name, good, detail):
        nonlocal ok
        ok = ok and good
        print(f"  [{'PASS' if good else 'FAIL'}] {name} -- {detail}")

    finite = all(np.isfinite([r["fwd_per_step"], r.get("pi_loss", 0.0),
                              r.get("vf_loss", 0.0)]).all() for r in rows)
    check("no NaN/Inf in any logged quantity", finite,
          f"{len(rows)} iterations")
    check("0 diverged worlds", env.n_diverged == 0,
          f"n_diverged = {env.n_diverged}")
    alphas = [r["alpha"] for r in rows if r["alpha"] is not None]
    check("alpha schedule is intact and decreasing",
          len(alphas) == len(rows) and alphas == sorted(alphas, reverse=True),
          f"{alphas[0]:.6f} -> {alphas[-1]:.6f}" if alphas else "no alpha")
    # M2E section 5 puts the 1v1 dense reward near -3.0 per agent-step. This is
    # a ballpark check, not a target: four agents on a wider pitch is a
    # different task and the number is not required to match, only to be in the
    # same world rather than orders of magnitude away.
    fwd = float(np.mean([r["fwd_per_step"] for r in rows]))
    check("forward reward per step is in the 1v1 ballpark", abs(fwd) < 30.0,
          f"mean {fwd:+.4f} per agent-step")
    drift, last_step = rows[-1]["design_drift"], rows[-1]["design_step"]
    check("the design head is still moving on the LAST update",
          last_step > 0.0 and drift > 0.0,
          f"cumulative max |dW| = {drift:.3e}, last step {last_step:.3e}")

    torch.save({"ac": ac.state_dict(), "args": vars(args)},
               os.path.join(args.out, "policy.pt"))
    print(f"\n{'ALL CHECKS PASSED' if ok else 'CHECKS FAILED'} -- "
          f"{args.out}/log.json, {time.time() - t0:.0f}s")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
