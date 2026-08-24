"""Gate for `team_policy.py` -- 2f step 3's regression, plus what it misses.

The doc asks for one check: a 1v1 net widened to 2v2 must reproduce its 1v1
behaviour on the same states. That check alone passes trivially if `expand_obs`
is the identity on the wrong columns, so this file also asserts the column
mapping directly and breaks it on demand.

    PYTHONPATH=. MUJOCO_GL=osmesa .venv/bin/python \
        -m rower_soccer.competevo_port.gate_team_policy [--worlds 64]

Every check runs on states from the real batched 2v2 env, not on `randn`: the
column mapping is a claim about what the env emits, and random tensors cannot
falsify it.
"""

import argparse

import torch

from rower_soccer.competevo_port.dev_ppo import DevActorCritic
from rower_soccer.competevo_port.team_policy import (OWN_DIM, ROLE_DIM,
                                                     TeamActorCritic,
                                                     others_permutation,
                                                     widen_from_1v1)

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def one_v_one_view(obs, design_dim=20):
    """The 52-dim 1v1 observation embedded in a POLICY-ORDER 2v2 one.

    Only meaningful after `expand_obs`, and only because the reorder puts
    `opp_near` immediately after the own-state block -- which is precisely the
    property under test, so this function is deliberately dumb: a prefix slice.
    """
    return obs[..., :1 + design_dim + OWN_DIM + 2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--worlds", type=int, default=64)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"device: {dev}")
    print("\n[0] column mapping, asserted directly")
    # A synthetic observation whose every entry is its own column index makes
    # the permutation readable rather than inferred.
    A = 4
    tac = TeamActorCritic(n_agents=A).to(dev)
    env_w = 1 + tac.design_dim + tac.env_sim_dim
    probe = torch.arange(env_w, dtype=torch.float32, device=dev)
    got = tac.expand_obs(probe, agent_idx=0)
    base = 1 + tac.design_dim
    perm = others_permutation(A)
    want = list(range(base + OWN_DIM))
    for o in perm:
        want += [base + OWN_DIM + 2 * o, base + OWN_DIM + 2 * o + 1]
    want = torch.tensor(want, dtype=torch.float32, device=dev)
    check("expand_obs permutes the others block as [opp_near, teammate, opp_far]",
          torch.equal(got[:len(want)], want),
          f"perm={perm}")
    check("role one-hot is appended last, front for agent 0",
          torch.equal(got[-ROLE_DIM:],
                      torch.tensor([1.0, 0.0], device=dev)))
    check("role one-hot is back for agent 2",
          torch.equal(tac.expand_obs(probe, 2)[-ROLE_DIM:],
                      torch.tensor([0.0, 1.0], device=dev)))
    check("widened widths are 58 obs / 37 control input",
          tac.obs_dim == 58 and tac.sim_obs_dim == 37,
          f"obs {tac.obs_dim}, control {tac.sim_obs_dim}")

    print("\n[1] real 2v2 states")
    from rower_soccer.competevo_port.team_env import TeamRunToGoalDevEnv
    env = TeamRunToGoalDevEnv(num_worlds=args.worlds, use_gpu=(dev == "cuda"),
                              seed=args.seed, down_rule="team_down",
                              win_rule="team_first", goal_credit="team")
    ac1 = DevActorCritic().to(dev)
    # Give the normalizer real statistics -- with n = 0 RunningNorm is the
    # identity and the whole `_widen_norm` question is untested.
    ac1.train()
    obs = env.reset()
    states = []
    for _ in range(args.steps):
        o = obs.float()
        states.append(o.clone())
        a = torch.zeros(env.n, env.n_agents, ac1.act_dim, device=dev,
                        dtype=env.dtype)
        obs, _, _, _ = env.step(a)
    flat56 = torch.cat(states, 0).reshape(-1, states[0].shape[-1])
    print(f"  {flat56.shape[0]} agent-states of width {flat56.shape[1]}")

    wide = TeamActorCritic(n_agents=env.n_agents).to(dev)
    x = wide.expand_obs(flat56, agent_idx=0)
    ac1.control_norm(x[..., 1 + 20:1 + 20 + 31])
    ac1.vf_norm(one_v_one_view(x))
    ac1.eval()
    check("1v1 normalizer has real statistics", ac1.control_norm.n.item() > 0,
          f"n={int(ac1.control_norm.n.item())}")

    print("\n[2] THE REGRESSION -- widened net vs the 1v1 net it came from")
    tac = widen_from_1v1(ac1, n_agents=env.n_agents).to(dev).eval()
    worst_a = worst_v = worst_lp = 0.0
    for agent_idx in range(env.n_agents):
        x = tac.expand_obs(flat56, agent_idx)
        ref_obs = one_v_one_view(x)
        with torch.no_grad():
            a_new, a_ref = tac.mean_action(x), ac1.mean_action(ref_obs)
            v_new, v_ref = tac.value(x), ac1.value(ref_obs)
            lp_new = tac.log_prob(x, a_ref)
            lp_ref = ac1.log_prob(ref_obs, a_ref)
        worst_a = max(worst_a, (a_new - a_ref).abs().max().item())
        worst_v = max(worst_v, (v_new - v_ref).abs().max().item())
        worst_lp = max(worst_lp, (lp_new - lp_ref).abs().max().item())
    # Note what is NOT done here: the teammate and far-opponent channels are
    # left at their real values. The doc's version zeroes them; this is the
    # stronger statement -- the widened net ignores the new channels because
    # every weight reading them is zero, so it agrees on states where they are
    # informative, not only on states where they are blank.
    check("mean_action identical on real (unzeroed) 2v2 states, all 4 roles",
          worst_a == 0.0, f"max |da| = {worst_a:.3e}")
    check("value identical", worst_v == 0.0, f"max |dv| = {worst_v:.3e}")
    check("log_prob identical", worst_lp == 0.0, f"max |dlp| = {worst_lp:.3e}")

    print("\n[3] negative controls -- each must FAIL to agree")
    x = tac.expand_obs(flat56, 0)
    with torch.no_grad():
        a_ref = ac1.mean_action(one_v_one_view(x))

        # (a) the bug this file exists to catch: scene order fed straight in,
        # so the teammate's (x, y) lands in the opponent's columns.
        bad = tac.sim_perm.clone()
        bad[OWN_DIM:OWN_DIM + 4] = bad[[OWN_DIM + 2, OWN_DIM + 3,
                                        OWN_DIM, OWN_DIM + 1]]
        saved, tac.sim_perm = tac.sim_perm, bad
        d_perm = (tac.mean_action(tac.expand_obs(flat56, 0))
                  - a_ref).abs().max().item()
        tac.sim_perm = saved

        # (b) a leading column that MUST matter: blank the near opponent.
        x_blank = x.clone()
        x_blank[..., 1 + 20 + OWN_DIM:1 + 20 + OWN_DIM + 2] = 0.0
        d_blank = (tac.mean_action(x_blank) - a_ref).abs().max().item()

        # (c) a trailing column that must NOT matter yet: the role one-hot is
        # read only by zeroed weights at init.
        x_role = x.clone()
        x_role[..., -ROLE_DIM:] = torch.tensor([0.0, 1.0], device=dev)
        d_role = (tac.mean_action(x_role) - a_ref).abs().max().item()

    check("(a) unpermuted others block CHANGES the action", d_perm > 1e-6,
          f"max |da| = {d_perm:.3e}")
    check("(b) blanking the near opponent CHANGES the action", d_blank > 1e-6,
          f"max |da| = {d_blank:.3e}")
    check("(c) flipping the role one-hot does NOT change it at init",
          d_role == 0.0, f"max |da| = {d_role:.3e}")

    print("\n[4] save/load round trip")
    # `sim_perm` and `roles` are registered non-persistent, so they are NOT in
    # the state dict and are rebuilt by __init__ from n_agents. If that ever
    # drifted, every post-hoc consumer (score_policies, role_metrics,
    # render_team) would load a policy whose column mapping silently differs
    # from the one it trained with -- and nothing else here would catch it,
    # because in-process the buffers are simply the same objects.
    import io
    buf = io.BytesIO()
    torch.save(tac.state_dict(), buf)
    buf.seek(0)
    reborn = TeamActorCritic(n_agents=env.n_agents).to(dev)
    missing, unexpected = reborn.load_state_dict(torch.load(buf))
    reborn.eval()
    x = tac.expand_obs(flat56, 0)
    with torch.no_grad():
        dd = (tac.mean_action(x) - reborn.mean_action(x)).abs().max().item()
        dvv = (tac.value(x) - reborn.value(x)).abs().max().item()
    check("state dict is complete (no missing or unexpected keys)",
          not missing and not unexpected,
          f"{len(missing)} missing, {len(unexpected)} unexpected")
    check("a reloaded policy is identical", dd == 0.0 and dvv == 0.0,
          f"max |da| {dd:.3e}, |dv| {dvv:.3e}")
    check("non-persistent buffers rebuild identically",
          torch.equal(tac.sim_perm, reborn.sim_perm)
          and torch.equal(tac.roles, reborn.roles))
    check("RunningNorm statistics survive the round trip",
          float(reborn.control_norm.n) == float(tac.control_norm.n)
          and float(reborn.control_norm.n) > 0,
          f"n = {float(reborn.control_norm.n):.0f}")

    n_ok = sum(ok for _, ok, _ in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} checks passed")
    if n_ok != len(RESULTS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
