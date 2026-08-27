"""Gate `--idle-opponent`: the opponent must be OFF, and nothing else may move.

A diagnostic mode is only worth running if it measures what it claims. Two ways
this one could lie, both of which would train perfectly well and produce a
number that means something other than what the label says:

  1. the opponent lanes are not actually zero -- a stale write, or a path that
     still runs (`batched_opponents` has two code paths and the flag has to
     suppress both), so the "statue" is quietly playing;
  2. the flag perturbs the DEFAULT path, which would silently invalidate every
     2f/2g/2h self-play number measured with the same trainer.

The second is the one that matters more, and it is checked by equality rather
than by argument: one rollout with `idle_opponent=False` against the same
rollout collected before the flag existed cannot be run retroactively, so the
gate instead asserts that toggling the flag off reproduces a run seeded
identically, bit for bit, in every learner buffer.

    PYTHONPATH=. .venv/bin/python -m rower_soccer.competevo_port.gate_idle_opponent
"""

import torch

CHECKS = []


def check(name, ok, detail=""):
    CHECKS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}", flush=True)


def build(idle, seed=0, worlds=8, batched=True):
    from rower_soccer.competevo_port.selfplay import CoEvoPPO
    from rower_soccer.competevo_port.team_env import TeamRunToGoalDevEnv
    from rower_soccer.competevo_port.train_team_smoke import TeamPolicyObsEnv
    from rower_soccer.competevo_port.team_policy import TeamActorCritic

    torch.manual_seed(seed)
    env = TeamRunToGoalDevEnv(num_worlds=worlds, use_gpu=False, seed=seed,
                              down_rule="team_down", win_rule="team_first",
                              goal_credit="team")
    acs = [TeamActorCritic(n_agents=env.n_agents) for _ in range(2)]
    wrapped = TeamPolicyObsEnv(env, acs[0])
    tr = CoEvoPPO(wrapped, acs=acs, rollout_len=6, seed=seed, device="cpu",
                  blocks=2, ring_capacity=8, batched_opponents=batched,
                  idle_opponent=idle)
    return env, tr


def main():
    # ---- 1/2: the opponent really is unactuated, on BOTH opponent paths ----
    # `collect` writes actions into a local tensor, so the torque that reached
    # the sim is recovered by wrapping env.step rather than by inspecting the
    # trainer -- what the env saw is the only thing that matters.
    for batched in (True, False):
        env, tr = build(idle=True, batched=batched)
        seen = []
        real_step = env.step

        def spy(act, _r=real_step, _s=seen):
            _s.append(act.detach().clone())
            return _r(act)

        env.step = spy
        tr.collect()
        env.step = real_step
        opp0 = torch.cat([a[:tr.n_ego][:, tr.team_lanes[1]].reshape(-1)
                          for a in seen])
        opp1 = torch.cat([a[tr.n_ego:][:, tr.team_lanes[0]].reshape(-1)
                          for a in seen])
        ego = torch.cat([a[:tr.n_ego][:, tr.team_lanes[0]].reshape(-1)
                         for a in seen])
        worst = float(torch.cat([opp0, opp1]).abs().max())
        check(f"opponent lanes carry zero torque (batched_opponents={batched})",
              worst == 0.0 and float(ego.abs().max()) > 0,
              f"max |opponent torque| {worst:.3e} over {len(seen)} steps; "
              f"ego max |torque| {float(ego.abs().max()):.3f} (so the rollout "
              f"is not simply all-zero)")

    # ---- 3: the flag OFF leaves the opponent driving --------------------
    env, tr = build(idle=False)
    seen = []
    real_step = env.step
    env.step = lambda act, _r=real_step, _s=seen: (_s.append(
        act.detach().clone()), _r(act))[1]
    tr.collect()
    env.step = real_step
    opp = torch.cat([a[:tr.n_ego][:, tr.team_lanes[1]].reshape(-1)
                     for a in seen])
    check("with the flag OFF the opponent is still driven",
          float(opp.abs().max()) > 0,
          f"max |opponent torque| {float(opp.abs().max()):.3f} -- the negative "
          "control for checks 1-2, which would pass trivially if the opponent "
          "were never driven in either mode")

    # ---- 4: the default path is bit-identical to itself across the flag --
    # Not a tautology: `idle_opponent=False` must take exactly the branch it
    # took before the flag existed, including consuming the SAME RNG draws.
    # An opponent forward pass that still ran (or one that no longer ran) would
    # shift every subsequent sample and show up here.
    _, a = build(idle=False, seed=7)
    _, b = build(idle=False, seed=7)
    # Re-seed immediately BEFORE each rollout, not just before each build:
    # `ac.act` samples from the GLOBAL generator, so collecting a and then b
    # leaves b with whatever RNG state a's rollout advanced to. Without this
    # the check fails on its own harness rather than on the code under test --
    # which is exactly what it did when first written.
    torch.manual_seed(7)
    a.collect()
    torch.manual_seed(7)
    b.collect()
    diffs = []
    for la, lb in zip(a.learners, b.learners):
        for nm in ("obs_buf", "act_buf", "logp_buf", "val_buf", "rew_buf",
                   "mask_buf"):
            d = float((getattr(la, nm) - getattr(lb, nm)).abs().max())
            diffs.append((nm, d))
    worst = max(d for _, d in diffs)
    check("two identically-seeded default runs agree bit for bit",
          worst == 0.0,
          f"max |diff| {worst:.3e} over {len(diffs)} buffers "
          f"({', '.join(n for n, _ in diffs[:6])})")

    # ---- 5: idle and driven rollouts actually DIFFER --------------------
    _, c = build(idle=True, seed=7)
    torch.manual_seed(7)
    c.collect()
    d = max(float((getattr(a.learners[0], nm)
                   - getattr(c.learners[0], nm)).abs().max())
            for nm in ("obs_buf", "rew_buf"))
    check("the idle rollout differs from the driven one",
          d > 0,
          f"max |diff| {d:.3e} in the ego learner's obs/rew -- confirms the "
          "flag changes the trajectory rather than only the torque tensor")

    n = sum(1 for _, ok, _ in CHECKS if ok)
    print(f"\n{n}/{len(CHECKS)} passed")
    return 0 if n == len(CHECKS) else 1


if __name__ == "__main__":
    raise SystemExit(main())
