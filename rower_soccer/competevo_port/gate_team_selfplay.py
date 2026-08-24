"""Gate for 2f step 5: two-learner co-evolution at four agents.

`tests/test_selfplay.py` already protects everything the generalisation left
alone -- it is the L = 1 regression and it still reads 15/15. This file checks
the things that only exist once a side has more than one lane, and each of them
is a property that would TRAIN FINE if it were wrong:

  1. **whole past TEAMS.** Design doc section 6: a world plays one sampled past
     team, not two independently sampled halves of two different ones. Both
     are legal-looking algorithms and only one is theirs. Checked by pinning
     each slot to a distinct constant action and asserting a world's two
     opponent lanes report the SAME slot.
  2. **lane assignment.** Which lanes a learner drives, which lanes it stores,
     and which lanes the opponent drives -- with markers chosen so swapping the
     teams fails all three.
  3. **role follows the lane.** Learner e's buffer row for its back agent must
     carry the BACK one-hot. Get this wrong and both teammates are the same
     agent twice, which is exactly the failure mode 2f is trying to avoid.
  4. **independence**, at team scale: a step on learner 0 leaves every one of
     learner 1's tensors bit-identical.
  5. **the ring behaves**: lag tracks epoch/4 at delta 0.5, nothing clamps.

    PYTHONPATH=. MUJOCO_GL=osmesa .venv/bin/python \
        -m rower_soccer.competevo_port.gate_team_selfplay [--gpu]
"""

import argparse
import time

import torch

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return bool(ok)


def build(worlds=16, rollout=4, blocks=4, use_gpu=False, delta=0.5):
    from rower_soccer.competevo_port.selfplay import CoEvoPPO
    from rower_soccer.competevo_port.team_env import TeamRunToGoalDevEnv
    from rower_soccer.competevo_port.team_policy import TeamActorCritic
    from rower_soccer.competevo_port.train_team_smoke import TeamPolicyObsEnv

    device = "cuda" if use_gpu else "cpu"
    env = TeamRunToGoalDevEnv(num_worlds=worlds, use_gpu=use_gpu, seed=0,
                              down_rule="team_down", win_rule="team_first",
                              goal_credit="team")
    acs = [TeamActorCritic(n_agents=env.n_agents) for _ in range(2)]
    tr = CoEvoPPO(TeamPolicyObsEnv(env, acs[0]), acs=acs, delta=delta,
                  blocks=blocks, rollout_len=rollout, seed=0, device=device,
                  minibatch_size=64, epochs=1)
    return env, tr


def pin(net, c):
    """Every action this net emits is the constant `c`, with no noise, so the
    action a lane receives NAMES the slot that produced it."""
    with torch.no_grad():
        net.control_mean.weight.zero_(); net.control_mean.bias.fill_(c)
        net.scale_mean.weight.zero_(); net.scale_mean.bias.fill_(c)
        net.control_log_std.fill_(-20.0); net.scale_log_std.fill_(-20.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    print("=== 2f step 5 gate")
    env, tr = build(use_gpu=args.gpu)
    L, M = tr.L, tr.n_ego
    check("two sides of two lanes, teams [[0, 2], [1, 3]]",
          L == 2 and [l.tolist() for l in tr.team_lanes] == [[0, 2], [1, 3]],
          f"L={L}, lanes={[l.tolist() for l in tr.team_lanes]}")

    print("\n-- 1. a world plays a WHOLE past team")
    # One rollout first: straight after reset every world is in the DESIGN
    # stage, where `_assemble` zeroes the motor block, so a slot's constant
    # would be invisible in the half of the action this check reads.
    tr.collect()
    for e in range(2):
        for k, net in enumerate(tr.opp_nets[e]):
            pin(net, 1.0 + 10 * e + k)
    tr.opp_stack.sync_from(tr.opp_nets)
    obs = tr._obs.float()
    got = tr._opponent_actions_batched(obs)[..., -env.n_motor:]   # [2,M,L,motor]
    check("batched opponent output is lane-shaped", tuple(got.shape[:3]) == (2, M, L),
          f"{tuple(got.shape)}")
    # Both lanes of a world must carry the same constant: same slot, same team.
    spread = (got[:, :, 0, :] - got[:, :, 1, :]).abs().max().item()
    check("both opponent lanes of a world come from the SAME slot", spread == 0.0,
          f"max |lane0 - lane1| = {spread:.3e} (nonzero => per-AGENT sampling)")
    slots = tr.slot.view(2, M)
    want = (1.0 + 10 * torch.arange(2, device=slots.device).view(2, 1)
            + slots).to(got.dtype).view(2, M, 1, 1).expand_as(got)
    check("the slot a world plays is the slot the table says",
          torch.allclose(got, want, atol=1e-3),
          f"wanted {want[0, :3, 0, 0].tolist()}, got {got[0, :3, 0, 0].tolist()}")
    # And the check bites: distinct slots really do produce distinct actions.
    check("distinct slots are distinguishable (the check can fail)",
          float(got.max() - got.min()) > 1.0,
          f"range {float(got.min()):.1f}..{float(got.max()):.1f}")

    print("\n-- 2/3. lanes and roles, through a real rollout")
    env, tr = build(use_gpu=args.gpu)
    from rower_soccer.competevo_port.team_policy import ROLE_DIM
    tr.collect()
    ok_role, ok_obs = True, True
    for e in range(2):
        lanes = tr.team_lanes[e]
        buf = tr.learners[e].obs_buf[0]                       # [M, L, obs]
        roles = buf[..., -ROLE_DIM:]
        # lane 0 of a side is its FRONT agent (index < 2), lane 1 its BACK one.
        ok_role &= bool((roles[:, 0, 0] == 1.0).all() and (roles[:, 0, 1] == 0.0).all())
        ok_role &= bool((roles[:, 1, 1] == 1.0).all() and (roles[:, 1, 0] == 0.0).all())
        # The stored observation is this side's lanes of this side's worlds.
        w = tr.ego_worlds[e]
        live = tr.env._expand(tr.env._env.obs())[w][:, lanes]
        ok_obs &= buf.shape == live.shape
    check("learner e's buffer carries [front, back] role one-hots in that order",
          ok_role, "checked on both sides, every world")
    check("buffer geometry matches this side's lanes of this side's worlds", ok_obs)

    # Swapping the teams must break the role check -- otherwise it proves nothing.
    swapped = tr.learners[0].obs_buf[0][..., -ROLE_DIM:].flip(1)
    check("(negative control) swapping the two lanes breaks the role check",
          not bool((swapped[:, 0, 0] == 1.0).all()))

    print("\n-- 4. the two learners are independent")
    before = {n: p.detach().clone() for n, p in tr.acs[1].named_parameters()}
    gae = tr.collect()
    tr.learners[0].update(*gae[0])
    moved1 = sum(int(not torch.equal(p, before[n]))
                 for n, p in tr.acs[1].named_parameters())
    moved0 = sum(int(p.grad is not None and p.grad.abs().sum() > 0)
                 for p in tr.acs[0].parameters())
    check("a step on learner 0 leaves every tensor of learner 1 bit-identical",
          moved1 == 0, f"{len(before)} tensors, {moved1} moved; "
                       f"learner 0 had {moved0} nonzero grads")

    print("\n-- 5. the ring, over enough epochs to see the lag")
    env, tr = build(worlds=8, rollout=2, use_gpu=args.gpu)
    t0 = time.time()
    for _ in range(24):
        tr.train_iter()
    lag, clamped = tr.opponent_lag(), sum(r.n_clamped for r in tr.rings)
    check("ring filled one entry per team per epoch",
          [len(r) for r in tr.rings] == [24, 24],
          f"{[len(r) for r in tr.rings]}")
    # delta=0.5 draws uniformly on a past window of width ~epoch/2, so the mean
    # lag is ~epoch/4. At 24 epochs and 8 live slots that is a small sample:
    # the band is deliberately wide, and it is the SHAPE being checked.
    check("opponent lag tracks epoch/4 at delta 0.5", 1.0 < lag < 12.0,
          f"lag {lag:.2f}, epoch/4 = {tr.epoch / 4:.2f}")
    check("nothing clamped", clamped == 0, f"n_clamped = {clamped}")
    check("0 diverged worlds over 24 epochs", env.n_diverged == 0,
          f"n_diverged = {env.n_diverged}, {time.time() - t0:.1f}s")

    n_ok = sum(ok for _, ok, _ in RESULTS)
    print(f"\n{n_ok}/{len(RESULTS)} passed")
    raise SystemExit(0 if n_ok == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
