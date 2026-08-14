"""Gate for the 2v2 scene and the downed-player / win rules.

Plain-python (no pytest in this venv):

    PYTHONPATH=. .venv/bin/python -m rower_soccer.competevo_port.tests.test_team2v2

Everything here runs on the CPU backend with a handful of worlds, so it is
seconds, not minutes.

Every check in this file is written so that it CAN fail. Three of them are
negative controls and each one names the mutation it is guarding against; the
`--break` flag applies those mutations so you can watch them fail rather than
take the word of a green run:

    ... --break bitmask     # use their 2-agent contact formula at n=4
    ... --break nomask      # stop zeroing a downed agent's torque
    ... --break payslacker  # pay a downed agent its survive bonus
"""

import argparse
import sys

import mujoco
import numpy as np
import torch

from rower_soccer.competevo_port.run_to_goal_env import SURVIVE_BONUS
from rower_soccer.competevo_port.scene import build_dev_scene
from rower_soccer.competevo_port.team_env import TeamRunToGoalDevEnv
from rower_soccer.competevo_port.team_scene import (build_dev_team_scene,
                                                    colliding_pairs,
                                                    dev_team_xml,
                                                    team_init_pose)

BREAK = set()
RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    return bool(ok)


def _pairs(n_agents, naive=False):
    return colliding_pairs(
        mujoco.MjModel.from_xml_string(dev_team_xml(n_agents, naive_bitmask=naive)))


# ---------------------------------------------------------------------------
def test_bitmask_matches_theirs_at_two_agents():
    """Our N-agent mask must reproduce THEIR 2-agent contact behaviour exactly.
    The integers differ (theirs is contype=1-i/conaffinity=i, ours is a bit per
    agent); the invariant is the set of colliding (owner, owner) pairs."""
    theirs = colliding_pairs(build_dev_scene(2)[0])
    ours = _pairs(2, naive="bitmask" in BREAK)
    check("n=2: our bitmask == their merger's collision set", ours == theirs,
          f"ours {sorted(ours)}")


def test_naive_bitmask_is_broken_at_four_agents():
    """The control. Their formula at n=4 must be measurably wrong, and ours
    must be right -- both halves are asserted, so this fails if either the
    'broken' thing silently works or ours silently breaks.

    What is wrong with theirs at n=4, concretely: `contype=1-i, conaffinity=i`
    gives agent 2 (contype -1, conaffinity 2) and agent 3 (contype -2,
    conaffinity 3). -1 is all bits set, so 2 and 3 SELF-collide, and the
    teammate pairs (0,2) and (1,3) do not collide at all."""
    naive = _pairs(4, naive=True)
    ours = _pairs(4, naive="bitmask" in BREAK)
    want = {(-1, -1)} | {(-1, i) for i in range(4)} | {
        (i, j) for i in range(4) for j in range(i + 1, 4)}
    self_hits = {p for p in naive if p[0] == p[1] and p[0] >= 0}
    missing = want - naive
    ok = (self_hits == {(2, 2), (3, 3)} and missing == {(0, 2), (1, 3)}
          and ours == want)
    check("n=4: their formula is broken, ours is not", ok,
          f"naive self-collides {sorted(self_hits)}, misses {sorted(missing)}; "
          f"ours == complete graph: {ours == want}")


def test_first_two_agents_are_the_1v1_pair():
    """A 2v2 scene truncated to its first two agents must be the validated 1v1
    scene: same spawns, same goals, same masses, same actuator order."""
    m2, meta2 = build_dev_scene(2)
    m4, meta4 = build_dev_team_scene(4)
    ok = True
    for i in (0, 1):
        a2, a4 = meta2.agents[i], meta4.agents[i]
        ok &= (a2.goal_x == a4.goal_x and a2.move_left == a4.move_left
               and a2.qpos == a4.qpos and a2.qvel == a4.qvel
               and a2.ctrl == a4.ctrl)
        ok &= np.allclose(m2.body_pos[a2.torso_body], m4.body_pos[a4.torso_body])
    # the whole robot, per agent, must be identical in mass
    mass2 = np.array([m2.body_mass[meta2.agents[i].body_ids].sum() for i in (0, 1)])
    mass4 = np.array([m4.body_mass[meta4.agents[i].body_ids].sum() for i in range(4)])
    ok &= np.allclose(mass2, mass4[:2]) and np.allclose(mass4, mass4[0])
    check("2v2 agents 0,1 are the 1v1 pair (slices, goals, spawns, mass)", ok,
          f"per-agent mass {mass4.round(4).tolist()}")


def test_obs_layout_at_two_agents_is_the_1v1_layout():
    """`build_dev_team_scene(2)` must produce the 52-dim 1v1 observation, so
    the team code path is a superset of the validated one rather than a fork."""
    _, meta2 = build_dev_scene(2)
    _, metaT = build_dev_team_scene(2)
    ok = (meta2.obs_dim == metaT.obs_dim == 52
          and meta2.act_dim == metaT.act_dim == 28
          and [a.other_qpos_xy for a in meta2.agents]
          == [a.other_qpos_xy for a in metaT.agents])
    check("n=2 team scene reproduces the 1v1 obs layout", ok,
          f"obs_dim {metaT.obs_dim}, other_xy "
          f"{[a.other_qpos_xy for a in metaT.agents]}")


def test_other_xy_is_teammate_first():
    """The 4-agent observation must be ROLE-symmetric: slot 0 is my teammate,
    slots 1-2 my opponents, for every agent. Theirs (`get_other_qpos()[:2]`)
    would give every agent except agent 0 the same 'other' -- agent 0."""
    m, meta = build_dev_team_scene(4)
    env = TeamRunToGoalDevEnv(num_worlds=2, use_gpu=False, seed=0)
    obs = env.reset()
    xy = obs[0, :, 50:].reshape(4, 3, 2).numpy()
    root = obs[0, :, 21:23].numpy()
    ok = True
    for i in range(4):
        want = [meta.teammate[i]] + list(meta.opponents[i])
        ok &= np.allclose(xy[i], root[want], atol=1e-9)
    check("obs 'others' block is [teammate, opp, opp] for every agent", ok,
          f"agent0 sees {xy[0].round(2).tolist()}")


# ---------------------------------------------------------------------------
def _drop(env, world, agent, z=0.1):
    """Put one agent's torso through the floor so it reads as fallen."""
    env.qpos[world, env.root_z_idx[agent]] = z
    env.backend.forward()


def _blank(env):
    return torch.zeros(env.n, env.n_agents, env.act_dim, dtype=env.dtype)


def _past_design(env, steps=1):
    obs = env.reset()
    for _ in range(steps):
        obs, *_ = env.step(_blank(env))
    return obs


def test_down_rule_any_vs_frozen_termination():
    """The blocking issue itself. Under their rule one fallen agent ends the
    episode; under `frozen` it must not."""
    out = {}
    for rule in ("any", "frozen", "ignore", "team_down"):
        env = TeamRunToGoalDevEnv(num_worlds=2, use_gpu=False, seed=0,
                                  down_rule=rule)
        _past_design(env)
        _drop(env, 0, 1)
        _, _, done, info = env.step(_blank(env))
        out[rule] = (bool(done[0]), bool(info["down"][0, 1]))
    ok = (out["any"][0] and not out["frozen"][0] and not out["team_down"][0]
          and not out["ignore"][0]
          and out["frozen"][1] and out["team_down"][1] and not out["ignore"][1])
    check("one agent down: 'any' ends the episode, 'frozen' does not", ok,
          f"{out}")


def test_frozen_agent_is_disabled_and_unpaid():
    """A downed agent must (a) have its torque zeroed and (b) earn exactly 0.

    NEGATIVE CONTROL: `--break nomask` removes (a) and `--break payslacker`
    removes (b); either makes this fail. Without (b) a corpse collects the +1
    survive bonus for the rest of the episode and lying down becomes a
    strategy."""
    env = TeamRunToGoalDevEnv(num_worlds=2, use_gpu=False, seed=0,
                              down_rule="frozen")
    if "nomask" in BREAK:
        env._mask_motors = lambda m: m
    if "payslacker" in BREAK:
        _orig = env.terms
        def _paying_terms(a, bad=None):
            t = _orig(a, bad)
            t["dense"] = t["dense"] + (1.0 - t["alive"]) * SURVIVE_BONUS
            return t
        env.terms = _paying_terms

    _past_design(env)
    _drop(env, 0, 1)
    a = torch.full((env.n, env.n_agents, env.act_dim), 0.9, dtype=env.dtype)
    _, rew, _, info = env.step(a)          # the step it goes down on
    _, rew2, _, info2 = env.step(a)        # and the step after
    ctrl = env.ctrl.reshape(env.n, env.n_agents, -1)
    dense = info2["dense"][0].numpy()
    ok = (abs(float(dense[1])) < 1e-12 and abs(float(dense[0])) > 1e-6
          and float(ctrl[0, 1].abs().max()) == 0.0
          and float(ctrl[0, 0].abs().max()) > 0.0)
    check("frozen agent: torque zeroed and dense reward exactly 0", ok,
          f"dense {dense.round(4).tolist()}  |ctrl| down={float(ctrl[0,1].abs().max()):.3f} "
          f"alive={float(ctrl[0,0].abs().max()):.3f}")


def test_win_rule_team_first_vs_exactly_one():
    """Two TEAMMATES crossing on the same step is a win, not a draw. Their
    `num_reached_goal != 1 -> pay nobody` rule calls it a draw."""
    res = {}
    for rule in ("exactly_one", "team_first"):
        env = TeamRunToGoalDevEnv(num_worlds=2, use_gpu=False, seed=0,
                                  down_rule="frozen", win_rule=rule)
        _past_design(env)
        # both of team 0 (agents 0, 2) over the +4 line
        for ag in (0, 2):
            env.qpos[0, env.qpos_idx[ag][0]] = 6.0
        env.backend.forward()
        _, rew, done, info = env.step(_blank(env))
        res[rule] = (info["parse"][0].numpy().round(1).tolist(), bool(done[0]),
                     info["winner"][0].numpy().tolist())
    ok = (res["exactly_one"][0] == [0.0] * 4 and res["exactly_one"][1]
          and res["team_first"][0] == [1000.0, -1000.0, 1000.0, -1000.0])
    check("both teammates cross: 'exactly_one' pays nobody, 'team_first' wins",
          ok, f"{res}")


def test_simultaneous_cross_by_both_teams_is_a_draw():
    env = TeamRunToGoalDevEnv(num_worlds=2, use_gpu=False, seed=0,
                              down_rule="frozen", win_rule="team_first")
    _past_design(env)
    env.qpos[0, env.qpos_idx[0][0]] = 6.0      # team 0 over +4
    env.qpos[0, env.qpos_idx[1][0]] = -6.0     # team 1 over -4
    env.backend.forward()
    _, _, done, info = env.step(_blank(env))
    ok = (float(info["parse"][0].abs().max()) == 0.0 and bool(done[0])
          and not bool(info["winner"][0].any()))
    check("both teams cross on the same step: draw, episode ends", ok,
          f"parse {info['parse'][0].numpy().round(1).tolist()}")


def test_goal_credit_variants():
    got = {}
    for credit in ("team", "scorer", "split"):
        env = TeamRunToGoalDevEnv(num_worlds=2, use_gpu=False, seed=0,
                                  down_rule="frozen", win_rule="team_first",
                                  goal_credit=credit)
        _past_design(env)
        env.qpos[0, env.qpos_idx[0][0]] = 6.0
        env.backend.forward()
        _, _, _, info = env.step(_blank(env))
        got[credit] = info["parse"][0].numpy().round(1).tolist()
    ok = (got["team"] == [1000.0, -1000.0, 1000.0, -1000.0]
          and got["scorer"] == [1000.0, -1000.0, 0.0, -1000.0]
          and got["split"] == [500.0, -500.0, 500.0, -500.0])
    check("goal_credit team/scorer/split pay what they say", ok, f"{got}")


def test_team_down_wipeout():
    """Both members of a team down => that team loses immediately, and the
    payout is the goal reward with the sign of the SURVIVING team."""
    env = TeamRunToGoalDevEnv(num_worlds=2, use_gpu=False, seed=0,
                              down_rule="team_down")
    _past_design(env)
    _drop(env, 0, 1)
    _, _, done1, _ = env.step(_blank(env))
    _drop(env, 0, 3)
    _, _, done2, info = env.step(_blank(env))
    p = info["parse"][0].numpy().round(1).tolist()
    ok = (not bool(done1[0]) and bool(done2[0])
          and p == [1000.0, -1000.0, 1000.0, -1000.0]
          and int(env.last_end[0]) == 2)
    check("team_down: one down continues, both down loses the match", ok,
          f"parse {p}, end code {int(env.last_end[0])}")


def test_recover_reposes_and_re_enables():
    env = TeamRunToGoalDevEnv(num_worlds=2, use_gpu=False, seed=0,
                              down_rule="recover", recover_steps=3)
    _past_design(env)
    _drop(env, 0, 1)
    zs, downs = [], []
    for _ in range(6):
        _, _, _, info = env.step(_blank(env))
        zs.append(round(float(env._root_z()[0, 1]), 3))
        downs.append(bool(info["down"][0, 1]))
    ok = (downs[0] and not any(downs[4:]) and zs[-1] > 0.28
          and env.n_recoveries >= 1)
    check("recover: agent is re-posed upright and re-enabled after N steps", ok,
          f"z {zs}  down {downs}  recoveries {env.n_recoveries}")


def test_downed_body_is_still_an_obstacle():
    """The whole point of "frozen" over "teleport the corpse out": the body
    stays in the way. Overlap a fallen agent with a standing one and count the
    contacts MuJoCo actually generates between them.

    NEGATIVE CONTROL: the same overlap between an agent and ITSELF must
    generate zero contacts (self-collision is off), and moving the fallen body
    10 m away must too. Both are asserted, so a model where everything collides
    with everything -- which would also pass a naive "contacts > 0" check --
    fails here.

    `env.cfrc_ext` cannot be used for this: MuJoCo only fills `cfrc_ext` in
    `mj_rnePostConstraint`, which neither stack calls, which is exactly why
    M2E_VALIDATION section 2 records the contact cost as a constant 0 on BOTH
    sides."""
    model, meta = build_dev_team_scene(4)
    data = mujoco.MjData(model)

    def n_between(other, dx, z_live=0.55, z_down=0.25):
        """`other` posed as a fallen ant (torso z = 0.25, below the 0.28 fall
        threshold), `agent0` posed at a walking torso height."""
        data.qpos[:] = model.qpos0
        a0, ao = meta.agents[0], meta.agents[other]
        data.qpos[a0.qpos[0] + 2] = z_live
        data.qpos[ao.qpos[0]] = data.qpos[a0.qpos[0]] + dx
        data.qpos[ao.qpos[0] + 1] = data.qpos[a0.qpos[0] + 1]
        data.qpos[ao.qpos[0] + 2] = z_down
        mujoco.mj_forward(model, data)
        own = lambda g: (model.geom(g).name.split("/")[0]
                         if model.geom(g).name.startswith("agent") else "w")
        return sum(1 for c in range(data.ncon)
                   if {own(data.contact[c].geom1), own(data.contact[c].geom2)}
                   == {"agent0", f"agent{other}"})

    opp_close, opp_far = n_between(1, 0.30), n_between(1, 10.0)
    mate_close = n_between(2, 0.30)
    ok = opp_close > 0 and mate_close > 0 and opp_far == 0
    check("a downed body is still a collidable obstacle", ok,
          f"agent0 vs fallen opponent: {opp_close} contacts at 0.30 m, "
          f"{opp_far} at 10 m; vs fallen TEAMMATE: {mate_close}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--break", dest="brk", default="",
                   help="comma list of deliberate breakages: "
                        "bitmask,nomask,payslacker")
    a = p.parse_args()
    BREAK.update(x for x in a.brk.split(",") if x)
    if BREAK:
        print(f"!! deliberately broken: {sorted(BREAK)} -- "
              f"the matching checks MUST fail\n")
    for fn in (test_bitmask_matches_theirs_at_two_agents,
               test_naive_bitmask_is_broken_at_four_agents,
               test_first_two_agents_are_the_1v1_pair,
               test_obs_layout_at_two_agents_is_the_1v1_layout,
               test_other_xy_is_teammate_first,
               test_down_rule_any_vs_frozen_termination,
               test_frozen_agent_is_disabled_and_unpaid,
               test_win_rule_team_first_vs_exactly_one,
               test_simultaneous_cross_by_both_teams_is_a_draw,
               test_goal_credit_variants,
               test_team_down_wipeout,
               test_recover_reposes_and_re_enables,
               test_downed_body_is_still_an_obstacle):
        fn()
    n = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"\n{n}/{len(RESULTS)} passed")
    sys.exit(0 if n == len(RESULTS) else 1)


if __name__ == "__main__":
    main()
