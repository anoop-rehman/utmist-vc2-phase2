"""Gate for the batched 2v2 self-play env (`warp_port/soccer2v2_env.py`, D1 1e).

Plain python, no pytest in this venv:

    MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m tests.test_soccer2v2
    ... --gpu          # additionally run the warp/GPU stability smoke
    ... --worlds 8 --steps 200      # size of that smoke

Everything except the GPU smoke runs on the CPU MuJoCo backend with one or two
worlds: the claims below are about layout, routing and rules, none of which are
GPU-specific, and the card is shared.

What this exists to defend
--------------------------
 1. PROPRIO IS BYTE-IDENTICAL TO THE DRILLS. This is the headline and the
    reason the file exists. The whole D1 pipeline is "train drills on a frozen
    decoder, then fine-tune 2v2 on the same decoder". The decoder's input is
    proprio. A 2v2 env whose proprio is a permutation -- or is one accelerometer
    clamp away -- of the drills' trains beautifully and transfers nothing, and
    nothing else in the stack can notice. So: the same creature state is written
    into a `shoot` drill env and into each of the four 2v2 slots, and the 65
    numbers are compared. Contact-free (where the two models' solves are
    genuinely comparable) the requirement is BITWISE; standing on the ground it
    is a measured bound, because the four creatures share one constraint solve
    and an iterative solver is not linear in its inputs.
 2. THE LANES ARE PER-CREATURE AND A SWAP IS DETECTABLE. Slot k's observation
    must be built from slot k's body and slot k's action must reach slot k's
    actuators. Both are index arithmetic over a flattened (world, player) batch,
    which is exactly the kind of thing that is off by one silently. Modelled on
    the CompetEvo stage-3 gate's `t_agent_slicing`: the markers are chosen so a
    permutation CANNOT be accidentally equal, and the test asserts that too.
 3. TEAM MIRRORING IS EXACT. Self-play is only cheap because one policy plays
    both teams, which is only correct if the 180-degree rotation that swaps the
    goals maps observations and actions onto each other. Checked from a state
    that has been stepped away from the (symmetric) kickoff, or the check would
    be vacuous.
 4. THE RULES ARE dm_control's. Goals, out of play and the reward are compared
    against dm_control's OWN detector (`PositionDetector._is_in_zone`) on a
    `Pitch` configured to this scene's geometry -- the same class `game/match.py`
    counts goals with -- not against a re-derivation of the rule.
 5. IT RUNS. N worlds x M steps, finite observations, zero diverged worlds.
"""

import argparse
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np      # noqa: E402
import torch            # noqa: E402

from rower_soccer.warp_port.soccer2v2_env import (WarpSoccer2v2Env,  # noqa: E402
                                                  drill_ball)
from rower_soccer.warp_port.worm_env_base import proprio_obs           # noqa: E402

_results = []


def check(name, fn):
    t0 = time.perf_counter()
    try:
        detail = fn() or ""
        ok, err = True, ""
    except Exception:                                               # noqa: BLE001
        import traceback
        ok, detail, err = False, "", traceback.format_exc()
    _results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} "
          f"({time.perf_counter() - t0:.1f}s) {detail}", flush=True)
    if err:
        print(err, flush=True)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
_ENV = {}


def env2v2(worlds=1, **kw):
    key = (worlds, tuple(sorted(kw.items())))
    if key not in _ENV:
        _ENV[key] = WarpSoccer2v2Env(num_worlds=worlds, use_gpu=False, seed=0,
                                     match_seconds=1.0, **kw)
    return _ENV[key]


def drill_env(worlds=1):
    """A `shoot` drill env on the same pitch, ball and creature. shoot is the
    drill whose scene is closest to the match (pitch + goals + drill ball), so
    any proprio difference is about the creature, not the world around it."""
    if "drill" not in _ENV:
        from rower_soccer.warp_port.shoot_env import WarpShootEnv
        _ENV["drill"] = WarpShootEnv(num_worlds=worlds,
                                     creature_xml="creature_configs/ant.xml",
                                     use_gpu=False, seed=0, episode_seconds=1.0,
                                     ball=drill_ball(), pitch_scale=0.3125)
    return _ENV["drill"]


def _rand_creature_state(rng, n_joints, z, xy=(0.0, 0.0)):
    """A full single-creature state: root pose + joints + all velocities."""
    q = np.zeros(7 + n_joints)
    q[0], q[1], q[2] = xy[0], xy[1], z
    quat = rng.normal(size=4)
    q[3:7] = quat / np.linalg.norm(quat)
    q[7:] = rng.uniform(-0.4, 0.4, size=n_joints)
    v = rng.uniform(-0.5, 0.5, size=6 + n_joints)
    return q.astype(np.float32), v.astype(np.float32)


def _write_creature(env, qpos_root, qvel_root, jq, jv, q, v):
    env.qpos[0, qpos_root:qpos_root + 7] = torch.as_tensor(q[:7])
    env.qpos[0, jq] = torch.as_tensor(q[7:])
    env.qvel[0, qvel_root:qvel_root + 6] = torch.as_tensor(v[:6])
    env.qvel[0, jv] = torch.as_tensor(v[6:])


def _write_ball(env, bq, bv, pos, vel):
    env.qpos[0, bq:bq + 3] = torch.as_tensor(pos)
    env.qpos[0, bq + 3] = 1.0
    env.qpos[0, bq + 4:bq + 7] = 0.0
    env.qvel[0, bv:bv + 3] = torch.as_tensor(vel)
    env.qvel[0, bv + 3:bv + 6] = 0.0


def _proprio_compare(z_creature, ball_z, park_z):
    """Write one creature state into the drill and into each 2v2 slot; return
    the largest |difference| over the four slots, and whether all four were
    bitwise equal."""
    d, e = drill_env(), env2v2()
    rng = np.random.default_rng(7)
    n_j = len(d.jq)
    q, v = _rand_creature_state(rng, n_j, z_creature, xy=(1.5, -2.0))
    ball_pos = np.array([2.6, -1.1, ball_z], np.float32)
    ball_vel = np.array([0.3, -0.2, 0.05], np.float32)

    d.qpos.zero_(); d.qvel.zero_()
    _write_creature(d, d.meta.qpos_root, d.meta.qvel_root, d.jq, d.jv, q, v)
    _write_ball(d, d.bq, d.bv, ball_pos, ball_vel)
    d._forward()
    ref = d._proprio_obs()[0].clone()
    ref_ball = d._ball_ego6()[0].clone()

    worst, worst_ball, bitwise = 0.0, 0.0, True
    for k in range(e.n_agents):
        e.qpos.zero_(); e.qvel.zero_()
        # The other three park far apart, well clear of the tested creature and
        # of each other, at the same height (so this configuration is either
        # contact-free for everyone or for no one).
        park = [(-9.0, 7.0), (9.0, 7.0), (-9.0, -7.0), (9.0, -7.0)]
        for j in range(e.n_agents):
            if j == k:
                continue
            e.qpos[0, e.qpos_root[j] + 0] = park[j][0]
            e.qpos[0, e.qpos_root[j] + 1] = park[j][1]
            e.qpos[0, e.qpos_root[j] + 2] = park_z
            e.qpos[0, e.qpos_root[j] + 3] = 1.0
        _write_creature(e, e.qpos_root[k], e.qvel_root[k],
                        e.pidx[k].jq, e.pidx[k].jv, q, v)
        _write_ball(e, e.bq, e.bv, ball_pos, ball_vel)
        e._forward()
        got = proprio_obs(e.qpos, e.qvel, e.xpos, e.xmat, e.sensordata,
                          e.pidx[k])[0]
        obs_k = e._player_obs(k)[0]
        assert torch.equal(got, obs_k[:e.n_proprio]), \
            f"slot {k}: the env's own obs does not start with its proprio"
        worst = max(worst, float((got - ref).abs().max()))
        worst_ball = max(worst_ball,
                         float((obs_k[e.n_proprio:e.n_proprio + 6]
                                - ref_ball).abs().max()))
        bitwise = bitwise and torch.equal(got, ref)
    return worst, worst_ball, bitwise, ref.numel()


def t_proprio_bitwise():
    """HEADLINE. Contact-free, the 2v2 proprio must be the drill's, bit for
    bit, in all four slots -- and so must the ball_ego block that follows it."""
    worst, worst_ball, bitwise, width = _proprio_compare(
        z_creature=3.0, ball_z=3.0, park_z=3.0)
    assert width == 65, f"ant proprio should be 65 wide, got {width}"
    assert bitwise, (f"proprio differs from the drill's by up to {worst:.3e} "
                     f"-- a transferred decoder would be fed a different vector")
    assert worst_ball == 0.0, f"ball_ego differs by {worst_ball:.3e}"
    return (f"{width}-wide proprio + 6-wide ball_ego identical in all 4 slots "
            f"(max |diff| exactly 0)")


def t_proprio_in_contact():
    """Standing on the ground, with the touch sensors live. Not bitwise by
    right: four creatures share one iterative constraint solve, so the drill's
    single-creature solve is a different numerical problem. The bound is the
    claim."""
    worst, worst_ball, bitwise, _ = _proprio_compare(
        z_creature=0.62, ball_z=0.15, park_z=0.62)
    assert worst < 1e-4, f"proprio drifts by {worst:.3e} in contact"
    return (f"max |diff| {worst:.2e} (proprio), {worst_ball:.2e} (ball_ego); "
            f"bitwise={bitwise}")


def t_obs_lanes():
    """Slot k's observation is built from slot k's body, and a permutation of
    the slots is DETECTABLY wrong (not accidentally equal)."""
    e = env2v2(worlds=2)
    e.reset()
    # A marker unique per (world, slot), in a proprio entry that is genuinely
    # per-creature: body_height, at index 3*nbody in the proprio block.
    h_idx = 3 * len(e.pidx[0].body_ids)
    marker = torch.tensor([[1.10 + 0.10 * w + 0.01 * k for k in range(e.n_agents)]
                           for w in range(e.n)])
    for k in range(e.n_agents):
        e.qpos[:, e.qpos_root[k] + 2] = marker[:, k]
    e._forward()
    obs = e.obs().view(e.n, e.n_agents, -1)

    for w in range(e.n):
        for k in range(e.n_agents):
            got = float(obs[w, k, h_idx])
            want = float(marker[w, k])
            assert abs(got - want) < 1e-5, \
                f"world {w} slot {k}: body_height {got:.4f} != {want:.4f}"
            for j in range(e.n_agents):
                if j != k:
                    assert abs(got - float(marker[w, j])) > 1e-4, \
                        "the marker cannot distinguish the lanes"
    # The task block, too: each player's ball_ego is its OWN frame's, checked
    # against an independent computation rather than against the env.
    e.qpos[:, e.bq + 0] = 3.0
    e.qpos[:, e.bq + 1] = -1.5
    e.qpos[:, e.bq + 2] = 0.15
    e._forward()
    obs = e.obs().view(e.n, e.n_agents, -1)
    P = e.n_proprio
    spread = []
    for k in range(e.n_agents):
        pos, rot = e.root_frames(k)
        want = torch.einsum("nij,nj->ni", rot.transpose(1, 2),
                            e.ball_xyz() - pos)
        got = obs[:, k, P:P + 3]
        assert torch.allclose(got, want, atol=1e-6), \
            f"slot {k}: ball_ego is not this creature's ego frame"
        spread.append(got[0].clone())
    # And a swap of two slots' ball_ego is a different vector.
    for a in range(e.n_agents):
        for b in range(a + 1, e.n_agents):
            assert float((spread[a] - spread[b]).abs().max()) > 1e-3, \
                f"slots {a} and {b} see the same ball_ego -- swap undetectable"
    return (f"body_height and ball_ego are per-slot in {e.n} worlds x "
            f"{e.n_agents} slots; every pairwise swap is detectable")


def t_action_lanes():
    """A torque sent to slot k moves slot k's creature and no other.

    Measured as a DIFFERENCE against an all-zero rollout from the same state,
    so gravity, contacts and the shared solve cancel out and what is left is
    the effect of the action alone.
    """
    e = env2v2()
    e.reset()
    q0, v0 = e.qpos.clone(), e.qvel.clone()
    A, U = e.n_agents, e.act_dim

    def rollout(act):
        e.set_state(q0, v0)
        for _ in range(3):
            e.step(act)
        return torch.cat([e.qvel[0, e.pidx[k].jv] for k in range(A)]).clone()

    base = rollout(torch.zeros(A, U))
    on, off = [], []
    for k in range(A):
        a = torch.zeros(A, U)
        a[k] = 0.8
        d = (rollout(a) - base).abs().view(A, -1)
        on.append(float(d[k].max()))
        off.append(float(torch.cat([d[j] for j in range(A) if j != k]).max()))
    assert min(on) > 1e-3, \
        f"a torque in some lane did not move its own creature (min {min(on):.2e})"
    assert max(off) < 1e-5, \
        (f"a torque in one lane moved another creature by {max(off):.2e} -- "
         f"the action lanes are crossed")
    assert min(on) > 100 * max(off), "the lane separation is not decisive"
    return (f"driven slot moves >= {min(on):.2e} rad/s, the other three move "
            f"<= {max(off):.2e}")


def _mirror_pair(steps=10, seed=3):
    """A stepped-away-from-kickoff state and its mirror, both live."""
    e = env2v2()
    e.reset()
    g = torch.Generator().manual_seed(seed)
    a = None
    for _ in range(steps):
        a = (torch.rand(e.n * e.n_agents, e.act_dim, generator=g) * 2 - 1) * 0.6
        e.step(a)
    # mj_step leaves xpos/sensordata one integration behind qpos (it computes
    # them, THEN integrates), so the two sides of this comparison have to be
    # forward-consistent or the difference measured is that lag, not the mirror.
    e._forward()
    q, v, c = e.qpos.clone(), e.qvel.clone(), e.ctrl.clone()
    obs = e.obs().view(e.n, e.n_agents, -1).clone()
    mq, mv = e.mirror_state()
    # ctrl is mirrored too: the accelerometer reads qacc, which includes the
    # actuator forces, so a mirrored state driven by un-mirrored torques is a
    # different physical situation (it is worth 0.03 in the scaled obs).
    e.set_state(mq, mv, e.mirror_actions(a))
    mobs = e.obs().view(e.n, e.n_agents, -1).clone()
    return e, (q, v, c), (mq, mv), obs, mobs


def t_mirror_obs():
    """obs[M(s)][mirror_slot(k)] == obs[s][k], from an ASYMMETRIC state.

    Run from the kickoff this would be vacuous -- the mirror spawn is already
    symmetric, so every pair matches for free. Ten steps of random torque break
    that first.
    """
    e, _, _, obs, mobs = _mirror_pair()
    # the state really is asymmetric, or the test proves nothing
    asym = float((obs[:, 0] - obs[:, 1]).abs().max())
    assert asym > 1e-2, f"the fixture is still symmetric ({asym:.2e})"
    worst = 0.0
    for k in range(e.n_agents):
        j = int(e.mirror_slot[k])
        worst = max(worst, float((mobs[:, j] - obs[:, k]).abs().max()))
    assert worst < 1e-4, f"mirrored obs differs by {worst:.3e}"
    # ... and the mirror is not the identity: without the slot swap it fails.
    naive = max(float((mobs[:, k] - obs[:, k]).abs().max())
                for k in range(e.n_agents))
    assert naive > 1e-2, "the slot swap makes no difference -- test is vacuous"
    return (f"max |diff| {worst:.2e} over 4 slots x {e.obs_dim} dims "
            f"(without the slot swap: {naive:.2e})")


def t_mirror_dynamics():
    """The mirror commutes with the physics: stepping s with action a and M(s)
    with the slot-permuted a leaves two states that are still mirrors. This is
    what makes "the same policy plays both teams" true of the match and not
    just of the observation function."""
    e, (q, v, c), (mq, mv), _, _ = _mirror_pair()
    g = torch.Generator().manual_seed(11)
    a = (torch.rand(e.n * e.n_agents, e.act_dim, generator=g) * 2 - 1) * 0.7

    e.set_state(q, v, c)
    e.step(a)
    # mirror the RESULT of the unmirrored step
    m_q1, m_v1 = e.mirror_state()

    e.set_state(mq, mv, e.mirror_actions(c))
    e.step(e.mirror_actions(a))
    q2, v2 = e.qpos.clone(), e.qvel.clone()

    dq = float((m_q1 - q2).abs().max())
    dv = float((m_v1 - v2).abs().max())
    moved = float((q2 - mq).abs().max())
    assert moved > 1e-4, "nothing moved; the comparison is vacuous"
    assert dq < 2e-4 and dv < 2e-3, f"mirror does not commute: dq {dq:.2e} dv {dv:.2e}"
    return f"after one 10-substep control step: dqpos {dq:.2e}, dqvel {dv:.2e}"


def _dm_pitch(e):
    """dm_control's own `Pitch`, configured to THIS scene's goal geometry.

    dm_control sizes a goal as (depth/2, half_width, height/2) around a centre
    at the back of the pitch, and `match.py` reads goals out of exactly this
    class. Handing it our numbers lets the comparison be against dm_control's
    implementation instead of against a second copy of my own reading of it.
    """
    from dm_control.locomotion.soccer.pitch import Pitch
    depth = e.pitch_half[0] - e.goal_x
    return Pitch(size=e.pitch_half,
                 goal_size=(depth / 2.0, e.goal_half_width, e.goal_height / 2.0))


def t_goal_box_matches_dm_control():
    """Both the box and the classification, against dm_control's detector."""
    e = env2v2()
    p = _dm_pitch(e)
    got_lo, got_hi = np.array(p.away_goal.lower), np.array(p.away_goal.upper)
    want_lo = np.array([e.goal_x, -e.goal_half_width, 0.0])
    want_hi = np.array([e.pitch_half[0], e.goal_half_width, e.goal_height])
    assert np.allclose(got_lo, want_lo, atol=1e-6), f"{got_lo} != {want_lo}"
    assert np.allclose(got_hi, want_hi, atol=1e-6), f"{got_hi} != {want_hi}"
    fl, fu = np.array(p.field.lower), np.array(p.field.upper)
    assert np.allclose(fu, np.array(e.field_half), atol=1e-6), \
        f"field {fu} != {e.field_half}"
    assert np.allclose(fl, -np.array(e.field_half), atol=1e-6)

    gx, hw, gh, px = e.goal_x, e.goal_half_width, e.goal_height, e.pitch_half[0]
    pts = [
        (0.0, 0.0, 0.15),              # centre spot
        (gx + 0.3, 0.0, 0.15),         # a goal, dead centre
        (gx - 0.05, 0.0, 0.15),        # on the pitch side of the line
        (gx + 0.3, hw - 0.05, 0.15),   # inside the near post
        (gx + 0.3, hw + 0.05, 0.15),   # outside the post: wide, not a goal
        (gx + 0.3, 0.0, gh - 0.05),    # under the bar
        (gx + 0.3, 0.0, gh + 0.05),    # over the bar
        (px + 0.5, 0.0, 0.15),         # behind the back wall
        (-gx - 0.3, 0.0, 0.15),        # away scores in the home goal
        (-gx - 0.3, -hw - 0.05, 0.15),
        (0.0, e.field_half[1] + 0.2, 0.15),   # over the touchline
        (0.0, e.field_half[1] - 0.2, 0.15),
        (gx + 0.3, 0.0, 0.0),          # exactly on the floor: strict bound
    ]
    n_goal = n_out = 0
    for x, y, z in pts:
        e.qpos[:, e.bq + 0] = x
        e.qpos[:, e.bq + 1] = y
        e.qpos[:, e.bq + 2] = z
        pos = np.array([x, y, z])
        dm_home = p.home_goal._is_in_zone(pos)      # noqa: SLF001
        dm_away = p.away_goal._is_in_zone(pos)      # noqa: SLF001
        # Pitch.detected_goal: the ball in the HOME goal scores for AWAY.
        dm_code = 1 if dm_away else (2 if dm_home else 0)
        got = int(e.detected_goal()[0])
        assert got == dm_code, \
            (f"goal at {pos}: env says {got}, dm_control says {dm_code}")
        # the field detector is inverted: in-zone means IN play
        dm_out = not p.field._is_in_zone(pos)       # noqa: SLF001
        got_out = bool(e.detected_off_court()[0])
        assert got_out == dm_out, \
            f"off-court at {pos}: env says {got_out}, dm_control says {dm_out}"
        n_goal += dm_code > 0
        n_out += dm_out
    return (f"goal box and field box equal dm_control's to 1e-6; "
            f"{len(pts)} constructed positions agree ({n_goal} goals, "
            f"{n_out} out of play)")


def t_goal_counting_and_reward():
    """Rising edge, team credit, dm_soccer's +1/-1 reward, and the re-spawn."""
    # The latch is only observable with the re-spawn off: with it on, the ball
    # is teleported back to the centre spot the instant it goes in, so a level
    # read would also count once and the test would prove nothing. In the game
    # the detector stays latched across the re-spawn (`retain_substep_detections`
    # -- see MatchSim._detect_goal), which is precisely the case this defends.
    e = env2v2(goal_respawn=False)
    e.reset()
    zero = torch.zeros(e.n_agents, e.act_dim)
    scored_steps, rewards = [], []
    for t in range(3):
        e.qpos[:, e.bq + 0] = e.goal_x + 0.3
        e.qpos[:, e.bq + 1] = 0.0
        e.qpos[:, e.bq + 2] = 0.15
        e.qvel[:, e.bv:e.bv + 6] = 0.0
        e._forward()
        _, r, _ = e.step(zero)
        scored_steps.append(int(e.scored_now[0]))
        rewards.append(r.clone())
    assert float(e.score[0, 0]) == 1.0 and float(e.score[0, 1]) == 0.0, \
        (f"three steps with the ball in the goal scored {e.score[0].tolist()} "
         f"-- the rising edge is not latching")
    assert scored_steps[0] == 1, f"the first step should register home: {scored_steps}"
    r0 = rewards[0].view(e.n, e.n_agents)[0]
    assert torch.allclose(r0, torch.tensor([1.0, 1.0, -1.0, -1.0])), \
        f"goal reward {r0.tolist()} != dm_soccer's +1 scorers / -1 conceders"
    assert torch.allclose(rewards[1].view(e.n, e.n_agents)[0],
                          torch.zeros(e.n_agents)), \
        "the reward was paid twice for one goal"

    # A goal in the HOME (-x) goal credits AWAY -- and with the default
    # MultiturnTask behaviour the world is re-kicked-off, not terminated.
    e = env2v2()
    e.reset()
    e.qpos[:, e.bq + 0] = -(e.goal_x + 0.3)
    e.qpos[:, e.bq + 1] = 0.0
    e.qpos[:, e.bq + 2] = 0.15
    e._forward()
    _, r, _ = e.step(zero)
    assert float(e.score[0, 1]) == 1.0 and float(e.score[0, 0]) == 0.0, \
        f"a goal in the home goal scored {e.score[0].tolist()}, expected away"
    assert torch.allclose(r.view(e.n, e.n_agents)[0],
                          torch.tensor([-1.0, -1.0, 1.0, 1.0]))
    # MultiturnTask re-spawns and does not terminate: the ball is back on the
    # centre spot and the match clock is still running.
    assert float(torch.linalg.norm(e.ball_xy()[0])) < 1e-5, \
        "the world was not re-kicked-off after the goal"
    assert bool(e.world_reset[0]), "world_reset was not raised for the reward mask"
    return "rising edge counts once, +1/-1 by team, ball re-spotted after a goal"


def t_out_of_play_throw_in():
    """dm_soccer's throw-in: the ball comes back by a random 10-30%, its
    velocity is zeroed, and nobody else is moved."""
    e = env2v2()
    e.reset()
    before = [e.root_xy(k)[0].clone() for k in range(e.n_agents)]
    x0, y0 = 0.5 * e.field_half[0], e.field_half[1] + 1.0
    e.qpos[:, e.bq + 0] = x0
    e.qpos[:, e.bq + 1] = y0
    e.qpos[:, e.bq + 2] = 0.15
    e.qvel[:, e.bv + 0] = 4.0
    e._forward()
    assert bool(e.detected_off_court()[0])
    e.step(torch.zeros(e.n_agents, e.act_dim))
    bx, by = e.ball_xy()[0].tolist()
    ux, uy = bx / x0, by / y0
    assert 0.7 <= ux <= 0.9 and 0.7 <= uy <= 0.9, \
        f"throw-in shrink ({ux:.3f}, {uy:.3f}) outside dm_control's U[0.7, 0.9]"
    assert float(torch.linalg.norm(e.ball_vel_xyz()[0])) < 1e-5, \
        "the throw-in did not zero the ball velocity"
    moved = max(float((e.root_xy(k)[0] - before[k]).abs().max())
                for k in range(e.n_agents))
    assert moved < 0.15, f"a throw-in teleported the players ({moved:.3f} m)"
    assert float(e.throw_ins[0]) == 1.0
    return f"shrink ({ux:.3f}, {uy:.3f}), ball velocity zeroed, players untouched"


def t_time_limit():
    """One `done`, for the whole batch, at match_seconds -- and goals do not
    end the episode (`terminate_on_goal=False` in match.py)."""
    e = env2v2()
    assert e.episode_steps == int(round(e.match_seconds / 0.025))
    e.reset()
    zero = torch.zeros(e.n_agents, e.act_dim)
    dones = []
    for t in range(e.episode_steps):
        _, _, d = e.step(zero)
        dones.append(d)
    assert not any(dones[:-1]), "the episode ended early"
    assert dones[-1], "the episode did not end at the time limit"
    return (f"done exactly once, at step {e.episode_steps} "
            f"= {e.match_seconds}s / 0.025s")


def t_env_runs(worlds, steps, use_gpu):
    """N worlds x M steps: finite observations, zero diverged worlds."""
    dev = "cuda" if use_gpu else "cpu"
    e = WarpSoccer2v2Env(num_worlds=worlds, use_gpu=use_gpu, seed=1,
                         match_seconds=max(1.0, steps * 0.025),
                         spawn="uniform")
    obs = e.reset()
    g = torch.Generator(device=dev).manual_seed(5)
    t0 = time.perf_counter()
    for t in range(steps):
        a = (torch.rand(worlds * e.n_agents, e.act_dim, generator=g,
                        device=dev) * 2 - 1)
        obs, r, done = e.step(a)
        assert torch.isfinite(obs).all(), f"non-finite obs at step {t}"
        assert torch.isfinite(r).all(), f"non-finite reward at step {t}"
    dt = time.perf_counter() - t0
    assert e.n_diverged == 0, f"{e.n_diverged} diverged worlds"
    st = e.match_stats()
    return (f"{worlds} worlds x {steps} steps on {dev}: 0 diverged, obs "
            f"{tuple(obs.shape)} finite, {worlds * steps / dt:,.0f} env-steps/s, "
            f"throw-ins/world {st['throw_ins']:.1f}, upright {st['upright']:.2f}")


def t_team_colour_is_inert():
    """Tinting the per-player materials must not move a single number.

    rgba is visual by construction, but this project has twice shipped an env
    that was numerically fine and visually wrong, and this is the mirror case --
    a visual change that must be numerically nothing. Step a coloured and an
    uncoloured scene from one identical state and require BITWISE equal qpos and
    qvel; anything less is an assumption, not a check.
    """
    import numpy as np
    import mujoco
    from rower_soccer.warp_port.scene import build_soccer_scene

    out = []
    for team_rgba in (None, __import__(
            "rower_soccer.warp_port.scene", fromlist=["TEAM_RGBA"]).TEAM_RGBA):
        model, _, _ = build_soccer_scene("creature_configs/ant.xml", n_players=4,
                                         ball=drill_ball(), team_rgba=team_rgba)
        data = mujoco.MjData(model)
        rng = np.random.default_rng(0)
        data.qpos[:] = model.qpos0 + 0.01 * rng.standard_normal(model.nq)
        data.qvel[:] = 0.01 * rng.standard_normal(model.nv)
        data.ctrl[:] = 0.3
        for _ in range(50):
            mujoco.mj_step(model, data)
        out.append((data.qpos.copy(), data.qvel.copy()))
    dq = float(np.abs(out[0][0] - out[1][0]).max())
    dv = float(np.abs(out[0][1] - out[1][1]).max())
    assert dq == 0.0 and dv == 0.0, f"dqpos {dq:.3e}, dqvel {dv:.3e}"
    return f"50 steps from one state: dqpos {dq:.1e}, dqvel {dv:.1e} (bitwise)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", action="store_true",
                   help="additionally run the warp/GPU stability smoke")
    p.add_argument("--worlds", type=int, default=8)
    p.add_argument("--steps", type=int, default=200)
    args = p.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")
    torch.manual_seed(0)

    check("proprio is BYTE-IDENTICAL to the drill's (contact-free)",
          t_proprio_bitwise)
    check("proprio matches the drill's in contact (MEASURED bound)",
          t_proprio_in_contact)
    check("obs lanes are per-creature (a slot swap is detectable)", t_obs_lanes)
    check("action lanes are per-creature (a torque moves only its own body)",
          t_action_lanes)
    check("team mirroring: mirrored state -> mirrored obs", t_mirror_obs)
    check("team mirroring: mirrored state + mirrored action -> mirrored step",
          t_mirror_dynamics)
    check("goal / off-court boxes agree with dm_control's own detector",
          t_goal_box_matches_dm_control)
    check("goals count on a rising edge and pay dm_soccer's +1/-1",
          t_goal_counting_and_reward)
    check("out of play -> dm_soccer throw-in", t_out_of_play_throw_in)
    check("the match ends at the time limit, and only there", t_time_limit)
    check("the env steps N worlds for M steps, finite, 0 diverged (cpu)",
          lambda: t_env_runs(2, 40, False))
    if args.gpu:
        check("the env steps N worlds for M steps, finite, 0 diverged (warp)",
              lambda: t_env_runs(args.worlds, args.steps, True))

    check("team colours are visual only (a coloured scene steps identically)",
          t_team_colour_is_inert)

    n_fail = sum(1 for _, ok in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} passed", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
