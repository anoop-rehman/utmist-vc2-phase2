"""Drill v4 gate: the timed kick and the urgent shoot (docs/DRILL_V4_SPEC.md).

Plain python, no pytest in this venv:

    MUJOCO_GL=egl PYTHONPATH=. .venv/bin/python -m tests.test_drill_v4
    ... --no-physics      # reward/fitness algebra only, no GPU, ~1 s

Two groups. The algebra group drives the reward objects against a stub env, so
it pins the exact bugs this change could reintroduce -- a goal bonus read after
the respawn has zeroed the clock (pays the full flat bonus, urgency silently
gone), a fitness that ranks a near miss above a scored goal, a ball->cmd
shaping term creeping back into the timed kick. The physics group builds the
real Warp envs and checks the deadline machinery end to end: that the segment
ends at exactly T, that T is what the pace band says it should be, that the
policy can SEE the clock, and that arrival is graded at T and nowhere else.
"""

import argparse
import math
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch  # noqa: E402

CONTROL_DT = 0.025
_results = []


def check(name, fn):
    t0 = time.perf_counter()
    try:
        detail = fn() or ""
        ok, err = True, ""
    except Exception:                                               # noqa: BLE001
        import traceback
        ok, detail, err = False, "", traceback.format_exc()
    dt = time.perf_counter() - t0
    _results.append((name, ok))
    print(f"[{'PASS' if ok else 'FAIL'}] {name} ({dt:.1f}s) {detail}", flush=True)
    if err:
        print(err, flush=True)


# ---------------------------------------------------------------------------
# algebra: the reward objects against a stub env
# ---------------------------------------------------------------------------
class StubEnv:
    """The handful of attributes the ball-drill rewards actually touch.

    Everything is at the origin and at rest, so every shaping term is exactly
    zero and the numbers under test are not buried in noise. `upright` reads
    rot[:, 2, 2] = 1 => the multiplier is 1.
    """

    def __init__(self, n=1):
        self.n = n
        z = torch.zeros(n)
        self.shaping_scale = 1.0
        self.credit = z.clone()
        self.cmd_dir = torch.zeros(n, 2)
        self.cmd_dir[:, 0] = 1.0
        self.seg_reset = torch.zeros(n, dtype=torch.bool)
        # shoot
        self.scored_now = torch.zeros(n, dtype=torch.bool)
        self.seg_scored = torch.zeros(n, dtype=torch.bool)
        self.seg_score_t = z.clone()
        self.last_score_t = z.clone()
        self.seg_goal_best = z.clone()
        self.goal_fit_sum = z.clone()
        self.n_segments = z.clone()
        self._goal_time_coef = 0.4
        self._reward_coef = 0.5
        # kick
        self.last_arrival = z.clone()
        self.target_fit_sum = z.clone()
        self.prev_target_fit_sum = z.clone()
        self.prev_n_segments = z.clone()
        self.ball_spawn_xy = torch.zeros(n, 2)

    # ball_task.SegmentedBallTask.anchor_excess, verbatim -- the stub must not
    # reimplement it, or the test would pass against a formula the env does
    # not use.
    def anchor_excess(self, free_radius=1.0, cap=5.0):
        from rower_soccer.warp_port.ball_task import SegmentedBallTask
        return SegmentedBallTask.anchor_excess(self, free_radius, cap)

    def _root_frames(self):
        return torch.zeros(self.n, 3), torch.eye(3).expand(self.n, 3, 3)

    def _root_vel_xy(self):
        return torch.zeros(self.n, 2)

    def _ball_xy(self):
        return torch.zeros(self.n, 2)

    def _ball_vel_xy(self):
        return torch.zeros(self.n, 2)

    # shoot_env.seg_fitness, verbatim -- the stub must not reimplement it, or
    # the test would pass against a formula the env does not use.
    def seg_fitness(self):
        from rower_soccer.warp_port.shoot_env import WarpShootEnv
        return WarpShootEnv.seg_fitness(self)


def t_shoot_goal_bonus_is_time_discounted():
    from rower_soccer.warp_port.ball_task import ShootReward
    k, bonus = 0.4, 5.0
    r = ShootReward(goal_bonus=bonus, goal_time_coef=k)
    r.reset(StubEnv())
    out = {}
    for t in (0.0, 1.0, 3.0, 5.0):
        env = StubEnv()
        r.reset(env)
        env.scored_now[:] = True
        env.last_score_t[:] = t
        out[t] = float(r(env)[0])
        assert abs(out[t] - bonus * math.exp(-k * t)) < 1e-5, (t, out[t])
    assert out[0.0] > out[1.0] > out[3.0] > out[5.0], out
    # The spec's calibration for k: a 1 s goal keeps ~0.7 of the bonus, a 5 s
    # goal ~0.15. If someone retunes k, this is the line that should argue.
    assert 0.6 < out[1.0] / bonus < 0.8, out[1.0] / bonus
    assert 0.08 < out[5.0] / bonus < 0.22, out[5.0] / bonus
    # A no-goal step pays nothing from this term, whatever the clock says.
    env = StubEnv()
    r.reset(env)
    env.last_score_t[:] = 1.0
    assert float(r(env)[0]) == 0.0
    return f"1s={out[1.0]:.2f} 3s={out[3.0]:.2f} 5s={out[5.0]:.2f} of {bonus}"


def t_shoot_bonus_reads_the_presnapshot_clock():
    """Regression: the segment is respawned BEFORE the reward runs, so a bonus
    that read the live per-segment clock would see 0 and pay full price for a
    5 s goal -- the urgency term would be dead code that still looked right."""
    from rower_soccer.warp_port.ball_task import ShootReward
    r = ShootReward(goal_bonus=5.0, goal_time_coef=0.4)
    env = StubEnv()
    r.reset(env)
    env.scored_now[:] = True
    env.last_score_t[:] = 4.0      # snapshot taken before the respawn
    env.seg_score_t[:] = 0.0       # live clock, already reset by the respawn
    paid = float(r(env)[0])
    assert paid < 0.5 * 5.0, f"{paid}: reward is reading the post-respawn clock"
    return f"paid {paid:.3f} for a 4 s goal (flat would be 5.0)"


def t_shoot_fitness_ranks_goals_over_misses():
    from rower_soccer.warp_port.ball_task import ShootReward
    r = ShootReward(goal_bonus=5.0, goal_time_coef=0.4, reward_coef=0.5)

    def seg(scored, t=0.0, d=0.0):
        env = StubEnv()
        env.seg_scored[:] = scored
        env.seg_score_t[:] = t
        env.seg_goal_best[:] = d
        return float(env.seg_fitness()[0])

    # The invariant, and the reason seg_fitness maps the scored branch into the
    # upper half rather than using the spec's bare exp(-k*t): the slowest
    # imaginable goal must still beat the best imaginable miss. With the bare
    # form these two are 0.135 and 0.5 and best.pt would prefer the miss.
    slowest_goal = seg(True, t=15.0)    # a goal at the episode limit
    perfect_miss = seg(False, d=0.0)    # ball ON the line but not over it
    assert slowest_goal > perfect_miss, (slowest_goal, perfect_miss)
    assert seg(True, t=0.5) > seg(True, t=2.0) > seg(True, t=5.0) > slowest_goal
    assert perfect_miss > seg(False, d=2.0) > seg(False, d=8.0) >= 0.0
    for v in (slowest_goal, perfect_miss, seg(True), seg(False, d=100.0)):
        assert 0.0 <= v <= 1.0, v
    # Episode aggregation: mean over closed segments plus the one in flight.
    env = StubEnv()
    r.reset(env)
    env.n_segments[:] = 2.0
    env.goal_fit_sum[:] = 1.5           # two closed segments
    env.seg_scored[:] = False
    env.seg_goal_best[:] = 0.0          # in flight, currently on the line
    got = float(r.fitness(env)[0])
    assert abs(got - (1.5 + 0.5) / 3.0) < 1e-6, got
    return (f"slowest goal {slowest_goal:.3f} > best miss {perfect_miss:.3f}; "
            f"episode mean {got:.3f}")


def t_timed_kick_drops_ball_to_cmd_shaping():
    from rower_soccer.warp_port.ball_task import (KickToPointReward,
                                                  TimedKickReward)
    r = TimedKickReward(w_arrive=3.0, w_ball_to_cmd=0.25, w_strike=0.0)
    assert r.w_b2c == 0.0, r.w_b2c
    # ... and the term is really gone from the shaping, not just from the field.
    env = StubEnv()
    r.reset(env)
    env._ball_vel_xy = lambda: torch.tensor([[5.0, 0.0]])   # straight at target
    assert float(r(env)[0]) == 0.0, "ball->cmd velocity is still being paid"
    # The v3 reward, by contrast, does pay for it -- so this is a real change.
    old = KickToPointReward(w_arrive=3.0, w_ball_to_cmd=0.25)
    old.reset(env)
    assert float(old(env)[0]) > 0.0
    # Arrival is paid, and only from the deadline snapshot.
    env.last_arrival[:] = 0.8
    assert abs(float(r(env)[0]) - 3.0 * 0.8) < 1e-6
    return "w_b2c forced to 0; arrival paid from last_arrival"


def _moving_env(root_xy, root_vel, ball_xy, spawn_xy):
    """Stub posed mid-segment: creature at root_xy moving at root_vel, ball
    already struck and sitting at ball_xy, segment spawned at spawn_xy."""
    env = StubEnv()
    env._root_frames = lambda: (torch.tensor([[*root_xy, 0.0]]),
                                torch.eye(3).expand(1, 3, 3))
    env._root_vel_xy = lambda: torch.tensor([list(root_vel)])
    env._ball_xy = lambda: torch.tensor([list(ball_xy)])
    env.ball_spawn_xy = torch.tensor([list(spawn_xy)])
    return env


def t_anchor_stops_paying_for_the_chase():
    """The v7 claim, stated as a reward comparison.

    Mid-segment: the ball has been struck and is 4 m downfield of where it
    spawned; the creature is standing on the spawn point. Running AFTER the
    ball is what the unanchored me->ball term pays for -- that is the dribble.
    """
    from rower_soccer.warp_port.ball_task import TimedKickReward
    # chasing: 1 m/s straight at the rolled-away ball.
    chase = dict(root_xy=(0.0, 0.0), root_vel=(1.0, 0.0),
                 ball_xy=(4.0, 0.0), spawn_xy=(0.0, 0.0))

    off = TimedKickReward(w_arrive=3.0, w_player_to_ball=0.15, w_strike=0.0)
    env = _moving_env(**chase)
    off.reset(env)
    paid_unanchored = float(off(env)[0])
    assert paid_unanchored > 0.0, "baseline is wrong: v4/v6 should pay to chase"
    assert abs(paid_unanchored - 0.15) < 1e-6, paid_unanchored

    on = TimedKickReward(w_arrive=3.0, w_player_to_ball=0.15, w_strike=0.0,
                         w_anchor=0.01, anchor_free_radius=1.0)
    env = _moving_env(**chase)
    on.reset(env)
    paid_anchored = float(on(env)[0])
    # Standing ON the spawn point => no penalty yet; the chase simply is not
    # paid. clamp(min=0) means running AWAY from the anchor is 0, not negative,
    # so the whole difference has to come from the approach term.
    assert abs(paid_anchored) < 1e-6, paid_anchored

    # ...while approaching the ball BEFORE it is struck is paid identically,
    # because the anchor and the ball are then the same point. If this ever
    # differs, the anchor has started fighting the approach.
    pre = dict(root_xy=(-2.0, 0.0), root_vel=(1.0, 0.0),
               ball_xy=(0.0, 0.0), spawn_xy=(0.0, 0.0))
    a, b = TimedKickReward(w_player_to_ball=0.15, w_strike=0.0), None
    env = _moving_env(**pre); a.reset(env)
    approach_off = float(a(env)[0])
    b = TimedKickReward(w_player_to_ball=0.15, w_strike=0.0, w_anchor=0.01,
                        anchor_free_radius=1.0)
    env = _moving_env(**pre); b.reset(env)
    approach_on = float(b(env)[0])
    # 2 m from the anchor, free radius 1 => 1 m excess * 0.01.
    assert abs((approach_off - 0.01) - approach_on) < 1e-6, (approach_off,
                                                             approach_on)
    return f"chase paid {paid_unanchored:.3f} -> {paid_anchored:.3f}"


def t_strike_offset_puts_the_approach_behind_the_ball():
    """v8: the approach point must sit on the FAR side of the ball from the
    target, so that walking to it and continuing forward IS the kick.

    Measured motivation (v7/best.pt, 55,303 strikes): positioning, the angle
    between ant->ball and ball->target, is median 103.9 deg against a 90 deg
    random baseline -- no positioning skill, and biased to the WRONG side.
    """
    from rower_soccer.warp_port.ball_task import TimedKickReward
    r = TimedKickReward(w_arrive=3.0, w_strike=0.0, strike_offset=0.5)
    env = _moving_env((0.0, 0.0), (0.0, 0.0), (2.0, 0.0), (2.0, 0.0))
    env.target_xy = torch.tensor([[6.0, 0.0]])       # target beyond the ball
    p = r._approach_xy(env)[0]
    # ball at x=2, target at x=6 => approach point at x=1.5, i.e. BEHIND the
    # ball as seen from the target. If the sign were flipped this would be 2.5
    # and the shaping would drive the creature to the target side, which is the
    # failure mode being fixed.
    assert abs(float(p[0]) - 1.5) < 1e-5, float(p[0])
    assert abs(float(p[1])) < 1e-5, float(p[1])

    # ...and the ordering that matters, stated directly: the approach point is
    # farther from the target than the ball is.
    d_ball = float(torch.linalg.norm(torch.tensor([2.0, 0.0]) - env.target_xy[0]))
    d_pt = float(torch.linalg.norm(p - env.target_xy[0]))
    assert d_pt > d_ball, (d_pt, d_ball)

    # Off-axis case: still colinear with (target -> ball), extended by exactly
    # strike_offset.
    env2 = _moving_env((0.0, 0.0), (0.0, 0.0), (3.0, 4.0), (3.0, 4.0))
    env2.target_xy = torch.tensor([[0.0, 0.0]])      # ball 5 m from target
    q = r._approach_xy(env2)[0]
    assert abs(float(torch.linalg.norm(q)) - 5.5) < 1e-5, float(torch.linalg.norm(q))

    # off by default => every existing arm is untouched
    off = TimedKickReward(w_arrive=3.0, w_strike=0.0)
    assert off.strike_offset == 0.0
    env3 = _moving_env((0.0, 0.0), (0.0, 0.0), (2.0, 0.0), (2.0, 0.0))
    env3.target_xy = torch.tensor([[6.0, 0.0]])
    assert abs(float(off._approach_xy(env3)[0][0]) - 2.0) < 1e-6
    return "approach point 0.5 m behind the ball, on the target line"


def t_anchor_penalty_shape():
    """Zero inside the free radius, linear outside, clipped, and NOT a
    function of uprightness or of shaping_scale."""
    from rower_soccer.warp_port.ball_task import TimedKickReward
    r = TimedKickReward(w_arrive=3.0, w_player_to_ball=0.0, w_strike=0.0,
                        w_anchor=0.01, anchor_free_radius=1.0)
    seen = {}
    for d in (0.0, 0.5, 1.0, 2.0, 5.0, 6.0, 20.0):
        env = _moving_env((d, 0.0), (0.0, 0.0), (d, 0.0), (0.0, 0.0))
        r.reset(env)
        seen[d] = float(r(env)[0])
    assert seen[0.0] == seen[0.5] == seen[1.0] == 0.0, seen
    assert abs(seen[2.0] + 0.01) < 1e-6, seen          # 1 m excess
    assert abs(seen[5.0] + 0.04) < 1e-6, seen          # 4 m excess
    # cap at 5 m of excess: 6 m and 20 m must cost the same, so one runaway
    # world cannot dominate the batch return.
    assert abs(seen[6.0] - seen[20.0]) < 1e-6, seen
    assert abs(seen[20.0] + 0.05) < 1e-6, seen

    # Annealing the shaping must NOT switch the anchor off -- it is part of the
    # objective, not a training aid.
    env = _moving_env((3.0, 0.0), (0.0, 0.0), (3.0, 0.0), (0.0, 0.0))
    r.reset(env)
    env.shaping_scale = 0.0
    assert abs(float(r(env)[0]) + 0.02) < 1e-6, float(r(env)[0])

    # Tipping over must not discount it (upright multiplies the rest, not this).
    env = _moving_env((3.0, 0.0), (0.0, 0.0), (3.0, 0.0), (0.0, 0.0))
    r.reset(env)
    rot = torch.eye(3).expand(1, 3, 3).clone()
    rot[:, 2, 2] = 0.1
    env._root_frames = lambda: (torch.tensor([[3.0, 0.0, 0.0]]), rot)
    assert abs(float(r(env)[0]) + 0.02) < 1e-6, float(r(env)[0])
    return "0 inside r_free, linear, capped at 5 m, survives annealing"


def t_anchor_is_off_by_default():
    """Every existing kick arm (v3/v4/v6) must be bit-identical without the
    flag, or this change silently rewrites runs already in flight."""
    from rower_soccer.warp_port.ball_task import TimedKickReward
    r = TimedKickReward(w_arrive=3.0, w_player_to_ball=0.15, w_strike=0.0)
    assert r.w_anchor == 0.0
    env = _moving_env((0.0, 0.0), (1.0, 0.0), (4.0, 0.0), (99.0, 99.0))
    r.reset(env)
    # spawn_xy is deliberately absurd: with the anchor off nothing may read it.
    assert abs(float(r(env)[0]) - 0.15) < 1e-6, float(r(env)[0])
    return "w_anchor=0 => live-ball approach, anchor never read"


def t_anchor_magnitude_is_calibrated():
    """A full-segment dribble should cost the same ORDER as a perfect pass is
    worth -- otherwise the term either does nothing or eats the objective."""
    from rower_soccer.warp_port.ball_task import TimedKickReward
    w_anchor, w_arrive = 0.01, 3.0
    r = TimedKickReward(w_arrive=w_arrive, w_player_to_ball=0.0, w_strike=0.0,
                        w_anchor=w_anchor, anchor_free_radius=1.0)
    # 125 steps ~ the middle of the 50-200 step segment band, creature walking
    # the ball 2.5 m past the free radius.
    env = _moving_env((3.5, 0.0), (0.0, 0.0), (3.5, 0.0), (0.0, 0.0))
    r.reset(env)
    per_step = -float(r(env)[0])
    total = per_step * 125
    perfect_pass = w_arrive * 1.0
    assert 0.3 < total / perfect_pass < 3.0, (total, perfect_pass)
    return f"125-step 2.5 m dribble costs {total:.2f} vs {perfect_pass:.1f} for a pass"


def t_timed_kick_fitness_is_mean_arrival():
    from rower_soccer.warp_port.ball_task import TimedKickReward
    r = TimedKickReward()
    env = StubEnv()
    env.n_segments[:] = 4.0
    env.target_fit_sum[:] = 2.0
    assert abs(float(r.fitness(env)[0]) - 0.5) < 1e-6
    # No segment closed yet this episode: fall back to the previous episode's
    # mean rather than reporting 0 at random (PPOTrainer samples fitness
    # wherever its rollout lands).
    env2 = StubEnv()
    env2.prev_n_segments[:] = 2.0
    env2.prev_target_fit_sum[:] = 1.4
    assert abs(float(r.fitness(env2)[0]) - 0.7) < 1e-6
    assert float(r.fitness(StubEnv())[0]) == 0.0
    return "mean over closed segments, prev-episode fallback"


# ---------------------------------------------------------------------------
# physics: the real Warp envs
# ---------------------------------------------------------------------------
def _kick(worlds, timed=True, **kw):
    from rower_soccer.warp_port.kick_env import WarpKickEnv
    from rower_soccer.warp_port.scene import BallSpec
    return WarpKickEnv(
        num_worlds=worlds, seed=3, use_graph=True,
        creature_xml="creature_configs/ant.xml",
        ball=BallSpec(radius=0.15, mass=0.045),
        arena="pitch", pitch_scale=0.3125,
        reward_kind="timed" if timed else "point",
        w_strike=0.0, w_arrive=3.0, w_upright=1.0,
        target_dist_range=(3.0, 6.0), **kw)


def t_kick_task_width():
    timed, point = _kick(2, timed=True), _kick(2, timed=False)
    assert len(point.task_indices) == 12, len(point.task_indices)
    assert len(timed.task_indices) == 14, len(timed.task_indices)
    # The first 12 columns must be untouched, or a v3 checkpoint's task encoder
    # is being fed a permuted vector rather than a widened one.
    assert timed.obs_dim - point.obs_dim == 2
    assert len(timed.proprio_indices) == len(point.proprio_indices)
    return f"point={point.obs_dim} timed={timed.obs_dim} (proprio " \
           f"{len(timed.proprio_indices)} + task 12/14)"


def t_kick_deadline_matches_the_pace_band(steps=400, worlds=16):
    """T = clamp(d_spawn / v_pace) exactly, segments end exactly at T, and the
    remaining-time obs is the same clock the env ends the segment on."""
    from rower_soccer.warp_port.ppo import OBS_SANITY_LIMIT
    env = _kick(worlds, pace_range=(1.5, 3.0), deadline_range=(0.5, 4.0))
    obs = env.reset()
    n_checked, n_closed, Ts = 0, 0, []
    prev_seg = env.n_segments.clone()
    prev_limit = env.seg_limit.clone()
    for _ in range(steps):
        act = torch.zeros(env.n, env.act_dim, device=env.device)
        obs, rew, done = env.step(act)
        assert torch.isfinite(obs).all(), "non-finite obs"
        assert obs.abs().max() < OBS_SANITY_LIMIT, float(obs.abs().max())
        assert torch.isfinite(rew).all(), "non-finite reward"

        # Every world that just respawned: check its freshly drawn deadline.
        fresh = env.seg_t == 0
        if bool(fresh.any()):
            i = fresh.nonzero(as_tuple=True)[0]
            d = env._target_dist_now()[i]
            pace = env.seg_pace[i]
            want = torch.round((d / pace).clamp(0.5, 4.0) / CONTROL_DT) * CONTROL_DT
            err = (env.seg_T[i] - want).abs().max()
            assert float(err) < 1e-4, f"deadline mismatch {float(err)}"
            assert float(env.seg_T[i].min()) >= 0.5 - 1e-6
            assert float(env.seg_T[i].max()) <= 4.0 + 1e-6
            Ts += env.seg_T[i].tolist()
            n_checked += int(i.numel())

        # A segment closed exactly on its deadline, never before it.
        closed = env.n_segments > prev_seg
        if bool(closed.any()):
            n_closed += int(closed.sum())
        # seg_t is reset on close, so compare against the PREVIOUS step's limit.
        early = (~closed) & (env.seg_t > prev_limit)
        assert not bool(early.any()), "segment ran past its deadline"
        prev_seg, prev_limit = env.n_segments.clone(), env.seg_limit.clone()

        # The clock the policy sees is the clock the env enforces.
        t_rem = obs[:, env.task_indices[-2]]
        assert torch.allclose(t_rem, env._time_left(), atol=1e-5)
        assert float(t_rem.min()) >= 0.0
        req = obs[:, env.task_indices[-1]]
        assert float(req.max()) <= 10.0 + 1e-5
        if done:
            obs = env.reset()
            prev_seg = env.n_segments.clone()
            prev_limit = env.seg_limit.clone()

    assert n_checked > 20 and n_closed > 20, (n_checked, n_closed)
    return (f"{n_checked} deadlines drawn, {n_closed} segments closed, "
            f"T in [{min(Ts):.2f}, {max(Ts):.2f}]s")


def t_ball_spawn_xy_tracks_spawns(steps=250, worlds=8):
    """The anchor is only meaningful if it equals the ball's position at the
    instant of every spawn, and holds still afterwards.

    Both spawn paths are covered: the episode reset, and the mid-episode
    segment restart (which does NOT move the creature, so the anchor is the
    only thing that changes). Also asserts the anchor does NOT follow a ball
    the creature has kicked -- a buffer quietly aliased to the live ball would
    pass every algebra test above and make the whole term a no-op.
    """
    env = _kick(worlds, w_anchor=0.01)
    env.reset()
    assert torch.allclose(env.ball_spawn_xy, env._ball_xy(), atol=1e-5), \
        "anchor wrong immediately after reset"

    prev_seg = env.n_segments.clone()
    n_restarts, n_drifted, max_drift = 0, 0, 0.0
    for k in range(steps):
        # Random actions do not get an untrained ant to the ball, so the ball
        # would never move and "the anchor does not follow the ball" would be
        # vacuously true. Roll it directly instead -- through qvel, NOT
        # _write_ball, because _write_ball is the spawn path and is SUPPOSED to
        # move the anchor. This is the struck-ball case.
        if k % 25 == 12:
            env.qvel[:, env._ball_vcols[0]] = 4.0
        _, _, done = env.step(torch.randn(env.n, env.act_dim,
                                          device=env.device) * 0.8)
        restarted = env.n_segments > prev_seg
        if bool(restarted.any()):
            i = restarted.nonzero(as_tuple=True)[0]
            err = (env.ball_spawn_xy[i] - env._ball_xy()[i]).norm(dim=-1).max()
            assert float(err) < 1e-4, f"anchor not rewritten on restart: {err}"
            n_restarts += int(i.numel())
        held = (~restarted).nonzero(as_tuple=True)[0]
        if held.numel():
            drift = (env.ball_spawn_xy[held]
                     - env._ball_xy()[held]).norm(dim=-1)
            max_drift = max(max_drift, float(drift.max()))
            n_drifted += int((drift > 0.25).sum())
        prev_seg = env.n_segments.clone()
        if done:
            env.reset()
            assert torch.allclose(env.ball_spawn_xy, env._ball_xy(), atol=1e-5)
            prev_seg = env.n_segments.clone()

    assert n_restarts > 10, f"too few segment restarts to conclude: {n_restarts}"
    # If the anchor tracked the live ball, max_drift would be ~0 everywhere and
    # this term would be measuring nothing.
    assert n_drifted > 0 and max_drift > 0.25, (n_drifted, max_drift)
    ex = float(env.anchor_excess(1.0).max())
    assert 0.0 <= ex <= 5.0, ex
    return (f"{n_restarts} restarts all rewrote the anchor; ball drifted up to "
            f"{max_drift:.2f} m from it")


def t_kick_arrival_is_graded_at_the_deadline(steps=300, worlds=8):
    """last_arrival is exp(-c * d_at_T) on the closing step and 0 otherwise --
    the only channel the reward has, so if it is stale or leaky the objective
    is not the one the spec describes."""
    env = _kick(worlds)
    env.reset()
    prev_seg = env.n_segments.clone()
    seen, arrivals = 0, []
    for _ in range(steps):
        d_before = env._target_dist_now().clone()
        env.step(torch.zeros(env.n, env.act_dim, device=env.device))
        closed = env.n_segments > prev_seg
        prev_seg = env.n_segments.clone()
        nz = env.last_arrival > 0
        assert bool((nz == closed).all()), "arrival paid off the deadline"
        if bool(closed.any()):
            i = closed.nonzero(as_tuple=True)[0]
            # d_before is the distance at the END of the closing step (measured
            # before the next step's physics), i.e. exactly d(T).
            want = torch.exp(-0.5 * d_before[i])
            assert float((env.last_arrival[i] - want).abs().max()) < 1e-5
            arrivals += env.last_arrival[i].tolist()
            seen += int(i.numel())
        fit = env.fitness()
        assert torch.isfinite(fit).all() and float(fit.min()) >= 0.0 \
            and float(fit.max()) <= 1.0, float(fit.max())
    assert seen > 10, seen
    return (f"{seen} deadlines graded, arrival mean "
            f"{sum(arrivals)/len(arrivals):.3f}, fitness in [0,1]")


def t_shoot_runs_and_fitness_is_bounded(steps=200, worlds=8):
    from rower_soccer.warp_port.scene import BallSpec
    from rower_soccer.warp_port.shoot_env import WarpShootEnv
    env = WarpShootEnv(num_worlds=worlds, seed=3, use_graph=True,
                       creature_xml="creature_configs/ant.xml",
                       ball=BallSpec(radius=0.15, mass=0.045),
                       arena="pitch", pitch_scale=0.3125,
                       goal_time_coef=0.4, w_upright=1.0)
    obs = env.reset()
    lo, hi = 1.0, 0.0
    for _ in range(steps):
        obs, rew, done = env.step(
            torch.zeros(env.n, env.act_dim, device=env.device))
        assert torch.isfinite(obs).all() and torch.isfinite(rew).all()
        # last_score_t only ever fires with scored_now.
        assert not bool(((env.last_score_t > 0) & ~env.scored_now).any())
        f = env.fitness()
        assert torch.isfinite(f).all(), "NaN fitness"
        lo, hi = min(lo, float(f.min())), max(hi, float(f.max()))
        assert 0.0 <= lo and hi <= 1.0, (lo, hi)
        if done:
            obs = env.reset()
    return f"fitness stayed in [{lo:.3f}, {hi:.3f}]"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-physics", action="store_true",
                   help="reward algebra only: no GPU, no Warp compile")
    args = p.parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    check("shoot: goal bonus is time-discounted",
          t_shoot_goal_bonus_is_time_discounted)
    check("shoot: bonus reads the pre-respawn clock",
          t_shoot_bonus_reads_the_presnapshot_clock)
    check("shoot: fitness ranks any goal over any miss",
          t_shoot_fitness_ranks_goals_over_misses)
    check("kick: timed reward drops ball->cmd shaping",
          t_timed_kick_drops_ball_to_cmd_shaping)
    check("kick: timed fitness is mean arrival at T",
          t_timed_kick_fitness_is_mean_arrival)
    check("kick: anchor stops paying for the chase (v7)",
          t_anchor_stops_paying_for_the_chase)
    check("kick: anchor penalty shape and channel (v7)", t_anchor_penalty_shape)
    check("kick: strike offset sits behind the ball (v8)",
          t_strike_offset_puts_the_approach_behind_the_ball)
    check("kick: anchor is off by default (v7)", t_anchor_is_off_by_default)
    check("kick: anchor magnitude is calibrated (v7)",
          t_anchor_magnitude_is_calibrated)
    if not args.no_physics:
        check("kick: ball_spawn_xy tracks every spawn path (v7)",
              t_ball_spawn_xy_tracks_spawns)
        check("kick: task width 12 (point) / 14 (timed)", t_kick_task_width)
        check("kick: deadline matches the pace band",
              t_kick_deadline_matches_the_pace_band)
        check("kick: arrival graded at the deadline",
              t_kick_arrival_is_graded_at_the_deadline)
        check("shoot: env steps, fitness bounded",
              t_shoot_runs_and_fitness_is_bounded)

    n_fail = sum(1 for _, ok in _results if not ok)
    print(f"\n{len(_results) - n_fail}/{len(_results)} passed", flush=True)
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
