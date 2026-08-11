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
    if not args.no_physics:
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
