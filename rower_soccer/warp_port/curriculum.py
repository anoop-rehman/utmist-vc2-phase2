"""Performance-gated target-speed curriculum, shared by follow and dribble.

Table S2 defines dribble as "similar to the 'follow' drill but the agent must
keep the ball close to the moving target", so the two drills should differ in
what is being kept near the target, NOT in how fast the target moves. One
implementation, used by both, is how that stays true.

Why the gate is TRACKING, not fitness
-------------------------------------
The first version gated on `fitness >= 0.9`, then `>= 0.78`. Both were wrong for
the same reason, and it took two failures to see it.

Follow's fitness is exp(-0.5 * d). Its ceiling is set by how precisely the body
can PARK on the target, which is a property of the body, not of whether the
target is too slow:

    ant v1 (small)   settles 0.03 m from target -> fitness 0.985
    ant v2 (2.7x)    settles 0.56 m from target -> fitness 0.756

So on ant_v2 a 0.90 gate was unreachable (the curriculum never fired in 72M
steps), and a 0.78 gate sat 0.004 above the operating point (fires on noise).
Any fitness threshold is either unreachable or marginal, because it is measuring
precision when the question is capability. Meanwhile the run looks perfectly
healthy the whole time -- the same shape of failure as a rising reward curve
over an objective that turned out to be empty.

The question "is the target too easy?" has a direct answer: IS THE CREATURE
KEEPING UP. If the distance to the target is not growing over the episode, the
creature is tracking it and the target can go faster. That is body-size
independent, needs no threshold tuned per creature, and is exactly what the
measurement below showed for ant_v2:

    band 0.2-1.2   dist 0.4 -> 0.4 m   keeping up (ant cruising at 40% of ceiling)
    band 1.0-2.0   dist 1.7 -> 2.2 m   roughly keeping up
    band 2.0-3.5   dist 3.7 -> 4.8 m   falling behind  <- the real ceiling, ~2.15 m/s

Set `speed_max` from a measured ceiling where you have one. The curriculum is a
safety net for finding it, not a substitute for measuring it.
"""


class SpeedCurriculum:
    """Raise env.speed_range while the creature is still keeping up.

    Envs must expose a mutable `speed_range` (both follow and dribble get it from
    `_init_moving_target`) and `tracking_error()` -- the current mean distance
    from the thing being tracked to the target. The new band takes effect at the
    next episode reset, where target velocities are drawn.
    """

    def __init__(self, enabled=False, mult=1.4, speed_max=8.0, patience=4,
                 grow_tol=0.25):
        self.enabled = enabled
        self.mult = mult
        self.speed_max = speed_max
        self.patience = patience
        # Metres of drift per tick tolerated before "falling behind". Not 0: the
        # distance fluctuates as the target turns, and demanding a strictly
        # non-increasing distance would stall the curriculum on noise -- the
        # marginal-gate failure again, in a different coordinate.
        self.grow_tol = grow_tol
        self._hist = []

    def update(self, env, eval_env=None):
        """Call once per monitor tick. Returns a log line, or None."""
        if not self.enabled:
            return None
        d = self._error(env)
        if d is None:
            return None
        self._hist.append(d)
        if len(self._hist) < self.patience + 1:
            return None
        window = self._hist[-(self.patience + 1):]
        # Keeping up == distance not trending upward across the window.
        drift = window[-1] - window[0]
        if drift > self.grow_tol:
            self._hist = self._hist[-(self.patience + 1):]
            return None
        lo, hi = env.speed_range
        nhi = min(hi * self.mult, self.speed_max)
        if nhi <= hi + 1e-6:
            return None                      # already at the cap
        # Keep the slow end proportional but never above half the fast end: a
        # band collapsed to a single speed stops teaching the range the creature
        # meets in a game.
        nlo = min(lo * self.mult, nhi * 0.5)
        env.speed_range = (nlo, nhi)
        if eval_env is not None:
            eval_env.speed_range = (nlo, nhi)
        self._hist.clear()
        return (f"[curriculum] tracking held (drift {drift:+.2f} m over "
                f"{self.patience} ticks) -> target speed "
                f"{lo:.2f}-{hi:.2f} => {nlo:.2f}-{nhi:.2f} m/s")

    @staticmethod
    def _error(env):
        fn = getattr(env, "tracking_error", None)
        return None if fn is None else float(fn().mean())


def add_args(p):
    """Attach the shared flags to a trainer's ArgumentParser."""
    p.add_argument("--speed-curriculum", action="store_true",
                   help="raise the target-speed band while the creature is "
                        "still KEEPING UP (distance to target not growing). "
                        "Gating on fitness does not work: fitness is bounded by "
                        "parking precision, which is a property of body size, so "
                        "any threshold is either unreachable or marginal.")
    p.add_argument("--speed-mult", type=float, default=1.4)
    p.add_argument("--speed-max", type=float, default=8.0,
                   help="cap on the fast end (m/s). Set from a MEASURED ceiling "
                        "where you have one (ant_v2: ~2.15 m/s).")
    p.add_argument("--speed-patience", type=int, default=4,
                   help="monitor ticks the distance must hold before a bump")
    p.add_argument("--speed-grow-tol", type=float, default=0.25,
                   help="metres of drift across the window still counted as "
                        "keeping up")


def from_args(args):
    return SpeedCurriculum(enabled=getattr(args, "speed_curriculum", False),
                           mult=args.speed_mult, speed_max=args.speed_max,
                           patience=args.speed_patience,
                           grow_tol=getattr(args, "speed_grow_tol", 0.25))
