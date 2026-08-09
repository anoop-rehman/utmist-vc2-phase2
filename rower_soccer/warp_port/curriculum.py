"""Performance-gated target-speed curriculum, shared by follow and dribble.

Table S2 defines dribble as "similar to the 'follow' drill but the agent must
keep the ball close to the moving target", so the two drills should differ in
what is being kept near the target, NOT in how fast the target moves. One
implementation, used by both, is how that stays true.

Why gated rather than scheduled. The stock 0.07-0.6 m/s band was calibrated for
the WORM: a body that topples, tops out near 1.0-1.6 m/s, and needed
target_max/achievable ~= 0.28 before follow was learnable at all. The ant is
statically stable, peaks at 1.34 m/s with reserve, and has never been asked to
SUSTAIN speed -- so its real ceiling is unknown and any fixed band is a guess.
Measured on follow_ant_v2: the ant closes to 0.03 m from the target (p90 0.16 m)
and stays there, scoring fitness 0.97 on a task solved at the moment of
arrival, which on video looks like vibrating in place.

So the band is raised only once the policy is comfortably solving the current
one, and `speed_max` is set deliberately ABOVE expectation: the useful outcome
is the curriculum stalling at the body's actual ceiling, not the cap binding and
hiding it.
"""


class SpeedCurriculum:
    """Raise env.speed_range when fitness holds above `gate`.

    Envs must expose a mutable `speed_range` (both follow and dribble get it
    from `_init_moving_target`) and a `fitness()`. The new band takes effect at
    the next episode reset, which is where target velocities are drawn.
    """

    def __init__(self, enabled=False, gate=0.90, mult=1.25, speed_max=2.5,
                 patience=8):
        self.enabled = enabled
        self.gate = gate
        self.mult = mult
        self.speed_max = speed_max
        self.patience = patience
        self._hist = []

    def update(self, env, eval_env=None):
        """Call once per monitor tick. Returns a log line, or None."""
        if not self.enabled:
            return None
        self._hist.append(float(env.fitness().mean()))
        if len(self._hist) < self.patience:
            return None
        recent = self._hist[-self.patience:]
        if sum(recent) / len(recent) < self.gate:
            return None
        lo, hi = env.speed_range
        nhi = min(hi * self.mult, self.speed_max)
        if nhi <= hi + 1e-6:
            return None                      # already at the cap
        # Keep the slow end proportional, but never let it exceed half the fast
        # end: a band collapsed to a single speed would stop teaching the policy
        # to handle the range it will meet in a game.
        nlo = min(lo * self.mult, nhi * 0.5)
        env.speed_range = (nlo, nhi)
        if eval_env is not None:
            eval_env.speed_range = (nlo, nhi)
        self._hist.clear()
        return (f"[curriculum] fitness {recent[-1]:.3f} >= {self.gate} -> "
                f"target speed {lo:.2f}-{hi:.2f} => {nlo:.2f}-{nhi:.2f} m/s")


def add_args(p):
    """Attach the shared flags to a trainer's ArgumentParser."""
    p.add_argument("--speed-curriculum", action="store_true",
                   help="raise the target-speed band whenever fitness holds "
                        "above --speed-gate for --speed-patience monitor ticks. "
                        "The stock band was calibrated for the worm; the ant "
                        "solves it on arrival (measured 0.03 m from target) and "
                        "then vibrates. Lets the curriculum find the ceiling.")
    p.add_argument("--speed-gate", type=float, default=0.90)
    p.add_argument("--speed-mult", type=float, default=1.25)
    p.add_argument("--speed-max", type=float, default=2.5,
                   help="cap on the fast end (m/s), set ABOVE expectation so the "
                        "curriculum stalls at the body's ceiling rather than the "
                        "cap binding and hiding it")
    p.add_argument("--speed-patience", type=int, default=8,
                   help="monitor ticks the gate must hold before a bump")


def from_args(args):
    return SpeedCurriculum(enabled=getattr(args, "speed_curriculum", False),
                           gate=args.speed_gate, mult=args.speed_mult,
                           speed_max=args.speed_max, patience=args.speed_patience)
