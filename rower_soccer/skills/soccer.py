"""Glue between a dm_control soccer env and `SkillController`.

Two jobs, both small:

  * `SoccerFrameSource` turns a `TimeStep` into one `PlayerFrame` per player,
    adding the root pose that dm_soccer's observation deliberately omits.
  * `match_drill_timesteps` puts the physics on the dt the drills trained at.

Backend independence: nothing here assumes MuJoCo CPU. A `PlayerFrame` is an
observation dict, a root pose, and the ball's world state — a warp-backed 2v2
game (the escalation option in ANT_SPRINT_WORKSTREAMS' sim2sim section) can
supply all three from its own state tensors, and `SkillController` is unchanged.
The only CPU-specific calls in this package are the `physics.bind(...)` lookups
in `SoccerFrameSource._refresh_bindings`; port that one method and the rest
follows.
"""

import os
import sys

import numpy as np

from rower_soccer.skills.api import PlayerFrame

__all__ = ["SoccerFrameSource", "match_drill_timesteps", "make_skill_soccer_env",
           "DRILL_PHYSICS_DT", "DRILL_CONTROL_DT"]

# What the warp drills integrate at (`worm_env_base.CONTROL_DT` / SUBSTEPS=10).
# dm_soccer's Task defaults to physics 0.005 (5 substeps) at the same 40 Hz
# control rate.
DRILL_CONTROL_DT = 0.025
DRILL_PHYSICS_DT = 0.0025

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def match_drill_timesteps(env, physics_dt: float = DRILL_PHYSICS_DT,
                          control_dt: float = DRILL_CONTROL_DT):
    """Run the soccer env at the drill's integration step.

    The policy was optimised against 10 substeps of 0.0025; soccer defaults to 5
    of 0.005. Same control rate either way, so this changes only integration
    accuracy — but it is one of the few knobs that costs nothing and removes a
    known train/deploy difference, so it is on by default in this package's
    helpers. Call with the soccer defaults to measure the difference.
    """
    env.task.set_timesteps(control_timestep=control_dt, physics_timestep=physics_dt)
    return env


def make_skill_soccer_env(home=("ant",), away=(), *, time_limit=45.0,
                          random_state=None, match_dt=True, **kwargs):
    """`envs.build.make_soccer_env` with the drill dt applied.

    Creature kinds not in `envs.build.CREATURE_XMLS` are resolved from
    `creature_configs/<kind>.xml`, so this works whether or not WS5's one-line
    ant entry has landed on the branch you are on.
    """
    from rower_soccer.envs import build as B
    from rower_soccer.skills.contract import creature_xml_path

    for kind in tuple(home) + tuple(away):
        if kind not in B.CREATURE_XMLS:
            B.CREATURE_XMLS[kind] = creature_xml_path(kind)

    env = B.make_soccer_env(home_team=tuple(home), away_team=tuple(away),
                            n_away=0 if not away else None,
                            time_limit=time_limit, random_state=random_state,
                            **kwargs)
    if match_dt:
        match_drill_timesteps(env)
    return env


class SoccerFrameSource:
    """Builds `PlayerFrame`s for every player of a dm_control soccer env.

    Constructed once per env; it caches each walker so the per-tick cost is two
    `physics.bind` lookups per player.
    """

    def __init__(self, env):
        self._env = env
        self._walkers = [p.walker for p in env.task.players]
        # `physics.bind` walks the mjcf element -> model index map on every call,
        # which at 40 Hz x 4 players is real time spent for a lookup that never
        # changes. The returned binding is a live view, so binding once per walker
        # and reading `.xpos` each tick gives the same numbers for free.
        # ...but the soccer env uses `RandomizedPitch`, which resizes the pitch
        # per episode and therefore RECOMPILES physics on reset. A binding held
        # across that recompile holds a dead weakref and raises on first read, so
        # the cache is keyed on the live physics object and rebuilt when it
        # changes. (Found the hard way: caching without this key raised
        # "weakly-referenced object no longer exists" on the second episode.)
        self._bind_key = None
        self._bindings = []
        self._ball_binding = None
        self._ball_joint = None

    def _refresh_bindings(self):
        physics = self._env.physics
        key = id(physics)
        if key == self._bind_key:
            return
        ball = self._env.task.ball
        self._bindings = [physics.bind(w.root_body) for w in self._walkers]
        self._ball_binding = physics.bind(ball.root_body)
        # The ball's freejoint qvel[:3] is its WORLD linear velocity — the same
        # quantity `worm_env_base._ball_vel_xyz` reads. `cvel` is not: it is a
        # spatial velocity about the subtree COM.
        self._ball_joint = physics.bind(ball.root_body.freejoint)
        self._bind_key = key

    def ball_state(self):
        """(world position (3,), world linear velocity (3,)) of the ball."""
        self._refresh_bindings()
        return (np.array(self._ball_binding.xpos),
                np.array(self._ball_joint.qvel)[:3])

    @property
    def n_players(self) -> int:
        return len(self._walkers)

    @property
    def walkers(self):
        return tuple(self._walkers)

    def _pose(self, i: int):
        self._refresh_bindings()
        b = self._bindings[i]
        return np.array(b.xpos), np.array(b.xmat)

    def frames(self, timestep):
        """One `PlayerFrame` per player, in env (home-first) order."""
        obs = timestep.observation
        if len(obs) != len(self._walkers):
            raise ValueError(
                f"timestep has {len(obs)} player observations but the env has "
                f"{len(self._walkers)} players")
        bp, bv = self.ball_state()
        out = []
        for i in range(len(self._walkers)):
            pos, mat = self._pose(i)
            out.append(PlayerFrame(obs=obs[i], root_pos=pos, root_mat=mat,
                                   ball_pos=bp, ball_vel=bv))
        return out

    def frame(self, timestep, i: int) -> PlayerFrame:
        pos, mat = self._pose(i)
        bp, bv = self.ball_state()
        return PlayerFrame(obs=timestep.observation[i], root_pos=pos, root_mat=mat,
                           ball_pos=bp, ball_vel=bv)

    def root_xy(self, i: int) -> np.ndarray:
        return self._pose(i)[0][:2]

    def ball_xy(self) -> np.ndarray:
        """World XY of the ball (scoreboards, HUDs, and the demo's own metrics)."""
        return self.ball_state()[0][:2]
