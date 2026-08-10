"""Builders for creature soccer environments.

Reuses the repo-root integration (creature.py walker, custom_soccer_env.py
factory with its arena.xml monkey-patch) from the phase-2 project.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Importing custom_soccer_env applies the arena.xml monkey-patch at import
# time (composer.arena._ARENA_XML_PATH) — must happen before any arena is
# constructed anywhere in the process.
from custom_soccer_env import create_soccer_env  # noqa: E402
from creature import Creature  # noqa: E402
from dm_control.locomotion.soccer.team import RGBA_BLUE, RGBA_RED  # noqa: E402

CREATURE_XMLS = {
    "rower": os.path.join(_REPO_ROOT, "creature_configs", "two_arm_rower_blueprint.xml"),
    "worm": os.path.join(_REPO_ROOT, "creature_configs", "three_seg_worm.xml"),
    # The ant sprint's validation body (docs/ANT_SPRINT_PLAN.md). Same 8
    # actuators / 65-wide proprio as the rower, so every trained architecture
    # dimension carries over to the creature swap unchanged.
    "ant": os.path.join(_REPO_ROOT, "creature_configs", "ant.xml"),
}

# Team composition: rower attacks, worm defends (project default).
DEFAULT_TEAM = ("rower", "worm")


def make_creature(kind="rower", team="home", expose_root_pose=False):
    """One creature walker.

    expose_root_pose enables the `absolute_root_pos` / `absolute_root_mat`
    observables. It is FALSE for drills and TRUE for the soccer env, and the
    asymmetry is deliberate:

    * The drills' observation is the trained obs contract (ant/rower: 65 proprio
      + task). Enabling two more keys there would silently widen it to 77 and
      every checkpoint would slice the wrong columns.
    * The soccer env needs them. dm_soccer's `CoreObservablesAdder` enables only
      `walker.observables.proprioception` + the kinematic sensors, and root pose
      is (correctly) not in proprioception -- but `soccer_bridge.py` and, after
      it, WS3's SkillController rebuild each skill's egocentric task block
      (`target_ego`, ball targets, ...) from the root pose, so without these two
      keys the bridge raises KeyError('absolute_root_pos') and no drill policy
      can be driven inside soccer at all. They are extra keys in the soccer
      observation dict, never part of any policy's input vector -- consumers
      select by name.
    """
    rgba = RGBA_BLUE if team == "home" else RGBA_RED
    creature = Creature(CREATURE_XMLS[kind], marker_rgba=rgba)
    if expose_root_pose:
        creature.observables.absolute_root_pos.enabled = True
        creature.observables.absolute_root_mat.enabled = True
    return creature


def drill_ball():
    """The ball every v3 ant drill trained on: dm_soccer's SoccerBall at radius
    0.15 m (mass/friction stock). This is `warp_port/scene.BallSpec(radius=0.15,
    mass=0.045)` expressed as the CPU env's own ball class — the proportion that
    matches dm_control fetch (ball/torso 0.52). Pass it to `make_soccer_env`
    whenever a drill-trained policy plays; the 0.35 default is 2.3x the ball
    those checkpoints ever touched."""
    from dm_control.locomotion.soccer.soccer_ball import SoccerBall
    return SoccerBall(radius=0.15, mass=0.045)


def make_soccer_env(home_team=DEFAULT_TEAM, away_team=DEFAULT_TEAM,
                    n_home=None, n_away=None, time_limit=45.0, random_state=None,
                    disable_walker_contacts=False, terminate_on_goal=True,
                    ball=None):
    """Creature soccer env. Teams are tuples of creature kinds, e.g.
    ("rower", "worm"). n_home/n_away (int) are a homogeneous-rower shorthand.
    Actions/observations are per-player lists, home players first."""
    if n_home is not None:
        home_team = ("rower",) * n_home
    if n_away is not None:
        away_team = ("rower",) * n_away
    home = [make_creature(k, "home", expose_root_pose=True) for k in home_team]
    away = [make_creature(k, "away", expose_root_pose=True) for k in away_team]
    return create_soccer_env(
        home_players=home,
        away_players=away,
        time_limit=time_limit,
        random_state=random_state,
        disable_walker_contacts=disable_walker_contacts,
        terminate_on_goal=terminate_on_goal,
        ball=ball,
    )
