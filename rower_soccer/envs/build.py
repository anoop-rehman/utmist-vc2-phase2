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
}

# Team composition: rower attacks, worm defends (project default).
DEFAULT_TEAM = ("rower", "worm")


def make_creature(kind="rower", team="home"):
    rgba = RGBA_BLUE if team == "home" else RGBA_RED
    return Creature(CREATURE_XMLS[kind], marker_rgba=rgba)


def make_soccer_env(home_team=DEFAULT_TEAM, away_team=DEFAULT_TEAM,
                    n_home=None, n_away=None, time_limit=45.0, random_state=None,
                    disable_walker_contacts=False, terminate_on_goal=True):
    """Creature soccer env. Teams are tuples of creature kinds, e.g.
    ("rower", "worm"). n_home/n_away (int) are a homogeneous-rower shorthand.
    Actions/observations are per-player lists, home players first."""
    if n_home is not None:
        home_team = ("rower",) * n_home
    if n_away is not None:
        away_team = ("rower",) * n_away
    home = [make_creature(k, "home") for k in home_team]
    away = [make_creature(k, "away") for k in away_team]
    return create_soccer_env(
        home_players=home,
        away_players=away,
        time_limit=time_limit,
        random_state=random_state,
        disable_walker_contacts=disable_walker_contacts,
        terminate_on_goal=terminate_on_goal,
    )
