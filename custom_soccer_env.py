import os
from dm_control import composer

# Monkey patch out original XML path
# cuz they removed the parameter to override it normally 💀
composer.arena._ARENA_XML_PATH = os.path.join(os.path.dirname(__file__), 'arena.xml')

from dm_control.locomotion.walkers import legacy_base
from dm_control.locomotion.soccer.team import Team, Player
from dm_control.locomotion.soccer.task import Task, MultiturnTask
from dm_control.locomotion.soccer.soccer_ball import SoccerBall
from dm_control.locomotion.soccer.pitch import Pitch, RandomizedPitch
from typing import List

def create_soccer_env(
    home_players: List[legacy_base.Walker],
    away_players: List[legacy_base.Walker],
    time_limit=45.,
    random_state=None,
    disable_walker_contacts=False,
    enable_field_box=False,
    keep_aspect_ratio=False,
    terminate_on_goal=True,
):
    """Construct a soccer environment with custom agents.

    Args:
            home_players: list of walkers/creatures for the home team
            away_players: list of walkers/creatures for the away team

            time_limit: (optional) Float, the maximum duration of each episode in seconds.
            random_state: (optional) an int seed or `np.random.RandomState` instance.
            disable_walker_contacts: (optional) if `True`, disable physical contacts
                    between walkers.
            enable_field_box: (optional) if `True`, enable physical bounding box for
                    the soccer ball (but not the players).
            keep_aspect_ratio: (optional) if `True`, maintain constant pitch aspect
                    ratio.
            terminate_on_goal: (optional) if `False`, continuous game play across
                    scoring events.

    Returns:
            A `composer.Environment` instance.
    """
    goal_size = None
    field_size = (40, 30)  # Using a fixed middle size between min and max
    ball = SoccerBall()

    task_factory = Task
    if not terminate_on_goal:
        task_factory = MultiturnTask

    home_team = [Player(Team.HOME, player) for player in home_players]
    away_team = [Player(Team.AWAY, player) for player in away_players]

    players = home_team + away_team

    if len(players) < 1:
        raise ValueError("No players on the scene")

    return composer.Environment(
        task=task_factory(
            players=players,
            arena=Pitch(
                size=field_size,
                field_box=enable_field_box,
                goal_size=goal_size),
            ball=ball,
            disable_walker_contacts=disable_walker_contacts),
        time_limit=time_limit,
        random_state=random_state
    )
