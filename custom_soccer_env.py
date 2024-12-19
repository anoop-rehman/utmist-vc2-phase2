import os
from dm_control import composer
import numpy as np

# Monkey patch out original XML path
# cuz they removed the parameter to override it normally 💀
composer.arena._ARENA_XML_PATH = os.path.join(os.path.dirname(__file__), 'arena.xml')

from dm_control.locomotion.walkers import legacy_base
from dm_control.locomotion.soccer.team import Team, Player
from dm_control.locomotion.soccer.task import Task, MultiturnTask
from dm_control.locomotion.soccer.soccer_ball import SoccerBall
from dm_control.locomotion.soccer.pitch import RandomizedPitch
from typing import List

class FixedResetEnvironment(composer.Environment):
    """Environment wrapper that resets random state before each episode."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Store the initial seed
        if isinstance(self._random_state, np.random.RandomState):
            self._initial_seed = self._random_state.get_state()
        else:
            self._initial_seed = self._random_state

    def reset(self):
        # Reset random state to initial value
        if isinstance(self._initial_seed, tuple):
            self._random_state.set_state(self._initial_seed)
        else:
            self._random_state = np.random.RandomState(self._initial_seed)
        return super().reset()

def create_soccer_env(
    home_players: List[legacy_base.Walker],
    away_players: List[legacy_base.Walker],
    time_limit=45.0,
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
    min_size = (32, 24)
    # max_size = (48, 36)
    max_size = (32, 24)
    ball = SoccerBall()

    task_factory = Task
    if not terminate_on_goal:
        task_factory = MultiturnTask

    home_team = [Player(Team.HOME, player) for player in home_players]
    away_team = [Player(Team.AWAY, player) for player in away_players]

    players = home_team + away_team

    if len(players) < 1:
        raise ValueError("No players on the scene")

    print("RANDOM STATE:", random_state)
    return FixedResetEnvironment(
        task=task_factory(
            players=players,
            arena=RandomizedPitch(
                min_size=min_size,
                max_size=max_size,
                keep_aspect_ratio=keep_aspect_ratio,
                field_box=enable_field_box,
                goal_size=goal_size),
            ball=ball,
            disable_walker_contacts=disable_walker_contacts),
        time_limit=time_limit,
        random_state=random_state
    )
