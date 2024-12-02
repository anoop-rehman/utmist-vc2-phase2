import numpy as np
from custom_soccer_env import create_soccer_env
from dm_control.locomotion.soccer.team import RGBA_BLUE, RGBA_RED
from dm_control import viewer
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from ant import Ant
from creature import Creature

home_player_1 = Creature("creature_configs/two_arm_rower_blueprint.xml", marker_rgba=RGBA_BLUE)
home_player_2 = Creature("creature_configs/two_arm_rower_blueprint.xml", marker_rgba=RGBA_BLUE)
away_player_1 = Creature("creature_configs/two_arm_rower_blueprint.xml", marker_rgba=RGBA_RED)
away_player_2 = Creature("creature_configs/two_arm_rower_blueprint.xml", marker_rgba=RGBA_RED)

# Instantiates soccer environment using custom creatures with episodes of 10 seconds
# each. Upon scoring, the environment reset player positions and the episode
# continues. In this example, players can physically block each other and the
# ball is trapped within an invisible box encapsulating the field.
env = create_soccer_env(
    home_players=[home_player_1],
    away_players = [],
    time_limit=60.0,
    disable_walker_contacts=False,
    enable_field_box=True,
    terminate_on_goal=False
)

# # Retrieves action_specs for all 4 players.
# action_specs = env.action_spec()

# # Step through the environment for one episode with random actions.
# timestep = env.reset()
# while not timestep.last():
#   actions = []
#   for action_spec in action_specs:
#     action = np.random.uniform(
#         action_spec.minimum, action_spec.maximum, size=action_spec.shape)
#     actions.append(action)
#   timestep = env.step(actions)

#   for i in range(len(action_specs)):
#     print(
#         "Player {}: reward = {}, discount = {}, observations = {}.".format(
#             i, timestep.reward[i], timestep.discount, timestep.observation[i]))

# Function to generate random actions for all players.

def random_policy(time_step):
    actions = []
    action_specs = env.action_spec()
    for action_spec in action_specs:
        action = np.random.uniform(
            -1, 1, size=action_spec.shape)
        actions.append(action)
    print(time_step.observation[0]['joints_vel'])
    return actions
    

# Use the viewer to visualize the environment with the random policy.
viewer.launch(env, policy=random_policy)
