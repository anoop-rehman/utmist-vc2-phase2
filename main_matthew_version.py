import numpy as np
from dm_control.locomotion import soccer as dm_soccer
from dm_control import viewer


'''
# 1) Initiate 4 independent agaents (aka box heads)

NOTE: dm_soccer.lead does the following:
    - a) Constructs a `team_size`-vs-`team_size` soccer environment.

'''

# Instantiates a 2-vs-2 BOXHEAD soccer environment with episodes of 10 seconds
# each. Upon scoring, the environment reset player positions and the episode
# continues. In this example, players can physically block each other and the
# ball is trapped within an invisible box encapsulating the field.


env = dm_soccer.load(team_size=2,                       # 2 v 2
                     time_limit=10.0,                   # 10 second duration of episodes
                     disable_walker_contacts=False,     # False: disable physical contact between walkers
                     enable_field_box=True,             # True: enable physical bounding box for the ball (not players)
                     terminate_on_goal=False,           # False: continous gameplay across scoring events
                     walker_type=dm_soccer.WalkerType.BOXHEAD)  # Type of walker



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
            action_spec.minimum, action_spec.maximum, size=action_spec.shape)
        actions.append(action)
    return actions

# Use the viewer to visualize the environment with the random policy.
viewer.launch(env, policy=random_policy)
