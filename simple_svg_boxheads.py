import numpy as np
from dm_control.locomotion import soccer as dm_soccer
from dm_control import viewer
from keras import layers
import tensorflow as tf
from collections import deque
import random
from pprint import pprint


# Instantiates a 2-vs-2 BOXHEAD soccer environment with episodes of 10 seconds
# each. Upon scoring, the environment reset player positions and the episode
# continues. In this example, players can physically block each other and the
# ball is trapped within an invisible box encapsulating the field.
env = dm_soccer.load(team_size=2,
                     time_limit=20.0,
                     disable_walker_contacts=False,
                     enable_field_box=True,
                     terminate_on_goal=False,
                     walker_type=dm_soccer.WalkerType.BOXHEAD)

STATE_DIM = sum(np.prod(env.observation_spec()[0][key].shape) for key in env.observation_spec()[0].keys() if 'stats' not in key)
ACTION_DIM = env.action_spec()[0].shape[0]

flag = False
def custom_policy(time_step):
    actions = []
    action_specs = env.action_spec()

    for action_spec in action_specs:
        action = np.ones(action_spec.shape)
        action[0] = 1.0 # forward accel
        action[1] = 0.0 # torque
        action[2] = 0.0 # jump
    
        global flag
        if not flag:
            print(time_step.observation[0])
            flag = True

        actions.append(action)
    return actions


# Use the viewer to visualize the environment
viewer.launch(env, policy=custom_policy)
