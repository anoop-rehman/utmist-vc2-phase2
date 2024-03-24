# import numpy as np
# from dm_control.locomotion import soccer as dm_soccer
# from dm_control import viewer


# # Instantiates a 2-vs-2 BOXHEAD soccer environment with episodes of 10 seconds
# # each. Upon scoring, the environment reset player positions and the episode
# # continues. In this example, players can physically block each other and the
# # ball is trapped within an invisible box encapsulating the field.
# env = dm_soccer.load(team_size=2,
#                      time_limit=10.0,
#                      disable_walker_contacts=False,
#                      enable_field_box=True,
#                      terminate_on_goal=False,
#                      walker_type=dm_soccer.WalkerType.ANT)

# # # Retrieves action_specs for all 4 players.
# # action_specs = env.action_spec()

# # # Step through the environment for one episode with random actions.
# # timestep = env.reset()
# # while not timestep.last():
# #   actions = []
# #   for action_spec in action_specs:
# #     action = np.random.uniform(
# #         action_spec.minimum, action_spec.maximum, size=action_spec.shape)
# #     actions.append(action)
# #   timestep = env.step(actions)

# #   for i in range(len(action_specs)):
# #     print(
# #         "Player {}: reward = {}, discount = {}, observations = {}.".format(
# #             i, timestep.reward[i], timestep.discount, timestep.observation[i]))

# # Function to generate random actions for all players.
# def random_policy(time_step):
#     actions = []
#     action_specs = env.action_spec()
#     for action_spec in action_specs:
#         action = np.random.uniform(
#             action_spec.minimum, action_spec.maximum, size=action_spec.shape)
#         actions.append(action)
#     return actions

# # Use the viewer to visualize the environment with the random policy.
# viewer.launch(env, policy=random_policy)


import numpy as np
from dm_control.locomotion import soccer as dm_soccer
from dm_control import viewer
from tensorflow.keras import layers
import tensorflow as tf
from collections import deque
import random


# Instantiates a 2-vs-2 BOXHEAD soccer environment with episodes of 10 seconds
# each. Upon scoring, the environment reset player positions and the episode
# continues. In this example, players can physically block each other and the
# ball is trapped within an invisible box encapsulating the field.
env = dm_soccer.load(team_size=2,
                     time_limit=10.0,
                     disable_walker_contacts=False,
                     enable_field_box=True,
                     terminate_on_goal=False,
                     walker_type=dm_soccer.WalkerType.ANT)


# # Assuming env has been correctly initialized
# observation_spec = env.observation_spec()
# action_spec = env.action_spec()

# # Observation spec is a list, let's handle it properly
# def get_observation_dim(obs_spec):
#     total_dims = 0
#     for obs in obs_spec:
#         # Each obs is an OrderedDict where each item has a shape attribute
#         for key, spec in obs.items():
#             total_dims += np.prod(spec.shape)
#     return total_dims

# # Calculate the total dimension of the observation space
# observation_dim = get_observation_dim(observation_spec)

# # Action spec gives us the shape directly
# action_dim = action_spec.shape[0]

# # Print the observation and action space dimensions
# print('Observation dimension:', observation_dim)
# print('Action dimension:', action_dim)

# Constants and Hyperparameters
STATE_DIM = env.observation_space.shape  # To be adjusted according to your environment's state space
ACTION_DIM = env.action_space.shape[0]  # To be adjusted according to your environment's action space
ACTOR_LR = 1e-4
CRITIC_LR = 2e-4
GAMMA = 0.99
TAU = 0.005
BUFFER_SIZE = 100000
MIN_BUFFER_SIZE = 1000
BATCH_SIZE = 64
# NUM_EPISODES = 1000
NUM_EPISODES = 3

# Replay buffer to store transitions
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

# Actor Model (Policy)
def create_actor(state_dim, action_dim):
    inputs = layers.Input(shape=(state_dim,))
    out = layers.Dense(256, activation="relu")(inputs)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(action_dim, activation="tanh")(out)  # Assuming action space is [-1, 1]
    
    model = tf.keras.Model(inputs, outputs)
    return model

# Critic Model (Value Function)
def create_critic(state_dim, action_dim):
    state_input = layers.Input(shape=(state_dim,))
    action_input = layers.Input(shape=(action_dim,))
    concatenated = layers.Concatenate()([state_input, action_input])
    
    out = layers.Dense(256, activation="relu")(concatenated)
    out = layers.Dense(256, activation="relu")(out)
    outputs = layers.Dense(1, activation="linear")(out)
    
    model = tf.keras.Model([state_input, action_input], outputs)
    return model


# The training step function
def train_step(states, actions, rewards, next_states, dones):
    # Compute target actions using actor_target
    # Compute Q-values using critic_target
    # Compute actor and critic losses
    # Update critic using critic_loss
    # Update actor using policy gradient
    # Soft update target networks
    pass  # This function needs to be fully implemented

# Initialize replay buffer
replay_buffer = ReplayBuffer(BUFFER_SIZE)

# Instantiate models and optimizers
actor = create_actor(STATE_DIM, ACTION_DIM)
critic = create_critic(STATE_DIM, ACTION_DIM)
actor_target = create_actor(STATE_DIM, ACTION_DIM)
critic_target = create_critic(STATE_DIM, ACTION_DIM)
actor_optimizer = tf.keras.optimizers.Adam(learning_rate=ACTOR_LR)
critic_optimizer = tf.keras.optimizers.Adam(learning_rate=CRITIC_LR)

# Random policy
def random_policy(time_step):
    actions = []
    action_specs = env.action_spec()
    for action_spec in action_specs:
        action = np.random.uniform(
            action_spec.minimum, action_spec.maximum, size=action_spec.shape)
        actions.append(action)
    return actions

# Training function to replace random_policy
def svg0_policy(time_step):
    if replay_buffer.size() < MIN_BUFFER_SIZE: 
        return random_policy(time_step) 
    
    # Otherwise, use the actor model to generate actions
    actions = []
    for walker_observation in time_step.observation['observations']:
        action = actor.predict(walker_observation[np.newaxis, ...])
        actions.append(action.squeeze())
    return actions


# # Main training loop
# for episode in range(NUM_EPISODES):
#     time_step = env.reset()
#     episode_reward = 0

#     while not time_step.last():
#         action = svg0_policy(time_step)
#         next_time_step = env.step(action)
#         reward = next_time_step.reward
#         done = next_time_step.last()

#         # Convert the TimeStep and action to suitable data structures
#         # before adding to replay buffer
#         replay_buffer.add(time_step.observation, action, reward, next_time_step.observation, done)
#         time_step = next_time_step

#         # Train as in the SVG(0) algorithm
#         if replay_buffer.size() >= MIN_BUFFER_SIZE:
#             states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
#             train_step(states, actions, rewards, next_states, dones)

#         episode_reward += reward

#     print(f'Episode: {episode} Total Reward: {episode_reward}')

# Use the viewer to visualize the environment
viewer.launch(env, policy=random_policy)
