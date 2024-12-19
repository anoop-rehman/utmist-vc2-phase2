from custom_soccer_env import create_soccer_env
from dm_control.locomotion.soccer.team import RGBA_BLUE, RGBA_RED
from creature import Creature
from train import train_creature, DMControlWrapper
from dm_control import viewer
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO


# Create creature and environment
home_player = Creature("creature_configs/two_arm_rower_blueprint.xml", marker_rgba=RGBA_BLUE)

env = create_soccer_env(
    home_players=[home_player],
    away_players=[],
    time_limit=60.0,
    disable_walker_contacts=False,
    enable_field_box=True,
    terminate_on_goal=False,
    random_state=3
)

# Train the creature
model = train_creature(env, save_path="trained_creatures/v1_2_1__720kTimesteps")

# Load a trained model
# wrapped_env = DMControlWrapper(env)
# vec_env = DummyVecEnv([lambda: wrapped_env])
# model = PPO(
#     "MlpPolicy",
#     vec_env,
#     verbose=1,
#     learning_rate=3e-4,
#     n_steps=2048,
#     batch_size=64,
#     n_epochs=10,
#     gamma=0.99,
#     gae_lambda=0.95,
#     clip_range=0.2
# )
model.load("trained_creatures/v1_2_1__720kTimesteps")

# Define a policy function for the viewer
def policy(time_step):
    # Convert observation to the format expected by the model
    obs_dict = time_step.observation[0]
    obs = np.concatenate([v.flatten() for v in obs_dict.values()])
    
    # Get action from model
    action, _states = model.predict(obs, deterministic=True)

    # Just randomly generate action (for testing)
    # action = np.random.uniform(-1, 1, size=(8,))

    vel_to_ball = time_step.observation[0]['stats_vel_to_ball'][0]
    ctrl_cost_weight = 0.5
    ctrl_cost = ctrl_cost_weight * np.sum(np.square(action))
    reward = vel_to_ball + 1.0 - ctrl_cost 

    print("-------------------------------")
    print("vel_to_ball:", vel_to_ball)
    print("test reward:", reward)


    # Return action wrapped in a list (for single agent)
    return [action]


# Launch the viewer
viewer.launch(env, policy=policy)
