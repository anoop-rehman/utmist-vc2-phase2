from custom_soccer_env import create_soccer_env
from dm_control.locomotion.soccer.team import RGBA_BLUE, RGBA_RED
from creature import Creature
from train import train_creature, DMControlWrapper, process_observation, calculate_reward
from dm_control import viewer
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os

# Add this new callback class for real-time plotting
class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self):
        # Log scalar value (here a random value)
        self.logger.record('reward', self.training_env.get_attr('reward')[0])
        return True

# Create creature and environment
home_player = Creature("creature_configs/two_arm_rower_blueprint.xml", marker_rgba=RGBA_BLUE)

env = create_soccer_env(
    home_players=[home_player],
    away_players=[],
    time_limit=60.0,
    disable_walker_contacts=False,
    enable_field_box=True,
    terminate_on_goal=False
)

# Train with checkpoints
print("Starting training with checkpoints...")
model = train_creature(
    env, 
    total_timesteps=20_000,  # This will create checkpoints at 4k, 8k, 12k, 16k, 20k
    checkpoint_freq=4000
)

# Define a policy function for the viewer
def policy(time_step):
    # Process observation and get action
    obs = process_observation(time_step)
    action, _states = model.predict(obs, deterministic=True)
    
    # Calculate and print reward
    reward, vel_to_ball = calculate_reward(time_step, action)
    print("-------------------------------")
    print("vel to ball:", vel_to_ball)
    print("test reward:", reward)

    return [action]

# Launch the viewer with the final model
print("\nLaunching viewer with final model...")
viewer.launch(env, policy=policy)
