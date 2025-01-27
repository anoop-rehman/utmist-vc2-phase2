from custom_soccer_env import create_soccer_env
from dm_control.locomotion.soccer.team import RGBA_BLUE, RGBA_RED
from creature import Creature
from train import train_creature, DMControlWrapper
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

# Initial training
print("Starting initial training (Run 1)...")
model = train_creature(
    env, 
    save_path="trained_creatures/run_10k",
    total_timesteps=10_000
)

# Sequential runs
for i in range(2, 9):  # Runs 2 through 8
    print(f"\nStarting Run {i}...")
    model = train_creature(
        env, 
        save_path=f"trained_creatures/run_{i*10}k",
        total_timesteps=10_000,
        load_path=f"trained_creatures/run_{(i-1)*10}k"
    )

# Define a policy function for the viewer
def policy(time_step):
    # Convert observation to the format expected by the model
    obs_dict = time_step.observation[0]
    obs = np.concatenate([v.flatten() for v in obs_dict.values()])
    
    # Get action from model
    action, _states = model.predict(obs, deterministic=True)
    
    vel_to_ball = time_step.observation[0]['stats_vel_to_ball'][0]
    ctrl_cost_weight = 0.5
    ctrl_cost = ctrl_cost_weight * np.sum(np.square(action))
    reward = vel_to_ball + 1.0 - ctrl_cost 

    print("-------------------------------")
    print("vel to ball:", vel_to_ball)
    print("test reward:", reward)

    return [action]

# Launch the viewer with the final model
print("\nLaunching viewer with final model...")
viewer.launch(env, policy=policy)
