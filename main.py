from custom_soccer_env import create_soccer_env
from dm_control.locomotion.soccer.team import RGBA_BLUE, RGBA_RED
from creature import Creature
from train import train_creature, DMControlWrapper, process_observation, calculate_reward, setup_env
from dm_control import viewer
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os
import argparse

# Add this new callback class for real-time plotting
class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self):
        # Log scalar value (here a random value)
        self.logger.record('reward', self.training_env.get_attr('reward')[0])
        return True

def create_env():
    """Create the creature and environment."""
    home_player = Creature("creature_configs/two_arm_rower_blueprint.xml", marker_rgba=RGBA_BLUE)
    return create_soccer_env(
        home_players=[home_player],
        away_players=[],
        time_limit=60.0,
        disable_walker_contacts=False,
        enable_field_box=True,
        terminate_on_goal=False
    )

def create_policy(model):
    """Create a policy function for the viewer."""
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
    return policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or view a creature model')
    parser.add_argument('--load-model', type=str, help='Path to model to load')
    parser.add_argument('--view-only', action='store_true', help='Only view the model, no training')
    parser.add_argument('--timesteps', type=int, default=20_000, help='Number of timesteps to train')
    parser.add_argument('--checkpoint-freq', type=int, default=4000, help='Frequency of checkpoints')
    args = parser.parse_args()

    # Create environment
    env = create_env()

    if args.view_only and args.load_model:
        # Load model and launch viewer
        print(f"\nLoading model from {args.load_model} for viewing...")
        vec_env = setup_env(env)
        model = PPO.load(args.load_model, env=vec_env)
        print("Launching viewer...")
        viewer.launch(env, policy=create_policy(model))
    else:
        # Train model (either new or resumed)
        print("Starting training...")
        if args.load_model:
            print(f"Resuming training from {args.load_model}")
        
        model = train_creature(
            env, 
            total_timesteps=args.timesteps,
            checkpoint_freq=args.checkpoint_freq,
            load_path=args.load_model
        )

        # Launch viewer after training
        print("\nLaunching viewer with trained model...")
        viewer.launch(env, policy=create_policy(model))
