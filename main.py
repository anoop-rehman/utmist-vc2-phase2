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
        self.episode_rewards = []
        self.episode_velocities = []
        
    def _on_step(self):
        # Get current reward and velocity
        reward = self.training_env.get_attr('reward')[0]
        vel_to_ball = self.training_env.get_attr('env')[0].last_vel_to_ball if hasattr(self.training_env.get_attr('env')[0], 'last_vel_to_ball') else 0
        
        # Log raw step metrics
        self.logger.record('train/raw_reward', reward)
        self.logger.record('train/raw_velocity_to_ball', vel_to_ball)
        
        # Track episode metrics
        if self.locals.get('done'):
            self.episode_rewards.append(reward)
            self.episode_velocities.append(vel_to_ball)
            
            # Log smoothed episode metrics
            if len(self.episode_rewards) > 0:
                self.logger.record('train/episode_reward_mean', np.mean(self.episode_rewards[-100:]))
                self.logger.record('train/episode_reward_max', np.max(self.episode_rewards[-100:]))
                self.logger.record('train/episode_reward_min', np.min(self.episode_rewards[-100:]))
                self.logger.record('train/episode_velocity_mean', np.mean(self.episode_velocities[-100:]))
        
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
        
        # Print absolute creature position
        if 'absolute_root_pos' in time_step.observation[0]:
            print(f"Creature position: {time_step.observation[0]['absolute_root_pos']}")

        return [action]
    return policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or view a creature model')
    parser.add_argument('--load-model', type=str, help='Path to model to load')
    parser.add_argument('--view-only', action='store_true', help='Only view the model, no training')
    parser.add_argument('--timesteps', type=int, default=20_000, help='Number of timesteps to train')
    parser.add_argument('--load-path', type=str, default=None, help='Path to load a saved model from')
    parser.add_argument('--checkpoint-freq', type=int, default=4000, help='How often to save checkpoints during training')
    parser.add_argument('--keep-checkpoints', action='store_true', help='Keep all checkpoints instead of deleting them')
    parser.add_argument('--checkpoint-stride', type=int, default=1, help='Save every Nth checkpoint (e.g. 3 means save checkpoint_freq * 3)')
    parser.add_argument('--tensorboard-log', type=str, default='tensorboard_logs', help='TensorBoard log directory')
    parser.add_argument('--start-timesteps', type=int, default=None, help='Starting timestep count (for resuming training)')
    args = parser.parse_args()

    # Create environment
    env = create_env()
    vec_env = setup_env(env)

    if args.view_only and args.load_model:
        # Load model and launch viewer
        print(f"\nLoading model from {args.load_model} for viewing...")
        model = PPO.load(args.load_model, env=vec_env)
        print("Launching viewer...")
        viewer.launch(env, policy=create_policy(model))
    else:
        # Train model (either new or resumed)
        print("Starting training...")
        if args.load_model:
            print(f"Resuming training from {args.load_model}")
        
        model = train_creature(
            env=vec_env,  # Use the wrapped environment
            total_timesteps=args.timesteps,
            checkpoint_freq=args.checkpoint_freq,
            load_path=args.load_model or args.load_path,  # Use load_model if provided, otherwise use load_path
            tensorboard_log=args.tensorboard_log,
            keep_checkpoints=args.keep_checkpoints,
            checkpoint_stride=args.checkpoint_stride,
            start_timesteps=args.start_timesteps
        )

        # Launch viewer after training
        print("\nLaunching viewer with trained model...")
        viewer.launch(env, policy=create_policy(model))
