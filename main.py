from custom_soccer_env import create_soccer_env
from dm_control.locomotion.soccer.team import RGBA_BLUE, RGBA_RED
from creature import Creature
from train import train_creature, DMControlWrapper, process_observation, calculate_reward, setup_env, default_hyperparameters, TensorboardCallback
from dm_control import viewer
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import os
import argparse

def create_env(training_phase="combined", view_only=False):
    """Create the environment for training or viewing."""
    home_player = Creature("creature_configs/two_arm_rower_blueprint.xml", marker_rgba=RGBA_BLUE)
    return create_soccer_env(
        home_players=[home_player],
        away_players=[],
        time_limit=8.0,
        disable_walker_contacts=True,
        enable_field_box=False,
        keep_aspect_ratio=False,
        terminate_on_goal=False
    )

def create_policy(model):
    """Create a policy function for the viewer."""
    def policy(time_step):
        # Process observation and get action
        obs = process_observation(time_step)
        action, _states = model.predict(obs, deterministic=True)
        return [action]
    return policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or view a creature model')
    parser.add_argument('--load-model', type=str, help='Path to model to load')
    parser.add_argument('--view-only', action='store_true', help='Only view the model, no training')
    parser.add_argument('--training-phase', type=str, default="combined", choices=["walking", "rotation", "combined"], help='Training phase to use')
    parser.add_argument('--n-updates', type=int, default=3, help=f'Number of policy updates to perform (each update = {default_hyperparameters["n_steps"]} timesteps)')
    parser.add_argument('--load-path', type=str, default=None, help='Path to load a saved model from')
    parser.add_argument('--checkpoint-freq', type=int, default=default_hyperparameters["n_steps"], help='How often to save checkpoints during training (defaults to one checkpoint per update)')
    parser.add_argument('--keep-checkpoints', action='store_true', help='Keep all checkpoints instead of deleting them')
    parser.add_argument('--checkpoint-stride', type=int, default=1, help='Save every Nth checkpoint (e.g. 3 means save every third update)')
    parser.add_argument('--keep-previous-model', action='store_true', help='Keep the previous model folder instead of deleting it')
    parser.add_argument('--tensorboard-log', type=str, default='tensorboard_logs', help='TensorBoard log directory')
    parser.add_argument('--start-timesteps', type=int, default=None, help='Starting timestep count (for resuming training)')
    parser.add_argument('--enable-viewer', action='store_true', help='Enable the DM Control viewer (requires a GUI environment)')
    args = parser.parse_args()

    # Create environment
    env = create_env(training_phase=args.training_phase, view_only=args.view_only)
    vec_env = setup_env(env, phase=args.training_phase)

    if args.view_only and args.load_model:
        # Load model and launch viewer
        print(f"\nLoading model from {args.load_model} for viewing...")
        model = PPO.load(args.load_model, env=vec_env)
        print("Launching viewer...")
        # Always launch viewer in view-only mode
        viewer.launch(env, policy=create_policy(model))
    else:
        # Convert n_updates to timesteps using n_steps from hyperparameters
        timesteps = args.n_updates * default_hyperparameters["n_steps"]
        print(f"Starting training for {args.n_updates} updates ({timesteps} timesteps)...")
        if args.load_model:
            print(f"Resuming training from {args.load_model}")
        
        model = train_creature(
            env=vec_env,  # Use the wrapped environment
            total_timesteps=timesteps,
            checkpoint_freq=args.checkpoint_freq,
            load_path=args.load_model or args.load_path,  # Use load_model if provided, otherwise use load_path
            tensorboard_log=args.tensorboard_log,
            keep_checkpoints=args.keep_checkpoints,
            checkpoint_stride=args.checkpoint_stride,
            start_timesteps=args.start_timesteps,
            keep_previous_model=args.keep_previous_model
        )

        # Launch viewer after training if enabled
        if args.enable_viewer:
            print("\nLaunching viewer with trained model...")
            viewer.launch(env, policy=create_policy(model))
        else:
            print("\nViewer disabled. Run with --enable-viewer to launch the viewer after training.")
