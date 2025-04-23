import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="glfw")
import os
from custom_soccer_env import create_soccer_env
from dm_control.locomotion.soccer.team import RGBA_BLUE, RGBA_RED
from creature import Creature
from train import train_creature, DMControlWrapper, process_observation, calculate_reward, setup_env, default_hyperparameters, TensorboardCallback
from dm_control import viewer
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
import argparse

def create_env():
    """Create the environment for training or viewing."""
    home_player = Creature("creature_configs/two_arm_rower_blueprint.xml", marker_rgba=RGBA_BLUE)
    return create_soccer_env(
        home_players=[home_player],
        away_players=[],
        time_limit=25.6,
        disable_walker_contacts=True,
        enable_field_box=False,
        keep_aspect_ratio=False,
        terminate_on_goal=False
    )

def create_policy(model, training_phase="rotation"):
    """Create a policy function for the viewer."""
    
    # Track state for phase and history
    phase = 0.0
    position_history = []
    velocity_history = []
    last_position = None
    start_position = None  # Add start position for displacement calculation
    history_buffer_size = 5
    velocity_buffer_size = 5
    dt = 0.025  # Default physics timestep, adjust if needed
    
    # Frame counter for limiting debug output frequency
    frame_counter = 0
    
    def policy(time_step):
        nonlocal phase, position_history, velocity_history, last_position, start_position, frame_counter
        
        # Process base observation
        orig_obs = process_observation(time_step)
        
        # Only add the additional observation components for walking phase
        if training_phase == "walking":
            # Update phase variable
            phase = (phase + dt) % 1.0
            phase_sin = np.sin(2 * np.pi * phase)
            phase_cos = np.cos(2 * np.pi * phase)
            
            # Process position history
            additional_obs = []
            
            # Add phase variables
            additional_obs.extend([phase_sin, phase_cos])
            
            # Track position for history
            if 'absolute_root_pos' in time_step.observation[0]:
                pos = time_step.observation[0]['absolute_root_pos']
                
                # Initialize start position if needed
                if start_position is None:
                    start_position = pos.copy()
                
                # Update position history
                position_history.append(pos.copy())
                if len(position_history) > history_buffer_size:
                    position_history.pop(0)
                
                # Calculate forward velocity from position change
                if last_position is not None:
                    forward_velocity = (pos[0][0] - last_position[0][0]) / dt
                    
                    # Update velocity history
                    velocity_history.append(forward_velocity)
                    if len(velocity_history) > velocity_buffer_size:
                        velocity_history.pop(0)
                else:
                    forward_velocity = 0.0
                last_position = pos.copy()
            else:
                forward_velocity = 0.0
                
            # Add position history (differences from current)
            if len(position_history) > 1:
                current_pos = position_history[-1]
                history_entries = min(len(position_history) - 1, history_buffer_size - 1)
                
                for i in range(history_entries):
                    past_pos = position_history[-(i+2)]
                    rel_pos = current_pos - past_pos
                    additional_obs.extend(rel_pos.flatten())
                    
                # Fill remaining history slots if needed
                remaining_slots = (history_buffer_size - 1) - history_entries
                for _ in range(remaining_slots):
                    additional_obs.extend([0.0, 0.0, 0.0])
            else:
                # No history yet, fill with zeros
                for _ in range(history_buffer_size - 1):
                    additional_obs.extend([0.0, 0.0, 0.0])
                    
            # Add velocity history
            vel_entries = min(len(velocity_history), velocity_buffer_size)
            if vel_entries > 0:
                additional_obs.extend(velocity_history[-vel_entries:])
                
            # Fill remaining velocity slots if needed
            remaining_vel_slots = velocity_buffer_size - vel_entries
            if remaining_vel_slots > 0:
                additional_obs.extend([0.0] * remaining_vel_slots)
                
            # Create the complete observation for the model
            full_obs = np.concatenate([orig_obs, np.array(additional_obs, dtype=np.float32)])
            
            # Verify shape matches expected dimensions
            expected_size = 211  # 192 base + 19 additional features
            if full_obs.shape[0] != expected_size:
                print(f"WARNING: Observation size mismatch: {full_obs.shape[0]} vs expected {expected_size}")
        else:
            # For rotation phase or combined phase, just use the original observation
            full_obs = orig_obs
                
        # Get action from model
        action, _states = model.predict(full_obs, deterministic=True)

        # Display rotation alignment information when in rotation phase
        if training_phase == "rotation" and 'absolute_root_mat' in time_step.observation[0]:
            frame_counter += 1
            # Only display every 10 frames to avoid console spam
            if frame_counter % 1 == 40:
                # Extract the z-axis from the rotation matrix
                rot_matrix = time_step.observation[0]['absolute_root_mat']
                # Get the x-component (alignment with x-axis)
                alignment = float(rot_matrix[0, 2])
                
                # Use raw alignment value (-1 to 1) 
                reward = alignment
                
                print(f"Rotation reward (alignment with x): {alignment:.3f}")
                print(f"Action: {action}")

        return [action]
    
    return policy

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train or view a creature model')
    parser.add_argument('--load-model', type=str, help='Path to model to load')
    parser.add_argument('--view-only', action='store_true', help='Only view the model, no training')
    parser.add_argument('--training-phase', type=str, default="combined", choices=["walking", "rotation", "combined"], help='Training phase to use')
    parser.add_argument('--n-updates', type=int, default=3, help=f'Number of policy updates to perform (each update = {default_hyperparameters["n_steps"]} timesteps)')
    parser.add_argument('--load-path', type=str, default=None, help='Path to load a saved model from')
    parser.add_argument('--checkpoint-freq', type=int, default=1000, help='Save a checkpoint every N updates (e.g., 40000 saves at updates 40000, 80000, etc.)')
    parser.add_argument('--keep-checkpoints', action='store_true', help='Keep all checkpoints instead of deleting them')
    parser.add_argument('--checkpoint-stride', type=int, default=1, help='DEPRECATED - Use checkpoint-freq instead')
    parser.add_argument('--keep-previous-model', action='store_true', help='Keep the previous model folder instead of deleting it')
    parser.add_argument('--tensorboard-log', type=str, default='tensorboard_logs', help='TensorBoard log directory')
    parser.add_argument('--start-timesteps', type=int, default=None, help='Starting timestep count (for resuming training)')
    parser.add_argument('--enable-viewer', action='store_true', help='Enable the DM Control viewer (requires a GUI environment)')
    parser.add_argument('--n-envs', type=int, default=1, help='Number of parallel environments for training')
    args = parser.parse_args()

    # Create raw environment
    env = create_env()
    
    # For viewer mode, we use a single environment with DummyVecEnv
    if args.view_only and args.load_model:
        # Setup a single environment and wrap it properly
        wrapped_env = setup_env(env, phase=args.training_phase)
        vec_env = DummyVecEnv([lambda: wrapped_env])
    else:
        # For training, use SubprocVecEnv with multiple environments
        print(f"\nUsing {args.n_envs} parallel environments for training")
        
        # Create environments with different seeds for diversity
        def make_env(seed):
            def _init():
                # Create a fresh environment for each parallel instance
                train_env = create_env()
                # Setup the environment without vectorizing it
                wrapped_env = setup_env(train_env, phase=args.training_phase)
                # Set the seed for randomization
                wrapped_env.seed(seed)
                # Return the wrapped environment directly
                return wrapped_env
            return _init
            
        # Create a vectorized environment with n_envs parallel environments
        vec_env = SubprocVecEnv([make_env(i) for i in range(args.n_envs)], start_method='forkserver')

    if args.view_only and args.load_model:
        # Load model and launch viewer
        print(f"\nLoading model from {args.load_model} for viewing...")
        model = PPO.load(args.load_model, env=vec_env)
        print("Launching viewer...")
        # Always launch viewer in view-only mode with the original unwrapped environment
        viewer.launch(env, policy=create_policy(model, args.training_phase))
    else:
        # Convert n_updates to timesteps using n_steps from hyperparameters
        # With parallel environments, we need to adjust the total_timesteps
        # since SB3 will collect n_envs times more samples per step
        timesteps_per_update = default_hyperparameters["n_steps"] 
        
        timesteps = args.n_updates * timesteps_per_update
        
        print(f"Starting training for {args.n_updates} updates...")
        print(f"Using {args.n_envs} parallel environments for sample collection")
        
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
            keep_previous_model=args.keep_previous_model,
            training_phase=args.training_phase,  # Pass the training phase
            n_envs=args.n_envs,  # Pass the number of environments
            target_updates=args.n_updates  # Ensure we stop at exactly n_updates
        )

        # Launch viewer after training if enabled
        if args.enable_viewer:
            print("\nLaunching viewer with trained model...")
            # For viewing, we need a single environment
            view_env = create_env()
            viewer.launch(view_env, policy=create_policy(model, args.training_phase))
        else:
            print("\nViewer disabled. Run with --enable-viewer to launch the viewer after training.")