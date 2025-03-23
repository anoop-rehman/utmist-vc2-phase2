import numpy as np
import gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import explained_variance
from gym import spaces
import os
from datetime import datetime
from model_card_generator import generate_model_card
import pytz
import shutil
from typing import Dict, Any

# Add tensorboard callback class
class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        
    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        for i, wrapper in enumerate(self.model.get_env().envs):
            if hasattr(wrapper, 'reward'):
                reward_value = wrapper.reward
                self.logger.record(f'env_{i}/reward', reward_value)
            if hasattr(wrapper, 'last_vel_to_ball'):
                vel_value = wrapper.last_vel_to_ball
                self.logger.record(f'env_{i}/velocity_metric', vel_value)
        return True

# Add quaternion utility function
def quaternion_to_forward_vector(quaternion):
    """Convert a quaternion to a forward vector (x-axis in local coordinates)."""
    # Extract quaternion components
    w, x, y, z = quaternion.flatten()
    
    # Forward vector is the x-axis (1,0,0) rotated by the quaternion
    forward_x = 1 - 2 * (y*y + z*z)
    forward_y = 2 * (x*y + w*z)
    forward_z = 2 * (x*z - w*y)
    
    # Return normalized vector
    forward = np.array([forward_x, forward_y, forward_z])
    return forward / np.linalg.norm(forward)

# Default hyperparameters for the PPO model.
default_hyperparameters = dict(
    learning_rate=3e-4,
    n_steps=8192,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
)

# Add train_creature function
def train_creature(env, total_timesteps, checkpoint_freq=8192, load_path=None, tensorboard_log="tensorboard_logs", 
                  keep_checkpoints=False, checkpoint_stride=1, start_timesteps=None, keep_previous_model=False):
    """Train a creature with PPO and save checkpoints."""
    # Create folders for checkpoints/logs
    save_folder = f"trained_creatures/{get_default_folder()}"
    checkpoint_folder = f"{save_folder}/checkpoints"
    final_model_path = f"{save_folder}/final_model"
    
    # Clear previous model from same run if needed
    if os.path.exists(save_folder) and not keep_previous_model:
        try:
            shutil.rmtree(save_folder)
            print(f"Cleared previous model folder: {save_folder}")
        except Exception as e:
            print(f"Error clearing model folder: {e}")
    
    # Create folders
    os.makedirs(save_folder, exist_ok=True)
    os.makedirs(checkpoint_folder, exist_ok=True)
    print(f"Saving model to {save_folder}")
    
    # Configure callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=checkpoint_folder,
        name_prefix="model",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1
    )
    
    # Create model
    model = create_ppo_model(env, tensorboard_log, load_path)
    
    # Setup callback
    callbacks = [checkpoint_callback, TensorboardCallback()]
    
    # Train model
    model.learn(
        total_timesteps=total_timesteps, 
        callback=callbacks,
        reset_num_timesteps=start_timesteps is None,  # Only reset if not continuing training
        tb_log_name=save_folder.replace("/", "_")
    )
    
    # Save final model
    final_model_name = f"{final_model_path}_{model.num_timesteps//default_hyperparameters['n_steps']}updates"
    model.save(final_model_name)
    print(f"\nFinal model saved to {final_model_name}.zip")
    
    # Clean up checkpoints if not keeping them
    if not keep_checkpoints and os.path.exists(checkpoint_folder):
        checkpoint_files = [f for f in os.listdir(checkpoint_folder) if f.endswith('.zip')]
        checkpoint_files.sort()
        
        # Keep only every Nth checkpoint where N is checkpoint_stride
        files_to_delete = []
        for i, file in enumerate(checkpoint_files):
            if (i + 1) % checkpoint_stride != 0:  # Keep 1st, (1+stride)th, etc.
                files_to_delete.append(file)
        
        # Delete unwanted checkpoints
        for file in files_to_delete:
            try:
                os.remove(os.path.join(checkpoint_folder, file))
            except Exception as e:
                print(f"Error removing checkpoint {file}: {e}")
        
        print(f"Cleaned up {len(files_to_delete)} checkpoints, keeping {len(checkpoint_files) - len(files_to_delete)}")
    
    # Generate a model card
    model_info = {
        "name": save_folder,
        "timesteps": model.num_timesteps,
        "hyperparameters": default_hyperparameters,
        "updates": model.num_timesteps // default_hyperparameters["n_steps"]
    }
    generate_model_card(model_info, save_folder)
    
    return model

def setup_env(env, phase="combined"):
    """Wrap environment based on training phase."""
    if phase == "walking":
        wrapped_env = WalkingPhaseWrapper(env)
    elif phase == "rotation":
        wrapped_env = RotationPhaseWrapper(env)
    else:
        wrapped_env = DMControlWrapper(env)  # Original combined phase
    return DummyVecEnv([lambda: wrapped_env])

def create_ppo_model(vec_env, tensorboard_log, load_path=None):
    """Create or load a PPO model with standard parameters."""
    if load_path:
        print(f"Loading pre-trained model from {load_path}")
        return PPO.load(load_path, env=vec_env, tensorboard_log=tensorboard_log, **default_hyperparameters)
    
    return PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        **default_hyperparameters
    )

def get_default_folder():
    """Generate a default folder name using datetime in EST timezone."""
    eastern_tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz=eastern_tz)
    return now.strftime("%Y%m%d__%p_%I_%M_%S").lower()

def process_observation(timestep):
    """Convert DM Control observation to the format expected by the model."""
    obs_dict = timestep.observation[0]
    return np.concatenate([v.flatten() for v in obs_dict.values()])

def calculate_reward(timestep, action, distance_in_window):
    """Calculate reward based on velocity to ball, control cost, and distance traveled."""
    vel_to_ball = timestep.observation[0]['stats_vel_to_ball'][0]
    
    # Scale down control cost to be less punishing
    ctrl_cost = 0.1 * np.sum(np.square(action))
    
    # Calculate stillness penalty based on distance traveled in window
    # Scale distance to be between 0 and 1 using a sigmoid-like function
    distance_factor = 2.0 / (1.0 + np.exp(-2.0 * distance_in_window)) - 1.0  # Maps any positive distance to (0,1)
    stillness_penalty = 0.25 * (1.0 - distance_factor)  # Max penalty of 0.25 when no distance covered
    
    # Combine rewards: velocity to ball + baseline - control cost - stillness penalty
    reward = vel_to_ball + 1.0 - ctrl_cost - stillness_penalty
    
    return reward, vel_to_ball

class DMControlWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.reward = 0
        self.last_vel_to_ball = 0
        self.episode_count = 0
        
        # Add position tracking
        self.position_history = []
        self.window_size = 20  # Track last 20 steps
        
        # Get action and observation specs
        action_spec = env.action_spec()[0]
        obs_spec = env.observation_spec()[0]
        
        # Define action space
        self.action_space = spaces.Box(
            low=np.float32(action_spec.minimum),
            high=np.float32(action_spec.maximum),
            dtype=np.float32
        )
        
        # Calculate observation size and create a sample observation to verify shape
        timestep = env.reset()
        self.obs_concat = process_observation(timestep)
        obs_size = self.obs_concat.shape[0]
        
        # Define observation space with the correct size
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )

    def get_distance_traveled(self):
        """Calculate total distance traveled in the moving window."""
        if len(self.position_history) < 2:
            return 0.0
        
        # Calculate total absolute distance in window
        total_distance = 0.0
        for i in range(1, len(self.position_history)):
            # Calculate Euclidean distance between consecutive positions
            pos_diff = self.position_history[i] - self.position_history[i-1]
            distance = np.sqrt(np.sum(np.square(pos_diff)))
            total_distance += distance
        
        return total_distance

    def step(self, action):
        timestep = self.env.step([action])
        
        # Track position for distance calculation
        if 'absolute_root_pos' in timestep.observation[0]:
            pos = timestep.observation[0]['absolute_root_pos']
            self.position_history.append(pos)
            if len(self.position_history) > self.window_size:
                self.position_history.pop(0)
        
        obs = process_observation(timestep)
        distance = self.get_distance_traveled()
        reward, vel_to_ball = calculate_reward(timestep, action, distance)
        done = timestep.last()
        info = {}

        self.reward = reward
        self.last_vel_to_ball = vel_to_ball
        return obs, reward, done, info

    def reset(self):
        timestep = self.env.reset()
        obs = process_observation(timestep)
        
        # Clear position history
        self.position_history = []
        
        # Initialize last_vel_to_ball
        _, self.last_vel_to_ball = calculate_reward(timestep, np.zeros(self.action_space.shape), 0.0)
        
        # Print episode start info
        self.episode_count += 1
        print(f"\nEpisode {self.episode_count} started:")
        if 'absolute_root_pos' in timestep.observation[0]:
            print(f"  Creature position: {timestep.observation[0]['absolute_root_pos']}")
        if 'ball_ego_pos' in timestep.observation[0]:
            print(f"  Ball position (ego): {timestep.observation[0]['ball_ego_pos']}")
        
        return obs

    def render(self, mode='human'):
        pass

class WalkingPhaseWrapper(DMControlWrapper):
    """Environment wrapper for walking phase of training."""
    def __init__(self, env):
        super().__init__(env)
        self.phase = 0.0
        self.phase_increment = 0.005  # Adjust as needed
        self.dt = 0.025  # Default physics timestep
        
        # Position tracking variables
        self.position_history = []
        self.velocity_history = []
        self.last_position = None
        self.start_position = None
        
        # History buffer sizes
        self.history_buffer_size = 5
        self.velocity_buffer_size = 5
        
        # Calculate the additional observation space size (beyond the processed original observation)
        # 2 phase variables + (history_buffer_size-1) positions × 3 values + velocity_buffer_size + 1 alignment
        self.additional_obs_size = 2 + (self.history_buffer_size - 1) * 3 + self.velocity_buffer_size + 1
        
        # Update observation space with the additional dimensions
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_concat.shape[0] + self.additional_obs_size,),
            dtype=np.float32
        )
        
        # Enable verbose mode for debugging (disabled by default)
        self.verbose = False
    
    def reset(self):
        timestep = self.env.reset()
        
        # Process the original observation
        orig_obs = process_observation(timestep)
        
        # Reset phase variable
        self.phase = 0.0
        phase_sin = np.sin(2 * np.pi * self.phase)
        phase_cos = np.cos(2 * np.pi * self.phase)
        
        # Clear history
        self.position_history = []
        self.velocity_history = []
        self.last_position = None
        self.start_position = None  # Reset start position
        
        # Create initial observation with phase variables
        additional_obs = []
        
        # Add phase variables
        additional_obs.extend([phase_sin, phase_cos])
        
        # Add empty position history placeholders - exactly (history_buffer_size-1) positions × 3 values
        for _ in range(self.history_buffer_size - 1):
            additional_obs.extend([0.0, 0.0, 0.0])  # x, y, z placeholders
        
        # Add empty velocity history
        additional_obs.extend([0.0] * self.velocity_buffer_size)
        
        # Initialize position history and position tracking
        if 'absolute_root_pos' in timestep.observation[0]:
            pos = timestep.observation[0]['absolute_root_pos']
            self.last_position = pos.copy()
            self.start_position = pos.copy()  # Set start position for displacement calculations
            self.position_history.append(pos.copy())
        
        # Combine original observation with additional features
        obs = np.concatenate([orig_obs, np.array(additional_obs, dtype=np.float32)])
        
        # Verify shape
        if obs.shape[0] != self.observation_space.shape[0]:
            print(f"WARNING: Observation shape mismatch: {obs.shape} vs expected {self.observation_space.shape}")
            print(f"Original observation shape: {orig_obs.shape}")
            print(f"Additional observation shape: {len(additional_obs)}")
        
        # Initialize last_vel_to_ball as 0 for the first step
        self.last_vel_to_ball = 0.0
        
        # Print episode start info
        self.episode_count += 1
        print(f"\nEpisode {self.episode_count} started:")
        if 'absolute_root_pos' in timestep.observation[0]:
            print(f"  Creature position: {timestep.observation[0]['absolute_root_pos']}")
        
        return obs
        
    def step(self, action):
        timestep = self.env.step([action])
        
        # Update phase variable
        self.phase = (self.phase + self.dt) % 1.0
        phase_sin = np.sin(2 * np.pi * self.phase)
        phase_cos = np.cos(2 * np.pi * self.phase)
        
        # Local variables to track forward movement
        forward_displacement = 0.0
        
        # Track position for distance calculation
        if 'absolute_root_pos' in timestep.observation[0]:
            pos = timestep.observation[0]['absolute_root_pos']
            
            # Get orientation quaternion
            if 'absolute_root_rot' in timestep.observation[0]:
                rot = timestep.observation[0]['absolute_root_rot']
                
                # Convert to forward direction vector
                forward_vec = quaternion_to_forward_vector(rot)
                
                # If we have a starting position, calculate displacement along forward direction
                if self.start_position is not None:
                    # Calculate displacement vector (global coordinates)
                    displacement_vec = pos - self.start_position
                    
                    # Project displacement onto creature's forward direction
                    forward_displacement = np.dot(displacement_vec.flatten(), forward_vec)
                else:
                    self.start_position = pos.copy()
            
            # Update position history
            self.position_history.append(pos.copy())
            if len(self.position_history) > self.history_buffer_size:
                self.position_history.pop(0)
            
            # Calculate forward velocity from position change (for logging purposes)
            if self.last_position is not None:
                # Calculate velocity as change in position over time using actual timestep
                forward_velocity = (pos[0][0] - self.last_position[0][0]) / self.dt  # x-axis velocity
                
                # Update velocity history
                self.velocity_history.append(forward_velocity)
                if len(self.velocity_history) > self.velocity_buffer_size:
                    self.velocity_history.pop(0)