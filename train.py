import numpy as np
import gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import explained_variance
from gym import spaces
import os
from datetime import datetime
from model_card_generator import generate_model_card
import pytz

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