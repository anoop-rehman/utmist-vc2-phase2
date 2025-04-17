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
    n_steps=960,
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
    
    # Add a counter to track number of calls (static variable)
    if not hasattr(process_observation, "counter"):
        process_observation.counter = 0
    
    # Print observations every 40 steps
    should_print = process_observation.counter % 40 == 0
    process_observation.should_print = should_print  # Set a flag for other functions
    
    if should_print:
        print(f"\n--- Observation {process_observation.counter} ---")
        print("Observation keys:", list(obs_dict.keys()))
        
        # Print full values for each key
        for k, v in obs_dict.items():
            if hasattr(v, 'shape'):
                print(f"{k}: shape={v.shape}, value={v.flatten()}")
            else:
                print(f"{k}: {v}")
    
    process_observation.counter += 1
    
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
    """Environment wrapper for the walking phase of training."""
    def __init__(self, env):
        # Initialize with parent but we'll override the observation space
        super().__init__(env)
        self.last_position = None
        self.start_position = None  # Track starting position for displacement calculations
        # Get the physics timestep from the environment
        self.dt = env.physics.timestep()
        
        # Initialize phase variable for cyclic movement tracking
        self.phase = 0.0
        
        # History buffer to store previous positions
        self.position_history = []
        self.history_buffer_size = 5  # Store last 5 positions
        
        # To calculate velocity averages
        self.velocity_history = []
        self.velocity_buffer_size = 5
        
        # Calculate additional observation size
        # 2 phase variables (sin, cos)
        self.phase_size = 2
        # Position history: (history_buffer_size-1) positions × 3 dimensions per position
        self.pos_history_size = (self.history_buffer_size - 1) * 3  
        # Velocity history: velocity_buffer_size values
        self.vel_history_size = self.velocity_buffer_size
        
        # Total additional size
        self.additional_obs_size = self.phase_size + self.pos_history_size + self.vel_history_size
        
        # Redefine observation space to include our additional features
        # First get a sample observation from parent
        timestep = env.reset()
        self.base_obs = process_observation(timestep)
        self.base_obs_size = self.base_obs.shape[0]
        
        # Define new observation space with extended size
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.base_obs_size + self.additional_obs_size,),
            dtype=np.float32
        )
        
        print(f"Observation space shape: {self.observation_space.shape}")
        print(f"Base observation size: {self.base_obs_size}")
        print(f"Additional features size: {self.additional_obs_size}")
        print(f"Total expected size: {self.base_obs_size + self.additional_obs_size}")
    
    def reset(self):
        timestep = self.env.reset()
        
        # Reset phase
        self.phase = 0.0
        phase_sin = np.sin(2 * np.pi * self.phase)
        phase_cos = np.cos(2 * np.pi * self.phase)
        
        # Get initial observation
        orig_obs = process_observation(timestep)
        
        # Clear histories
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
            else:
                forward_velocity = 0.0
            self.last_position = pos.copy()

            # If orientation not available, fallback to global displacement
            if 'absolute_root_rot' not in timestep.observation[0]:
                # Calculate displacement from origin
                if self.start_position is not None:
                    displacement_vec = pos - self.start_position
                    forward_displacement = np.linalg.norm(displacement_vec)
                else:
                    self.start_position = pos.copy()
        else:
            forward_velocity = 0.0
            forward_displacement = 0.0
        
        # Process the original observation
        orig_obs = process_observation(timestep)
        
        # Create additional observation components
        additional_obs = []
        
        # Add phase variables
        additional_obs.extend([phase_sin, phase_cos])
        
        # Add position history (differences from current) - (history_buffer_size-1) × 3 values
        if len(self.position_history) > 1:
            current_pos = self.position_history[-1]
            history_entries = min(len(self.position_history) - 1, self.history_buffer_size - 1)
            
            for i in range(history_entries):
                past_pos = self.position_history[-(i+2)]  # Go backwards from second-to-last
                # Calculate relative position (flattened)
                rel_pos = current_pos - past_pos
                additional_obs.extend(rel_pos.flatten())
                
            # Fill remaining history slots if needed
            remaining_slots = (self.history_buffer_size - 1) - history_entries
            for _ in range(remaining_slots):
                additional_obs.extend([0.0, 0.0, 0.0])  # x, y, z placeholders
        else:
            # No history yet, fill with zeros
            for _ in range(self.history_buffer_size - 1):
                additional_obs.extend([0.0, 0.0, 0.0])
        
        # Add velocity history
        vel_entries = min(len(self.velocity_history), self.velocity_buffer_size)
        if vel_entries > 0:
            additional_obs.extend(self.velocity_history[-vel_entries:])
            
        # Fill remaining velocity slots if needed
        remaining_vel_slots = self.velocity_buffer_size - vel_entries
        if remaining_vel_slots > 0:
            additional_obs.extend([0.0] * remaining_vel_slots)
        
        # Verify and ensure consistent observation size
        if len(additional_obs) != self.additional_obs_size:
            print(f"WARNING: Additional obs size mismatch: {len(additional_obs)} vs expected {self.additional_obs_size}")
            # Ensure correct size by truncating or padding
            if len(additional_obs) > self.additional_obs_size:
                additional_obs = additional_obs[:self.additional_obs_size]
            else:
                additional_obs.extend([0.0] * (self.additional_obs_size - len(additional_obs)))
        
        # Combine original observation with additional features
        obs = np.concatenate([orig_obs, np.array(additional_obs, dtype=np.float32)])
        
        # Scale up forward displacement reward with minimal control penalty
        ctrl_cost = 0.05 * np.sum(np.square(action))  # Reduced control cost
        
        # Increase reward magnitude for forward movement
        reward = forward_displacement * 2.0 + 0.5 - ctrl_cost
        
        done = timestep.last()
        info = {}

        self.reward = reward
        self.last_vel_to_ball = forward_velocity  # Keep tracking velocity for logging
        return obs, reward, done, info

class RotationPhaseWrapper(DMControlWrapper):
    """Environment wrapper for the rotation phase of training."""
    def __init__(self, env):
        super().__init__(env)
        # Track orientation matrix history instead of quaternions
        self.matrix_history = []
        self.initial_orientation = None
        self.verbose = True
        
    def step(self, action):
        timestep = self.env.step([action])
        
        # Track position for distance calculation
        current_pos = None
        if 'absolute_root_pos' in timestep.observation[0]:
            pos = timestep.observation[0]['absolute_root_pos']
            current_pos = pos.copy()
            self.position_history.append(pos)
            if len(self.position_history) > self.window_size:
                self.position_history.pop(0)
        
        obs = process_observation(timestep)
        
        # Store the observation dict for debugging
        self.last_obs_dict = timestep.observation[0]
        
        # Initialize variables
        angular_movement = 0.0
        angular_stillness_penalty = 0.0
        
        # Track orientation history
        if 'absolute_root_mat' in timestep.observation[0]:
            current_mat = timestep.observation[0]['absolute_root_mat'].copy()
            self.matrix_history.append(current_mat)
            # Keep history manageable
            if len(self.matrix_history) > 20:  # 20-step window
                self.matrix_history.pop(0)
            
            # Calculate angular movement over window
            if len(self.matrix_history) >= 2:
                # Get past matrix
                past_mat = self.matrix_history[max(0, len(self.matrix_history)-10)]  # 10 steps back
                
                # Calculate angular difference between matrices
                # This uses the trace of R1^T * R2 to find rotation angle
                current_mat_reshaped = current_mat.reshape(3, 3)
                past_mat_reshaped = past_mat.reshape(3, 3)
                
                # Calculate R1^T * R2
                rotation_diff = np.dot(current_mat_reshaped.T, past_mat_reshaped)
                
                # Trace of the matrix gives us 1 + 2*cos(theta)
                trace = np.trace(rotation_diff)
                trace = min(3.0, max(-1.0, trace))  # Clamp to valid range
                
                # Convert to angle in radians
                angle_diff = np.arccos((trace - 1.0) / 2.0)
                
                # Convert to degrees for more intuitive values
                angular_movement = np.degrees(angle_diff)
                
                # Calculate alignment reward
                alignment_reward = self.calculate_alignment_reward(current_mat)
                
                # Apply stillness penalty based on angular movement AND current alignment
                angle_threshold = 5.0  # Degrees of expected rotation in window
                if angular_movement < angle_threshold:
                    stillness_factor = (angle_threshold - angular_movement) / angle_threshold
                    
                    # Scale penalty based on alignment quality
                    alignment_quality = max(0.0, alignment_reward)  # Only consider positive alignment
                    penalty_scaling = 1.0 - (alignment_quality * alignment_quality)  # Squared for sharper dropoff
                    
                    # Apply scaled penalty
                    angular_stillness_penalty = 2.0 * stillness_factor * penalty_scaling
            else:
                # Not enough history yet, just calculate alignment without stillness penalty
                alignment_reward = self.calculate_alignment_reward(current_mat)
        else:
            print("absolute_root_mat not found in timestep.observation[0]!")
            alignment_reward = 0.0
        
        # Combine alignment reward with angular stillness penalty
        reward = alignment_reward - angular_stillness_penalty
        
        done = timestep.last()
        info = {}

        self.reward = reward
        self.last_vel_to_ball = alignment_reward  # For consistency with previous code
        return obs, reward, done, info
    
    def reset(self):
        timestep = self.env.reset()
        obs = process_observation(timestep)
        
        # Clear position history
        self.position_history = []
        
        # Randomize the creature's orientation by applying a random rotation
        # This is done via the internal physics engine
        if hasattr(self.env, 'physics'):
            try:
                # Try to get the root body of the creature
                # In DM Control soccer environment, the creature is the first walker
                if hasattr(self.env, '_task') and hasattr(self.env._task, 'players'):
                    # Access the first player's root body
                    player = self.env._task.players[0]
                    root_body = self.env.physics.bind(player.walker.root_body)
                    
                    # Generate a random quaternion for orientation
                    # Method: Generate random rotation axis and angle
                    axis = np.random.randn(3)
                    axis = axis / np.linalg.norm(axis)  # Normalize to unit vector
                    angle = np.random.uniform(0, 2 * np.pi)  # Random angle between 0 and 2π
                    
                    # Convert to quaternion (w,x,y,z format)
                    quat = np.zeros(4)
                    quat[0] = np.cos(angle/2)  # w component
                    quat[1:] = axis * np.sin(angle/2)  # x,y,z components
                    
                    # Apply the rotation to the root body
                    root_body.xquat = quat
                    
                    # Run a single physics step to apply the changes
                    self.env.physics.step()
                    
                    if self.verbose:
                        print(f"Applied random rotation: axis={axis}, angle={angle:.2f} rad")
                else:
                    print("Warning: Could not access creature's body for orientation randomization")
            except Exception as e:
                print(f"Error randomizing orientation: {e}")
        
        # Store initial orientation
        if 'absolute_root_mat' in timestep.observation[0]:
            self.initial_orientation = timestep.observation[0]['absolute_root_mat'].copy()
            # Calculate and log initial alignment
            initial_alignment = self.calculate_alignment_reward(self.initial_orientation)
            self.last_vel_to_ball = initial_alignment
        else:
            self.initial_orientation = None
            self.last_vel_to_ball = 0.0
        
        # Print episode start info
        self.episode_count += 1
        print(f"\nEpisode {self.episode_count} started:")
        if 'absolute_root_pos' in timestep.observation[0]:
            print(f"  Creature position: {timestep.observation[0]['absolute_root_pos']}")
        if 'absolute_root_mat' in timestep.observation[0]:
            print(f"  Initial alignment reward: {self.last_vel_to_ball:.4f}")
        
        # We need to re-process the observation after applying the random rotation
        if hasattr(self.env, 'physics') and hasattr(self.env, '_task') and hasattr(self.env._task, 'players'):
            try:
                # Get fresh observation after randomization
                timestep = self.env.physics.get_state()
                obs = process_observation(timestep)
            except Exception as e:
                print(f"Error getting fresh observation: {e}")
        
        return obs

    def calculate_alignment_reward(self, rotation_matrix):
        """Calculate reward based on aligning local z-axis with global x-axis using rotation matrix."""
        # Reshape from flat array to 3x3 matrix 
        mat = rotation_matrix.reshape(3, 3)

        # The third column of the matrix is the local z-axis in global coordinates
        local_z = mat[:, 2]  # Extract third column (index 2)
        
        # Alignment is simply the x-component of the local z-axis
        alignment = local_z[0]
        
        # Print alignment in verbose mode for debugging - only every 40 steps
        if self.verbose and hasattr(process_observation, "should_print") and process_observation.should_print:
            print(f"Root z-axis (from matrix): [{local_z[0]:.3f}, {local_z[1]:.3f}, {local_z[2]:.3f}], " +
                  f"Alignment with x: {alignment:.3f}, Reward: {alignment:.3f}")
        
        return alignment

class TrainingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        
    def _on_step(self):
        if self.locals.get('done'):
            self.episode_rewards.append(self.locals.get('rewards')[0])
            if len(self.episode_rewards) % 10 == 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                print(f"Episode {len(self.episode_rewards)}, Mean Reward: {mean_reward:.2f}")
        return True

class TensorboardCallback(BaseCallback):
    def __init__(self, start_timesteps=0, verbose=0):
        super().__init__(verbose)
        self.start_timesteps = start_timesteps
        
        # For tracking complete episodes
        self.episode_rewards = []
        self.episode_velocities = []
        
        # For tracking current episode
        self.current_episode_rewards = []
        self.current_episode_velocities = []
        self.episode_count = 0
        self.last_obs = None
        
    def _on_training_start(self):
        # Initialize episode tracking
        self.current_episode_rewards = []
        self.current_episode_velocities = []
        
    def _on_step(self):
        # Get current reward and velocity from the wrapped environment
        env = self.training_env.envs[0]  # Get the actual environment from DummyVecEnv
        reward = env.reward
        vel_to_ball = env.last_vel_to_ball
        
        # Track current episode stats
        self.current_episode_rewards.append(reward)
        self.current_episode_velocities.append(vel_to_ball)
        
        # Get current environment steps
        env_steps = self.num_timesteps
        
        # Log step-level metrics
        self.logger.record('train/reward', reward)
        self.logger.record('train/velocity_to_ball', vel_to_ball)
        
        # Log training metrics from the model
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            for key, value in self.model.logger.name_to_value.items():
                self.logger.record(key, value)
        
        # Handle episode completion
        if self.locals.get('done'):
            self.episode_count += 1
            
            # Calculate statistics for this episode
            episode_reward_sum = sum(self.current_episode_rewards)
            episode_reward_mean = np.mean(self.current_episode_rewards)
            episode_reward_min = np.min(self.current_episode_rewards)
            episode_reward_max = np.max(self.current_episode_rewards)
            episode_velocity_mean = np.mean(self.current_episode_velocities)
            
            # Store episode summary metrics
            self.episode_rewards.append(episode_reward_sum)
            self.episode_velocities.append(episode_velocity_mean)
            
            # Log this episode's statistics
            self.logger.record('train/episode_reward_sum', episode_reward_sum)
            self.logger.record('train/episode_reward_mean', episode_reward_mean)
            self.logger.record('train/episode_reward_min', episode_reward_min)
            self.logger.record('train/episode_reward_max', episode_reward_max)
            self.logger.record('train/episode_velocity_mean', episode_velocity_mean)
            self.logger.record('train/episode_length', len(self.current_episode_rewards))
            
            if self.episode_count % 10 == 0:
                print(f"\nEpisode {self.episode_count} stats:")
                print(f"  Reward: sum={episode_reward_sum:.4f}, mean={episode_reward_mean:.4f}, min={episode_reward_min:.4f}, max={episode_reward_max:.4f}")
                print(f"  Length: {len(self.current_episode_rewards)} steps")
                if hasattr(self.model, 'policy'):
                    print(f"  Value Loss: {self.model.value_loss if hasattr(self.model, 'value_loss') else 'N/A'}")
            
            # Log policy metrics if available
            if hasattr(self.model, 'policy'):
                explained_var = explained_variance(
                    self.model.rollout_buffer.values.flatten(),
                    self.model.rollout_buffer.returns.flatten()
                )
                self.logger.record('train/explained_variance', explained_var)
                
                if hasattr(self.model, 'clip_range'):
                    current_clip_range = self.model.clip_range(1) if callable(self.model.clip_range) else self.model.clip_range
                    self.logger.record('train/clip_fraction', float(current_clip_range))
                
                if hasattr(self.model, 'entropy_loss'):
                    self.logger.record('train/entropy_loss', self.model.entropy_loss)
                    
                if hasattr(self.model, 'policy_loss'):
                    self.logger.record('train/policy_loss', self.model.policy_loss)
                    
                if hasattr(self.model, 'value_loss'):
                    self.logger.record('train/value_loss', self.model.value_loss)
            
            # Reset episode tracking for the next episode
            self.current_episode_rewards = []
            self.current_episode_velocities = []
        
        # Make sure to dump all metrics to tensorboard
        self.logger.dump(self.num_timesteps)
        return True

class CheckpointCallback(BaseCallback):
    """Custom callback for saving model checkpoints during training."""
    
    def __init__(self, save_dir, checkpoint_freq=4000, start_timesteps=0, total_timesteps=None, keep_checkpoints=False, checkpoint_stride=1, verbose=0):
        """Initialize the callback.
        
        Args:
            save_dir: Directory to save checkpoints in
            checkpoint_freq: How often to save checkpoints
            start_timesteps: Starting timestep count (for resumed training)
            total_timesteps: Total timesteps to train for
            keep_checkpoints: Whether to keep all checkpoints or delete previous ones
            checkpoint_stride: Save every Nth checkpoint (e.g. 3 means save checkpoint_freq * 3)
            verbose: Verbosity level
        """
        super().__init__(verbose)
        self.save_dir = save_dir
        self.checkpoint_freq = checkpoint_freq
        self.start_timesteps = start_timesteps
        self.total_timesteps = total_timesteps
        self.keep_checkpoints = keep_checkpoints
        self.checkpoint_stride = checkpoint_stride
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
    def _init_callback(self) -> None:
        """Initialize callback attributes."""
        pass
        
    def _on_step(self) -> bool:
        """Save a checkpoint if it's time to do so."""
        # Calculate current checkpoint number
        current_checkpoint = self.start_timesteps + self.n_calls
        
        # Only save if we've reached a checkpoint frequency and it matches our stride
        if current_checkpoint % self.checkpoint_freq == 0:
            checkpoint_number = current_checkpoint // self.checkpoint_freq
            if checkpoint_number % self.checkpoint_stride == 0:
                # Don't save a checkpoint if we're at the end (final model will be saved)
                if current_checkpoint == self.start_timesteps + self.total_timesteps:
                    return True
                    
                n_updates = current_checkpoint // default_hyperparameters["n_steps"]
                checkpoint_path = os.path.join(self.save_dir, f"model_{n_updates}updates.zip")
                
                # If not keeping checkpoints, delete the previous one
                if not self.keep_checkpoints:
                    prev_updates = (current_checkpoint - (self.checkpoint_freq * self.checkpoint_stride)) // default_hyperparameters["n_steps"]
                    if prev_updates > 0:
                        prev_path = os.path.join(self.save_dir, f"model_{prev_updates}updates.zip")
                        if os.path.exists(prev_path):
                            os.remove(prev_path)
                            if self.verbose > 0:
                                print(f"\nRemoved checkpoint at {prev_updates} updates")
                
                # Save the current checkpoint
                self.model.save(checkpoint_path)
                if self.verbose > 0:
                    print(f"\nSaved checkpoint at {n_updates} updates")
        
        return True

def train_creature(env, total_timesteps=5000, checkpoint_freq=4000, load_path=None, save_dir=None, tensorboard_log=None, start_timesteps=None, keep_checkpoints=False, checkpoint_stride=1, keep_previous_model=False):
    """Train a creature using PPO.
    
    Args:
        env: The environment to train in
        total_timesteps: Total timesteps to train for
        checkpoint_freq: How often to save checkpoints
        load_path: Path to load a model from
        save_dir: Directory to save models in
        tensorboard_log: TensorBoard log directory
        start_timesteps: Starting timestep count (for resuming training)
        keep_checkpoints: Whether to keep all checkpoints
        checkpoint_stride: How many checkpoints to skip between saves
        keep_previous_model: Whether to keep the previous model folder
    """
    # Record start time
    start_time = datetime.now()

    # Create save directory if not provided
    if save_dir is None:
        save_dir = os.path.join("trained_creatures", get_default_folder())
    os.makedirs(save_dir, exist_ok=True)
    
    # Store the previous model's folder for cleanup
    prev_model_folder = None
    if load_path:
        print(f"\nLoading model from {load_path}")
        model = PPO.load(load_path, env=env)
        prev_model_folder = os.path.dirname(load_path)
        # Set tensorboard log directory for loaded model
        if tensorboard_log:
            model.tensorboard_log = tensorboard_log
        if start_timesteps is None and "updates" in load_path:
            try:
                prev_updates = int(load_path.split("updates")[0].split("_")[-1])
                start_timesteps = prev_updates * default_hyperparameters["n_steps"]
                print(f"Continuing from {prev_updates} updates ({start_timesteps} timesteps)")
            except:
                print("Could not parse starting updates from load_path")
                start_timesteps = 0
    else:
        print("\nCreating new model")
        model = PPO("MlpPolicy", env, tensorboard_log=tensorboard_log, **default_hyperparameters)
        start_timesteps = start_timesteps or 0
    
    # Setup callbacks
    tensorboard_callback = TensorboardCallback(start_timesteps=start_timesteps)
    callbacks = [
        CheckpointCallback(
            save_dir=save_dir,
            checkpoint_freq=checkpoint_freq,
            start_timesteps=start_timesteps,
            total_timesteps=total_timesteps,
            keep_checkpoints=keep_checkpoints,
            checkpoint_stride=checkpoint_stride,
            verbose=1
        ),
        tensorboard_callback
    ]
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name=os.path.basename(save_dir),
        reset_num_timesteps=False  # Continue timesteps from previous run
    )
    
    # Save final model with environment steps in filename
    env_steps = start_timesteps + total_timesteps
    training_iterations = total_timesteps * model.n_epochs
    total_updates = env_steps // default_hyperparameters["n_steps"]
    final_path = os.path.join(save_dir, f"final_model_{total_updates}updates.zip")
    model.save(final_path)
    
    # Verify the saved model
    if os.path.exists(final_path):
        file_size = os.path.getsize(final_path)
        expected_min_size = 100 * 1024  # 100KB minimum
        if file_size > expected_min_size:
            print(f"\nSaved final model to {final_path} (size: {file_size/1024:.1f}KB)")
            print(f"Environment Steps: {env_steps}")
            print(f"Training Iterations: {training_iterations} ({model.n_epochs} epochs per step)")
            
            # Clean up intermediate checkpoints unless keep_checkpoints is True
            if not keep_checkpoints:
                for filename in os.listdir(save_dir):
                    if filename.startswith("model_") and filename.endswith("updates.zip"):
                        checkpoint_path = os.path.join(save_dir, filename)
                        try:
                            os.remove(checkpoint_path)
                            print(f"Cleaned up checkpoint: {filename}")
                        except Exception as e:
                            print(f"Note: Could not clean up checkpoint {filename}: {e}")
            
            # Clean up previous model folder if it exists and is different from current
            if prev_model_folder and prev_model_folder != save_dir and not keep_previous_model:
                import shutil
                try:
                    shutil.rmtree(prev_model_folder)
                    print(f"Cleaned up previous model folder: {prev_model_folder}")
                except Exception as e:
                    print(f"Note: Could not clean up previous model folder: {e}")
            elif prev_model_folder and prev_model_folder != save_dir and keep_previous_model:
                print(f"Keeping previous model folder: {prev_model_folder}")
        else:
            print(f"\nWarning: Final model file seems too small ({file_size/1024:.1f}KB). Previous model folder kept.")
    else:
        print(f"\nWarning: Failed to save final model to {final_path}. Previous model folder kept.")
    
    # Store the folder name in the model for reference
    model.last_save_folder = os.path.basename(save_dir)
    
    # Store the tensorboard callback in the model for model card generation
    model.last_callback = tensorboard_callback
    
    # After training is complete, record end time
    end_time = datetime.now()
    
    # Generate model card with actual timing information and steps
    generate_model_card(
        model=model,
        save_dir=save_dir,
        start_time=start_time,
        end_time=end_time,
        start_timesteps=start_timesteps or 0,
        total_timesteps=env_steps - (start_timesteps or 0),  # Use env_steps for model card
        tensorboard_log=tensorboard_log,
        checkpoint_freq=checkpoint_freq,
        keep_checkpoints=keep_checkpoints,
        checkpoint_stride=checkpoint_stride,
        load_path=load_path
    )
    
    return model 