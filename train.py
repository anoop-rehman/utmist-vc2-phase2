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
    # n_steps=196608,  # Doubled from 8192 to extend collection phase
    # batch_size=1536,  # Maximal batch size for efficient GPU usage
    n_steps=1024,
    batch_size=24576,
    # batch_size=512,
    n_epochs=10,  
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    # Much larger network to fully utilize GPU
    policy_kwargs=dict(
        net_arch=[dict(
            # Policy network (much larger)
            pi=[64, 64],
            
            # Value network (also expanded)
            vf=[64, 64]
        )],
        # activation_fn=th.nn.ReLU
        activation_fn=th.nn.Tanh

    ),
)

def setup_env(env, phase="combined"):
    """Wrap environment based on training phase."""
    if phase == "walking":
        wrapped_env = WalkingPhaseWrapper(env)
    elif phase == "rotation":
        wrapped_env = RotationPhaseWrapper(env)
    else:
        wrapped_env = DMControlWrapper(env)  # Original combined phase
    
    # Return the wrapped environment directly, without vectorizing
    return wrapped_env

def create_ppo_model(vec_env, tensorboard_log, load_path=None):
    """Create or load a PPO model with standard parameters."""
    if load_path:
        print(f"Loading pre-trained model from {load_path}")
        return PPO.load(load_path, env=vec_env, tensorboard_log=tensorboard_log, **default_hyperparameters)
    
    return PPO(
        "MlpPolicy",
        vec_env,
        verbose=0,
        tensorboard_log=tensorboard_log,
        **default_hyperparameters
    )

def get_default_folder():
    """Generate a default folder name using datetime in EST timezone."""
    eastern_tz = pytz.timezone('US/Eastern')
    now = datetime.now(tz=eastern_tz)
    
    # Format with custom hour handling
    year = now.strftime("%Y%m%d")
    am_pm = now.strftime("%p").lower()
    
    # Convert 12-hour format to use 00 instead of 12
    hour_12 = now.hour % 12
    if hour_12 == 0:  # If it's noon or midnight
        hour_12 = 0   # Use 00 instead of 12
        
    minute = now.strftime("%M")
    second = now.strftime("%S")
    
    # Format with leading zeros
    return f"{year}__{am_pm}_{hour_12:02d}_{minute}_{second}"

def process_observation(timestep):
    """Convert DM Control observation to the format expected by the model."""
    obs_dict = timestep.observation[0]
    
    # Add a counter to track number of calls (static variable)
    if not hasattr(process_observation, "counter"):
        process_observation.counter = 0
    
    # Print observations every 40 steps
    # should_print = process_observation.counter % 40 == 0
    should_print = False
    process_observation.should_print = should_print  # Set a flag for other functions
    
    # Filter the observation - keep only the core components we want
    filtered_dict = {}
    core_observations = ['absolute_root_mat', 'bodies_pos', 'joints_pos']
    
    for key in core_observations:
        if key in obs_dict:
            filtered_dict[key] = obs_dict[key]
    
    # Only print the filtered observations
    if should_print:
        print(f"\n--- Observation {process_observation.counter} (Filtered) ---")
        print("Filtered observation keys:", list(filtered_dict.keys()))
        
        # Print filtered values
        for k, v in filtered_dict.items():
            if hasattr(v, 'shape'):
                print(f"{k}: shape={v.shape}, value={v.flatten()}")
            else:
                print(f"{k}: {v}")
    
    process_observation.counter += 1
    
    # Use only the filtered observations
    return np.concatenate([v.flatten() for v in filtered_dict.values()])

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
        
        # Calculate observation size with filtered observations
        timestep = env.reset()
        
        # Get filtered keys for debugging
        obs_dict = timestep.observation[0]
        core_observations = ['absolute_root_mat', 'bodies_pos', 'joints_pos']
        filtered_dict = {}
        for key in core_observations:
            if key in obs_dict:
                filtered_dict[key] = obs_dict[key]
                print(f"Keeping observation: {key} with shape {obs_dict[key].shape}")
            else:
                print(f"Warning: {key} not found in observation dictionary")
        
        self.obs_concat = process_observation(timestep)
        obs_size = self.obs_concat.shape[0]
        
        print(f"Filtered observation space size: {obs_size}")
        # Expected sizes: 9 (root_mat) + 27 (bodies_pos) + 8 (joints_pos) = 44
        
        # Define observation space with the correct size
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # Random number generator for seeding
        self.np_random = np.random.RandomState()

    def seed(self, seed=None):
        """Seed the environment's random number generator."""
        self.np_random = np.random.RandomState(seed)
        # Debug print to verify seeding
        print(f"DMControlWrapper: Seeding environment with seed {seed}")
        
        # Also seed the underlying DM Control environment physics engine
        if hasattr(self.env, 'physics') and hasattr(self.env.physics, 'set_random_state'):
            # For DM Control environments, also seed the physics engine
            try:
                self.env.physics.set_random_state(seed)
                print(f"  -> Successfully seeded physics engine with {seed}")
            except Exception as e:
                print(f"  -> Failed to seed physics engine: {e}")
        
        return [seed]

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
        
        # Debug print - verify this is a fresh reset with proper randomization
        env_id = id(self)  # Unique identifier for this environment instance
        seed_val = self.np_random.randint(0, 10000) if hasattr(self, 'np_random') else "UNSEEDED"
        print(f"DMControlWrapper {env_id}: Resetting environment (random check: {seed_val})")
        
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
        # First get base observation size (filtered)
        self.base_obs_size = self.observation_space.shape[0]
        
        # Define new observation space with extended size
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.base_obs_size + self.additional_obs_size,),
            dtype=np.float32
        )
        
        print(f"WalkingPhaseWrapper observation space shape: {self.observation_space.shape}")
        print(f"Base filtered observation size: {self.base_obs_size}")
        print(f"Additional features size: {self.additional_obs_size}")
        print(f"Total expected size: {self.base_obs_size + self.additional_obs_size}")
    
    def seed(self, seed=None):
        """Seed both the environment and internal RNG."""
        super().seed(seed)
        # Also seed phase and other random elements
        if seed is not None:
            # Add some offset to avoid same seeds for different aspects
            self.phase = self.np_random.uniform(0, 1.0)  # Randomize initial phase
        return [seed]
    
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
        # Track initial orientation for debuggging
        self.initial_orientation = None
        self.verbose = True
        
        print("\n==== Using simplified rotation reward (alignment only, -1 to 1 range) ====\n")
        
    def seed(self, seed=None):
        """Seed the environment's random number generator."""
        self.np_random = np.random.RandomState(seed)
        # Debug print to verify seeding
        print(f"RotationPhaseWrapper: Seeding environment with seed {seed}")
        
        # Also seed the underlying DM Control environment physics engine
        if hasattr(self.env, 'physics') and hasattr(self.env.physics, 'set_random_state'):
            # For DM Control environments, also seed the physics engine
            try:
                self.env.physics.set_random_state(seed)
                print(f"  -> Successfully seeded physics engine with {seed}")
            except Exception as e:
                print(f"  -> Failed to seed physics engine: {e}")
        
        return [seed]
    
    def step(self, action):
        timestep = self.env.step([action])
        
        # Get observation 
        obs = process_observation(timestep)
        
        # Initialize reward to default value
        alignment_reward = 0.0
        
        # Extract alignment directly from rotation matrix
        if 'absolute_root_mat' in timestep.observation[0]:
            # Get the rotation matrix
            rot_matrix = timestep.observation[0]['absolute_root_mat'].copy()
            
            # The x-component of the z-axis is directly at index 2 of the flattened matrix
            # This represents the alignment of the z-axis with the x-axis
            alignment_reward = float(rot_matrix[0, 2])
            
            # Print debug info periodically
            if hasattr(process_observation, "should_print") and process_observation.should_print:
                print(f"Rotation reward: z-axis x-component = {alignment_reward:.3f}")
        else:
            print("absolute_root_mat not found in timestep.observation[0]!")
        
        # Use simple alignment reward directly
        reward = alignment_reward
        
        done = timestep.last()
        info = {}

        self.reward = reward
        self.last_vel_to_ball = alignment_reward  # For consistency with previous code
        return obs, reward, done, info
    
    def reset(self):
        timestep = self.env.reset()
        obs = process_observation(timestep)
        
        # Debug print - verify this is a fresh reset with proper randomization
        env_id = id(self)  # Unique identifier for this environment instance
        seed_val = self.np_random.randint(0, 10000) if hasattr(self, 'np_random') else "UNSEEDED"
        print(f"RotationPhaseWrapper {env_id}: Resetting environment (random check: {seed_val})")
        
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
                    if not hasattr(self, 'np_random'):
                        print(f"WARNING: RotationPhaseWrapper {env_id} has no np_random - creating one with random seed")
                        self.np_random = np.random.RandomState()
                    
                    axis = self.np_random.randn(3)  # Use seeded RNG instead of global np.random
                    axis = axis / np.linalg.norm(axis)  # Normalize to unit vector
                    angle = self.np_random.uniform(0, 2 * np.pi)  # Use seeded RNG for random angle
                    
                    # Debug print to verify randomization
                    print(f"RotationPhaseWrapper {env_id}: Random rotation: axis={axis}, angle={angle:.2f}")
                    
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
        
        # Store initial orientation - now directly the alignment value
        if 'absolute_root_mat' in timestep.observation[0]:
            rot_matrix = timestep.observation[0]['absolute_root_mat'].copy()
            # Extract x-component of z-axis directly
            initial_alignment = float(rot_matrix[0, 2])
            self.initial_orientation = initial_alignment
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
            # Convert numpy array to float for proper formatting
            print(f"  Initial alignment reward: {float(self.last_vel_to_ball):.4f}")
        
        # We need to re-process the observation after applying the random rotation
        if hasattr(self.env, 'physics') and hasattr(self.env, '_task') and hasattr(self.env._task, 'players'):
            try:
                # Get fresh observation after randomization
                # Instead of using physics.get_state() directly, reset the environment to get a valid timestep
                fresh_timestep = self.env.physics.step()  # Take a zero-action step
                if hasattr(fresh_timestep, 'observation') and fresh_timestep.observation:
                    obs = process_observation(fresh_timestep)
                else:
                    # If stepping doesn't return a valid timestep, use the existing observation
                    print("Warning: Could not get fresh observation, using existing one")
            except Exception as e:
                print(f"Error getting fresh observation: {e}")
        
        return obs

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
    def __init__(self, start_timesteps=0, verbose=0, target_updates=None):
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
        
        # Track total timesteps for interruption handling
        self.current_total_timesteps = start_timesteps
        
        # For hardware monitoring
        self.hw_monitor_interval = 50  # Log hardware metrics every 50 steps
        self.has_gpu = th.cuda.is_available()
        import psutil
        self.psutil = psutil
        
        # For enforcing exact number of updates
        self.target_updates = target_updates
        
    def _on_training_start(self):
        # Initialize episode tracking
        self.current_episode_rewards = []
        self.current_episode_velocities = []
        
    def _on_step(self):
        # Check if we've reached our target number of updates
        if self.target_updates is not None:
            current_update = self.num_timesteps // default_hyperparameters["n_steps"]
            if current_update >= self.target_updates:
                print(f"\nReached target of {self.target_updates} updates. Stopping training.")
                return False  # Return False to stop training
        
        # Get rewards and velocities from ALL environments
        if hasattr(self.training_env, 'envs'):
            # For DummyVecEnv - direct access
            envs = self.training_env.envs
            rewards = [env.reward for env in envs]
            vel_to_balls = [env.last_vel_to_ball for env in envs]
        else:
            # For SubprocVecEnv - remote access
            rewards = self.training_env.get_attr('reward')
            vel_to_balls = self.training_env.get_attr('last_vel_to_ball')
        
        # Track metrics for all environments
        for i, (reward, vel) in enumerate(zip(rewards, vel_to_balls)):
            # Initialize tracking lists for each env if needed
            if not hasattr(self, 'current_episode_rewards_all'):
                self.current_episode_rewards_all = [[] for _ in range(len(rewards))]
                self.current_episode_velocities_all = [[] for _ in range(len(rewards))]
            
            # Add metrics to respective environment's tracking
            self.current_episode_rewards_all[i].append(reward)
            self.current_episode_velocities_all[i].append(vel)
        
        # For step-level logging, use average across all environments
        reward = np.mean(rewards) if rewards else 0.0
        vel_to_ball = np.mean(vel_to_balls) if vel_to_balls else 0.0
        
        # Track current episode stats
        self.current_episode_rewards.append(reward)
        self.current_episode_velocities.append(vel_to_ball)
        
        # Get current environment steps
        env_steps = self.num_timesteps
        
        # Update total timesteps count (including initial steps)
        # self.current_total_timesteps = self.start_timesteps + self.num_timesteps
        self.current_total_timesteps = self.num_timesteps
        
        # Log step-level metrics
        self.logger.record('train/reward', reward)
        self.logger.record('train/velocity_to_ball', vel_to_ball)
        
        # Log hardware metrics periodically to avoid overhead
        if self.num_timesteps % self.hw_monitor_interval == 0:
            try:
                # CPU usage (per core and total)
                cpu_percent = self.psutil.cpu_percent(interval=0.1)
                self.logger.record('system/cpu_percent', cpu_percent)
                
                # Memory usage
                mem = self.psutil.virtual_memory()
                self.logger.record('system/memory_used_percent', mem.percent)
                self.logger.record('system/memory_available_gb', mem.available / (1024**3))
                
                # GPU metrics if available
                if self.has_gpu:
                    try:
                        gpu_util = float(th.cuda.utilization())
                        self.logger.record('system/gpu_util_percent', gpu_util)
                    except:
                        # Fallback to nvidia-smi if torch method fails
                        try:
                            import subprocess
                            result = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'])
                            result = result.decode('utf-8').strip().split(',')
                            gpu_util = float(result[0])
                            gpu_mem_used = float(result[1])
                            gpu_mem_total = float(result[2])
                            
                            self.logger.record('system/gpu_util_percent', gpu_util)
                            self.logger.record('system/gpu_memory_used_mb', gpu_mem_used)
                            self.logger.record('system/gpu_memory_percent', 100 * gpu_mem_used / gpu_mem_total)
                        except:
                            pass
            except Exception as e:
                # Don't let hardware monitoring failures interrupt training
                if self.verbose > 0:
                    print(f"Warning: Failed to log hardware metrics: {e}")
        
        # Log training metrics from the model
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            for key, value in self.model.logger.name_to_value.items():
                self.logger.record(key, value)
        
        # Handle episode completions (check all environments)
        dones = self.locals.get('dones', [False])
        for i, done in enumerate(dones):
            if done:
                # Skip if this environment hasn't accumulated data yet
                if not hasattr(self, 'current_episode_rewards_all') or i >= len(self.current_episode_rewards_all):
                    continue
            
                # Calculate statistics for this particular environment's episode
                episode_rewards = self.current_episode_rewards_all[i]
                episode_velocities = self.current_episode_velocities_all[i]
                
                if episode_rewards:
                    episode_reward_sum = sum(episode_rewards)
                    episode_reward_mean = np.mean(episode_rewards)
                    episode_reward_min = np.min(episode_rewards)
                    episode_reward_max = np.max(episode_rewards)
                    episode_velocity_mean = np.mean(episode_velocities) if episode_velocities else 0.0
                    
                    # Store episode summary metrics - track for all environments
                    if not hasattr(self, 'all_episode_rewards'):
                        self.all_episode_rewards = [[] for _ in range(len(dones))]
                        self.all_episode_velocities = [[] for _ in range(len(dones))]
                    
                    self.all_episode_rewards[i].append(episode_reward_mean)
                    self.all_episode_velocities[i].append(episode_velocity_mean)
                    
                    # Log aggregated statistics after we've collected data from all environments
                    self.episode_count += 1
                    if self.episode_count % 192 == 0:  # After all 192 envs report
                        # Calculate global averages across all environments
                        all_rewards = [r for rewards in self.all_episode_rewards for r in rewards[-1:]]
                        all_velocities = [v for velocities in self.all_episode_velocities for v in velocities[-1:]]
                        
                        # Log aggregated metrics
                        self.logger.record('train/episode_reward_mean', np.mean(all_rewards))
                        self.logger.record('train/episode_reward_min', np.min(all_rewards))
                        self.logger.record('train/episode_reward_max', np.max(all_rewards))
                        self.logger.record('train/episode_velocity_mean', np.mean(all_velocities))
                        
                        # Print summary for user feedback
                        print(f"\nAggregated stats across {len(all_rewards)} environments:")
                        print(f"  Reward: mean={np.mean(all_rewards):.4f}, min={np.min(all_rewards):.4f}, max={np.max(all_rewards):.4f}")
                    
                    # Reset this environment's tracking
                    self.current_episode_rewards_all[i] = []
                    self.current_episode_velocities_all[i] = []
        
        # Make sure to dump all metrics to tensorboard
        self.logger.dump(self.num_timesteps)
        return True

class CheckpointCallback(BaseCallback):
    """Custom callback for saving model checkpoints during training."""
    
    def __init__(self, save_dir, checkpoint_freq=4000, start_timesteps=0, total_timesteps=None, keep_checkpoints=False, checkpoint_stride=1, verbose=0, n_envs=1):
        """Initialize the callback.
        
        Args:
            save_dir: Directory to save checkpoints in
            checkpoint_freq: How often to save checkpoints
            start_timesteps: Starting timestep count (for resumed training)
            total_timesteps: Total timesteps to train for
            keep_checkpoints: Whether to keep all checkpoints or delete previous ones
            checkpoint_stride: Save every Nth checkpoint (e.g. 3 means save checkpoint_freq * 3)
            verbose: Verbosity level
            n_envs: Number of parallel environments
        """
        super().__init__(verbose)
        self.save_dir = save_dir
        self.checkpoint_freq = checkpoint_freq
        self.start_timesteps = start_timesteps
        self.total_timesteps = total_timesteps
        self.keep_checkpoints = keep_checkpoints
        self.checkpoint_stride = checkpoint_stride
        self.n_envs = n_envs
        self.last_update_saved = 0
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
    def _init_callback(self) -> None:
        """Initialize callback attributes."""
        pass
        
    def _on_step(self) -> bool:
        """Save a checkpoint if it's time to do so."""
        # Calculate total environment steps (accounting for vectorization)
        env_steps = (self.start_timesteps + self.n_calls) * self.n_envs
        
        # Calculate current number of updates
        n_updates = env_steps // self.model.n_steps
        
        # Save if we've completed a new update and it matches our checkpoint_stride pattern
        if n_updates > self.last_update_saved and (n_updates % self.checkpoint_stride == 0):
            self.last_update_saved = n_updates
            checkpoint_path = os.path.join(self.save_dir, f"model_{n_updates}updates.zip")
            
            # If not keeping checkpoints, delete the previous one
            if not self.keep_checkpoints and n_updates > self.checkpoint_stride:
                prev_updates = n_updates - self.checkpoint_stride
                prev_path = os.path.join(self.save_dir, f"model_{prev_updates}updates.zip")
                if os.path.exists(prev_path):
                    try:
                        os.remove(prev_path)
                        if self.verbose > 0:
                            print(f"\nRemoved checkpoint at {prev_updates} updates")
                    except Exception as e:
                        print(f"Warning: Could not remove previous checkpoint: {e}")
            
            # Save the current checkpoint
            self.model.save(checkpoint_path)
            if self.verbose > 0:
                print(f"\nSaved checkpoint at {n_updates} updates")
        
        return True

class RolloutDebugCallback(BaseCallback):
    """Callback that prints detailed information at the end of each rollout.
    Helps verify that all environments are contributing data correctly.
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        # Track how many rollouts we've seen
        self.rollout_idx = 0

    def _on_rollout_end(self) -> None:
        self.rollout_idx += 1

        # Length of rollout buffer == n_steps * n_envs
        buffer_size = len(self.model.rollout_buffer.rewards)
        n_steps_cfg = self.model.n_steps
        n_envs_cfg = self.model.n_envs
        expected_size = n_steps_cfg * n_envs_cfg

        print("\n=== Rollout Debug ===")
        print(f"Rollout #{self.rollout_idx} ended.")
        print(f"Configured n_steps: {n_steps_cfg}")
        print(f"Configured n_envs:  {n_envs_cfg}")
        print(f"Rollout buffer size: {buffer_size}")
        print(f"Expected size (n_steps * n_envs): {expected_size}")
        
        # The env_indices attribute doesn't exist in SB3's RolloutBuffer
        # Check the shape of buffer components to understand distribution
        obs_shape = getattr(self.model.rollout_buffer.observations, 'shape', 'unknown')
        actions_shape = getattr(self.model.rollout_buffer.actions, 'shape', 'unknown')
        rewards_shape = self.model.rollout_buffer.rewards.shape if hasattr(self.model.rollout_buffer.rewards, 'shape') else 'unknown'
        
        print(f"Observations shape: {obs_shape}")
        print(f"Actions shape: {actions_shape}")
        print(f"Rewards shape: {rewards_shape}")
        print("====================\n")
    
    def _on_step(self) -> bool:
        """Required method that is called at each step."""
        return True  # Return True to continue training

def train_creature(env, total_timesteps=5000, checkpoint_freq=4000, load_path=None, save_dir=None, tensorboard_log=None, start_timesteps=None, keep_checkpoints=False, checkpoint_stride=1, keep_previous_model=False, training_phase="combined", n_envs=1, target_updates=None):
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
        training_phase: Which training phase is being used ("combined", "walking", or "rotation")
        n_envs: Number of parallel environments being used
        target_updates: The exact number of updates to train for (overrides total_timesteps for stopping)
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
        # Use default hyperparameters with updated policy_kwargs
        model = PPO("MlpPolicy", 
                   env, 
                   tensorboard_log=tensorboard_log, 
                   verbose=0,  # Changed from 1 to 0 to disable table logs
                   **default_hyperparameters)
        start_timesteps = start_timesteps or 0
    
    # Log the number of parallel environments
    if n_envs > 1:
        print(f"\nTraining with {n_envs} parallel environments")
        print(f"Each timestep will collect {n_envs} samples")
        print(f"Expected speedup: ~{n_envs}x (minus overhead)")

    # Setup callbacks
    tensorboard_callback = TensorboardCallback(start_timesteps=start_timesteps, target_updates=target_updates)
    callbacks = [
        CheckpointCallback(
            save_dir=save_dir,
            checkpoint_freq=default_hyperparameters["n_steps"],  # This doesn't actually matter much with the fix
            start_timesteps=start_timesteps,
            total_timesteps=total_timesteps,
            keep_checkpoints=keep_checkpoints,
            checkpoint_stride=checkpoint_stride,
            verbose=1,
            n_envs=n_envs  # Pass n_envs as a parameter
        ),
        tensorboard_callback,
        RolloutDebugCallback(verbose=1)
    ]
    
    # Store the model for saving in case of interruption
    # Note: We'll update this attribute in case of an interruption
    model.actual_timesteps_trained = 0
    
    # Train the model
    interrupted = False
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callbacks,
            tb_log_name=os.path.basename(save_dir),
            reset_num_timesteps=False  # Continue timesteps from previous run
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by keyboard! Saving model and generating model card...")
        interrupted = True
        
        # Get how many timesteps we've actually trained
        # This will be more accurate than estimating
        if hasattr(tensorboard_callback, 'current_total_timesteps'):
            model.actual_timesteps_trained = tensorboard_callback.current_total_timesteps - start_timesteps
            print(f"Trained for {model.actual_timesteps_trained} steps of planned {total_timesteps}")
        else:
            model.actual_timesteps_trained = total_timesteps
            print("No tensorboard callback found, using total timesteps")
    
    # Use correct number of timesteps
    if interrupted:
        env_steps = start_timesteps + model.actual_timesteps_trained
        # Calculate total updates based on total steps (not separate divisions)
        total_updates = env_steps // default_hyperparameters["n_steps"]
        print(f"Training was interrupted after approximately {model.actual_timesteps_trained} steps.")
        print(f"Total steps including previous training: {env_steps}")
        print(f"Total updates: {total_updates}")
    else:
        env_steps = start_timesteps + total_timesteps
        # Calculate total updates based on total steps
        total_updates = env_steps // default_hyperparameters["n_steps"]
    
    # Save final model with environment steps in filename
    training_iterations = (total_timesteps if not interrupted else model.actual_timesteps_trained) * model.n_epochs
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
            
            # Add information about parallel environments
            if n_envs > 1:
                print(f"Parallel Environments: {n_envs}")
                print(f"Total samples collected: {env_steps * n_envs}")
            
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
    
    # Use the right number of timesteps for model card
    actual_timesteps = model.actual_timesteps_trained if interrupted else total_timesteps
    
    # Generate model card with actual timing information and steps
    generate_model_card(
        model=model,
        save_dir=save_dir,
        start_time=start_time,
        end_time=end_time,
        start_timesteps=start_timesteps or 0,
        trained_timesteps=actual_timesteps,  # Use actual timesteps trained
        tensorboard_log=tensorboard_log,
        checkpoint_freq=checkpoint_freq,
        keep_checkpoints=keep_checkpoints,
        checkpoint_stride=checkpoint_stride,
        load_path=load_path,
        interrupted=interrupted,  # Pass the interrupted flag
        training_phase=training_phase,  # Pass the training phase
        n_envs=n_envs  # Pass the number of environments
    )
    
    return model 