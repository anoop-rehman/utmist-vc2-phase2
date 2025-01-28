import numpy as np
import gym
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import spaces
import os
from datetime import datetime
from model_card_generator import generate_model_card

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

def setup_env(env):
    """Wrap environment and create vectorized env."""
    wrapped_env = DMControlWrapper(env)
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
    """Generate a default folder name using datetime."""
    now = datetime.now()
    return now.strftime("%Y%m%d__%I_%M_%S%p").lower()

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
        if 'bodies_pos' in timestep.observation[0]:
            pos = timestep.observation[0]['bodies_pos'][0]  # Get root body position
            self.position_history.append(pos)
            # Keep only window_size positions
            if len(self.position_history) > self.window_size:
                self.position_history.pop(0)
        
        obs = process_observation(timestep)
        distance = self.get_distance_traveled()
        reward, vel_to_ball = calculate_reward(timestep, action, distance)
        done = timestep.last()
        info = {}
        
        print("-------------------------------")
        print("vel to ball:", vel_to_ball)
        print("train reward:", reward)

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
        return obs

    def render(self, mode='human'):
        pass

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
        self.episode_rewards = []
        self.episode_velocities = []
        
    def _on_step(self):
        # Get current reward and velocity from the wrapped environment
        env = self.training_env.envs[0]  # Get the actual environment from DummyVecEnv
        reward = env.reward
        vel_to_ball = env.last_vel_to_ball
        
        # Log step metrics
        global_step = self.start_timesteps + self.n_calls
        self.logger.record('train/reward', reward, global_step)
        self.logger.record('train/velocity_to_ball', vel_to_ball, global_step)
        
        # Track episode metrics
        if self.locals.get('done'):
            self.episode_rewards.append(reward)
            self.episode_velocities.append(vel_to_ball)
            
            # Log episode metrics
            if len(self.episode_rewards) > 0:
                self.logger.record('train/episode_reward_mean', np.mean(self.episode_rewards[-100:]), global_step)
                self.logger.record('train/episode_reward_max', np.max(self.episode_rewards[-100:]), global_step)
                self.logger.record('train/episode_reward_min', np.min(self.episode_rewards[-100:]), global_step)
                self.logger.record('train/episode_velocity_mean', np.mean(self.episode_velocities[-100:]), global_step)
        
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
                    
                checkpoint_path = os.path.join(self.save_dir, f"model_{current_checkpoint}steps.zip")
                
                # If not keeping checkpoints, delete the previous one
                if not self.keep_checkpoints:
                    prev_checkpoint = current_checkpoint - (self.checkpoint_freq * self.checkpoint_stride)
                    if prev_checkpoint > 0:
                        prev_path = os.path.join(self.save_dir, f"model_{prev_checkpoint}steps.zip")
                        if os.path.exists(prev_path):
                            os.remove(prev_path)
                            if self.verbose > 0:
                                print(f"\nRemoved checkpoint at {prev_checkpoint} steps")
                
                # Save the current checkpoint
                self.model.save(checkpoint_path)
                if self.verbose > 0:
                    print(f"\nSaved checkpoint at {current_checkpoint} steps")
        
        return True

def train_creature(env, total_timesteps=5000, checkpoint_freq=4000, load_path=None, save_dir=None, tensorboard_log=None, start_timesteps=None, keep_checkpoints=False, checkpoint_stride=1):
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
    """
    # Record start time
    start_time = datetime.now()

    # Create save directory if not provided
    if save_dir is None:
        save_dir = os.path.join("trained_creatures", get_default_folder())
    os.makedirs(save_dir, exist_ok=True)
    
    # Load or create model
    if load_path:
        print(f"\nLoading model from {load_path}")
        model = PPO.load(load_path)
        model.set_env(env)
        # Set tensorboard log directory for loaded model
        if tensorboard_log:
            model.tensorboard_log = tensorboard_log
        if start_timesteps is None and "steps" in load_path:
            try:
                start_timesteps = int(load_path.split("steps")[0].split("_")[-1])
                print(f"Continuing from {start_timesteps} steps")
            except:
                print("Could not parse starting timesteps from load_path")
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
        reset_num_timesteps=False
    )
    
    # Save final model with cumulative step count in filename
    final_path = os.path.join(save_dir, f"final_model_{start_timesteps + total_timesteps}_steps")
    model.save(final_path)
    print(f"\nSaved final model to {final_path}")
    
    # Store the folder name in the model for reference
    model.last_save_folder = os.path.basename(save_dir)
    
    # Store the tensorboard callback in the model for model card generation
    model.last_callback = tensorboard_callback
    
    # After training is complete, record end time
    end_time = datetime.now()
    
    # Generate model card with actual timing information
    generate_model_card(
        model=model,
        save_dir=save_dir,
        start_time=start_time,
        end_time=end_time,
        start_timesteps=start_timesteps or 0,
        total_timesteps=total_timesteps,
        tensorboard_log=tensorboard_log,
        checkpoint_freq=checkpoint_freq,
        keep_checkpoints=keep_checkpoints,
        checkpoint_stride=checkpoint_stride,
        load_path=load_path
    )
    
    return model