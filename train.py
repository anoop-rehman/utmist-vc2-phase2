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
            # Keep only window_size positions
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
        self.last_obs = None
        
    def _on_step(self):
        # Get current reward and velocity from the wrapped environment
        env = self.training_env.envs[0]  # Get the actual environment from DummyVecEnv
        reward = env.reward
        vel_to_ball = env.last_vel_to_ball
        
        # Get current environment steps
        env_steps = self.start_timesteps + self.num_timesteps
        
        # Log step-level metrics
        self.logger.record('train/reward', reward)
        self.logger.record('train/velocity_to_ball', vel_to_ball)
        
        # Log training metrics from the model
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            for key, value in self.model.logger.name_to_value.items():
                self.logger.record(key, value)
        
        # Track episode metrics
        if self.locals.get('done'):
            self.episode_rewards.append(reward)
            self.episode_velocities.append(vel_to_ball)
            
            # Calculate episode statistics
            if len(self.episode_rewards) > 0:
                recent_rewards = self.episode_rewards[-100:]  # Last 100 episodes
                recent_velocities = self.episode_velocities[-100:]
                
                # Log episode metrics
                self.logger.record('train/episode_reward_mean', np.mean(recent_rewards))
                self.logger.record('train/episode_reward_min', np.min(recent_rewards))
                self.logger.record('train/episode_reward_max', np.max(recent_rewards))
                self.logger.record('train/episode_velocity_mean', np.mean(recent_velocities))
                self.logger.record('train/episode_length', self.n_calls)
                
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
        
        # Make sure to dump all metrics to tensorboard
        self.logger.dump(env_steps)
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
    
    # Store the previous model's folder for cleanup
    prev_model_folder = None
    if load_path:
        print(f"\nLoading model from {load_path}")
        model = PPO.load(load_path)
        model.set_env(env)
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
        reset_num_timesteps=False
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
            if prev_model_folder and prev_model_folder != save_dir:
                import shutil
                try:
                    shutil.rmtree(prev_model_folder)
                    print(f"Cleaned up previous model folder: {prev_model_folder}")
                except Exception as e:
                    print(f"Note: Could not clean up previous model folder: {e}")
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