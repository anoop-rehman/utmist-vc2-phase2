import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import spaces
import os
from datetime import datetime

# Default hyperparameters
PPO_PARAMS = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2
}

def setup_env(env):
    """Wrap environment and create vectorized env."""
    wrapped_env = DMControlWrapper(env)
    return DummyVecEnv([lambda: wrapped_env])

def create_ppo_model(vec_env, tensorboard_log, load_path=None):
    """Create or load a PPO model with standard parameters."""
    if load_path:
        print(f"Loading pre-trained model from {load_path}")
        return PPO.load(load_path, env=vec_env, tensorboard_log=tensorboard_log, **PPO_PARAMS)
    
    return PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        tensorboard_log=tensorboard_log,
        **PPO_PARAMS
    )

def get_default_folder():
    """Generate a default folder name using datetime."""
    now = datetime.now()
    return now.strftime("%Y%m%d__%I_%M%p").lower()

def process_observation(timestep):
    """Convert DM Control observation to the format expected by the model."""
    obs_dict = timestep.observation[0]
    return np.concatenate([v.flatten() for v in obs_dict.values()])

def calculate_reward(timestep, action):
    """Calculate reward based on velocity to ball and control cost."""
    vel_to_ball = timestep.observation[0]['stats_vel_to_ball'][0]
    ctrl_cost_weight = 0.5
    ctrl_cost = ctrl_cost_weight * np.sum(np.square(action))
    reward = vel_to_ball + 1.0 - ctrl_cost
    return reward, vel_to_ball

class DMControlWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.reward = 0  # Add this to store current reward for tensorboard
        
        # Get action and observation specs
        action_spec = env.action_spec()[0]  # Get first player's action spec
        obs_spec = env.observation_spec()[0]  # Get first player's observation spec
        
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

    def step(self, action):
        timestep = self.env.step([action])
        
        obs = process_observation(timestep)
        reward, vel_to_ball = calculate_reward(timestep, action)
        done = timestep.last()
        info = {}
        
        print("-------------------------------")
        print("vel to ball:", vel_to_ball)
        print("train reward:", reward)

        self.reward = reward  # Store the reward
        return obs, reward, done, info

    def reset(self):
        timestep = self.env.reset()
        return process_observation(timestep)

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
        
    def _on_step(self):
        # Use global step count for tensorboard
        global_step = self.start_timesteps + self.n_calls
        self.logger.record('reward', self.training_env.get_attr('reward')[0], global_step)
        return True

class CheckpointCallback(BaseCallback):
    def __init__(self, save_dir, checkpoint_freq=4000, start_timesteps=0, total_timesteps=None, verbose=0):
        super().__init__(verbose)
        self.checkpoint_freq = checkpoint_freq
        self.save_dir = save_dir
        self.start_timesteps = start_timesteps
        self.total_timesteps = total_timesteps
        
    def _on_step(self):
        if self.n_calls % self.checkpoint_freq == 0:
            current_checkpoint = self.start_timesteps + self.n_calls
            previous_checkpoint = current_checkpoint - self.checkpoint_freq
            
            # Don't save if this is the final checkpoint (will be saved as final_model)
            if current_checkpoint == self.start_timesteps + self.total_timesteps:
                return True
                
            # Save current checkpoint
            checkpoint_path = os.path.join(self.save_dir, f"model_{current_checkpoint}steps")
            self.model.save(checkpoint_path)
            print(f"\nSaved checkpoint at {current_checkpoint} steps")
            
            # Delete previous checkpoint if it exists
            if previous_checkpoint > self.start_timesteps:
                old_path = os.path.join(self.save_dir, f"model_{previous_checkpoint}steps")
                if os.path.exists(old_path + ".zip"):
                    os.remove(old_path + ".zip")
                    print(f"Removed previous checkpoint at {previous_checkpoint} steps")
        return True

def train_creature(env, save_dir=None, total_timesteps=240_000, load_path=None, checkpoint_freq=4000, start_timesteps=None, tensorboard_log=None):
    """
    Train a creature while maintaining the most recent checkpoint.
    When loading from a previous model, it will:
    1. Continue the global step count from where the previous model left off
    2. Use the same tensorboard log directory if provided, enabling continuous training graphs
    3. Save the final model with the cumulative step count in its name
    
    Args:
        env: The environment to train in
        save_dir: Directory to save models (if None, uses datetime-based folder)
        total_timesteps: Total timesteps to train for in this session
        load_path: Path to load a pre-trained model (optional)
        checkpoint_freq: Frequency at which to save checkpoints (default 4000)
        start_timesteps: Starting timestep count (optional, will try to parse from load_path if not provided)
        tensorboard_log: Directory for tensorboard logs (optional, will create new one if not provided)
                        When resuming training, pass the same directory to continue the training graphs
    """
    # Get starting timesteps from parameter or load_path
    if start_timesteps is not None:
        print(f"Using provided start_timesteps: {start_timesteps}")
    else:
        start_timesteps = 0
        if load_path and "steps" in load_path:
            try:
                start_timesteps = int(load_path.split("steps")[0].split("_")[-1])
                print(f"Parsed start_timesteps from load_path: {start_timesteps}")
            except:
                print("Could not parse starting timesteps from load_path")
    
    # Generate default save directory if none provided (datetime-based folder)
    if save_dir is None:
        save_dir = os.path.join("trained_creatures", get_default_folder())
    
    # Store the folder name for reference
    model_folder = os.path.basename(save_dir)
    
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup environment and model
    vec_env = setup_env(env)
    
    # Use provided tensorboard_log or create default
    if tensorboard_log is None:
        run_name = f"PPO_{model_folder}"
        tensorboard_log = f"./tensorboard_logs/{run_name}"
    
    model = create_ppo_model(vec_env, tensorboard_log, load_path)

    # Train the model with callbacks for logging and checkpointing
    callbacks = [
        TensorboardCallback(start_timesteps=start_timesteps),  # Tracks global step count
        TrainingCallback(),
        CheckpointCallback(save_dir, checkpoint_freq, start_timesteps, total_timesteps)
    ]
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=False if load_path else True  # Continue step count when loading model
    )

    # Clean up last checkpoint
    last_checkpoint = total_timesteps - (total_timesteps % checkpoint_freq)
    if last_checkpoint == total_timesteps:
        last_checkpoint -= checkpoint_freq
    checkpoint_path = os.path.join(save_dir, f"model_{last_checkpoint}steps.zip")
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        print(f"\nRemoved checkpoint at {last_checkpoint} steps")

    # Save final model with cumulative step count in filename
    last_steps = start_timesteps + total_timesteps
    final_path = os.path.join(save_dir, f"final_model_{last_steps}_steps")
    final_path += ".zip"  # Add .zip extension
    model.save(final_path)
    print(f"\nSaved final model to {final_path}")
    model.last_save_folder = os.path.basename(save_dir)  # Store the folder name in the model for reference
    return model 