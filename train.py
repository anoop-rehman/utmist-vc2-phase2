import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import spaces
import os
from datetime import datetime

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
        obs_dict = timestep.observation[0]
        self.obs_concat = np.concatenate([v.flatten() for v in obs_dict.values()])
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
        
        obs_dict = timestep.observation[0]
        obs = np.concatenate([v.flatten() for v in obs_dict.values()])
        
        vel_to_ball = timestep.observation[0]['stats_vel_to_ball'][0]
        ctrl_cost_weight = 0.5
        ctrl_cost = ctrl_cost_weight * np.sum(np.square(action))
        reward = vel_to_ball + 1.0 - ctrl_cost 

        done = timestep.last()
        info = {}  # Add any additional info you want to track
        
        print("-------------------------------")
        print("vel to ball:", vel_to_ball)
        print("train reward:", reward)

        self.reward = reward  # Store the reward
        return obs, reward, done, info

    def reset(self):
        timestep = self.env.reset()
        obs_dict = timestep.observation[0]
        obs = np.concatenate([v.flatten() for v in obs_dict.values()])
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
    def __init__(self, verbose=0):
        super().__init__(verbose)
        
    def _on_step(self):
        # Log reward
        self.logger.record('reward', self.training_env.get_attr('reward')[0])
        return True

def get_default_folder():
    """Generate a default folder name using datetime."""
    now = datetime.now()
    return now.strftime("%Y%m%d__%I_%M%p").lower()

class CheckpointCallback(BaseCallback):
    def __init__(self, save_dir, checkpoint_freq=4000, verbose=0):
        super().__init__(verbose)
        self.checkpoint_freq = checkpoint_freq
        self.save_dir = save_dir
        
    def _on_step(self):
        if self.n_calls % self.checkpoint_freq == 0:
            # Calculate which checkpoints to keep/remove
            current_checkpoint = self.n_calls
            previous_checkpoint = current_checkpoint - self.checkpoint_freq
            
            # Save current checkpoint
            checkpoint_path = os.path.join(self.save_dir, f"model_{current_checkpoint}steps")
            self.model.save(checkpoint_path)
            print(f"\nSaved checkpoint at {current_checkpoint} steps")
            
            # Delete previous checkpoint if it exists
            if previous_checkpoint > 0:
                old_path = os.path.join(self.save_dir, f"model_{previous_checkpoint}steps")
                if os.path.exists(old_path + ".zip"):
                    os.remove(old_path + ".zip")
                    print(f"Removed previous checkpoint at {previous_checkpoint} steps")
        
        return True

def train_creature_with_checkpoints(env, save_dir=None, total_timesteps=240_000, load_path=None, checkpoint_freq=4000):
    """
    Train a creature while maintaining the most recent checkpoint.
    
    Args:
        env: The environment to train in
        save_dir: Directory to save models (if None, uses datetime-based folder)
        total_timesteps: Total timesteps to train for
        load_path: Path to load a pre-trained model (optional)
        checkpoint_freq: Frequency at which to save checkpoints (default 4000)
    """
    # Generate default save directory if none provided
    if save_dir is None:
        save_dir = os.path.join("trained_creatures", get_default_folder())
    
    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Wrap environment for Stable Baselines3
    wrapped_env = DMControlWrapper(env)
    vec_env = DummyVecEnv([lambda: wrapped_env])

    # Create a unique run name
    run_name = f"PPO_{os.path.basename(save_dir)}"
    tensorboard_log = f"./tensorboard_logs/{run_name}"

    if load_path:
        print(f"Loading pre-trained model from {load_path}")
        model = PPO.load(
            load_path, 
            env=vec_env,
            tensorboard_log=tensorboard_log,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2
        )
    else:
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log=tensorboard_log
        )

    # Set up callbacks
    callbacks = [
        TensorboardCallback(),
        TrainingCallback(),
        CheckpointCallback(save_dir, checkpoint_freq)
    ]

    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=False if load_path else True
    )

    # Save the final model
    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    
    # Delete the last checkpoint after confirming final model was saved
    if os.path.exists(final_path + ".zip"):
        last_checkpoint = os.path.join(save_dir, f"model_{total_timesteps}steps")
        if os.path.exists(last_checkpoint + ".zip"):
            os.remove(last_checkpoint + ".zip")
            print(f"\nRemoved last checkpoint at {total_timesteps} steps")
    
    return model

def train_creature(env, save_path="trained_creature", total_timesteps=240_000, load_path=None):
    # Wrap environment for Stable Baselines3
    wrapped_env = DMControlWrapper(env)
    vec_env = DummyVecEnv([lambda: wrapped_env])

    # Create a unique run name based on whether it's initial or resumed training
    run_name = f"PPO_{os.path.basename(save_path)}"
    tensorboard_log = f"./tensorboard_logs/{run_name}"

    if load_path:
        print(f"Loading pre-trained model from {load_path}")
        # Load the pre-trained model
        model = PPO.load(
            load_path, 
            env=vec_env,
            tensorboard_log=tensorboard_log,
            # Set hyperparameters in the constructor
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2
        )
    else:
        # Initialize new PPO model with tensorboard logging
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log=tensorboard_log
        )

    # Train the model with both callbacks
    callbacks = [
        TensorboardCallback(),
        TrainingCallback()
    ]
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=False if load_path else True
    )

    # Save the trained model
    model.save(save_path)
    return model 