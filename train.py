import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import spaces
import os

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
        reset_num_timesteps=False  # This ensures the timestep counting continues from the loaded model
    )

    # Save the trained model
    model.save(save_path)
    return model 