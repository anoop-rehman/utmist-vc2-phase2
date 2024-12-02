import numpy as np
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from gym import spaces

class DMControlWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        
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
        
        # reward = float(timestep.reward[0]) if timestep.reward is not None else 0.0
        reward = timestep.observation[0]['stats_vel_to_ball'][0]
        # print(timestep.observation[0]['stats_vel_to_ball'])
        done = timestep.last()
        info = {}  # Add any additional info you want to track
        
        print("train reward:", reward)
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

def train_creature(env, save_path="trained_creature", total_timesteps=1_000_000):
    # Wrap environment for Stable Baselines3
    wrapped_env = DMControlWrapper(env)
    vec_env = DummyVecEnv([lambda: wrapped_env])

    # Initialize PPO model
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
        clip_range=0.2
    )

    # Train the model
    callback = TrainingCallback()
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback
    )

    # Save the trained model
    model.save(save_path)
    return model 