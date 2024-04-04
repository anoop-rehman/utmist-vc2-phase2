import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import BaseCallback

class RewardLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RewardLoggerCallback, self).__init__(verbose)
        self.rewards = []
        self.episode_rewards = []
        self.current_episode_reward = 0

    def _on_step(self) -> bool:
        self.current_episode_reward += np.sum(self.locals["rewards"])
        if any(self.locals["dones"]):
            self.episode_rewards.append(self.current_episode_reward)
            self.current_episode_reward = 0
        return True

    def _on_training_end(self) -> None:
        self.rewards = self.episode_rewards

# Parallel environments
vec_env = make_vec_env("Ant-v4", n_envs=4)

model = PPO("MlpPolicy", vec_env, verbose=1)
reward_logger = RewardLoggerCallback()
model.learn(total_timesteps=25000*10*10, callback=reward_logger) # 25000 timesteps = around 4 epochs
model.save("ppo_ant_vel_400epochs_3target_fullrewardshifted")

del model # remove to demonstrate saving and loading

plt.plot(reward_logger.episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Rewards Over Time')
plt.show()

model.save("ppo_ant_vel_400epochs_3target_fullrewardshifted")

obs = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = vec_env.step(action)
    vec_env.render("human")