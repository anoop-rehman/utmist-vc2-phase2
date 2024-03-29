import gym
from gym import spaces
import numpy as np
from dm_control.locomotion import soccer as dm_soccer
import matplotlib.pyplot as plt

class BoxHeadSoccerEnv(gym.Env):
    def __init__(self, team_size=2, time_limit=0.2, disable_walker_contacts=False, enable_field_box=True, terminate_on_goal=False):
        super(BoxHeadSoccerEnv, self).__init__()

        self.env = dm_soccer.load(
            team_size=team_size,
            time_limit=time_limit,
            disable_walker_contacts=disable_walker_contacts,
            enable_field_box=enable_field_box,
            terminate_on_goal=terminate_on_goal,
            walker_type=dm_soccer.WalkerType.BOXHEAD)

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.env.action_spec()[0].shape[0],), dtype=np.float32)
        
        # Assuming observation is a flattened vector (modify if it's not the case)
        obs_dim_per_agent = sum(np.prod(self.env.observation_spec()[0][key].shape) for key in self.env.observation_spec()[0].keys() if 'stats' not in key)
        total_obs_dim = obs_dim_per_agent * 4  # Adjust 4 to the number of agents if different
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_obs_dim,), dtype=np.float32)


    def step(self, action):
        time_step = self.env.step(action)
        obs = self._get_observation(time_step)
        # reward = time_step.reward or 0
        reward = np.sum([r.item() for r in time_step.reward])

        done = time_step.last()
        info = {}  # Placeholder for any additional info you might want to return
        truncated = False  # Assuming episodes are only done and not truncated, adjust as necessary

        # Debug: Print out the reward to ensure it's what you expect
        # print("Reward:", reward)

        return obs, reward, done, truncated, info


    def reset(self, **kwargs):
        time_step = self.env.reset()  # Assuming this calls dm_control's reset
        initial_observation = self._get_observation(time_step)
        # If you need to return an info dict, it would look something like this:
        return initial_observation, {}

    def _get_observation(self, time_step):
        # Implement a function to extract and possibly flatten the observation
        obs_list = []
        for agent_obs in time_step.observation:  # Iterate through each agent's observation
            agent_obs_concatenated = np.concatenate([agent_obs[key].flatten() for key in agent_obs.keys() if 'stats' not in key])
            obs_list.append(agent_obs_concatenated)
        obs = np.concatenate(obs_list)
        return obs

    def render(self, mode='human'):
        image = self.env.physics.render(height=480, width=640, camera_id=0)
        
        if mode == 'human':
            plt.imshow(image)
            plt.show()
        elif mode == 'rgb_array':
            return image