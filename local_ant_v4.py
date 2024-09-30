import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box

DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
    # "distance": 16.0,
}

class AntEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 20,
    }

    def __init__(
        self,
        # xml_file="ant.xml",
        xml_file="/Users/anooprehman/Documents/uoft/extracurricular/design_teams/utmist2/utmist-vc2-phase2/two_arm_rower.xml",
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            ctrl_cost_weight,
            use_contact_forces,
            contact_cost_weight,
            healthy_reward,
            terminate_when_unhealthy,
            healthy_z_range,
            contact_force_range,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._healthy_reward = healthy_reward
        self._terminate_when_unhealthy = terminate_when_unhealthy
        self._healthy_z_range = healthy_z_range
        self._contact_force_range = contact_force_range
        self._reset_noise_scale = reset_noise_scale
        self._use_contact_forces = use_contact_forces
        self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

        self.target_x_velocity = 3.0
        self.averaged_x_velocity = 0.0
        self.x_velocities = []

        # obs_shape = 27
        # if not exclude_current_positions_from_observation:
        #     obs_shape += 2
        # if use_contact_forces:
        #     obs_shape += 84
        # obs_shape += 2  # For target_x_velocity and averaged_x_velocity

        obs_shape = 16

        observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_shape,), dtype=np.float64
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            # 5,
            25,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    @property
    def healthy_reward(self):
        return (
            float(self.is_healthy or self._terminate_when_unhealthy)
            * self._healthy_reward
        )

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    @property
    def contact_forces(self):
        raw_contact_forces = self.data.cfrc_ext
        min_value, max_value = self._contact_force_range
        contact_forces = np.clip(raw_contact_forces, min_value, max_value)
        return contact_forces

    @property
    def contact_cost(self):
        contact_cost = self._contact_cost_weight * np.sum(
            np.square(self.contact_forces)
        )
        return contact_cost

    @property
    def is_healthy(self):
        state = self.state_vector()
        min_z, max_z = self._healthy_z_range
        is_healthy = np.isfinite(state).all() and min_z <= state[2] <= max_z
        return is_healthy

    @property
    def terminated(self):
        terminated = not self.is_healthy if self._terminate_when_unhealthy else False
        return terminated

    def step(self, action):
        xy_position_before = self.get_body_com("seg0")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("seg0")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        # Update velocities deque and calculate exponentially weighted moving average
        self.x_velocities.append(x_velocity)
        alpha = 0.1  # smoothing factor for EWMA
        if len(self.x_velocities) == 1:
            self.averaged_x_velocity = x_velocity
        else:
            self.averaged_x_velocity = alpha * x_velocity + (1 - alpha) * self.averaged_x_velocity

        # Scaled reward based on the difference from the target velocity
        velocity_diff = np.abs(self.averaged_x_velocity - self.target_x_velocity)
        forward_reward = 1.0 / (velocity_diff + 1.0)

        healthy_reward = self.healthy_reward

        rewards = forward_reward + healthy_reward

        costs = ctrl_cost = self.control_cost(action)

        terminated = self.terminated  # use the proper termination condition
        observation = self._get_obs()  # get the proper observation

        info = {
            "reward_forward": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "reward_survive": healthy_reward,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "forward_reward": forward_reward,
        }

        reward = rewards - costs
        print("--------------------")
        print('forward_reward:', info["reward_forward"])
        print('healthy_reward:', info["reward_survive"])
        print('ctrl_cost:', info["reward_ctrl"])
        print('reward:', reward)
        print("")
        print('target_x_velocity:', self.target_x_velocity)
        print('actual_x_velocity:', x_velocity)
        print('averaged_x_velocity:', self.averaged_x_velocity)
        print('len(x_velocities)', len(self.x_velocities))

        if False:  # placeholder for render mode check
            self.render()

        return observation, reward, terminated, False, info

    # def _get_obs(self):
    #     position = self.data.qpos.flat.copy()
    #     velocity = self.data.qvel.flat.copy()

    #     if self._exclude_current_positions_from_observation:
    #         position = position[2:]

    #     basic_obs = np.concatenate((position, velocity))

    #     if self._use_contact_forces:
    #         contact_force = self.contact_forces.flat.copy()
    #         return np.concatenate((basic_obs, contact_force, [self.target_x_velocity, self.averaged_x_velocity]))
    #     else:
    #         return np.concatenate((basic_obs, [self.target_x_velocity, self.averaged_x_velocity]))


    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        # print("Position shape:", position.shape, "Values:", position)
        # print("Velocity shape:", velocity.shape, "Values:", velocity)

        if self._exclude_current_positions_from_observation:
            position = position[2:]  # Adjust based on the model's qpos structure
            # print("Position shape after exclusion:", position.shape, "Values:", position)

        basic_obs = np.concatenate((position, velocity))
        # print("Basic observation shape:", basic_obs.shape, "Values:", basic_obs)

        if self._use_contact_forces:
            contact_force = self.contact_forces.flat.copy()
            # print("Contact force shape:", contact_force.shape, "Values:", contact_force)
            final_observation = np.concatenate((basic_obs, contact_force, [self.target_x_velocity, self.averaged_x_velocity]))
            # print("Final observation shape with contact forces:", final_observation.shape, "Values:", final_observation)
            return final_observation
        else:
            final_observation = np.concatenate((basic_obs, [self.target_x_velocity, self.averaged_x_velocity]))
            # print("Final observation shape without contact forces:", final_observation.shape, "Values:", final_observation)
            return final_observation




    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(low=noise_low, high=noise_high, size=self.model.nq)
        qvel = self.init_qvel + self._reset_noise_scale * self.np_random.standard_normal(self.model.nv)

        self.set_state(qpos, qvel)

        self.x_velocities = []  # Reset the velocities list
        self.averaged_x_velocity = 0.0  # Reset the averaged velocity

        observation = self._get_obs()
        return observation
