from custom_soccer_env import create_soccer_env
from dm_control.locomotion.soccer.team import RGBA_BLUE, RGBA_RED
from creature import Creature
from train import train_creature, DMControlWrapper
from dm_control import viewer
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3 import PPO


# Create creature and environment
home_player = Creature("creature_configs/two_arm_rower_blueprint.xml", marker_rgba=RGBA_BLUE)

env = create_soccer_env(
    home_players=[home_player],
    away_players=[],
    time_limit=180.0,
    disable_walker_contacts=False,
    enable_field_box=True,
    terminate_on_goal=False,
    random_state=3
)

# Train the creature
# model = train_creature(env, save_path="trained_creatures/v1_2_1_1__480kTimesteps1")

# Load a trained model
wrapped_env = DMControlWrapper(env)
vec_env = DummyVecEnv([lambda: wrapped_env])
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
model.load("trained_creatures/v1_2_1_1__480kTimesteps")

vel_to_balls = []
averaged_vel_to_ball = 0.0

# Define a policy function for the viewer
def policy(time_step):
    global vel_to_balls
    global averaged_vel_to_ball

    # Convert observation to the format expected by the model
    obs_dict = time_step.observation[0]
    obs = np.concatenate([v.flatten() for v in obs_dict.values()])
    
    # Get action from model
    action, _states = model.predict(obs, deterministic=True)
    
    obs_dict = time_step.observation[0]
    obs = np.concatenate([v.flatten() for v in obs_dict.values()])
    
    vel_to_ball = time_step.observation[0]['stats_vel_to_ball'][0]

    vel_to_balls.append(vel_to_ball)
    alpha = 0.1  # smoothing factor for EWMA
    if len(vel_to_balls) == 1:
        averaged_vel_to_ball = vel_to_ball
    else:
        averaged_vel_to_ball = alpha * vel_to_ball + (1 - alpha) * averaged_vel_to_ball

    reward = averaged_vel_to_ball

    if (len(vel_to_balls) > 1400 and averaged_vel_to_ball < 0.1):
        reward -= 5.0
        

    done = time_step.last()
    info = {}  # Add any additional info you want to track
    
    print("-------------------------------")
    print("vel_to_ball:", vel_to_ball)
    print('self.averaged_vel_to_ball:', averaged_vel_to_ball)
    print('len(self.vel_to_balls)', len(vel_to_balls))
    print("test reward:", reward)



    # Return action wrapped in a list (for single agent)
    return [action]


# Launch the viewer
viewer.launch(env, policy=policy)
