from custom_soccer_env import create_soccer_env
from dm_control.locomotion.soccer.team import RGBA_BLUE, RGBA_RED
from creature import Creature
from train import train_creature, DMControlWrapper
from dm_control import viewer
import numpy as np

# Create creature and environment
home_player = Creature("creature_configs/two_arm_rower_blueprint.xml", marker_rgba=RGBA_BLUE)

env = create_soccer_env(
    home_players=[home_player],
    away_players=[],
    time_limit=60.0,
    disable_walker_contacts=False,
    enable_field_box=True,
    terminate_on_goal=False
)

# Train the creature
model = train_creature(env)

# Define a policy function for the viewer
def policy(time_step):
    # Convert observation to the format expected by the model
    obs_dict = time_step.observation[0]
    obs = np.concatenate([v.flatten() for v in obs_dict.values()])
    
    # Get action from model
    action, _states = model.predict(obs, deterministic=True)
    
    # Return action wrapped in a list (for single agent)
    return [action]

# Launch the viewer
viewer.launch(env, policy=policy)
