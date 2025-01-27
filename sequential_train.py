from train import train_creature
from custom_soccer_env import create_soccer_env
from dm_control.locomotion.soccer.team import RGBA_BLUE
from creature import Creature

# Create environment
home_player = Creature("creature_configs/two_arm_rower_blueprint.xml", marker_rgba=RGBA_BLUE)
env = create_soccer_env(
    home_players=[home_player],
    away_players=[],
    time_limit=60.0,
    disable_walker_contacts=False,
    enable_field_box=True,
    terminate_on_goal=False
)

# Use a single tensorboard log directory for all batches
tensorboard_log = "./tensorboard_logs/PPO_sequential_training"

# Initial training
print("\nStarting batch 1/3...")
model = train_creature(
    env,
    save_dir="trained_creatures/batch_1",
    total_timesteps=5000,
    checkpoint_freq=5000,
    tensorboard_log=tensorboard_log
)

# Sequential training
for i in range(2):  # 2 more times for total of 3
    batch_num = i + 2
    prev_steps = (batch_num - 1) * 5000  # Previous total steps
    curr_steps = batch_num * 5000  # Current total steps
    last_model = f"trained_creatures/batch_{batch_num-1}/final_model_{prev_steps}_steps"
    print(f"\nStarting batch {batch_num}/3, loading from {last_model}")
    model = train_creature(
        env,
        save_dir=f"trained_creatures/batch_{batch_num}",
        total_timesteps=5000,
        checkpoint_freq=5000,
        load_path=last_model,
        start_timesteps=prev_steps,
        tensorboard_log=tensorboard_log
    ) 