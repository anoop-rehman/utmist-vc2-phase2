import inspect
import os
from datetime import datetime
import pytz
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import xml.etree.ElementTree as ET
from custom_soccer_env import create_soccer_env

def get_tensorboard_metrics(tensorboard_log, run_name):
    """Read metrics from tensorboard logs."""
    if not tensorboard_log or not run_name:
        return None
    
    # Get the latest run directory
    run_dir = os.path.join(tensorboard_log, f"{run_name}_0")
    if not os.path.exists(run_dir):
        return None
    
    # Load the events file
    event_acc = EventAccumulator(run_dir)
    event_acc.Reload()
    
    # Get reward metrics
    rewards = []
    if 'train/reward' in event_acc.Tags()['scalars']:
        rewards = [s.value for s in event_acc.Scalars('train/reward')]
    elif 'train/episode_reward_mean' in event_acc.Tags()['scalars']:  # Fallback to episode mean
        rewards = [s.value for s in event_acc.Scalars('train/episode_reward_mean')]
    
    if not rewards:
        return None
    
    return {
        'final_reward': rewards[-1],
        'best_reward': max(rewards),
        'worst_reward': min(rewards)
    }

def get_reward_function_text(training_phase="combined"):
    """Extract the appropriate reward function based on training phase."""
    with open("train.py", "r") as f:
        content = f.read()
    
    if training_phase == "combined":
        # Keep existing behavior - find calculate_reward function
        start = content.find("def calculate_reward")
        if start == -1:
            return "Reward function not found in train.py"
        
        # Get the function body by finding the next function definition or class
        end = content.find("\n\n", start)  # Look for double newline
        if end == -1:  # If not found, try finding next def or class
            end = min(x for x in [
                content.find("\ndef", start + 1),
                content.find("\nclass", start + 1),
                len(content)
            ] if x != -1)
        
        function_text = content[start:end].strip()
        
    elif training_phase == "rotation":
        # Find the step method in RotationPhaseWrapper
        rotation_class_start = content.find("class RotationPhaseWrapper")
        if rotation_class_start == -1:
            return "RotationPhaseWrapper not found in train.py"
            
        step_method_start = content.find("    def step(self, action):", rotation_class_start)
        if step_method_start == -1:
            return "step method not found in RotationPhaseWrapper"
            
        # Find the end of the step method by looking for the next method at the same indentation level
        step_method_end = content.find("\n    def ", step_method_start + 1)
        if step_method_end == -1:  # If not found, try finding the end of the class
            step_method_end = content.find("\nclass ", step_method_start + 1)
            if step_method_end == -1:  # If still not found, use the end of the file
                step_method_end = len(content)
                
        function_text = content[step_method_start:step_method_end].strip()
        
    elif training_phase == "walking":
        # Find the step method in WalkingPhaseWrapper
        walking_class_start = content.find("class WalkingPhaseWrapper")
        if walking_class_start == -1:
            return "WalkingPhaseWrapper not found in train.py"
            
        step_method_start = content.find("    def step(self, action):", walking_class_start)
        if step_method_start == -1:
            return "step method not found in WalkingPhaseWrapper"
            
        # Find the end of the step method by looking for the next method at the same indentation level
        step_method_end = content.find("\n    def ", step_method_start + 1)
        if step_method_end == -1:  # If not found, try finding the end of the class
            step_method_end = content.find("\nclass ", step_method_start + 1)
            if step_method_end == -1:  # If still not found, use the end of the file
                step_method_end = len(content)
                
        function_text = content[step_method_start:step_method_end].strip()
    else:
        return f"Unknown training phase: {training_phase}"
    
    # Format the function text for markdown
    lines = function_text.split("\n")
    formatted_lines = ["    " + line for line in lines]
    return "\n".join(formatted_lines)

def get_env_params():
    """Get environment parameters from source files."""
    # Read motor and physics params from XML
    tree = ET.parse('creature_configs/two_arm_rower_blueprint.xml')
    root = tree.getroot()
    
    # Get motor params
    motor = root.find('.//motor')
    control_range = motor.get('ctrlrange')
    gear_ratio = motor.get('gear')
    
    # Get physics params
    geom = root.find('.//geom')
    friction = geom.get('friction')
    joint = root.find('.//joint')
    damping = joint.get('damping')
    stiffness = joint.get('stiffness')
    density = geom.get('density')
    
    # Get environment params from create_soccer_env
    env_params = inspect.getsource(create_soccer_env)
    # Extract default values using string parsing
    import re
    time_limit = float(re.search(r'time_limit=(\d+\.?\d*)', env_params).group(1))
    disable_walker_contacts = re.search(r'disable_walker_contacts=(\w+)', env_params).group(1) == 'True'
    enable_field_box = re.search(r'enable_field_box=(\w+)', env_params).group(1) == 'True'
    terminate_on_goal = re.search(r'terminate_on_goal=(\w+)', env_params).group(1) == 'True'
    
    # Get pitch size from RandomizedPitch
    try:
        from dm_control.locomotion.soccer import pitch
        pitch_params = inspect.getsource(pitch.RandomizedPitch)
        min_size = re.search(r'min_size=\((\d+),\s*(\d+)\)', pitch_params)
        if min_size:
            pitch_size = (int(min_size.group(1)), int(min_size.group(2)))
        else:
            pitch_size = (40, 30)  # Default if not found
    except:
        pitch_size = (40, 30)  # Fallback default
    
    return {
        'time_limit': time_limit,
        'pitch_size': pitch_size,
        'walker_contacts': not disable_walker_contacts,
        'field_box': enable_field_box,
        'terminate_on_goal': terminate_on_goal,
        'control_range': control_range,
        'gear_ratio': gear_ratio,
        'friction': friction,
        'joint_damping': damping,
        'joint_stiffness': stiffness,
        'body_density': density
    }

def generate_model_card(model, save_dir, start_time, end_time, start_timesteps=0, total_timesteps=None, tensorboard_log=None, checkpoint_freq=None, keep_checkpoints=False, checkpoint_stride=1, load_path=None, interrupted=False, training_phase="combined"):
    """Generate a markdown file with model details.
    
    Args:
        model: The trained model
        save_dir: Directory where model is saved
        start_time: Actual training start time
        end_time: Actual training end time
        start_timesteps: Starting timestep count
        total_timesteps: Total timesteps trained for
        tensorboard_log: TensorBoard log directory
        checkpoint_freq: How often checkpoints were saved
        keep_checkpoints: Whether all checkpoints were kept
        checkpoint_stride: How many checkpoints were skipped between saves
        load_path: Path to the model loaded for continued training
        interrupted: Whether training was interrupted (e.g., by KeyboardInterrupt)
        training_phase: The training phase used ("combined", "walking", or "rotation")
    """
    from train import default_hyperparameters
    card_path = os.path.join(save_dir, "model_card.md")
    
    # Calculate actual training duration
    duration = end_time - start_time
    hours = duration.seconds // 3600
    minutes = (duration.seconds % 3600) // 60
    seconds = duration.seconds % 60
    
    # Convert times to EST
    eastern_tz = pytz.timezone('US/Eastern')
    
    # Convert start time to EST (handling both naive and aware datetime objects)
    if start_time.tzinfo is None:
        # If naive datetime, assume it's in local time and convert to EST
        local_tz = datetime.now().astimezone().tzinfo
        start_time_local = start_time.replace(tzinfo=local_tz)
        start_time_est = start_time_local.astimezone(eastern_tz)
    else:
        # If already timezone-aware, just convert to EST
        start_time_est = start_time.astimezone(eastern_tz)
        
    # Convert end time to EST (handling both naive and aware datetime objects)
    if end_time.tzinfo is None:
        # If naive datetime, assume it's in local time and convert to EST
        local_tz = datetime.now().astimezone().tzinfo
        end_time_local = end_time.replace(tzinfo=local_tz)
        end_time_est = end_time_local.astimezone(eastern_tz)
    else:
        # If already timezone-aware, just convert to EST
        end_time_est = end_time.astimezone(eastern_tz)
    
    with open(card_path, "w") as f:
        f.write("# Model Card\n\n")
        
        # Add interrupt warning if applicable
        if interrupted:
            f.write("⚠️ **TRAINING INTERRUPTED** ⚠️\n\n")
            f.write("This training run was interrupted before completion. The model may not be fully trained.\n\n")
        
        # Training Information
        f.write("## Training Information\n")
        
        # Training Command in its own subsection
        f.write("### Training Command\n")
        n_updates = total_timesteps // default_hyperparameters["n_steps"]  # Convert timesteps back to updates
        command = f"python main.py --training-phase {training_phase} --n-updates {n_updates}"
        if tensorboard_log and tensorboard_log != 'tensorboard_logs':  # Only include if not default
            command += f" --tensorboard-log {tensorboard_log}"
        if load_path:
            command += f" --load-path {load_path}"
        if start_timesteps:
            command += f" --start-timesteps {start_timesteps}"
        f.write(f"```bash\n{command}\n```\n")
        
        # Rest of training info
        f.write("### Details\n")
        f.write(f"- Start Time: {start_time_est.strftime('%I:%M:%S %p')} EST\n")
        f.write(f"- End Time: {end_time_est.strftime('%I:%M:%S %p')} EST\n")
        f.write(f"- Duration: {hours}h {minutes}m {seconds}s\n")
        # Add policy updates info
        prev_updates = start_timesteps // default_hyperparameters["n_steps"] if start_timesteps else 0
        
        # For total updates, calculate actual count based on total steps (not integer division)
        total_steps = start_timesteps + total_timesteps
        total_updates_actual = total_steps // default_hyperparameters["n_steps"]
        
        # Training updates is the difference between total and previous
        training_updates = total_updates_actual - prev_updates
        
        f.write(f"- Previous Updates: {prev_updates} ({start_timesteps} env timesteps)\n")
        f.write(f"- Training Updates: {training_updates} ({total_timesteps} env timesteps)\n")
        f.write(f"- Total Updates: {total_updates_actual} ({total_steps} env timesteps)\n")
        
        # Add completion status
        if interrupted:
            f.write(f"- Training Status: **INTERRUPTED** after {total_timesteps} steps\n")
        else:
            f.write(f"- Training Status: COMPLETED\n")
            
        final_model_path = os.path.join(save_dir, f'final_model_{total_updates_actual}updates.zip')
        f.write(f"- Final Model Path: `{final_model_path}`\n")
        if start_timesteps > 0:
            if load_path:
                f.write(f"- Previous Model Path: `{load_path}`\n")
            else:
                f.write("- Previous Model Path: Not found\n")
        else:
            f.write("- Previous Model Path: N/A\n")
        if tensorboard_log:
            # Get run name from save_dir
            run_name = os.path.basename(save_dir)
            f.write(f"- TensorBoard Log: `{os.path.join(tensorboard_log, f'{run_name}_0')}`\n")
        
        # Reward Function
        reward_func = get_reward_function_text(training_phase)
        if reward_func:
            f.write("\n## Reward Function\n")
            f.write("```python\n")
            f.write(reward_func)
            f.write("\n```\n")
        
        # Final Training Metrics with subsections
        if hasattr(model, 'logger') and model.logger is not None:
            f.write("\n## Final Training Metrics\n")
            
            # Reward metrics subsection
            f.write("### Reward\n")
            # Get run name from save_dir (format: YYYYMMDD__HH_MM_SSpm)
            run_name = os.path.basename(save_dir)
            reward_metrics = get_tensorboard_metrics(tensorboard_log, run_name)
            if reward_metrics:
                f.write(f"- Final Reward: {reward_metrics['final_reward']:.3f}\n")
                f.write(f"- Best Reward: {reward_metrics['best_reward']:.3f}\n")
                f.write(f"- Worst Reward: {reward_metrics['worst_reward']:.3f}\n")
            else:
                f.write("No reward metrics found in tensorboard logs.\n")
            
            # Other metrics subsection
            f.write("\n### Other Metrics\n")
            for key, value in model.logger.name_to_value.items():
                if 'reward' not in key.lower():
                    f.write(f"- {key}: {value:.3f}\n")
        
        # Model Architecture
        f.write("\n## Model Architecture\n")
        f.write("### Overview\n")
        f.write("- Algorithm: Proximal Policy Optimization (PPO)\n")
        policy = model.policy
        f.write(f"- Policy Network: {type(policy).__name__}\n")
        f.write(f"- Input Shape: {policy.observation_space.shape}\n")
        f.write(f"- Output Shape: {policy.action_space.shape}\n")
        f.write(f"- Activation Function: {policy.activation_fn.__name__}\n\n")
        
        # MLP Architecture
        f.write("### Detailed MLP Architecture\n")
        f.write("- Policy Network:\n")
        for i, layer in enumerate(policy.mlp_extractor.policy_net):
            f.write(f"  - Layer {i+1}: {layer}\n")
        f.write("- Value Network:\n")
        for i, layer in enumerate(policy.mlp_extractor.value_net):
            f.write(f"  - Layer {i+1}: {layer}\n")
        
        # Hyperparameters
        f.write("\n## Model Hyperparameters\n")
        for param, value in default_hyperparameters.items():
            if isinstance(value, dict):
                f.write(f"- {param}:\n")
                for k, v in value.items():
                    f.write(f"  - {k}: {v}\n")
            else:
                f.write(f"- {param}: {value}\n")
        
        # Environment Parameters
        f.write("\n## Environment Parameters\n")
        env_params = get_env_params()
        f.write("### General Parameters:\n")
        f.write(f"- Time Limit: {env_params['time_limit']:.2f} seconds\n")
        f.write(f"- Pitch Size: {env_params['pitch_size'][0]}x{env_params['pitch_size'][1]}\n")
        f.write(f"- Walker Contacts: {'Enabled' if env_params['walker_contacts'] else 'Disabled'}\n")
        f.write(f"- Field Box: {'Enabled' if env_params['field_box'] else 'Disabled'}\n")
        f.write(f"- Goal Termination: {'Enabled' if env_params['terminate_on_goal'] else 'Disabled'}\n\n")
        
        f.write("### Motor Control Parameters:\n")
        f.write(f"- Control Range: {env_params['control_range']}\n")
        f.write(f"- Gear Ratio: {env_params['gear_ratio']}\n\n")
        
        f.write("### Physics Parameters:\n")
        f.write(f"- Ground Friction: {env_params['friction']}\n")
        f.write(f"- Joint Damping: {env_params['joint_damping']}\n")
        f.write(f"- Joint Stiffness: {env_params['joint_stiffness']}\n")
        f.write(f"- Body Density: {env_params['body_density']}\n")
        
        # Other Notes section for manual additions
        f.write("\n## Other Notes\n")
        f.write("<!-- Add any interesting observations about this training run here -->\n")
    
    print(f"\nGenerated model card at {card_path}")
    return card_path 