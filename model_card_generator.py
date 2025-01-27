import inspect
import os
from datetime import datetime

def get_reward_function_text():
    """Extract the reward function implementation from train.py."""
    with open("train.py", "r") as f:
        content = f.read()
    
    # Find the calculate_reward function definition
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
    
    # Format the function text for markdown
    lines = function_text.split("\n")
    formatted_lines = ["    " + line for line in lines]
    return "\n".join(formatted_lines)

def generate_model_card(model, save_dir, start_time, end_time, start_timesteps, total_timesteps, load_path=None, tensorboard_log=None):
    """Generate a markdown file with model details."""
    card_path = os.path.join(save_dir, "model_card.md")
    
    # Get model architecture details
    policy = model.policy
    extractor = policy.mlp_extractor
    
    # Calculate training duration
    duration = end_time - start_time
    hours = duration.seconds // 3600
    minutes = (duration.seconds % 3600) // 60
    seconds = duration.seconds % 60
    
    with open(card_path, "w") as f:
        f.write("# Model Card\n\n")
        
        # Training Information
        f.write("## Training Information\n")
        f.write(f"- Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"- Duration: {hours}h {minutes}m {seconds}s\n")
        f.write(f"- Starting Timesteps: {start_timesteps}\n")
        f.write(f"- Training Timesteps: {total_timesteps}\n")
        f.write(f"- Total Timesteps: {start_timesteps + total_timesteps}\n")
        if load_path:
            f.write(f"- Loaded From: {load_path}\n")
        if tensorboard_log:
            f.write(f"- TensorBoard Log: {tensorboard_log}\n")
        f.write("\n")
        
        # Model Architecture
        f.write("## Model Architecture\n")
        f.write("### Policy Network\n")
        f.write(f"- Input Shape: {policy.observation_space.shape}\n")
        f.write(f"- Output Shape: {policy.action_space.shape}\n")
        f.write(f"- Network Type: {type(policy).__name__}\n")
        f.write(f"- Activation Function: {policy.activation_fn.__name__}\n\n")
        
        # MLP Architecture
        f.write("### MLP Architecture\n")
        f.write(f"- Policy Network: {extractor.policy_net}\n")
        f.write(f"- Value Network: {extractor.value_net}\n\n")
        
        # Hyperparameters
        f.write("## Hyperparameters\n")
        from train import PPO_PARAMS
        for param, value in PPO_PARAMS.items():
            f.write(f"- {param}: {value}\n")
        f.write("\n")
        
        # Training Metrics
        f.write("## Final Training Metrics\n")
        if hasattr(model, "logger") and model.logger is not None:
            for key, value in model.logger.name_to_value.items():
                f.write(f"- {key}: {value}\n")
        f.write("\n")
        
        # Environment Details
        f.write("## Environment Details\n")
        f.write("- Type: Custom Soccer Environment\n")
        f.write("- Description: A 3D physics-based environment where a creature learns to move and interact with a soccer ball.\n")
        f.write("- Observation Space: Joint angles, velocities, and ball position\n")
        f.write("- Action Space: Joint torques\n\n")
        
        # Reward Function
        f.write("## Reward Function\n")
        f.write("Implementation from train.py:\n")
        f.write("```python\n")
        f.write(get_reward_function_text())
        f.write("\n```\n\n")
        
        # Training Command
        f.write("## Training Command\n")
        command = "python main.py"
        if total_timesteps:
            command += f" --timesteps {total_timesteps}"
        if load_path:
            command += f" --load-path {load_path}"
        if tensorboard_log:
            command += f" --tensorboard-log {tensorboard_log}"
        f.write(f"```bash\n{command}\n```\n\n")
        
        # Additional Notes
        f.write("## Additional Notes\n")
        if isinstance(model, str):
            f.write("- Model Type: Unknown (loaded from path)\n")
        else:
            f.write(f"- Algorithm: {model.__class__.__name__}\n")
        if tensorboard_log:
            f.write(f"- TensorBoard: Enabled (logs at {tensorboard_log})\n")
        else:
            f.write("- TensorBoard: Disabled\n")
        f.write("- Checkpoints: Saved during training\n")
        
    print(f"\nGenerated model card at {card_path}")
    
    return model 