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

def generate_model_card(model, save_dir, start_time, end_time, start_timesteps, total_timesteps, tensorboard_log=None, checkpoint_freq=4000, keep_checkpoints=False, checkpoint_stride=1, load_path=None):
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
    """
    card_path = os.path.join(save_dir, "model_card.md")
    
    # Calculate actual training duration
    duration = end_time - start_time
    hours = duration.seconds // 3600
    minutes = (duration.seconds % 3600) // 60
    seconds = duration.seconds % 60
    
    with open(card_path, "w") as f:
        f.write("# Model Card\n\n")
        
        # Training Information
        f.write("## Training Information\n")
        
        # Training Command in its own subsection
        f.write("### Training Command\n")
        command = f"python main.py --timesteps {total_timesteps}"
        if checkpoint_freq != 4000:  # Only add if different from default
            command += f" --checkpoint-freq {checkpoint_freq}"
        if keep_checkpoints:
            command += " --keep-checkpoints"
        if checkpoint_stride != 1:  # Only add if different from default
            command += f" --checkpoint-stride {checkpoint_stride}"
        if tensorboard_log:
            command += f" --tensorboard-log {tensorboard_log}"
        if load_path:
            command += f" --load-path {load_path}"
        if start_timesteps:
            command += f" --start-timesteps {start_timesteps}"
        f.write(f"```bash\n{command}\n```\n")
        
        # Rest of training info
        f.write("### Details\n")
        f.write(f"- Start Time: {start_time.strftime('%I:%M:%S %p')} EST\n")
        f.write(f"- End Time: {end_time.strftime('%I:%M:%S %p')} EST\n")
        f.write(f"- Duration: {hours}h {minutes}m {seconds}s\n")
        f.write(f"- Previous Steps (steps already trained before this session): {start_timesteps}\n")
        f.write(f"- Training Steps (new steps trained in this session): {total_timesteps}\n")
        f.write(f"- Total Steps (cumulative steps after training): {start_timesteps + total_timesteps}\n")
        final_model_path = os.path.join(save_dir, f'final_model_{start_timesteps + total_timesteps}_steps.zip')
        f.write(f"- Final Model Path: `{final_model_path}`\n")
        if start_timesteps > 0:
            if load_path:
                f.write(f"- Previous Model Path: `{load_path}`\n")
            else:
                f.write("- Previous Model Path: Not found\n")
        else:
            f.write("- Previous Model Path: N/A\n")
        if tensorboard_log:
            f.write(f"- TensorBoard Log: {tensorboard_log}\n")
        
        # Reward Function
        reward_func = get_reward_function_text()
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
            rewards = [v for k, v in model.logger.name_to_value.items() if 'reward' in k.lower()]
            if rewards:
                final_reward = rewards[-1]
                best_reward = max(rewards)
                worst_reward = min(rewards)
                f.write(f"- Final Reward: {final_reward:.3f}\n")
                f.write(f"- Best Reward: {best_reward:.3f}\n")
                f.write(f"- Worst Reward: {worst_reward:.3f}\n")
            
            # Other metrics subsection
            f.write("\n### Other Metrics\n")
            for key, value in model.logger.name_to_value.items():
                if 'reward' not in key.lower():
                    f.write(f"- {key}: {value:.3f}\n")
        
        # Model Architecture
        f.write("\n## Model Architecture\n")
        f.write("- Algorithm: Proximal Policy Optimization (PPO)\n")
        policy = model.policy
        f.write(f"- Policy Network: {type(policy).__name__}\n")
        f.write(f"- Input Shape: {policy.observation_space.shape}\n")
        f.write(f"- Output Shape: {policy.action_space.shape}\n")
        f.write(f"- Activation Function: {policy.activation_fn.__name__}\n")
        
        # MLP Architecture
        f.write("\n### MLP Architecture\n")
        f.write("- Policy Network:\n")
        for i, layer in enumerate(policy.mlp_extractor.policy_net):
            f.write(f"  - Layer {i+1}: {layer}\n")
        f.write("- Value Network:\n")
        for i, layer in enumerate(policy.mlp_extractor.value_net):
            f.write(f"  - Layer {i+1}: {layer}\n")
        
        # Hyperparameters
        f.write("\n## Hyperparameters\n")
        from train import PPO_PARAMS
        for param, value in PPO_PARAMS.items():
            f.write(f"- {param}: {value}\n")
    
    print(f"\nGenerated model card at {card_path}")
    return card_path 