# Model Card

⚠️ **TRAINING INTERRUPTED** ⚠️

This training run was interrupted before completion. The model may not be fully trained.

## Training Information
### Training Command
```bash
python main.py --training-phase rotation --n-updates 71 --load-path trained_creatures/20250417__pm_04_32_30/final_model_116updates.zip --start-timesteps 950853
```
### Details
- Start Time: 07:13:32 PM EST
- End Time: 09:40:22 PM EST
- Duration: 2h 26m 49s
- Previous Updates: 116 (950853 env timesteps)
- Training Updates: 71 (589249 env timesteps)
- Total Updates: 188 (1540102 env timesteps)
- Training Status: **INTERRUPTED** after 589249 steps
- Final Model Path: `trained_creatures/20250417__pm_07_13_33/final_model_188updates.zip`
- Previous Model Path: `trained_creatures/20250417__pm_04_32_30/final_model_116updates.zip`
- TensorBoard Log: `tensorboard_logs/20250417__pm_07_13_33_0`

## Reward Function
```python
    def step(self, action):
            timestep = self.env.step([action])
            
            # Get observation 
            obs = process_observation(timestep)
            
            # Initialize reward to default value
            alignment_reward = 0.0
            
            # Extract alignment directly from rotation matrix
            if 'absolute_root_mat' in timestep.observation[0]:
                # Get the rotation matrix
                rot_matrix = timestep.observation[0]['absolute_root_mat'].copy()
                
                # The x-component of the z-axis is directly at index 2 of the flattened matrix
                # This represents the alignment of the z-axis with the x-axis
                alignment_reward = float(rot_matrix[0, 2])
                
                # Print debug info periodically
                if hasattr(process_observation, "should_print") and process_observation.should_print:
                    print(f"Rotation reward: z-axis x-component = {alignment_reward:.3f}")
            else:
                print("absolute_root_mat not found in timestep.observation[0]!")
            
            # Use simple alignment reward directly
            reward = alignment_reward
            
            done = timestep.last()
            info = {}
    
            self.reward = reward
            self.last_vel_to_ball = alignment_reward  # For consistency with previous code
            return obs, reward, done, info
```

## Final Training Metrics
### Reward
- Final Reward: 0.986
- Best Reward: 1.000
- Worst Reward: -0.999

### Other Metrics

## Model Architecture
### Overview
- Algorithm: Proximal Policy Optimization (PPO)
- Policy Network: ActorCriticPolicy
- Input Shape: (44,)
- Output Shape: (8,)
- Activation Function: Tanh

### Detailed MLP Architecture
- Policy Network:
  - Layer 1: Linear(in_features=44, out_features=64, bias=True)
  - Layer 2: Tanh()
  - Layer 3: Linear(in_features=64, out_features=64, bias=True)
  - Layer 4: Tanh()
- Value Network:
  - Layer 1: Linear(in_features=44, out_features=64, bias=True)
  - Layer 2: Tanh()
  - Layer 3: Linear(in_features=64, out_features=64, bias=True)
  - Layer 4: Tanh()

## Model Hyperparameters
- learning_rate: 0.0003
- n_steps: 8192
- batch_size: 64
- n_epochs: 10
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2

## Environment Parameters
### General Parameters:
- Time Limit: 4.00 seconds
- Pitch Size: 40x30
- Walker Contacts: Enabled
- Field Box: Disabled
- Goal Termination: Enabled

### Motor Control Parameters:
- Control Range: -1.0 1.0
- Gear Ratio: 16000

### Physics Parameters:
- Ground Friction: 1 0.5 0.5
- Joint Damping: 1
- Joint Stiffness: 1
- Body Density: 50.0

## Other Notes
<!-- Add any interesting observations about this training run here -->
