# Model Card

⚠️ **TRAINING INTERRUPTED** ⚠️

**Reason:** Training was manually interrupted by keyboard.

## Training Information
### Training Command
```bash
python main.py --training-phase rotation --n-updates 3648 --n-envs 192
```
### Details
- Start Time: 03:58:23 PM EST
- End Time: 04:24:46 PM EST
- Duration: 0h 26m 22s
- Previous Updates: 0 (0 vectorized timesteps)
- Training Updates: 3648 (3735552 vectorized timesteps)
- Total Updates: 3648 (3735552 vectorized timesteps)
- Parallel Environments: 192
- Total Environment Interactions: 717225984
- Training Status: **INTERRUPTED** after 3648 updates
- Final Model Path: `trained_creatures/20250421__pm_03_58_24/final_model_3648updates.zip`
- Previous Model Path: N/A
- TensorBoard Log: `tensorboard_logs/20250421__pm_03_58_24_0`

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
- Final Reward: 0.420
- Best Reward: 0.497
- Worst Reward: -0.153

### Other Metrics
- train/learning_rate: 0.000

## Model Architecture
### Overview
- Algorithm: Proximal Policy Optimization (PPO)
- Policy Network: ActorCriticPolicy
- Input Shape: (44,)
- Output Shape: (8,)
- Activation Function: Tanh

### Detailed MLP Architecture
- Policy Network:
  - Layer 1: Linear(in_features=44, out_features=2048, bias=True)
  - Layer 2: Tanh()
  - Layer 3: Linear(in_features=2048, out_features=2048, bias=True)
  - Layer 4: Tanh()
  - Layer 5: Linear(in_features=2048, out_features=1024, bias=True)
  - Layer 6: Tanh()
  - Layer 7: Linear(in_features=1024, out_features=512, bias=True)
  - Layer 8: Tanh()
  - Layer 9: Linear(in_features=512, out_features=256, bias=True)
  - Layer 10: Tanh()
- Value Network:
  - Layer 1: Linear(in_features=44, out_features=2048, bias=True)
  - Layer 2: Tanh()
  - Layer 3: Linear(in_features=2048, out_features=1536, bias=True)
  - Layer 4: Tanh()
  - Layer 5: Linear(in_features=1536, out_features=1024, bias=True)
  - Layer 6: Tanh()
  - Layer 7: Linear(in_features=1024, out_features=512, bias=True)
  - Layer 8: Tanh()

## Model Hyperparameters
- learning_rate: 0.0003
- n_steps: 1024
- batch_size: 24576
- n_epochs: 20
- gamma: 0.99
- gae_lambda: 0.95
- clip_range: 0.2
- policy_kwargs:
  - net_arch: [{'pi': [2048, 2048, 1024, 512, 256], 'vf': [2048, 1536, 1024, 512]}]
  - activation_fn: <class 'torch.nn.modules.activation.Tanh'>

## Environment Parameters
### General Parameters:
- Time Limit: 25.60 seconds
- Pitch Size: 40x30
- Walker Contacts: Enabled
- Field Box: Disabled
- Goal Termination: Enabled

### Motor Control Parameters:
- Control Range: -1.0 1.0
```xml
<actuator>
        <motor name="motor0_to_1" joint="seg0_to_1" gear="1111.6875" />
        <motor name="motor0_to_2" joint="seg0_to_2" gear="800.0" />
        <motor name="motor2_to_3" joint="seg2_to_3" gear="422.959" />
        <motor name="motor0_to_4" joint="seg0_to_4" gear="55642.824" />
        <motor name="motor4_to_5" joint="seg4_to_5" gear="29418.2855" />
        <motor name="motor0_to_6" joint="seg0_to_6" gear="103984.20" />
        <motor name="motor6_to_7" joint="seg6_to_7" gear="54976.289" />
        <motor name="motor0_to_8" joint="seg0_to_8" gear="3385.1395" />
    </actuator>

    
```

### Physics Parameters:
- Ground Friction: 1 0.5 0.5
- Joint Damping: 1
- Joint Stiffness: 1
- Body Density: 50.0

## Version Control
- Branch: `give-good-observations`
- Commit: `3d37dd9`
- Date: Mon Apr 21 19:58:19 2025
- Message: hyperparameter testing

## Other Notes
<!-- Add any interesting observations about this training run here -->
