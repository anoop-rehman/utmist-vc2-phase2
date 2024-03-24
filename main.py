import numpy as np
from dm_control.locomotion import soccer as dm_soccer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load the environment
env = dm_soccer.load(team_size=2, time_limit=5.0, disable_walker_contacts=False,
                     enable_field_box=True, terminate_on_goal=False,
                     walker_type=dm_soccer.WalkerType.BOXHEAD)

# Initialize a dictionary to store stats for each player over time
stats_over_time = {key: [] for key in env.observation_spec()[0].keys() if 'stats' in key}

# Function to step the environment and collect data
def step_env():
    actions = np.random.uniform(-1.0, 1.0, size=(4, env.action_spec()[0].shape[0]))  # Example random policy
    timestep = env.step(actions)
    # Update stats_over_time with new observations
    for key in stats_over_time.keys():
        stats_over_time[key].append([timestep.observation[i][key] for i in range(len(env.action_spec()))])

# Prepare the figure and axes for plotting
fig, axes = plt.subplots(len(stats_over_time) // 2, 2, figsize=(20, 10))
if len(stats_over_time) % 2 != 0:
    fig.delaxes(axes.flatten()[-1])  # Remove the last ax if an odd number of plots

def init():
    for ax in axes.flat:
        ax.clear()  # Clear current axes
    return axes.flat

# Update function for the animation
def update(frame):
    step_env()  # Step the environment and update stats
    for ax, (key, values) in zip(axes.flat, stats_over_time.items()):
        ax.clear()  # Clear current axes to redraw
        values = np.array(values)  # Ensure it's a numpy array for easier manipulation
        if values.ndim > 2 and values.size > 0:
            for dim in range(values.shape[2]):
                ax.plot(values[:, 0, dim], label=f'Dimension {dim+1}')
        elif values.size > 0:
            ax.plot(values.squeeze())
        ax.set_title(key)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Value')
    return axes.flat

# Number of steps to simulate
num_frames = 200

ani = FuncAnimation(fig, update, frames=range(num_frames), init_func=init, blit=False)

# Save the animation
ani.save('simulation.gif', writer='pillow', fps=20)

plt.close(fig)  # Close the figure to prevent it from displaying after saving
