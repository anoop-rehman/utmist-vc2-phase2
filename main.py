import numpy as np
from dm_control.locomotion import soccer as dm_soccer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio

# Load the environment
env = dm_soccer.load(team_size=2, time_limit=2.0, disable_walker_contacts=False,
                     enable_field_box=True, terminate_on_goal=False,
                     walker_type=dm_soccer.WalkerType.BOXHEAD)

# Initialize a list to hold the captured viewer frames
viewer_frames = []

# Initialize a dictionary to store stats for each player over time
stats_over_time = {key: [] for key in env.observation_spec()[0].keys() if 'stats' in key}

# Prepare the figure and axes for plotting
fig, axes = plt.subplots(max(len(stats_over_time) // 2, 1), 2, figsize=(20, 10))
if len(stats_over_time) % 2 != 0:
    fig.delaxes(axes.flatten()[-1])  # Remove the last ax if an odd number of plots

# Function to update the plots with the new data
def update_plots(frame):
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

# Function to step the environment and collect data
def step_env():
 
    time_step = env.reset()
    for i in range(200):  # Number of steps to simulate
        actions = np.random.uniform(-1.0, 1.0, size=(4, env.action_spec()[0].shape[0]))  # Example random policy
        timestep = env.step(actions)
        print(f'Step {i}')

        # Update stats_over_time with new observations
        for key in stats_over_time.keys():
            stats_over_time[key].append([timestep.observation[i][key] for i in range(len(env.action_spec()))])

        # Render the current frame and append it to the list of frames
        viewer_frame = env.physics.render(camera_id=0, width=480, height=360)
        viewer_frames.append(viewer_frame)

        # if time_step.last():
        #     env.reset()

# Call step_env to start simulation and data collection
step_env()

# Create and save the animation of the plots
ani = FuncAnimation(fig, update_plots, frames=np.arange(200), blit=False)
ani.save('simulation.gif', writer='pillow', fps=5)
plt.close(fig)  # Close the figure

# Path to save the viewer output GIF
viewer_output_gif_path = 'viewer_output.gif'
# Save the viewer frames as a GIF after all frames have been collected
imageio.mimsave(viewer_output_gif_path, viewer_frames, fps=5)