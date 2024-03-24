import numpy as np
from dm_control.locomotion import soccer as dm_soccer
from dm_control import viewer
import matplotlib.pyplot as plt

# Instantiates a 2-vs-2 BOXHEAD soccer environment with episodes of 10 seconds
# each. Upon scoring, the environment reset player positions and the episode
# continues. In this example, players can physically block each other and the
# ball is trapped within an invisible box encapsulating the field.
env = dm_soccer.load(team_size=2,
                    #  time_limit=10.0,
                     time_limit=5.0,
                     disable_walker_contacts=False,
                     enable_field_box=True,
                     terminate_on_goal=False,
                     walker_type=dm_soccer.WalkerType.BOXHEAD)

# Initialize a dictionary to store stats for each player over time.
stats_over_time = {key: [] for key in env.observation_spec()[0].keys() if 'stats' in key}

# Function to update the plots with the new data
def update_plots(fig, axes, stats_over_time):
    for ax, (key, values) in zip(axes.flat, stats_over_time.items()):
        ax.clear()  # Clear current axes to redraw
        values = np.array(values)  # Ensure it's a numpy array for easier manipulation
        if values.ndim > 2:
            for dim in range(values.shape[2]):
                ax.plot(values[:, 0, dim], label=f'Dimension {dim+1}')
        else:
            ax.plot(values.squeeze())
        ax.set_title(key)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Value')
    plt.draw()
    plt.pause(0.01)  # Pause briefly to allow the plot to be updated

# Prepare the figure and axes for plotting
plt.ion()  # Turn on interactive plotting mode
fig, axes = plt.subplots(len(stats_over_time) // 2, 2, figsize=(20, 10))
if len(stats_over_time) % 2 != 0:
    fig.delaxes(axes.flatten()[-1])  # Remove the last ax if an odd number of plots

# Collect observations and update plots in real-time
action_specs = env.action_spec()
timestep = env.reset()
while not timestep.last():
    actions = np.random.uniform(-1.0, 1.0, size=(4, action_specs[0].shape[0]))  # Example random policy
    timestep = env.step(actions)
    # Update stats_over_time with new observations
    for key in stats_over_time.keys():
        for i in range(len(action_specs)):
            stats_over_time[key].append(timestep.observation[i][key])
    # Update plots with new data
    update_plots(fig, axes, stats_over_time)

plt.ioff()  # Turn off interactive mode
plt.show()