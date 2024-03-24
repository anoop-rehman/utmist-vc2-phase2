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
stats_over_time = {key: [] for key in env.observation_spec()[0].keys()}


# Function to generate random actions for all players.
def random_policy(time_step):
    actions = []
    action_specs = env.action_spec()
    for action_spec in action_specs:
        action = np.random.uniform(
            action_spec.minimum, action_spec.maximum, size=action_spec.shape)
        actions.append(action)
    return actions

# Collect observations and store them
def collect_and_store_observations():
    action_specs = env.action_spec()
    timestep = env.reset()
    while not timestep.last():
        actions = random_policy(action_specs)
        timestep = env.step(actions)
        # For each stat, append the current value to its list in the dictionary.
        for key in stats_over_time.keys():
            for i in range(len(action_specs)):  # Assuming you want stats for all players
                stats_over_time[key].append(timestep.observation[i][key])


# Commented out for now, uncomment to visualize
# viewer.launch(env, policy=lambda ts: random_policy(env.action_spec()))

viewer.launch(env, policy=random_policy)

collect_and_store_observations()

# Plotting the statistics over time
def plot_stats_over_time(stats_over_time):
    # Convert all lists to numpy arrays for easier manipulation.
    for key in stats_over_time:
        stats_over_time[key] = np.array(stats_over_time[key])

    # Determine the total number of plots needed.
    total_plots = sum(value.shape[2] if value.ndim > 2 else 1 for value in stats_over_time.values())

    # Choose a layout that fits all plots reasonably well.
    num_columns = 2
    num_rows = (total_plots + num_columns - 1) // num_columns  # Ceiling division

    plt.figure(figsize=(15, num_rows * 3))  # Adjust figure size based on number of rows.
    
    plot_counter = 1  # Counter to track the subplot position.
    for key, values in stats_over_time.items():
        if values.ndim > 2:  # Multi-dimensional data
            for dim in range(values.shape[2]):
                plt.subplot(num_rows, num_columns, plot_counter)
                plt.plot(values[:, 0, dim])
                plt.title(f'{key} Dimension {dim+1}')
                plt.xlabel('Timestep')
                plt.ylabel('Value')
                plot_counter += 1
        else:  # Single-dimensional data
            plt.subplot(num_rows, num_columns, plot_counter)
            plt.plot(values.squeeze())  # Squeeze in case of (N, 1) shape arrays.
            plt.title(key)
            plt.xlabel('Timestep')
            plt.ylabel('Value')
            plot_counter += 1

    plt.tight_layout()
    plt.savefig('stats_over_time.pdf')

plot_stats_over_time(stats_over_time)
