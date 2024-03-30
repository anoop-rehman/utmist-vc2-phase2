import numpy as np
from dm_control.locomotion import soccer as dm_soccer
from dm_control import viewer
import matplotlib.pyplot as plt

# Constants
TEAM_SIZE = 2

# Instantiates a 2-vs-2 BOXHEAD soccer environment with episodes of 10 seconds
# each. Upon scoring, the environment reset player positions and the episode
# continues. In this example, players can physically block each other and the
# ball is trapped within an invisible box encapsulating the field.
env = dm_soccer.load(team_size=TEAM_SIZE,
                    #  time_limit=10.0,
                     time_limit=5.0,
                     disable_walker_contacts=False,
                     enable_field_box=True,
                     terminate_on_goal=False,
                     walker_type=dm_soccer.WalkerType.BOXHEAD)

# Initialize a dictionary to store stats for each player over time.
stats_over_time = {key: [] for key in env.observation_spec()[0].keys() if 'stats' in key for player in range(TEAM_SIZE)}

# Function to update the plots with the new data

flag = False
def update_plots(fig, axes, stats_over_time):

    team1_velocities = []
    team2_velocities = []
    global flag

    if not flag:
        print(stats_over_time.items())

    '''
    stats_over_time:
        Is a dict of key value pairs, where the key is the name of the stat and the value is a list of values for each player.
        The first TEAM_SIZE values in the list are player i's velocity to ball for team1, the next TEAM_SIZE values are for player j's velocities to ball for team2.
        
        etc. for each of the other stats.
    '''    
    # Home Home, Away Away
    for i in range(len(stats_over_time['stats_vel_to_ball'])):
        if i < TEAM_SIZE:
            team1_velocities.append(stats_over_time['stats_vel_to_ball'][i])
        else:
            team2_velocities.append(stats_over_time['stats_vel_to_ball'][i])
    if not flag:
        flag = True
        print(team1_velocities)
        print(team2_velocities)
    # for i in range(0, )
    for ax, (key, values) in zip(axes.flat, stats_over_time.items()):
        ax.clear()  # Clear current axes to redraw
        values = np.array(values)  # Ensure it's a numpy array for easier manipulation
        num_players = TEAM_SIZE
        num_data_points = len(values)
        num_values_per_player = num_data_points // num_players
        for i in range(num_players):
            player_values = values[i::num_players]  # Extract values for each player
            ax.plot(player_values, label=f'Player {i+1}')
        ax.set_title(key)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Value')
        ax.legend()
    # plt.draw()
    # plt.pause(0.01)  # Pause briefly to allow the plot to be updated

# Prepare the figure and axes for plotting
# plt.ion()  # Turn on interactive plotting mode
fig, axes = plt.subplots(len(stats_over_time) // 2, 2, figsize=(20, 10))
if len(stats_over_time) % 2 != 0:
    fig.delaxes(axes.flatten()[-1])  # Remove the last ax if an odd number of plots

# Collect observations and update plots in real-time
action_specs = env.action_spec()
timestep = env.reset()
time = 0
while not timestep.last():
    actions = np.random.uniform(-1.0, 1.0, size=(4, action_specs[0].shape[0]))  # Example random policy
    timestep = env.step(actions)
    if time % TEAM_SIZE == 0:
        # Update stats_over_time with new observations
        for key in stats_over_time.keys():
            for i in range(len(action_specs)):
                    stats_over_time[key].append(timestep.observation[i][key])
        # Update plots with new data
        update_plots(fig, axes, stats_over_time)
    time += 1


# plt.ioff()  # Turn off interactive mode
# plt.show()