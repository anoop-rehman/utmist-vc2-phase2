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
    global flag
    
    if not flag:
        print(stats_over_time.items())
        flag = True
        
    '''
    stats_over_time:
        Is a dict of key value pairs, where the key is the name of the stat and the value is a list of values for each player.
        The first TEAM_SIZE values in the list are player i's velocity to ball for team1, the next TEAM_SIZE values are for player j's velocities to ball for team2.
        
        etc. for each of the other stats.
    '''   

    for ax, (key, values) in zip(axes.flat, stats_over_time.items()):
        ax.clear()  # Clear current axes to redraw
        
        #if (key == 'closest_velocity_to_ball' or ):

        dark_red = "#8c0303"
        medium_red = '#db1818'
        light_red = '#ff2e2e'

        dark_blue = '#11018c'
        medium_blue = '#2d18c9'
        light_blue = '#462eff'


        color_team1 = [dark_red, medium_red, light_red]
        color_team2 = [dark_blue, medium_blue, light_blue]

        # color_team2 = ['blue', 'green', 'cyan']
        
        # Plot for each player in home team
        for i in range(TEAM_SIZE):
            player_values_home = values[i::2*TEAM_SIZE]  # Extract values for each player in home team
            color = color_team1[i]
            ax.plot(player_values_home, label=f'Home Team, Player {i+1}', color=color)
        
        # Plot for each player in away team
        for i in range(TEAM_SIZE):
            color = color_team2[i]
            player_values_away = values[TEAM_SIZE + i::2*TEAM_SIZE]  # Extract values for each player in away team
            ax.plot(player_values_away, label=f'Away Team, Player {i+1}', color=color)
        
        ax.set_title(key)
        ax.set_xlabel('Timestep')
        ax.set_ylabel('Value')
        ax.legend()
    
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

plt.ioff()  # Turn off interactive mode
plt.show()