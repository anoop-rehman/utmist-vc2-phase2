import numpy as np
from dm_control.locomotion import soccer as dm_soccer
from dm_control import viewer

# Implement PBT:
# 1) Initialize N Independant Agents from a population


# 2) Initialize agent paramemter: theta_i
#    Initialize agent rating: rating_i
#    Sample initial hyperparatmer theta_i_h from the initial hyper_parameter distribution

# 3) For loop: Agents play TrainingMatches 
# Update network parameters by "Retrace_SVGO"
# -> For match result in Training match
# -> -> UpdateRating(ri, rj, si, sj)

# 4) For each agent
# If A_j eligible:
# -> from Agetns Select(Ai, Aj)
# -> -> if Aj != NULL
# -> -> -> Ai inheritis from Aj INHERIT()
# -> -> -> -> -> theta_i_h Mutate(theta_i_h)

class PBT_Agent(object):
    def __init__(self, soul_agent, agent_rating):
        self.agent = soul_agent
        self.rating = agent_rating
        self.lr = 0  # Initialize lr to 0, you can modify this as needed

    def update_rating(self, new_rating):
        self.rating = new_rating



###-----------------------------###
# 0) Initialize hyperparameters ###
###-----------------------------###


'''
NOTE: dm_soccer.lead does the following:
    - a) Constructs a `team_size`-vs-`team_size` soccer environment.
'''
binomial_n = 1
team_size = 2
N = team_size*2 # population size
R_init = 100 # initial rating of the agents
inherit_prob = 0.5
perturb_prob = 0.1
hyperparameters_range = {"lr": [1e-7, 1e-1], 
                         "gamma": [0.9, 0.999]}


# Create the N agents


###---------------------------------------------------###
### 1) Initiate 4 independent agaents (aka box heads) ###
###---------------------------------------------------###
'''
NOTE: dm_soccer.lead does the following:
    - a) Constructs a `team_size`-vs-`team_size` soccer environment.
    - QUESTION: how do we assign the hyperparameters to the agents???
        - SOLUTION: we won't lmao, just store in a global "hyperparameter"
'''

env = dm_soccer.load(team_size=team_size,                       # 2 v 2
                     time_limit=10.0,                   # 10 second duration of episodes
                     disable_walker_contacts=False,     # False: disable physical contact between walkers
                     enable_field_box=True,             # True: enable physical bounding box for the ball (not players)
                     terminate_on_goal=False,           # False: continous gameplay across scoring events
                     walker_type=dm_soccer.WalkerType.BOXHEAD)  # Type of walker

###----------------------------------------------------------###
### 2) intialize agent parameters (rating and learning rate) ###
###----------------------------------------------------------###

def init_hyperparameters_and_ratings(N, hyperparameters_range, R_init):
    '''
    - Select sample hyper-parameter from the hyperparameter range
    - Set initial agent rating as R_init
    - NOTE: N is the population size
    - NOTE: this will have to be further implemented into the other code later
    '''
    #agent_info = []
    agent_lr = []
    agent_gamma = []
    agent_rating = []

    for i in range(N):
        agent_number = str(i)
        lr = np.random.uniform(low=hyperparameters_range["lr"][0],  high=hyperparameters_range["lr"][1], size = None)
        gamma = np.random.uniform(low=hyperparameters_range["gamma"][0], high=hyperparameters_range["gamma"][1], size=None)
        #agent_info.append({"agent":agent_number, "rating":R_init, "lr": lr, "gamma": gamma})
        agent_lr.append(lr)
        agent_gamma.append(gamma)
        agent_rating.append(R_init)
        

    print("TODO: set these variables to the correct place")
    return agent_lr, agent_gamma, agent_rating

agent_gamma, agent_lr, agent_rating = init_hyperparameters_and_ratings(N, hyperparameters_range, R_init)

uwu = 4
###----------------------###
### 3) Training For loop ###
###----------------------###



'''
def PBT_MARL(Agents, N, hyperparameters_range, R_init number_of_steps): 
    # Agents are the actual agents, 
    # N is the number of Agents
    # Number of steps is the number of PBT training steps


    # Initialize:
    agents_theta_i, agents_theta_h_i, agents_rating = init_hyperparameters_and_ratings(N, hyperparameters_range, R_init)

    # Other variables
    elo_k = 10

    # Run training matches
    num_of_training_matches = 100
    match_results = training_match(num_of_training_matches)

    while i in range(number_of_steps):
        for match_result in match_results:
            UpdateRating(agent_rating_i, agent_rating_j, score_i, score_j)
        
        for agent_i in Agents():
            if is_eligible(agent):
                agent_j = Select(Agents, agent)
                if agent_j != NULL:
                    inherit_agent(agent_i, agent_j)
                    agents_theta_h_i =  mutate_hyperparameter(agents_theta_h_i)

    return
'''
    

'''
# 2) Initilize agentn parameters
'''






# # Retrieves action_specs for all 4 players.
# action_specs = env.action_spec()

# # Step through the environment for one episode with random actions.
# timestep = env.reset()
# while not timestep.last():
#   actions = []
#   for action_spec in action_specs:
#     action = np.random.uniform(
#         action_spec.minimum, action_spec.maximum, size=action_spec.shape)
#     actions.append(action)
#   timestep = env.step(actions)

#   for i in range(len(action_specs)):
#     print(
#         "Player {}: reward = {}, discount = {}, observations = {}.".format(
#             i, timestep.reward[i], timestep.discount, timestep.observation[i]))

# Function to generate random actions for all players.
def random_policy(time_step):
    actions = []
    action_specs = env.action_spec()
    for action_spec in action_specs:
        action = np.random.uniform(
            action_spec.minimum, action_spec.maximum, size=action_spec.shape)
        actions.append(action)
    return actions

# Use the viewer to visualize the environment with the random policy.
viewer.launch(env, policy=random_policy)
