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


# Make a wrapper class for the agents

class PBT_Agent(object): # Larger class to also contain rating
    def __init__(self, soul_agent, agent_rating):
        self.agent = soul_agent
        self.rating = agent_rating
    
    def update_rating(self, new_rating):
        self.rating = new_rating

# Helper functions:
def pbt_init():
    pass

def training_match():
    # Do Training Matches
    # Use Retrace_SVGO to update parameters

    pass

    return match_results

def retrace_SVGO_ours():
    pass

def update_rating(PBT_agent_i, PBT_agent_j, score_i, score_j, elo_k):
    # agent_i and agent_j are PBT_Agents

    s = (np.sign(score_i-score_j) + 1)/2
    s_elo = 1/(1 + 10**(agent_j.rating-agent_i,rating)/400)
    agent_i.rating = agent_i.rating + K*(s-s_elo)
    agent_j.rating = agent_j.rating - K*(s-s_elo)

def is_eligible(PBT_agent):
    # processed 2 x 10^9 frames of learning since beginning of training
    # processed 4 x 10^8 frames of learning since the last time it was eligible for evolution

    # Frames time setp how? 

    # WIP

    eligible = True;

    return eligible

def is_parent(PBT_agent):
    # processed 4 x 10^8 frames of learning since it last evolved
    # WIP

    parent = True;

    return parent

def select_agents(Agents, PBT_agent):

    T_select = 10

    # Create new set without PBT_agent
    new_Agents = []
    for agent in Agents:
        if agent != PBT_agent:
            new_Agents.append(agent)
    
    random_agent_index = np.random.randint(len(new_Agents))
    agent_j = new_Agents[random_agent_index]
    s_elo = 1/(1 + 10**(agent_j.rating-agent_i,rating)/400)

    if is_parent(agent_j) and s_elo < T_select:
        return agent_j
    else:
        return NULL

def inherit_agent():
    pass

def mutate_hyperparameter():
    pass


# Training:
def PBT_MARL(Agents, N, number_of_steps): 
    # Agents are the actual agents, 
    # N is the number of Agents
    # Number of steps is the number of PBT training steps


    # Initialize:
    agents_theta_i, agents_theta_h_i, agents_rating = pbt_init()

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

    

    



