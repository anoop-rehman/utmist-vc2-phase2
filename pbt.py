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

# Helper functions:
def pbt_init():
    pass

def training_match():
    pass

def Retrace_SVGO_ours():
    pass

def update_rating():
    pass

def is_eligible():
    pass

def select_agents():
    pass

def inherit_agent():
    pass

def mutate_hyperparameter():
    pass


# Training:
