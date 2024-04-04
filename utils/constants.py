# Ignoring the following observations gives a 93-dimension environment observation space
inputs_to_ignore = [
    'body_height', 
    'end_effectors_pos', 
    'joints_pos', 
    'joints_vel', 
    'prev_action',  
    'world_zaxis', 
    'ball_ego_angular_velocity',  
    'teammate_0_end_effectors_pos',
    'stats_vel_to_ball', 
    'stats_closest_vel_to_ball', 
    'stats_veloc_forward', 
    'stats_vel_ball_to_goal', 
    'stats_home_avg_teammate_dist',
    'stats_teammate_spread_out', 
    'stats_home_score', 
    'stats_away_score'
    ]

# Eye-balled hyperparameters from graphs of paper
ACTOR_LR = 0.000265
CRITIC_LR = 0.000135
DISCOUNT_FACTOR = 0.995
ENTROPY_REGULARIZER_COST = 0.00085
