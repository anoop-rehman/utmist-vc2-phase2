import numpy as np
from dm_control.locomotion import soccer as dm_soccer
from dm_control import viewer

#NOTE: this file is essentially the modified PBT Marl from ChuaCheowHuan
# https://github.com/ChuaCheowHuan/PBT_MARL_watered_down/blob/master/PBT_MARL.py


binomial_n = 1
team_size = 2
N = team_size*2 # population size
R_init = 100 # initial rating of the agents
inherit_prob = 0.5
perturb_prob = 0.1
hyperparameters_range = {"lr": [1e-7, 1e-1], 
                         "gamma": [0.9, 0.999]}




class PBT_MARL:
    def __init__(self, population_size, K, T_select)