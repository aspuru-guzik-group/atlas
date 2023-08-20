#!/usr/bin/env python

import os, sys
import pickle
import numpy as np
import pandas as pd

from olympus.surfaces import Surface
from olympus.campaigns import Campaign, ParameterSpace 
from olympus.objects import ParameterDiscrete

from atlas.planners.gp.planner import GPPlanner


# CONFIG
SURFACE_KIND = 'Schwefel'
DIM = 6
NUM_RUNS = 50
BUDGET = 30


surface = Surface(kind=SURFACE_KIND, param_dim=DIM)


# HELPER FUNCTIONS
def compute_cost(observations):
    return np.sum(observations.get_params()[:, 0].astype(float))

def measure(params, s):
    x0 = params.param_0
    x1 = params.param_1
    x2 = params.param_2
    x3 = params.param_3
    x4 = params.param_4
    x5 = params.param_5
    if s == 1.:
        measurement = surface.run([x0,x1,x2,x3,x4,x5])[0][0] # high fidelity
    if s == 0.1:
        measurement = surface.run([x0,x1,x2,x3,x4,x5])[0][0] + 0.5 # low fidelity
    return measurement

# SET PARAM SPACE
param_space = surface.param_space

# BEGIN EXPERIMENT
all_data = []
for run_ix in range(NUM_RUNS):
    
    planner = GPPlanner(goal='minimize', acquisition_optimizer_kind='pymoo')
    planner.set_param_space(param_space)
    
    campaign = Campaign()
    campaign.set_param_space(param_space)
    
    cumul_cost = []
    while campaign.num_obs < BUDGET:
        samples = planner.recommend(campaign.observations)
        for sample in samples:
            measurement = measure(sample, s=1.0)
            campaign.add_observation(sample, measurement)
            cumul_cost.append(campaign.num_obs)
            
    # store results in dataframe
    s_col = np.ones(len(campaign.observations.get_values()))
    x0_col = campaign.observations.get_params()[:, 0]
    x1_col = campaign.observations.get_params()[:, 1]
    x2_col = campaign.observations.get_params()[:, 2]
    x3_col = campaign.observations.get_params()[:, 3]
    x4_col = campaign.observations.get_params()[:, 4]
    x5_col = campaign.observations.get_params()[:, 5]

    obj0_col = campaign.observations.get_values()

    data = pd.DataFrame({
        'cumul_cost': np.array(cumul_cost),
        's': s_col,
        'x0': x0_col,
        'x2': x2_col,
        'x3': x3_col,
        'x4': x4_col,
        'x5': x5_col,
        'obj': obj0_col,
    })
    all_data.append(data)
    pickle.dump(all_data, open('bo_cont_high_results.pkl', 'wb'))