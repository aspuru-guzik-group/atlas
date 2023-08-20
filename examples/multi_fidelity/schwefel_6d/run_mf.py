#!/usr/bin/env python

import os, sys
import pickle
import numpy as np
import pandas as pd

from olympus.surfaces import Surface
from olympus.campaigns import Campaign, ParameterSpace 
from olympus.objects import ParameterDiscrete

from atlas.planners.multi_fidelity.planner import MultiFidelityPlanner


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


# BUILD PARAMETER SPACE
param_space = ParameterSpace()
param_space.add(ParameterDiscrete(name='s', options=[0.1, 1.]))
for param in surface.param_space:
    param_space.add(param)


# BEGIN EXPERIMENT


all_data = []
for run_ix in range(NUM_RUNS):
    
    planner = MultiFidelityPlanner(
        goal='minimize',
        fidelity_params=0,
        fidelities=[0.1, 1.],
        acquisition_optimizer_kind='pymoo',
    )
    planner.set_param_space(param_space)
    
    campaign = Campaign()
    campaign.set_param_space(param_space)
    
    target_rec_measurements = []
    
    cumul_cost = []
    max_cumul_cost = 0.
    iter_ = 0
    while max_cumul_cost < BUDGET: 
        if iter_ % 4 == 0:
            planner.set_ask_fidelity(1.0)
        else:
            planner.set_ask_fidelity(0.1)
        samples = planner.recommend(campaign.observations)
        for sample in samples:

            measurement = measure(sample, sample.s)
            campaign.add_observation(sample, measurement)
            cumul_cost.append( compute_cost(campaign.observations) )
            max_cumul_cost = np.amax(cumul_cost)
            
            iter_ += 1
            
            if campaign.num_obs >= 5+1:
                # make a prediction on the target fidelity and measure greedy
                rec_sample = planner.recommend_target_fidelity(batch_size=1)[0]
                rec_measurement = measure(rec_sample, s=1.0)
                target_rec_measurements.append(rec_measurement)
                
            else:
                # just record the current measurement
                target_rec_measurements.append(measurement)
            
            
    # store results in dataframe
    s_col = campaign.observations.get_params()[:, 0]
    x0_col = campaign.observations.get_params()[:, 1]
    x1_col = campaign.observations.get_params()[:, 2]
    x2_col = campaign.observations.get_params()[:, 3]
    x3_col = campaign.observations.get_params()[:, 4]
    x4_col = campaign.observations.get_params()[:, 5]
    x5_col = campaign.observations.get_params()[:, 6]

    obj0 = np.array(target_rec_measurements) #campaign.observations.get_values()

    data = pd.DataFrame({
        'cumul_cost': np.array(cumul_cost),
        's': s_col,
        'x0': x0_col,
        'x2': x2_col,
        'x3': x3_col,
        'x4': x4_col,
        'x5': x5_col,
        'obj': target_rec_measurements,
    })
    all_data.append(data)
    pickle.dump(all_data, open('mf_cont_high_results.pkl', 'wb'))



