#!/usr/bin/env python

import numpy as np
import pandas as pd


from olympus.datasets import Dataset
from olympus.objects import (
    ParameterContinuous,
    ParameterDiscrete, 
    ParameterCategorical, 
    ParameterVector
)
from olympus.campaigns import ParameterSpace, Campaign

from atlas.planners.multi_fidelity.planner import MultiFidelityPlanner

# config
dset = Dataset(kind='perovskites')
NUM_RUNS = 1
BUDGET = 20
NUM_INIT_DESIGN = 5

def measure(params, s):
    func_params = ParameterVector().from_dict(
        {
            'organic': sample.organic, 
            'cation': sample.cation,
            'anion': sample.anion,
        }
    )
    # TODO: update this to actually be the 
    # low fidelity measurement 
    if s == 1.0:
        measurement = dset.run(func_params, noiseless=True)
    elif s == 0.1:
        measurement = dset.run(func_params, noiseless=True)
    return measurement


# build parameter space
param_space = ParameterSpace()

# fidelity param
param_space.add(
    ParameterDiscrete(
        name='s',
        options=[0.1, 1.0],
        low=0.1, 
        high=1.0,
    )
)
# organic
param_space.add(dset.param_space[0])
# cation
param_space.add(dset.param_space[1])
# anion
param_space.add(dset.param_space[2])

for run_ix in range(NUM_RUNS):

    campaign = Campaign()
    campaign.set_param_space(param_space)

    planner = MultiFidelityPlanner(
        goal='minimize',
        init_design_strategy='random',
        num_init_design=NUM_INIT_DESIGN,
        batch_size=1,
        acquisition_optimizer_kind='pymoo',
        fidelity_params=0,
        fidelities=[0.1, 1.],
    )

    planner.set_param_space(param_space)

    iter_ = 0
    while len(campaign.observations.get_values()) < BUDGET:

        print(f'\nRUN : {run_ix+1}/{NUM_RUNS}\tITER : {iter_+1}/{BUDGET}\n')

        samples = planner.recommend(campaign.observations)
        for sample in samples:
            measurement = measure(sample, sample.s)
            campaign.add_observation(sample, measurement)

            iter_+=1