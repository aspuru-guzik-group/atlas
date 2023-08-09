#!/usr/bin/env python

import os, sys
import pickle
import numpy as np
import pandas as pd

import torch
from olympus.datasets import Dataset
from olympus.objects import (
	ParameterContinuous,
	ParameterDiscrete, 
	ParameterCategorical, 
	ParameterVector
)
from olympus.campaigns import ParameterSpace, Campaign

from atlas import tkwargs
from atlas.planners.dkt.planner import DKTPlanner



# optimal parameter settings for each aryl halide
opt_params = {
	0: {'base':0, 'ligand': 1, 'additive': 2},
	1: {'base':1, 'ligand': 1, 'additive': 4},
	2: {'base':2, 'ligand': 3, 'additive': 5},
	3: {'base':2, 'ligand': 3, 'additive': 17},
	4: {'base':2, 'ligand': 2, 'additive': 17},
	5: {'base':2, 'ligand': 3, 'additive': 17},
	6: {'base':2, 'ligand': 1, 'additive': 3},
	7: {'base':2, 'ligand': 1, 'additive': 1},
	8: {'base':2, 'ligand': 3, 'additive': 1},
	9: {'base':2, 'ligand': 1, 'additive': 1},
	10: {'base':2, 'ligand': 1, 'additive': 1},
	11: {'base':2, 'ligand': 2, 'additive': 3},
	12: {'base':0, 'ligand': 1, 'additive': 3},
	13: {'base':2, 'ligand': 1, 'additive': 0},
	14: {'base':2, 'ligand': 1, 'additive': 0},
}

NUM_RUNS = 1
BUDGET = 200
NUM_INIT_DESIGN = 10

#-----------------
# helper functions
#-----------------

def experiment(param, target_ix, lookup):
	base = int(param['base'][1:])
	ligand = int(param['ligand'][1:])
	additive = int(param['additive'][1:])
	sub_df = lookup[
		(lookup['base_ix']==base) &
		(lookup['ligand_ix']==ligand) &
		(lookup['additive_ix']==additive) &
		(lookup['aryl_halide_ix']==target_ix)
	]
	if sub_df.shape[0] == 1:
		yield_ = sub_df.loc[:, 'yield'].to_numpy()[0]
	elif sub_df.shape[0] < 1:
		yield_ = 0.0

	return yield_

def check_convergence(sample, opt_params):
	base = int(sample['base'][1:])
	ligand = int(sample['ligand'][1:])
	additive = int(sample['additive'][1:])
	if np.all([
		base == opt_params['base'],
		ligand == opt_params['ligand'],
		additive == opt_params['additive'],
	]):
		return True
	else:
		return False


# load in the tasks (one hot encoded)
tasks = pickle.load(open(f'tasks_aryl_one_hot.pkl', 'rb'))

# 8th aryl halide --> aryl halide index for the target task
aryl_halide_ix = 8

# load in the lookup table
with open(f'main_df.pkl', 'rb') as content:
	lookup = pickle.load(content)

SOURCE_TASKS_ = tasks.copy()
print('ARYL HALIDE IX : ', aryl_halide_ix)
del SOURCE_TASKS_[aryl_halide_ix]

# convert torch tensor dtypes
SOURCE_TASKS = []
for task in SOURCE_TASKS_:
	new_task = {
		'params': torch.tensor(task['params'], **tkwargs),
		'values': torch.tensor(task['values'], **tkwargs),
	}
	SOURCE_TASKS.append(new_task)



#----------------------------
# Create the parameter space
#----------------------------

# generate the parameter space
param_space = ParameterSpace()
# base  (3 options)
param_space.add(
	ParameterCategorical(
		name='base',
		options = [f'B{i}' for i in range(3)],
		descriptors = [None for i in range(3)] # one-hot-encoded desc
	)
)
# ligand/catalyst (4 options)
param_space.add(
	ParameterCategorical(
		name='ligand',
		options = [f'L{i}' for i in range(4)],
		descriptors = [None for i in range(4)] # one-hot-encoded desc
	)
)
# additive (23 options)
param_space.add(
	ParameterCategorical(
		name='additive',
		options = [f'A{i}' for i in range(23)],
		descriptors = [None for i in range(23)] # one-hot-encoded desc
	)
)


# begin experiments

for run_ix in range(NUM_RUNS):
	
	campaign = Campaign()
	campaign.set_param_space(param_space)

	planner = DKTPlanner(
		goal='maximize',
		init_design_strategy='random',
		num_init_design=NUM_INIT_DESIGN,
		batch_size=1, 
		acquisition_type='ei', 
		acquisition_optimizer_kind='gradient',
		#meta-learning
		train_tasks=SOURCE_TASKS, 
		valid_tasks=SOURCE_TASKS[:5],  
		model_path=f'./tmp_models/',
		from_disk=False,
		x_dim=30,
		hyperparams={'model':{
				'epochs': 5000,
			}
		} 
	)
	planner.set_param_space(param_space)

	iter_=0
	while len(campaign.observations.get_values()) < BUDGET:
		samples = planner.recommend(campaign.observations)
		for sample in samples:
			sample_arr = sample.to_array()
			# try to make the measurement, otherwise return a value of 0.0
			measurement = experiment(sample, aryl_halide_ix, lookup)
			campaign.add_observation(sample, measurement)

			iter_+=1

	os.system('rm -r ./tmp_models/')

	

