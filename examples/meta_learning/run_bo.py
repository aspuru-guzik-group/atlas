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
from olympus.datasets import Dataset

from atlas import tkwargs
from atlas.planners.gp.planner import GPPlanner


#------------------
# BEST OBJS
{
    'a': {
		'best_idx': 597, 
		'best_obj': 55.56585889, 
		'best_params': 'ParamVector(aryl_halide = FC(F)(F)c1ccc(I)cc1, additive = Cc1ccon1, base = CN1CCCN2CCCN=C12, ligand = CC(C1=C(C2=C(OC)C=CC(OC)=C2P(C34CC5CC(C4)CC(C5)C3)C67CC8CC(C7)CC(C8)C6)C(C(C)C)=CC(C(C)C)=C1)C)'}, 
	'b': {
    	'best_idx': 740, 
        'best_obj': 68.2481271, 
        'best_params': 'ParamVector(aryl_halide = COc1ccc(I)cc1, additive = C(N(Cc1ccccc1)c2ccon2)c3ccccc3, base = CN1CCCN2CCCN=C12, ligand = CC(C1=C(C2=C(OC)C=CC(OC)=C2P(C34CC5CC(C4)CC(C5)C3)C67CC8CC(C7)CC(C8)C6)C(C(C)C)=CC(C(C)C)=C1)C)'}, 
	'c': {
		'best_idx': 276, 
		'best_obj': 86.59757822, 
		'best_params': 'ParamVector(aryl_halide = CCc1ccc(Br)cc1, additive = CCOC(=O)c1onc(C)c1, base = CN1CCCN2CCCN=C12, ligand = CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=CC=CC=C2P(C(C)(C)C)C(C)(C)C)'}, 
	'd': {
		'best_idx': 575, 
		'best_obj': 99.99999, 
		'best_params': 'ParamVector(aryl_halide = Ic1ccccn1, additive = o1cc(cn1)c2ccccc2, base = CN1CCCN2CCCN=C12, ligand = CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=CC=CC=C2P(C(C)(C)C)C(C)(C)C)'}, 
	'e': {
		'best_idx': 275, 
		'best_obj': 98.73132029, 
		'best_params': 'ParamVector(aryl_halide = Brc1cccnc1, additive = o1nccc1c2ccccc2, base = CN1CCCN2CCCN=C12, ligand = CC(C)C(C=C(C(C)C)C=C1C(C)C)=C1C2=CC=CC=C2P(C(C)(C)C)C(C)(C)C)'}
}
#------------------


# config
TARGET_TASK = sys.argv[1]

NUM_RUNS = 50
BUDGET = 200
NUM_INIT_DESIGN = 10

dataset = Dataset(kind=f'buchwald_{TARGET_TASK}')

print('TARGET TASK : ', TARGET_TASK)


all_data = []

for run_ix in range(NUM_RUNS):

	campaign = Campaign()
	campaign.set_param_space(dataset.param_space)

	planner = GPPlanner(
		goal='maximize',
		init_design_strategy='random',
		num_init_design=NUM_INIT_DESIGN,
		batch_size=1, 
		acquisition_type='ei', 
		acquisition_optimizer_kind='pymoo',
	)
	planner.set_param_space(dataset.param_space)

	iter_ = 0
	while len(campaign.observations.get_values()) < BUDGET:
		samples = planner.recommend(campaign.observations)
		for sample in samples:
			measurement = dataset.run(sample, noiseless=True)[0][0]
			print('sample : ', sample)
			print(measurement)
			campaign.add_observation(sample, measurement)

			iter_ += 1

	# store the results in dataframe
	x0_col = campaign.observations.get_params()[:, 0]
	x1_col = campaign.observations.get_params()[:, 1]
	x2_col = campaign.observations.get_params()[:, 2]
	x3_col = campaign.observations.get_params()[:, 3]

	obj0_col = campaign.observations.get_values()

	data = pd.DataFrame({
		'aryl_halide': x0_col,
		'additive': x1_col,
		'base': x2_col,
		'ligand': x3_col,
		'yield': obj0_col,
	})
	all_data.append(data)

	pickle.dump(all_data, open(f'bo_target_{TARGET_TASK}_results.pkl', 'wb'))