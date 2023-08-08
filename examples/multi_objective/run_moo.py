#!/usr/bin/env python

import pickle
import numpy as np
import pandas as pd

from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import (
	ParameterCategorical,
	ParameterContinuous,
	ParameterDiscrete,
	ParameterVector,
)
from olympus.datasets import Dataset

from atlas.planners.gp.planner import GPPlanner

# config
dataset = Dataset(kind='dye_lasers')
NUM_RUNS = 2
BUDGET = 50
NUM_INIT_DESIGN = 10

all_data = []

for run_ix in range(NUM_RUNS):
	
	campaign = Campaign()
	campaign.set_param_space(dataset.param_space)

	planner = GPPlanner(
		goal='minimize', 
		init_design_strategy='random',
		num_init_design=NUM_INIT_DESIGN,
		use_descriptors=False,
		batch_size=1,
		acquisition_type='ei',
		acquisition_optimizer_kind='pymoo',
		is_moo=True,
		scalarizer_kind='Hypervolume',
		value_space=dataset.value_space,
		goals=['max', 'min', 'max'],
	)
	planner.set_param_space(dataset.param_space)

	iter_ = 0
	while len(campaign.observations.get_values()) < BUDGET: 

		print(f'\nRUN : {run_ix+1}/{NUM_RUNS}\tITER : {iter_+1}/{BUDGET}\n')

		samples = planner.recommend(campaign.observations)

		for sample in samples:
			measurement = dataset.run(sample, noiseless=True)[0]
			print('SAMPLE : ', sample)
			print('MEASUREMENT : ', measurement)
			campaign.add_and_scalarize(sample, measurement, planner.scalarizer)

			iter_ += 1
		
	x0_col = campaign.observations.get_params()[:, 0]
	x1_col = campaign.observations.get_params()[:, 1]
	x2_col = campaign.observations.get_params()[:, 2]

	obj0_col = campaign.observations.get_values()[:, 0]
	obj1_col = campaign.observations.get_values()[:, 1]
	obj2_col = campaign.observations.get_values()[:, 2]

	data = pd.DataFrame({
		'frag_a': x0_col,
		'frag_b': x1_col,
		'frag_c': x2_col,
		'peak_score': obj0_col,
		'spectral_overlap': obj1_col,
		'fluo_rate': obj2_col,
	})
	all_data.append(data)
	pickle.dump(all_data, open('moo_results.pkl', 'wb'))
