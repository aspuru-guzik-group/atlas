#!/usr/bin/env python

import pickle
import numpy as np
import pandas as pd
from copy import deepcopy


from olympus.datasets import Dataset
from olympus.objects import (
	ParameterContinuous,
	ParameterDiscrete, 
	ParameterCategorical, 
	ParameterVector
)
from olympus.campaigns import ParameterSpace, Campaign

from atlas.planners.gp.planner import GPPlanner

# config
dset = Dataset(kind='perovskites')
NUM_RUNS = 50
# BUDGET = 30
COST_BUDGET = 200.
NUM_INIT_DESIGN = 5

# lookup table
# organic --> cation --> anion --> bandgap_hse06/bandgap_gga
LOOKUP = pickle.load(open('lookup/lookup_table.pkl', 'rb'))
# print(lookup.keys())
# print(lookup['Ethylammonium']['Ge']['F'].keys())

def measure(params, s=None):
	# high-fidelity is hse06
	measurement = np.amin(
		LOOKUP[params.organic.capitalize()][params.cation][params.anion]['bandgap_hse06']
	)
	return measurement

def get_min_hse06_bandgap(param_space):
	organic_options = [o.capitalize() for o in param_space[0].options]
	cation_options = [o.capitalize() for o in param_space[1].options]
	anion_options = [o.capitalize() for o in param_space[2].options]

	hse06_bandgaps = []
	for organic_option in organic_options:
		for cation_option in cation_options:
			for anion_option in anion_options:
				hse06_bandgaps.append(
					np.amin(
						LOOKUP[organic_option][cation_option][anion_option]['bandgap_hse06']
					)
				)
	min_hse06_bandgap = np.amin(hse06_bandgaps)
	return min_hse06_bandgap

def compute_cost(params):
	costs = params[:,0].astype(float)
	return np.sum(costs)


# build parameter space
param_space = ParameterSpace()
# organic
param_space.add(dset.param_space[0])
# cation
param_space.add(dset.param_space[1])
# anion
param_space.add(dset.param_space[2])


all_data = []
min_hse06_bandgap = get_min_hse06_bandgap(param_space)
print('MIN HSE06 BANDGAP : ', min_hse06_bandgap)

for run_ix in range(NUM_RUNS):

	campaign = Campaign()
	campaign.set_param_space(param_space)

	planner = GPPlanner(
		goal='minimize',
		init_design_strategy='random',
		num_init_design=NUM_INIT_DESIGN,
		use_descriptors=False,
		batch_size=1,
		acquisition_optimizer_kind='gradient',
	)

	planner.set_param_space(param_space)

	COST = 0.

	iter_ = 0
	while COST < COST_BUDGET:

		print(f'\nRUN : {run_ix+1}/{NUM_RUNS}\tITER : {iter_+1}\tCOST : {COST}\n')

		samples = planner.recommend(campaign.observations)
		for sample in samples:
			measurement = measure(sample, None)
			campaign.add_observation(sample, measurement)

			print('SAMPLE : ', sample)
			print('MEASUREMENT : ', measurement)

			iter_+=1

		# kill the run if we have found the lowest hse06 bandgap
		# on the most recent high-fidelity measurement
		if measurement == min_hse06_bandgap:
			print('found the min hse06 bandgap!')
			break

		COST +=1.



	x0_col = campaign.observations.get_params()[:, 0]
	x1_col = campaign.observations.get_params()[:, 1]
	x2_col = campaign.observations.get_params()[:, 2]

	obj0_col = campaign.observations.get_values()

	data = pd.DataFrame({
		'organic': x0_col,
		'cation': x1_col,
		'anion': x2_col,
	})
	all_data.append(data)
	pickle.dump(all_data, open('high_fidelity_results_wo_desc.pkl', 'wb'))