#!/usr/bin/env python

import json
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

from atlas.planners import GPPlanner

# config
dataset = Dataset(kind='perovskites')
NUM_RUNS = 5
BUDGET = 50
# COST_BUDGET = 50 # 200.
NUM_INIT_DESIGN = 10
# NUM_CHEAP = 2 # 3

# lookup table
# organic --> cation --> anion --> bandgap_hse06/bandgap_gga
LOOKUP = pickle.load(open('lookup/lookup_table.pkl', 'rb'))
# print(lookup.keys())
# print(lookup['Ethylammonium']['Ge']['F'].keys())


def build_options_descriptors(json_file):
	with open(json_file, 'r') as f:
		content = json.load(f)

	options = list(content.keys())
	descriptors = [list(content[opt].values()) for opt in options]

	return options, descriptors

def measure(params, s):
	# high-fidelity is hse06, low-fidelity is gga
	if s == 1.0:
		measurement = np.amin(
			LOOKUP[params.organic.capitalize()][params.cation][params.anion]['bandgap_hse06']
		)
	elif s == 0.1:
		measurement = np.amin(
			LOOKUP[params.organic.capitalize()][params.cation][params.anion]['bandgap_gga']
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
for param in dataset.param_space: # add perovskite component parameters ('organic', 'cation', and 'anion')
	param_space.add(param)


all_data = []
min_hse06_bandgap = get_min_hse06_bandgap(param_space)
print('MIN HSE06 BANDGAP : ', min_hse06_bandgap)
# >>> MIN HSE06 BANDGAP :  1.5249
# hydrazinium	Sn	I


for run_ix in range(NUM_RUNS):
	campaign = Campaign()
	campaign.set_param_space(param_space)

	planner = GPPlanner(
		goal='minimize',
		init_design_strategy='random',
		num_init_design=NUM_INIT_DESIGN,
		use_descriptors=True,
		batch_size=1,
		acquisition_optimizer_kind='pymoo',
	)

	planner.set_param_space(param_space)

	COST = 0.

	target_rec_measurements = []
	iter_ = 0
	while len(campaign.observations.get_values()) < BUDGET:

		print(f'\nRUN : {run_ix+1}/{NUM_RUNS}\tITER : {iter_+1}\tCOST : {COST}\n')

		samples = planner.recommend(campaign.observations)
		for sample in samples:
			measurement = measure(sample, 1.0)
			campaign.add_observation(sample, measurement)

			print('SAMPLE : ', sample)
			print('MEASUREMENT : ', measurement)

			iter_+=1

		target_rec_measurements.append(measurement)
		# kill the run if we have found the lowest hse06 bandgap
		# on the most recent high-fidelity measurement
		if measurement == min_hse06_bandgap:
			print('found the min hse06 bandgap!')
			break


	x0_col = campaign.observations.get_params()[:, 0]
	x1_col = campaign.observations.get_params()[:, 1]
	x2_col = campaign.observations.get_params()[:, 2]

	obj0_col = np.array(target_rec_measurements) #campaign.observations.get_values()

	data = pd.DataFrame({
		'organic': x0_col,
		'cation': x1_col,
		'anion': x2_col,
		'obj': obj0_col,
	})
	all_data.append(data)
	pickle.dump(all_data, open('sf_results.pkl', 'wb'))
	