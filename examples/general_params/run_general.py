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
from olympus.emulators import Emulator

from atlas.planners.gp.planner import GPPlanner

# config

emulator_i = Emulator(dataset='suzuki_i', model='BayesNeuralNet')
emulator_ii = Emulator(dataset='suzuki_ii', model='BayesNeuralNet')
emulator_iii = Emulator(dataset='suzuki_iii', model='BayesNeuralNet')
emulator_iv = Emulator(dataset='suzuki_iv', model='BayesNeuralNet')

NUM_RUNS = 50
BUDGET = 50
NUM_INIT_DESIGN = 5


"""
Categorical (
		name='ligand',
		num_opts: 8, 
		options=['L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7'], 
		descriptors=[None, None, None, None, None, None, None, None],
	)
Continuous (name='res_time', low=60.0, high=600.0, is_periodic=False)
Continuous (name='temperature', low=30.0, high=110.0, is_periodic=False)
Continuous (name='catalyst_loading', low=0.498, high=2.515, is_periodic=False)
"""

def measure(params, s):
	func_params = ParameterVector().from_dict({
		'ligand': params.ligand,
		'res_time': params.res_time,
		'temperature': params.temperature,
		'catalyst_loading': params.catalyst_loading,
	})
	if s == 'i':
		measurement, _, __ = emulator_i.run(func_params, num_samples=10)
	elif s == 'ii':
		measurement, _, __ = emulator_ii.run(func_params, num_samples=10)
	elif s == 'iii':
		measurement, _, __ = emulator_iii.run(func_params, num_samples=10)
	elif s == 'iv':
		measurement, _, __ = emulator_iv.run(func_params, num_samples=10)
	return measurement


# build parameter space
param_space = ParameterSpace()
# general parameter
param_space.add(
	ParameterCategorical(
		name='s',
		options=['i', 'ii', 'iii', 'iv'],
		descriptors=[None, None, None, None],
	)
)
param_space.add(
	ParameterCategorical(
		name='ligand',
		num_opts = 8, 
		options=['L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'L6', 'L7'], 
		descriptors=[None, None, None, None, None, None, None, None],
	)
)
param_space.add(
	ParameterContinuous(
		name='res_time', low=60.0, high=600.0,
	)
)
param_space.add(
	ParameterContinuous(
		name='temperature', low=30.0, high=110.0,
	)
)
param_space.add(
	ParameterContinuous(
		name='catalyst_loading', low=0.498, high=2.515,
	)
)

all_true_measurements = []

for run_ix in range(NUM_RUNS):
	
	true_measurements = []

	campaign = Campaign()
	campaign.set_param_space(param_space)

	planner = GPPlanner(
		goal='minimize',
		init_design_strategy='random',
		num_init_design=NUM_INIT_DESIGN,
		batch_size=1,
		acquisition_type='general',
		acquisition_optimizer_kind='pymoo',
		general_parameters=[0],
		is_moo=True,
		scalarizer_kind='Hypervolume',
		value_space=emulator_i.value_space,
		goals=['max', 'max']
	)
	planner.set_param_space(param_space)

	iter_ = 0
	while len(campaign.observations.get_values()) < BUDGET:
		
		print(f'\nRUN : {run_ix+1}/{NUM_RUNS}\tITER : {iter_+1}/{BUDGET}\n')

		samples = planner.recommend(campaign.observations)

		for sample in samples:
			s = sample.s
			measurement = measure(sample, s)[0]
			print('measurement : ', measurement)
			campaign.add_and_scalarize(sample, measurement, planner.scalarizer)

			all_s_measurements = []
			for s in param_space[0].options:
				all_s_measurements.append(
					measure(sample, s)[0]
				)

			print(np.array(all_s_measurements))
			print(np.array(all_s_measurements).shape)
			mean_measurement = np.mean(np.array(all_s_measurements),axis=0)
			print('mean measurement : ', mean_measurement)
			true_measurements.append(mean_measurement)

			iter_ += 1
	
	#all_true_measurements.append(true_measurements)

	# store the results in dataframe
	s_col = campaign.observations.get_params()[:, 0]
	x0_col = campaign.observations.get_params()[:, 1]
	x1_col = campaign.observations.get_params()[:, 2]
	x2_col = campaign.observations.get_params()[:, 3]
	x3_col = campaign.observations.get_params()[:, 4]

	true_measurements = np.array(true_measurements)
	obj0_col = true_measurements[:,0]
	obj1_col = true_measurements[:,1]

	data = pd.DataFrame({
		's': s_col,
		'ligand': x0_col,
		'res_time': x1_col,
		'temperature': x2_col,
		'catalyst_loading': x3_col,
		'yield': obj0_col,
		'turnover': obj1_col,
	})
	all_true_measurements.append(data)

	pickle.dump(all_true_measurements, open('general_results.pkl', 'wb'))





