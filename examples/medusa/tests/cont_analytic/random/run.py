#!/usr/bin/env python

import os, sys
import shutil

import numpy as np
import pandas as pd
import pickle

from atlas.optimizers.medusa.planner import MedusaPlanner

import olympus
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import ParameterContinuous, ParameterCategorical
from olympus.surfaces import Branin, Michalewicz, Levy, Dejong # cont
from olympus.utils.misc import get_hypervolume, get_pareto_set

#-----------------
# Estimated values
#-----------------


# theoretical minimum
theo_min_params = [
	[ # Branin
		{'params': np.array([0.12389382, 0.81833333]), 'value': 0.39788735772973816},
		{'params': np.array([0.54277284, 0.15166667]), 'value': 0.39788735772973816}, 
		{'params': np.array([0.961652, 0.165]), 'value': 0.39788735775266204},
	],
	[
		# Dejong
		{'params': np.array([0.5, 0.5]), 'value': 0.0}
	],
	[
		# Michalewicz
		{'params': np.array([0.70120676, 0.4999999 ]), 'value': -1.8013034100904854}
	],
	[
		# Levy
		{'params': np.array([0.55, 0.55]), 'value': 1.4997597826618576e-32}
	],
]

theo_min_val = -1.4034160523607473

# theoretical maximum
theo_max_val = 308.129+4.72+0.+150.

# w_ref for hypervolume computation
w_ref = np.array([theo_max_val, 4. ])

# estimated best vals

est_best_dict = {
					1: {
						'G': [[0, 1, 2, 3]],
						'X_func': np.array([[0.52986536, 0.24390908]]),
						'f_x': 6.800270000320467
					},
					2: {
						'G': [[0], [1, 2, 3]],
						'X_func': np.array([[0.95126962, 0.13396389],
								  [0.51074837, 0.50608643]]),
						'f_x': 0.7368735958997857}
					,
					3: {
						'G': [[0], [1, 2], [3]],
						'X_func': np.array([[0.97038967, 0.15323792],
								[0.40699327, 0.48506922],
								[0.54527206, 0.57068025]]),
						'f_x': 1.021902483359942
					},
					4: {
						'G': [[0], [1], [2], [3]],
						'X_func': np.array([[0.56051091, 0.09369598],
							[0.44796786, 0.39556352],
							[0.66549864, 0.49774986],
							[0.54350259, 0.51359794]]),
						'f_x': 1.382150783615172
						}
				}

est_best_objs = np.array([
	[ 6.800270000320467, 1.],
	[0.7368735958997857, 2.],
	[1.021902483359942, 3.],
	[1.382150783615172, 4.]
])

# hypervolume for the best case
est_best_hypervolume = get_hypervolume(est_best_objs, w_ref)

print('est_best_hypervolume : ', est_best_hypervolume)



#------------------
# Helper functions
#------------------

def save_pkl_file(data_all_repeats):
	"""save pickle file with results so far"""

	if os.path.isfile('results.pkl'):
		shutil.move('results.pkl', 'bkp-results.pkl')  # overrides existing files

	# store run results to disk
	with open("results.pkl", "wb") as content:
		pickle.dump(data_all_repeats, content)


def load_data_from_pkl_and_continue(N):
	"""load results from pickle file"""

	data_all_repeats = []
	# if no file, then we start from scratch/beginning
	if not os.path.isfile('results.pkl'):
		return data_all_repeats, N

	# else, we load previous results and continue
	with open("results.pkl", "rb") as content:
		data_all_repeats = pickle.load(content)

	missing_N = N - len(data_all_repeats)

	return data_all_repeats, missing_N


def measure_single_obj(sample, surf_map):

	# non-functional param will always be x0
	# i.e. index of the surface to measure
	si = int( sample.x0[2:] )

	# functional params will always be x1 and x2 
	# NOTE: these should already be floats, just making sure
	X_func = np.array([float(sample.x1), float(sample.x2)]) 

	return surf_map[si].run(X_func)[0][0]


def measure_single_obj_non_olymp(X_func, si, surf_map, is_categorical):
    if is_categorical:
        X_func = list([str(elem) for elem in X_func])
    return surf_map[si].run(X_func)[0][0]


def measure_objective(xgs, G, surf_map, is_categorical):
    """ ... 
    """
    f_x = 0.
    for g_ix, Sg in enumerate(G):
        for si in Sg:
            f_x += measure_single_obj_non_olymp(
		    	xgs[g_ix], si, surf_map, is_categorical,
			)
    return f_x


#----------
# Settings
#----------

# 2d surfaces as the \Tilde{f} objective functions
surf_map = {
	0: Branin(),
	1: Dejong(),
	2: Michalewicz(),
	3: Levy(),
}


with_descriptors_func = False
with_descriptors_nonfunc = False
num_desc_nonfunc = 2 
func_param_type = 'continuous'
use_random_acqf = True

budget = 50
num_init_design = 5
repeats = 30
random_seed = None # i.e. use a different random seed each time

# check whether we are appending to previous results
data_all_repeats, missing_repeats = load_data_from_pkl_and_continue(repeats)

def set_param_space(func_param_type='continuous'):
	param_space = ParameterSpace()
	if func_param_type == 'continuous':
		# 2 continuous functional parameters
		x1 = ParameterContinuous(name='x1', low=0.0, high=1.0)
		x2 = ParameterContinuous(name='x2', low=0.0, high=1.0)
		# 1 categorical non-functional parameter 
		if with_descriptors_nonfunc:
			descriptors = [[float(i) for _ in range(num_desc_nonfunc)] for i in range(len(surf_map))]
		else:
			descriptors = [None for _ in range(len(surf_map))]
		x0 = ParameterCategorical(
			name='x0',
			options = [f's_{i}' for i in range(len(surf_map))],
			descriptors=descriptors
		)

	elif func_param_type == 'categorical':
		# 2 categorical functional parameters 
		if with_descriptors_func:
			descriptors = [[float(i), float(i)] for i in range(21)]
		else:
			descriptors = [None for _ in range(21)]
		x1 = ParameterCategorical(
			name='x1',
			options=[f'x_{i}' for i in range(21)],
			descriptors=descriptors,
		)
		x2 = ParameterCategorical(
			name='x2',
			options=[f'x_{i}' for i in range(21)],
			descriptors=descriptors,
		)
		# 1 categorical non-functional parameter 
		if with_descriptors_nonfunc:
			descriptors = [[float(i) for _ in range(num_desc_nonfunc)] for i in range(len(surf_map))]
		else:
			descriptors = [None for _ in range(len(surf_map))]
		x0 = ParameterCategorical(
			name='x0',
			options = [f's_{i}' for i in range(len(surf_map))],
			descriptors=descriptors
		)
	param_space.add(x0)
	param_space.add(x1)
	param_space.add(x2)

	return param_space


for num_repeat in range(missing_repeats):

	param_space = set_param_space(func_param_type=func_param_type)

	planner = MedusaPlanner(
		goal='minimize',
		general_parameters=[0],
		batch_size=1,
		num_init_design=num_init_design,
		use_random_acqf=use_random_acqf,

	)
	planner.set_param_space(param_space)

	campaign = Campaign()
	campaign.set_param_space(param_space)


	iter=0
	converged = False
	raw_hypervolumes = []
	relative_hypervolumes = []
	all_meas_obj = []
	while len(campaign.observations.get_values()) < budget and not converged:

		samples = planner.recommend(campaign.observations)

		for sample in samples:
			print(f'===============================')
			print(f'   Repeat {len(data_all_repeats)+1} -- Iteration {iter+1}')
			print(f'===============================')

			measurement = measure_single_obj(sample, surf_map)

			print('sample : ', sample)
			print('measurement : ', measurement)

			campaign.add_observation(sample.to_array(), measurement)

			if len(campaign.observations.get_values()) > num_init_design:
				# TODO: make predictions about the best X_func and Ng to use at the end of each iteration
				# this should be done with genetic algorithm similar to the one used for acquisition 
				# optimization

				proposals = planner.optimize_proposals()

				# TODO: for these proposals, compute the true objective function values and convert
				# to a 2d array of "objectives", with first column being f_x and second being Ng

				meas_objs = []
				for Ng, proposal in proposals.items():
					f_x = measure_objective(
							xgs=proposal['X_func'], 
							G=proposal['G'], 
							surf_map=surf_map,
							is_categorical=False,
					)
					meas_objs.append([f_x, float(Ng)])
				meas_objs=np.array(meas_objs)
				all_meas_obj.append(meas_objs)

				# TODO: record the raw and relative hypervolume values
				raw_hypervolume = get_hypervolume(meas_objs, w_ref)
				relative_hypervolume = raw_hypervolume/est_best_hypervolume 
				raw_hypervolumes.append(raw_hypervolume)
				relative_hypervolumes.append(relative_hypervolume)

				print('RELATIVE HYPERVOLUME : ', relative_hypervolume)

			else:
				raw_hypervolumes.append(0.)
				relative_hypervolumes.append(0.)
				meas_objs=np.zeros((4, 2))
				all_meas_obj.append(meas_objs)

			iter+=1

	
	# store the results into a DataFrame
	x0_col = campaign.observations.get_params()[:, 0]
	x1_col = campaign.observations.get_params()[:, 1]
	x2_col = campaign.observations.get_params()[:, 2]
	
	# TODO: this will be the overall objective , i.e. f(x1, x2, ... xNg, G)
	obj_col = campaign.observations.get_values(as_array=True)

	# TODO: maybe add the single objectives as well as columns ???

	data = pd.DataFrame({
		'x0': x0_col, 
		'x1': x1_col,
		'x1': x1_col, 
		'obj': obj_col,
		'all_meas_objs': all_meas_obj,
		'raw_hypervolume': raw_hypervolumes,
		'relative_hypervolumes': relative_hypervolumes,
	})
	data_all_repeats.append(data)

	# save results to disk
	save_pkl_file(data_all_repeats)
