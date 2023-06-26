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

#------
# Data
#------


# subset of QM8 dataset with CCSD E1 energies as objectives and 2 simple descriptors
# molecular weight and logP
qm8_subset = {
	'[H]C1([H])C2([H])C([H])([H])C3([H])C([H])([H])C1([H])C3([H])C2([H])[H]': {
        'descriptors': [108.1805, 2.0524], 'objective': 0.29467054,
	},
	'[H]C1([H])C2([H])OC3([H])C1([H])C([H])([H])C3([H])C2([H])[H]': {
        'descriptors': [110.1534, 1.1837], 'objective': 0.27034605,
	},
	'[H]C12OC3([H])C([H])([H])C([H])(C1([H])[H])C3([H])O2': {
        'descriptors': [112.1262,0.5201], 'objective': 0.28242072,
	},
	'[H]C1([H])C2([H])OC3([H])C([H])([H])C1([H])C([H])([H])C23[H]': {
        'descriptors': [110.1534,1.1837], 'objective': 0.25508264,
	},
	'[H]C1([H])C2([H])OC3([H])C1([H])OC3([H])C2([H])[H]': {
        'descriptors': [112.1262,0.3150], 'objective': 0.26216201,
	},
	'[H]C1([H])N2C([H])([H])C3([H])OC1([H])C3([H])C2([H])[H]': {
        'descriptors': [111.1416,-0.3629], 'objective': 0.22701265,
	},
	'[H]C([H])([H])C12C([H])([H])C3([H])N4C3([H])C1([H])C42[H]': {
        'descriptors': [107.1528,0.3991], 'objective': 0.2437688,
	},
}



#------------------
# Helper functions
#------------------

def measure_single_obj(sample, qm8_subset):
	""" simulate a DFT single point calculation using a lookup table
	"""
	# non-functional param will always be x0
	# i.e. the qm8 molecule to measure
	si = str(sample.x0)

	# functional params, i.e. the B3LYP adjustable parameters
	# NOTE: these should already be floats, just making sure
	X_func = np.array(
		[float(sample.dft_param_1), float(sample.dft_param_2), float(sample.dft_param_3)],
	) 

	# return noisy measurement of the CCSD E1 energy to simulate changing the functional parameters
	return qm8_subset[si]['objective'] + np.random.normal(0., 0.01) 


def set_param_space(func_param_type='continuous'):
	""" set the Olympus parameter space for the optimization experiment
	"""
	param_space = ParameterSpace()
	if func_param_type == 'continuous':
		# 3 continuous B3LYP functional parameters
		x1 = ParameterContinuous(name='dft_param_1', low=0.0, high=1.0)
		x2 = ParameterContinuous(name='dft_param_2', low=0.0, high=1.0)
		x3 = ParameterContinuous(name='dft_param_3', low=0.0, high=1.0)
		# 1 categorical non-functional parameter --> qm8 molecules as options
		if with_descriptors_nonfunc:
			descriptors = [value['descriptors'] for value in qm8_subset.values()]
		else:
			descriptors = [None for _ in range(len(qm8_subset))]
		x0 = ParameterCategorical(
			name='x0',
			options=[smiles for smiles in qm8_subset.keys()],
			descriptors=descriptors
		)

	param_space.add(x0)
	param_space.add(x1)
	param_space.add(x2)
	param_space.add(x3)

	return param_space



#----------
# Settings
#----------

with_descriptors_func = False
with_descriptors_nonfunc = False # use descriptors for the non-functional/general parameters
func_param_type = 'continuous'  # continuous B3LYP parameters

budget = 50  # number of 'calculations' to perform total
num_init_design = 5 # number of inital design points
repeats = 40
random_seed = None # i.e. use a different random seed each time


#--------------------
# BEGIN OPTIMIZATION 
#--------------------

for num_repeat in range(repeats):

	param_space = set_param_space(func_param_type=func_param_type)

	planner = MedusaPlanner(
		goal='minimize',
		general_parameters=[0],
		batch_size=1,
		num_init_design=num_init_design,
		use_descriptors=False,
		use_random_acqf=False,

	)
	planner.set_param_space(param_space)

	campaign = Campaign()
	campaign.set_param_space(param_space)


	iter=0
	while len(campaign.observations.get_values()) < budget:

		# optimizer recommending parameters to measure with DFT single point
		# qm8 molecule and B3LYP parameters
		samples = planner.recommend(campaign.observations)

		for sample in samples:
			print(f'===============================')
			print(f'   Repeat {num_repeat+1} -- Iteration {iter+1}')
			print(f'===============================')

			# TODO: get geometry of qm8 molecule and write to Gaussian input file  
			# along with B3LYP parameters

			# simulated measurement with lookup table
			# TODO: send input file to cluster, start job and collect results
			measurement = measure_single_obj(sample, qm8_subset)

			print('\n')
			print('sample : ', sample)
			print('measurement : ', measurement)
			print('\n')

			campaign.add_observation(sample.to_array(), measurement)

			iter+=1
