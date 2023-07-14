#!/usr/bin/env python

import numpy as np
import pytest
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import (
	ParameterCategorical,
	ParameterContinuous,
	ParameterDiscrete,
)
from olympus.surfaces import Surface

from atlas.planners.multi_fidelity.planner import MultiFidelityPlanner


def run_continuous(acquisition_optimizer):

	num_init_design=5
	batch_size=1

	def surface(params):
		if params['s'] == 0.5:
			# low fidelity
			return np.sin(3.*params['x0']) - np.exp(-2.*params['x1'])
		elif params['s'] == 1.0: 
			# target fidelity
			return np.sin(2.*params['x0']) - np.exp(-1.*params['x1'])

	param_space = ParameterSpace()
	# two continuous parameters and one fidelity param

	param_space.add(ParameterContinuous(name='x0', low=0., high=1.))
	param_space.add(ParameterContinuous(name='x1', low=0., high=1.))
	param_space.add(ParameterDiscrete(name='s', options=[0.5, 1.0]))

	planner = MultiFidelityPlanner(
		goal='minimize', 
		init_design_strategy='random',
		num_init_design=num_init_design, 
		batch_size=batch_size, 
		acquisition_optimizer_kind=acquisition_optimizer,
		fidelity_params=2,
		fidelities=[0.5, 1.0], # should remove this, can get from the fidelity params and param space...
		fixed_cost=5.0,
	)

	planner.set_param_space(param_space)

	campaign = Campaign()
	campaign.set_param_space(param_space)

	BUDGET = num_init_design + batch_size * 6

	iter_ = 0
	while len(campaign.observations.get_values()) < BUDGET:
		
		if len(campaign.observations.get_values()) > num_init_design:
			if iter_ % 2 == 0: # iterating between low and high fidelity samples
				planner.set_ask_fidelity(1.0) # target fidelity
			else:
				planner.set_ask_fidelity(0.5) # lower fidelity

		samples = planner.recommend(campaign.observations)
		for sample in samples:
			measurement = surface(sample)
			campaign.add_observation(sample, measurement) 

		iter_ += 1   




if __name__ == '__main__':

	run_continuous('pymoo')
	#run_categorical('pymoo')