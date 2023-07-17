#!/usr/bin/env python

import os
import numpy as np
import pytest
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import (
	ParameterCategorical,
	ParameterContinuous,
	ParameterDiscrete,
)
from olympus.scalarizers import Scalarizer
from olympus.surfaces import Surface

from atlas.planners.rgpe.planner import RGPEPlanner
from atlas.utils.synthetic_data import trig_factory
from atlas.utils.synthetic_data import olymp_cat_source_task_gen
from problem_generator import ProblemGenerator, KnownConstraintsGenerator, HybridSurface


def run_continuous(init_design_strategy):

	def surface(x):
		return np.sin(8 * x)

	# define the meta-training tasks
	train_tasks = trig_factory(
		num_samples=20,
		as_numpy=True,
		# scale_range=[[7, 9], [7, 9]],
		# scale_range=[[-9, -7], [-9, -7]],
		scale_range=[[-8.5, -7.5], [7.5, 8.5]],
		shift_range=[-0.02, 0.02],
		amplitude_range=[0.2, 1.2],
	)

	valid_tasks = trig_factory(
		num_samples=5,
		as_numpy=True,
		# scale_range=[[7, 9], [7, 9]],
		# scale_range=[[-9, -7], [-9, -7]],
		scale_range=[[-8.5, -7.5], [7.5, 8.5]],
		shift_range=[-0.02, 0.02],
		amplitude_range=[0.2, 1.2],
	)

	param_space = ParameterSpace()
	# add continuous parameter
	param_0 = ParameterContinuous(name="param_0", low=0.0, high=1.0)
	#param_0 = ParameterDiscrete(name='param0', options=[0.0, 0.25, 0.5, 0.75, 1.0])
	param_space.add(param_0)

	planner = RGPEPlanner(
		goal="minimize",
		init_design_strategy=init_design_strategy,
		num_init_design=5,
		batch_size=1,
		acquisition_type='ei',
		acquisition_optimizer_kind='gradient',
		# meta-learning stuff
		train_tasks=train_tasks,
		valid_tasks=valid_tasks,
		cache_weights=False, 
		hyperparams={},
	)

	planner.set_param_space(param_space)

	# make the campaign
	campaign = Campaign()
	campaign.set_param_space(param_space)

	BUDGET = 12

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)
		for sample in samples:
			sample_arr = sample.to_array()
			measurement = surface(sample_arr)
			campaign.add_observation(sample_arr, measurement)

			print('SAMPLE : ', sample)
			print('MEASUREMENT : ', measurement)

	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET


def run_discrete(init_design_strategy):
	def surface(x):
		return np.sin(8 * x)

	# define the meta-training tasks
	train_tasks = trig_factory(
		num_samples=20,
		as_numpy=True,
		# scale_range=[[7, 9], [7, 9]],
		# scale_range=[[-9, -7], [-9, -7]],
		scale_range=[[-8.5, -7.5], [7.5, 8.5]],
		shift_range=[-0.02, 0.02],
		amplitude_range=[0.2, 1.2],
	)
	valid_tasks = trig_factory(
		num_samples=5,
		as_numpy=True,
		# scale_range=[[7, 9], [7, 9]],
		# scale_range=[[-9, -7], [-9, -7]],
		scale_range=[[-8.5, -7.5], [7.5, 8.5]],
		shift_range=[-0.02, 0.02],
		amplitude_range=[0.2, 1.2],
	)

	param_space = ParameterSpace()
	# add continuous parameter
	param_0 = ParameterDiscrete(name="param_0", options=list(np.linspace(0, 1, 40)))
	param_space.add(param_0)

	planner = RGPEPlanner(
		goal="minimize",
		init_design_strategy=init_design_strategy,
		num_init_design=5,
		batch_size=1,
		acquisition_type='ei',
		acquisition_optimizer_kind='pymoo',
		# meta-learning stuff
		train_tasks=train_tasks,
		valid_tasks=valid_tasks,
		cache_weights=False, 
		hyperparams={},
	)

	planner.set_param_space(param_space)

	# make the campaign
	campaign = Campaign()
	campaign.set_param_space(param_space)

	BUDGET = 12

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)
		for sample in samples:
			sample_arr = sample.to_array()
			measurement = surface(sample_arr)
			campaign.add_observation(sample_arr, measurement)

			print('SAMPLE : ', sample)
			print('MEASUREMENT : ', measurement)

	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET


def run_categorical(init_design_strategy):

	train_tasks, valid_tasks = olymp_cat_source_task_gen(
		num_train_tasks=20,
		num_valid_tasks=5,
		use_descriptors=False,
	)

	problem_gen = ProblemGenerator(problem_type='categorical')
	surface_callable, param_space = problem_gen.generate_instance()

	planner = RGPEPlanner(
		goal="minimize",
		init_design_strategy=init_design_strategy,
		num_init_design=5,
		batch_size=1,
		acquisition_type='ei',
		acquisition_optimizer_kind='pymoo',
		# meta-learning stuff
		train_tasks=train_tasks,
		valid_tasks=valid_tasks,
		cache_weights=False, 
		hyperparams={},
	)

	planner.set_param_space(param_space)

	# make the campaign
	campaign = Campaign()
	campaign.set_param_space(param_space)

	BUDGET = 12

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)
		for sample in samples:
			sample_arr = sample.to_array()
			measurement = surface_callable.run(sample_arr)
			campaign.add_observation(sample_arr, measurement)

			print('SAMPLE : ', sample)
			print('MEASUREMENT : ', measurement)

	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET


def run_mixed_cat_cont(init_design_strategy):

	train_tasks, valid_tasks = olymp_cat_source_task_gen(
		num_train_tasks=20,
		num_valid_tasks=5,
		use_descriptors=False,
	)
	
	problem_gen = ProblemGenerator(problem_type='mixed_cat_cont')
	surface_callable, param_space = problem_gen.generate_instance()

	planner = RGPEPlanner(
		goal="minimize",
		init_design_strategy=init_design_strategy,
		num_init_design=5,
		batch_size=1,
		acquisition_type='ei',
		acquisition_optimizer_kind='pymoo',
		# meta-learning stuff
		train_tasks=train_tasks,
		valid_tasks=valid_tasks,
		cache_weights=False, 
		hyperparams={},
	)

	planner.set_param_space(param_space)

	# make the campaign
	campaign = Campaign()
	campaign.set_param_space(param_space)

	BUDGET = 12

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)
		for sample in samples:
			sample_arr = sample.to_array()
			measurement = surface_callable.run(sample_arr)
			campaign.add_observation(sample_arr, measurement)

			print('SAMPLE : ', sample)
			print('MEASUREMENT : ', measurement)

	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET



if __name__ == "__main__":
	# run_continuous('lhs')
	run_categorical('random')
	# run_discrete('random')
	#test_continuous_hypervolume()
