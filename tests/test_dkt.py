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

from atlas.planners.dkt.planner import DKTPlanner
from atlas.utils.synthetic_data import trig_factory
from problem_generator import ProblemGenerator
from atlas.utils.synthetic_data import olymp_cat_source_task_gen, mixed_source_code



CONT = {
	"init_design_strategy": [
		"random",
		#"sobol",
		#"lhs",
	],  # init design strategies
	"batch_size": [1],  # batch size
	"use_descriptors": [False],  # use descriptors
	"acquisition_type": ['ucb'],
	"acquisition_optimizer": ['pymoo'],#['pymoo', 'genetic'],
}

DISC = {
	"init_design_strategy": ["random"],
	"batch_size": [1],
	"use_descriptors": [False],
	"acquisition_type": ['ucb'],
	"acquisition_optimizer": ['pymoo'],#['pymoo', 'genetic'],
}

CAT = {
	"init_design_strategy": ["random"],
	"batch_size": [1],
	"use_descriptors": [False, True],
	"acquisition_type": ['ucb'],
	"acquisition_optimizer": ['pymoo'],#['pymoo', 'genetic'],
}

MIXED_CAT_CONT = {
	"init_design_strategy": ["random"],
	"batch_size": [1],
	"use_descriptors":[False, True],
	"acquisition_type": ['ucb'],
	"acquisition_optimizer": ['pymoo'],#['pymoo', 'genetic'],
}

MIXED_DISC_CONT = {
	"init_design_strategy": ["random"],
	"batch_size": [1],
	"use_descriptors": [False],
	"acquisition_type": ['ucb'],
	"acquisition_optimizer": ['pymoo'],#['pymoo', 'genetic'],
}


MIXED_CAT_DISC = {
	"init_design_strategy": ["random"],
	"batch_size": [1],
	"use_descriptors": [False, True],
	"acquisition_type": ['ucb'],
	"acquisition_optimizer":['pymoo'],# ['pymoo', 'genetic'],
}

MIXED_CAT_DISC_CONT = {
	"init_design_strategy": ["random"],
	"batch_size": [1],
	"use_descriptors": [False, True],
	"acquisition_type": ['ucb'],
	"acquisition_optimizer": ['pymoo'],#['pymoo', 'genetic'],
}

BATCHED = {
	"problem_type": [
		'cont', 'disc', 'cat', 'mixed_cat_cont',
		'mixed_disc_cont', 'mixed_cat_disc', 'mixed_cat_disc_cont'],
	"init_design_strategy": ["random"],
	"batch_size": [2, 4],
	"acquisition_optimizer": ['pymoo'],#['pymoo', 'genetic'],
}

    
@pytest.mark.parametrize("init_design_strategy", CONT["init_design_strategy"])
@pytest.mark.parametrize("batch_size", CONT["batch_size"])
@pytest.mark.parametrize("use_descriptors", CONT["use_descriptors"])
@pytest.mark.parametrize("acquisition_type", CONT["acquisition_type"])
@pytest.mark.parametrize("acquisition_optimizer", CONT["acquisition_optimizer"])
def test_continuous(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer):
	run_continuous(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer)


@pytest.mark.parametrize("init_design_strategy", DISC["init_design_strategy"])
@pytest.mark.parametrize("batch_size", DISC["batch_size"])
@pytest.mark.parametrize("use_descriptors", DISC["use_descriptors"])
@pytest.mark.parametrize("acquisition_type", DISC["acquisition_type"])
@pytest.mark.parametrize("acquisition_optimizer", DISC["acquisition_optimizer"])
def test_discrete(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer):
	run_discrete(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer)


@pytest.mark.parametrize("init_design_strategy", CAT["init_design_strategy"])
@pytest.mark.parametrize("batch_size", CAT["batch_size"])
@pytest.mark.parametrize("use_descriptors", CAT["use_descriptors"])
@pytest.mark.parametrize("acquisition_type", CAT["acquisition_type"])
@pytest.mark.parametrize("acquisition_optimizer", CAT["acquisition_optimizer"])
def test_categorical(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer):
	run_categorical(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer)


@pytest.mark.parametrize("init_design_strategy", MIXED_CAT_CONT["init_design_strategy"])
@pytest.mark.parametrize("batch_size", MIXED_CAT_CONT["batch_size"])
@pytest.mark.parametrize("use_descriptors", MIXED_CAT_CONT["use_descriptors"])
@pytest.mark.parametrize("acquisition_type", MIXED_CAT_CONT["acquisition_type"])
@pytest.mark.parametrize("acquisition_optimizer", MIXED_CAT_CONT["acquisition_optimizer"])
def test_mixed_cat_cont(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer):
	run_mixed_cat_cont(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer)

@pytest.mark.parametrize("init_design_strategy", MIXED_DISC_CONT["init_design_strategy"])
@pytest.mark.parametrize("batch_size", MIXED_DISC_CONT["batch_size"])
@pytest.mark.parametrize("use_descriptors", MIXED_DISC_CONT["use_descriptors"])
@pytest.mark.parametrize("acquisition_type", MIXED_DISC_CONT["acquisition_type"])
@pytest.mark.parametrize("acquisition_optimizer", MIXED_DISC_CONT["acquisition_optimizer"])
def test_mixed_disc_cont(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer):
	run_mixed_disc_cont(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer)


@pytest.mark.parametrize("init_design_strategy", MIXED_CAT_DISC["init_design_strategy"])
@pytest.mark.parametrize("batch_size", MIXED_CAT_DISC["batch_size"])
@pytest.mark.parametrize("use_descriptors", MIXED_CAT_DISC["use_descriptors"])
@pytest.mark.parametrize("acquisition_type", MIXED_CAT_DISC["acquisition_type"])
@pytest.mark.parametrize("acquisition_optimizer", MIXED_CAT_DISC["acquisition_optimizer"])
def test_mixed_cat_disc(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer):
	run_mixed_cat_disc(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer)


@pytest.mark.parametrize("init_design_strategy", MIXED_CAT_DISC_CONT["init_design_strategy"])
@pytest.mark.parametrize("batch_size", MIXED_CAT_DISC_CONT["batch_size"])
@pytest.mark.parametrize("use_descriptors", MIXED_CAT_DISC_CONT["use_descriptors"])
@pytest.mark.parametrize("acquisition_type", MIXED_CAT_DISC_CONT["acquisition_type"])
@pytest.mark.parametrize("acquisition_optimizer", MIXED_CAT_DISC_CONT["acquisition_optimizer"])
def test_mixed_cat_disc_cont(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer):
	run_mixed_cat_disc_cont(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer)

@pytest.mark.parametrize("problem_type", BATCHED["problem_type"])
@pytest.mark.parametrize("init_design_strategy", BATCHED["init_design_strategy"])
@pytest.mark.parametrize("batch_size", BATCHED["batch_size"])
@pytest.mark.parametrize("acquisition_optimizer", BATCHED["acquisition_optimizer"])
def test_batched(problem_type, init_design_strategy, batch_size, acquisition_optimizer):
	run_batched(problem_type, init_design_strategy, batch_size, acquisition_optimizer)



def run_batched(problem_type, init_design_strategy, batch_size, acquisition_optimizer):
	if problem_type == 'cont':
		run_continuous(init_design_strategy, batch_size, False, 'ucb', acquisition_optimizer, num_init_design=4)
	elif problem_type == 'disc':
		run_discrete(init_design_strategy, batch_size, False, 'ucb', acquisition_optimizer, num_init_design=4)
	elif problem_type == 'cat':
		run_categorical(init_design_strategy, batch_size, False, 'ucb', acquisition_optimizer, num_init_design=4)
		run_categorical(init_design_strategy, batch_size, True, 'ucb', acquisition_optimizer, num_init_design=4)
	elif problem_type == 'mixed_cat_cont': 
		run_mixed_cat_cont(init_design_strategy, batch_size, False, 'ucb', acquisition_optimizer, num_init_design=4)
		run_mixed_cat_cont(init_design_strategy, batch_size, True, 'ucb', acquisition_optimizer, num_init_design=4)
	elif problem_type == 'mixed_disc_cont':
		run_mixed_disc_cont(init_design_strategy, batch_size, False, 'ucb', acquisition_optimizer, num_init_design=4)
	elif problem_type == 'mixed_cat_disc':
		run_mixed_cat_disc(init_design_strategy, batch_size, False, 'ucb', acquisition_optimizer, num_init_design=4)
		run_mixed_cat_disc(init_design_strategy, batch_size, True, 'ucb', acquisition_optimizer, num_init_design=4)
	elif problem_type == 'mixed_cat_disc_cont':
		run_mixed_cat_disc_cont(init_design_strategy, batch_size, False, 'ucb', acquisition_optimizer, num_init_design=4)
		run_mixed_cat_disc_cont(init_design_strategy, batch_size, True, 'ucb', acquisition_optimizer, num_init_design=4)
	else:
		pass



def run_continuous(
		init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, num_init_design=5
	):

	def surface(x):
		return np.sin(8 * x)

	# define the meta-training tasks
	train_tasks = trig_factory(
		num_samples=20,
		# scale_range=[[7, 9], [7, 9]],
		# scale_range=[[-9, -7], [-9, -7]],
		scale_range=[[-8.5, -7.5], [7.5, 8.5]],
		shift_range=[-0.02, 0.02],
		amplitude_range=[0.2, 1.2],
	)

	valid_tasks = trig_factory(
		num_samples=5,
		# scale_range=[[7, 9], [7, 9]],
		# scale_range=[[-9, -7], [-9, -7]],
		scale_range=[[-8.5, -7.5], [7.5, 8.5]],
		shift_range=[-0.02, 0.02],
		amplitude_range=[0.2, 1.2],
	)


	param_space = ParameterSpace()
	# add continuous parameter
	param_0 = ParameterContinuous(name="param_0", low=0.0, high=1.0)
	param_space.add(param_0)

	planner = DKTPlanner(
		goal="minimize",
		init_design_strategy=init_design_strategy,
		num_init_design=num_init_design,
		batch_size=batch_size,
		use_descriptors=use_descriptors,
		acquisition_type=acquisition_type,
		acquisition_optimizer_kind=acquisition_optimizer,
		# meta-learning stuff
		from_disk=False,
 		hyperparams={
  			"model": {
  				"epochs": 6000,
  			}
  		},
  		warm_start=False, 
  		model_path='./tmp_models',
  		train_tasks=train_tasks,
  		valid_tasks=valid_tasks,
	)

	planner.set_param_space(param_space)

	# make the campaign
	campaign = Campaign()
	campaign.set_param_space(param_space)

	BUDGET = num_init_design + batch_size * 4

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)
		for sample in samples:
			sample_arr = sample.to_array()
			measurement = surface(sample_arr)
			campaign.add_observation(sample_arr, measurement)

	os.system('rm -r ./tmp_models')
	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET


def run_discrete(
		init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, num_init_design=5
):
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

	planner = DKTPlanner(
		goal="minimize",
		init_design_strategy=init_design_strategy,
		num_init_design=num_init_design,
		batch_size=batch_size,
		use_descriptors=use_descriptors,
		acquisition_type=acquisition_type,
		acquisition_optimizer_kind=acquisition_optimizer,
		# meta-learning stuff
		from_disk=False,
 		hyperparams={
  			"model": {
  				"epochs": 6000,
  			}
  		},
  		warm_start=False, 
  		model_path='./tmp_models',
  		train_tasks=train_tasks,
  		valid_tasks=valid_tasks,
	)

	planner.set_param_space(param_space)

	# make the campaign
	campaign = Campaign()
	campaign.set_param_space(param_space)

	BUDGET = num_init_design + batch_size * 4

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)
		for sample in samples:
			sample_arr = sample.to_array()
			measurement = surface(sample_arr)
			campaign.add_observation(sample_arr, measurement)

	os.system('rm -r ./tmp_models')

	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET


def run_categorical(
		init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, num_init_design=5
):

	train_tasks, valid_tasks = olymp_cat_source_task_gen(
		num_train_tasks=20,
		num_valid_tasks=5,
		num_opts=5,
		use_descriptors=False,
	)
	

	problem_gen = ProblemGenerator(problem_type='categorical')
	surface_callable, param_space = problem_gen.generate_instance()

	planner = DKTPlanner(
		goal="minimize",
		init_design_strategy=init_design_strategy,
		num_init_design=num_init_design,
		batch_size=batch_size,
		use_descriptors=use_descriptors,
		acquisition_type=acquisition_type,
		acquisition_optimizer_kind=acquisition_optimizer,
		# meta-learning stuff
		from_disk=False,
 		hyperparams={
  			"model": {
  				"epochs": 6000,
  			}
  		},
  		warm_start=False, 
  		model_path='./tmp_models',
  		train_tasks=train_tasks,
  		valid_tasks=valid_tasks,
	)

	planner.set_param_space(param_space)

	# make the campaign
	campaign = Campaign()
	campaign.set_param_space(param_space)

	BUDGET = num_init_design + batch_size * 4

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)
		for sample in samples:
			sample_arr = sample.to_array()
			measurement = surface_callable.run(sample_arr)
			campaign.add_observation(sample_arr, measurement)

	os.system('rm -r ./tmp_models')
	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET


def run_mixed_disc_cont(
		init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, num_init_design=5
):

	train_tasks, valid_tasks = mixed_source_code(problem_type='mixed_disc_cont')
	problem_gen = ProblemGenerator(problem_type='mixed_disc_cont')
	surface_callable, param_space = problem_gen.generate_instance()
	

	planner = DKTPlanner(
		goal="minimize",
		init_design_strategy=init_design_strategy,
		num_init_design=num_init_design,
		batch_size=batch_size,
		use_descriptors=use_descriptors,
		acquisition_type=acquisition_type,
		acquisition_optimizer_kind=acquisition_optimizer,
		# meta-learning stuff
		from_disk=False,
 		hyperparams={
  			"model": {
  				"epochs": 6000,
  			}
  		},
  		warm_start=False, 
  		model_path='./tmp_models',
  		train_tasks=train_tasks,
  		valid_tasks=valid_tasks,
	)

	planner.set_param_space(param_space)

	# make the campaign
	campaign = Campaign()
	campaign.set_param_space(param_space)

	BUDGET = num_init_design + batch_size * 4

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)
		for sample in samples:
			sample_arr = sample.to_array()
			measurement = surface_callable.run(sample_arr)
			campaign.add_observation(sample_arr, measurement)

	os.system('rm -r ./tmp_models')
	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET


def run_mixed_cat_disc(
		init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, num_init_design=5
):

	train_tasks, valid_tasks = mixed_source_code(problem_type='mixed_cat_disc')			
	problem_gen = ProblemGenerator(problem_type='mixed_cat_disc')
	surface_callable, param_space = problem_gen.generate_instance()
	

	planner = DKTPlanner(
		goal="minimize",
		init_design_strategy=init_design_strategy,
		num_init_design=num_init_design,
		batch_size=batch_size,
		use_descriptors=use_descriptors,
		acquisition_type=acquisition_type,
		acquisition_optimizer_kind=acquisition_optimizer,
		# meta-learning stuff
		from_disk=False,
 		hyperparams={
  			"model": {
  				"epochs": 6000,
  			}
  		},
  		warm_start=False, 
  		model_path='./tmp_models',
  		train_tasks=train_tasks,
  		valid_tasks=valid_tasks,
	)

	planner.set_param_space(param_space)

	# make the campaign
	campaign = Campaign()
	campaign.set_param_space(param_space)

	BUDGET = num_init_design + batch_size * 4

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)
		for sample in samples:
			sample_arr = sample.to_array()
			measurement = surface_callable.run(sample_arr)
			campaign.add_observation(sample_arr, measurement)

	os.system('rm -r ./tmp_models')
	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET


def run_mixed_cat_cont(
		init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, num_init_design=5
):

	train_tasks, valid_tasks = mixed_source_code(problem_type='mixed_cat_cont')
	problem_gen = ProblemGenerator(problem_type='mixed_cat_cont')
	surface_callable, param_space = problem_gen.generate_instance()
	

	planner = DKTPlanner(
		goal="minimize",
		init_design_strategy=init_design_strategy,
		num_init_design=num_init_design,
		batch_size=batch_size,
		use_descriptors=use_descriptors,
		acquisition_type=acquisition_type,
		acquisition_optimizer_kind=acquisition_optimizer,
		# meta-learning stuff
		from_disk=False,
 		hyperparams={
  			"model": {
  				"epochs": 6000,
  			}
  		},
  		warm_start=False, 
  		model_path='./tmp_models',
  		train_tasks=train_tasks,
  		valid_tasks=valid_tasks,
	)

	planner.set_param_space(param_space)

	# make the campaign
	campaign = Campaign()
	campaign.set_param_space(param_space)

	BUDGET = num_init_design + batch_size * 4

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)
		for sample in samples:
			sample_arr = sample.to_array()
			measurement = surface_callable.run(sample_arr)
			campaign.add_observation(sample_arr, measurement)

	os.system('rm -r ./tmp_models')
	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET


def run_mixed_cat_disc_cont(
		init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, num_init_design=5
):

	train_tasks, valid_tasks = mixed_source_code(problem_type='mixed_cat_disc_cont')
	problem_gen = ProblemGenerator(problem_type='mixed_cat_disc_cont')
	surface_callable, param_space = problem_gen.generate_instance()
	

	planner = DKTPlanner(
		goal="minimize",
		init_design_strategy=init_design_strategy,
		num_init_design=num_init_design,
		batch_size=batch_size,
		use_descriptors=use_descriptors,
		acquisition_type=acquisition_type,
		acquisition_optimizer_kind=acquisition_optimizer,
		# meta-learning stuff
		from_disk=False,
 		hyperparams={
  			"model": {
  				"epochs": 6000,
  			}
  		},
  		warm_start=False, 
  		model_path='./tmp_models',
  		train_tasks=train_tasks,
  		valid_tasks=valid_tasks,
	)


	planner.set_param_space(param_space)

	# make the campaign
	campaign = Campaign()
	campaign.set_param_space(param_space)

	BUDGET = num_init_design + batch_size * 4

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)
		for sample in samples:
			sample_arr = sample.to_array()
			measurement = surface_callable.run(sample_arr)
			campaign.add_observation(sample_arr, measurement)

	os.system('rm -r ./tmp_models')
	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET


if __name__ == '__main__':
	run_categorical(
		init_design_strategy='random', 
		batch_size=1,
		use_descriptors=False, 
		acquisition_type='ucb', 
		acquisition_optimizer='pymoo', 
		num_init_design=5
	)