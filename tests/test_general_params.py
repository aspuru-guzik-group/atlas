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
from problem_generator import ProblemGenerator

from atlas.planners.gp.planner import BoTorchPlanner

from problem_generator import ProblemGenerator, KnownConstraintsGenerator

IS_CONSTRAINED = [True, False]

CONT = {
	"init_design_strategy": [
		"random",
	],  # init design strategies
	"batch_size": [1],  # batch size
	"use_descriptors": [False],  # use descriptors
	"acquisition_type": ['general'], # fixed for this problem type
	"acquisition_optimizer": ['pymoo'],#['pymoo', 'genetic'],
	"is_constrained": IS_CONSTRAINED, 
}


DISC = {
	"init_design_strategy": [
		"random",
	], 
	"batch_size": [1], 
	"use_descriptors": [False], 
	"acquisition_type": ['general'], 
	"acquisition_optimizer": ['pymoo'],
	"is_constrained": IS_CONSTRAINED, 
}

CAT = {
	"init_design_strategy": [
		"random",
	], 
	"batch_size": [1],  
	"use_descriptors": [False, True],
	"acquisition_type": ['general'], 
	"acquisition_optimizer": ['pymoo'],
	"is_constrained": IS_CONSTRAINED, 
}

MIXED_CAT_CONT = {
	"init_design_strategy": [
		"random",
	], 
	"batch_size": [1],  
	"use_descriptors": [False, True],
	"acquisition_type": ['general'], 
	"acquisition_optimizer": ['pymoo'],
	"is_constrained": IS_CONSTRAINED, 
}


MIXED_DISC_CONT = {
	"init_design_strategy": [
		"random",
	], 
	"batch_size": [1],  
	"use_descriptors": [False],
	"acquisition_type": ['general'], 
	"acquisition_optimizer": ['pymoo'],
	"is_constrained": IS_CONSTRAINED, 
}

MIXED_CAT_DISC = {
	"init_design_strategy": [
		"random",
	], 
	"batch_size": [1],  
	"use_descriptors": [False, True],
	"acquisition_type": ['general'], 
	"acquisition_optimizer": ['pymoo'],
	"is_constrained": IS_CONSTRAINED, 
}

MIXED_CAT_DISC_CONT = {
	"init_design_strategy": [
		"random",
	], 
	"batch_size": [1],  
	"use_descriptors": [False, True],
	"acquisition_type": ['general'], 
	"acquisition_optimizer": ['pymoo'],
	"is_constrained": IS_CONSTRAINED, 
}

BATCHED = {
	"problem_type": [
		'cont', 'disc', 'cat', 'mixed_cat_cont',
		'mixed_disc_cont', 'mixed_cat_disc', 'mixed_cat_disc_cont'],
	"init_design_strategy": ["random"], 
	"batch_size": [2, 4],  
	"acquisition_optimizer": ['pymoo'],
	"is_constrained": IS_CONSTRAINED, 
}



	
@pytest.mark.parametrize("init_design_strategy", CONT["init_design_strategy"])
@pytest.mark.parametrize("batch_size", CONT["batch_size"])
@pytest.mark.parametrize("use_descriptors", CONT["use_descriptors"])
@pytest.mark.parametrize("acquisition_type", CONT["acquisition_type"])
@pytest.mark.parametrize("acquisition_optimizer", CONT["acquisition_optimizer"])
@pytest.mark.parametrize("is_constrained", CONT["is_constrained"])
def test_cat_general_cont_func(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, is_constrained):
	run_cat_general_cont_func(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, is_constrained)


@pytest.mark.parametrize("init_design_strategy", DISC["init_design_strategy"])
@pytest.mark.parametrize("batch_size", DISC["batch_size"])
@pytest.mark.parametrize("use_descriptors", DISC["use_descriptors"])
@pytest.mark.parametrize("acquisition_type", DISC["acquisition_type"])
@pytest.mark.parametrize("acquisition_optimizer", DISC["acquisition_optimizer"])
@pytest.mark.parametrize("is_constrained", DISC["is_constrained"])
def test_cat_general_disc_func(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, is_constrained):
	run_cat_general_disc_func(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, is_constrained)


@pytest.mark.parametrize("init_design_strategy", CAT["init_design_strategy"])
@pytest.mark.parametrize("batch_size", CAT["batch_size"])
@pytest.mark.parametrize("use_descriptors", CAT["use_descriptors"])
@pytest.mark.parametrize("acquisition_type", CAT["acquisition_type"])
@pytest.mark.parametrize("acquisition_optimizer", CAT["acquisition_optimizer"])
@pytest.mark.parametrize("is_constrained", CAT["is_constrained"])
def test_cat_general_cat_func(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, is_constrained):
	run_cat_general_cat_func(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, is_constrained)


@pytest.mark.parametrize(
    "init_design_strategy", MIXED_CAT_CONT["init_design_strategy"]
)
@pytest.mark.parametrize("batch_size", MIXED_CAT_CONT["batch_size"])
@pytest.mark.parametrize("use_descriptors", MIXED_CAT_CONT["use_descriptors"])
@pytest.mark.parametrize("acquisition_type", MIXED_CAT_CONT["acquisition_type"])
@pytest.mark.parametrize("acquisition_optimizer", MIXED_CAT_CONT["acquisition_optimizer"])
@pytest.mark.parametrize("is_constrained", MIXED_CAT_CONT["is_constrained"])
def test_cat_general_mixed_cat_cont_func(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, is_constrained):
	run_cat_general_mixed_cat_cont_func(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, is_constrained)


@pytest.mark.parametrize(
    "init_design_strategy", MIXED_DISC_CONT["init_design_strategy"]
)
@pytest.mark.parametrize("batch_size", MIXED_DISC_CONT["batch_size"])
@pytest.mark.parametrize("use_descriptors", MIXED_DISC_CONT["use_descriptors"])
@pytest.mark.parametrize("acquisition_type", MIXED_DISC_CONT["acquisition_type"])
@pytest.mark.parametrize("acquisition_optimizer", MIXED_DISC_CONT["acquisition_optimizer"])
@pytest.mark.parametrize("is_constrained", MIXED_DISC_CONT["is_constrained"])
def test_cat_general_mixed_disc_cont_func(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, is_constrained):
	run_cat_general_mixed_disc_cont_func(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, is_constrained)


@pytest.mark.parametrize(
    "init_design_strategy", MIXED_CAT_DISC["init_design_strategy"]
)
@pytest.mark.parametrize("batch_size", MIXED_CAT_DISC["batch_size"])
@pytest.mark.parametrize("use_descriptors", MIXED_CAT_DISC["use_descriptors"])
@pytest.mark.parametrize("acquisition_type", MIXED_CAT_DISC["acquisition_type"])
@pytest.mark.parametrize("acquisition_optimizer", MIXED_CAT_DISC["acquisition_optimizer"])
@pytest.mark.parametrize("is_constrained", MIXED_CAT_DISC["is_constrained"])
def test_cat_general_mixed_cat_disc_func(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, is_constrained):
	run_cat_general_mixed_cat_disc_func(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, is_constrained)


@pytest.mark.parametrize(
    "init_design_strategy", MIXED_CAT_DISC_CONT["init_design_strategy"]
)
@pytest.mark.parametrize("batch_size", MIXED_CAT_DISC_CONT["batch_size"])
@pytest.mark.parametrize("use_descriptors", MIXED_CAT_DISC_CONT["use_descriptors"])
@pytest.mark.parametrize("acquisition_type", MIXED_CAT_DISC_CONT["acquisition_type"])
@pytest.mark.parametrize("acquisition_optimizer", MIXED_CAT_DISC_CONT["acquisition_optimizer"])
@pytest.mark.parametrize("is_constrained", MIXED_CAT_DISC_CONT["is_constrained"])
def test_cat_general_mixed_cat_disc_cont_func(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, is_constrained):
	run_cat_general_mixed_cat_disc_cont_func(init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, is_constrained)

@pytest.mark.parametrize("problem_type", BATCHED["problem_type"])
@pytest.mark.parametrize(
    "init_design_strategy", BATCHED["init_design_strategy"]
)
@pytest.mark.parametrize("batch_size", BATCHED["batch_size"])
@pytest.mark.parametrize("acquisition_optimizer", BATCHED["acquisition_optimizer"])
@pytest.mark.parametrize("is_constrained", BATCHED["is_constrained"])
def test_batched(problem_type, init_design_strategy, batch_size, acquisition_optimizer, is_constrained):
	run_batched(problem_type, init_design_strategy, batch_size, acquisition_optimizer, is_constrained)


# ------------------------
# GENERAL PARAM SURFACES
# ------------------------


def surface_general(x, s, surfaces):
	return surfaces[s].run(x)


# ------------------------


def run_batched(problem_type, init_design_strategy, batch_size, acquisition_optimizer, is_constrained):
	if problem_type == 'cont':
		run_cat_general_cont_func(init_design_strategy, batch_size, False, 'general', acquisition_optimizer, is_constrained, num_init_design=4)
	elif problem_type == 'disc':
		run_cat_general_disc_func(init_design_strategy, batch_size, False, 'general', acquisition_optimizer, is_constrained, num_init_design=4)
	elif problem_type == 'cat':
		run_cat_general_cat_func(init_design_strategy, batch_size, False, 'general', acquisition_optimizer, is_constrained, num_init_design=4)
		run_cat_general_cat_func(init_design_strategy, batch_size, True, 'general', acquisition_optimizer, is_constrained, num_init_design=4)
	elif problem_type == 'mixed_cat_cont': 
		run_cat_general_mixed_cat_cont_func(init_design_strategy, batch_size, False, 'general', acquisition_optimizer, is_constrained, num_init_design=4)
		run_cat_general_mixed_cat_cont_func(init_design_strategy, batch_size, True, 'general', acquisition_optimizer, is_constrained, num_init_design=4)
	elif problem_type == 'mixed_disc_cont':
		run_cat_general_mixed_disc_cont_func(init_design_strategy, batch_size, False, 'general', acquisition_optimizer, is_constrained, num_init_design=4)
	elif problem_type == 'mixed_cat_disc':
		run_cat_general_mixed_cat_disc_func(init_design_strategy, batch_size, False, 'general', acquisition_optimizer, is_constrained, num_init_design=4)
		run_cat_general_mixed_cat_disc_func(init_design_strategy, batch_size, True, 'general', acquisition_optimizer, is_constrained, num_init_design=4)
	elif problem_type == 'mixed_cat_disc_cont':
		run_cat_general_mixed_cat_disc_cont_func(init_design_strategy, batch_size, False, 'general', acquisition_optimizer, is_constrained, num_init_design=4)
		run_cat_general_mixed_cat_disc_cont_func(init_design_strategy, batch_size, True, 'general', acquisition_optimizer, is_constrained, num_init_design=4)
	else:
		pass

def run_cat_general_cont_func(
		init_design_strategy, 
		batch_size, 
		use_descriptors, 
		acquisition_type, 
		acquisition_optimizer, 
		is_constrained,
		num_init_design=5,
	):
	""" single categorical general parameter
	"""
	param_space = ParameterSpace()

	# general parameter
	param_space.add(
		ParameterCategorical(
			name='s',
			options=[str(i) for i in range(3)],
			descriptors=[[float(i),float(i)] for i in range(3)],   
		)
	)
	# functional parameters
	param_space.add(ParameterContinuous(name='x_0',low=0.,high=1.))
	param_space.add(ParameterContinuous(name='x_1',low=0.,high=1.))

	surfaces = {}
	problem_gen = ProblemGenerator(problem_type='continuous')
	for general_param_option in param_space[0].options:
		surface_callable, _ = problem_gen.generate_instance()
		surfaces[general_param_option] = surface_callable
	
	if is_constrained:
		known_constraints = [KnownConstraintsGenerator(is_general=True).get_constraint('continuous')]
	else:
		known_constraints = None
		
	campaign = Campaign()
	campaign.set_param_space(param_space)

	planner = BoTorchPlanner(
		goal='minimize',
		init_design_strategy=init_design_strategy,
		num_init_design=num_init_design,
		batch_size=batch_size,
		use_descriptors=use_descriptors,
		acquisition_type=acquisition_type,
		acquisition_optimizer_kind=acquisition_optimizer,
		general_parameters=[0],
		known_constraints=known_constraints
	)
	planner.set_param_space(param_space)

	BUDGET = num_init_design + batch_size * 4
	true_measurements = []

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)
		for sample in samples:
			measurement = surface_general(
				[float(sample.x_0), float(sample.x_1)],
				sample.s,
				surfaces,
			)
			all_measurements = []
			for s in param_space[0].options:
				all_measurements.append(
					surface_general(
						[float(sample.x_0), float(sample.x_1)],
						s,
						surfaces,
					)
				)
			true_measurements.append(np.mean(all_measurements))

			campaign.add_observation(sample, measurement)

	
	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET
	
	if is_constrained:
		meas_params = campaign.observations.get_params()
		kcs = [known_constraints[0](param) for param in meas_params]
		assert all(kcs)


def run_cat_general_disc_func(
		init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, is_constrained, num_init_design=5):
	""" single categorical general parameter
	"""
	param_space = ParameterSpace()

	# general parameter
	param_space.add(
		ParameterCategorical(
			name='s',
			options=[str(i) for i in range(3)],
			descriptors=[[float(i),float(i)] for i in range(3)],   
		)
	)
	# functional parameters
	param_space.add(ParameterDiscrete(
		name='x_0',
		low=0.,
		high=1.,
		options=list(np.linspace(0.,1.,8)),
	))
	param_space.add(ParameterDiscrete(
		name='x_1',
		low=0.,
		high=1.,
		options=list(np.linspace(0.,1.,8)),
	))

	surfaces = {}
	problem_gen = ProblemGenerator(problem_type='discrete')
	for general_param_option in param_space[0].options:
		surface_callable, _ = problem_gen.generate_instance()
		surfaces[general_param_option] = surface_callable
		
	if is_constrained:
		known_constraints = [KnownConstraintsGenerator(is_general=True).get_constraint('discrete')]
	else:
		known_constraints = None

	campaign = Campaign()
	campaign.set_param_space(param_space)

	planner = BoTorchPlanner(
		goal='minimize',
		init_design_strategy=init_design_strategy,
		num_init_design=num_init_design,
		batch_size=batch_size,
		use_descriptors=use_descriptors,
		acquisition_type=acquisition_type,
		acquisition_optimizer_kind=acquisition_optimizer,
		general_parameters=[0],
		known_constraints=known_constraints
	)
	planner.set_param_space(param_space)

	BUDGET = num_init_design + batch_size * 4
	true_measurements = []

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)
		for sample in samples:
			measurement = surface_general(
				[float(sample.x_0), float(sample.x_1)],
				sample.s,
				surfaces,
			)
			all_measurements = []
			for s in param_space[0].options:
				all_measurements.append(
					surface_general(
						[float(sample.x_0), float(sample.x_1)],
						s,
						surfaces,
					)
				)
			true_measurements.append(np.mean(all_measurements))

			campaign.add_observation(sample, measurement)

	
	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET

	if is_constrained:
		meas_params = campaign.observations.get_params()
		kcs = [known_constraints[0](param) for param in meas_params]
		assert all(kcs)


def run_cat_general_cat_func(
		  init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, is_constrained, num_init_design=5):
	""" single categorical general parameter
	"""
	param_space = ParameterSpace()

	# general parameter
	param_space.add(
		ParameterCategorical(
			name='s',
			options=[str(i) for i in range(3)],
			descriptors=[[float(i),float(i)] for i in range(3)],   
		)
	)
	# functional parameters
	param_space.add(
		ParameterCategorical(
			name='x_0',
			options=[f'x{i}' for i in range(5)],
			descriptors=[[float(i),float(i)] for i in range(5)],   
		)
	)
	param_space.add(
		ParameterCategorical(
			name='x_1',
			options=[f'x{i}' for i in range(5)],
			descriptors=[[float(i),float(i)] for i in range(5)],   
		)
	)

	surfaces = {}
	problem_gen = ProblemGenerator(problem_type='categorical')
	for general_param_option in param_space[0].options:
		surface_callable, _ = problem_gen.generate_instance()
		surfaces[general_param_option] = surface_callable

	if is_constrained:
		known_constraints = [KnownConstraintsGenerator(is_general=True).get_constraint('categorical')]
	else:
		known_constraints = None

	campaign = Campaign()
	campaign.set_param_space(param_space)

	planner = BoTorchPlanner(
		goal='minimize',
		init_design_strategy=init_design_strategy,
		num_init_design=num_init_design,
		batch_size=batch_size,
		use_descriptors=use_descriptors,
		acquisition_type=acquisition_type,
		acquisition_optimizer_kind=acquisition_optimizer,
		general_parameters=[0],
		known_constraints=known_constraints
	)
	planner.set_param_space(param_space)

	BUDGET = num_init_design + batch_size * 4
	true_measurements = []

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)
		for sample in samples:
			measurement = surface_general([sample.x_0, sample.x_1], sample.s, surfaces)

			print('SAMPLE : ', sample)
			print('MEASUREMENT : ', measurement)

			all_measurements = []
			for s in param_space[0].options:
				all_measurements.append(
					surface_general([sample.x_0, sample.x_1], s, surfaces)
				)
			true_measurements.append(np.mean(all_measurements))

			campaign.add_observation(sample, measurement)

	
	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET

	if is_constrained:
		meas_params = campaign.observations.get_params()
		kcs = [known_constraints[0](param) for param in meas_params]
		assert all(kcs)


def run_cat_general_mixed_cat_cont_func(
		  init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, is_constrained, num_init_design=5):
	""" single categorical general parameter
	"""
	param_space = ParameterSpace()

	# general parameter
	param_space.add(
		ParameterCategorical(
			name='s',
			options=[str(i) for i in range(3)],
			descriptors=[[float(i),float(i)] for i in range(3)],   
		)
	)
	# functional parameters
	param_space.add(
		ParameterCategorical(
			name='x_0',
			options=[f'x{i}' for i in range(5)],
			descriptors=[[float(i),float(i)] for i in range(5)],   
		)
	)
	param_space.add(
		ParameterCategorical(
			name='x_1',
			options=[f'x{i}' for i in range(5)],
			descriptors=[[float(i),float(i)] for i in range(5)],   
		)
	)
	param_space.add(ParameterContinuous(name='x_2'))
	param_space.add(ParameterContinuous(name='x_3'))

	surfaces = {}
	problem_gen = ProblemGenerator(problem_type='mixed_cat_cont')
	for general_param_option in param_space[0].options:
		surface_callable, _ = problem_gen.generate_instance()
		surfaces[general_param_option] = surface_callable

	if is_constrained:
		known_constraints = [KnownConstraintsGenerator(is_general=True).get_constraint('cat_cont')]
	else:
		known_constraints = None

	campaign = Campaign()
	campaign.set_param_space(param_space)

	planner = BoTorchPlanner(
		goal='minimize',
		init_design_strategy=init_design_strategy,
		num_init_design=num_init_design,
		batch_size=batch_size,
		use_descriptors=use_descriptors,
		acquisition_type=acquisition_type,
		acquisition_optimizer_kind=acquisition_optimizer,
		general_parameters=[0],
		known_constraints=known_constraints
	)
	planner.set_param_space(param_space)

	BUDGET = num_init_design + batch_size * 4
	true_measurements = []

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)
		for sample in samples:
			measurement = surface_general([sample.x_0, sample.x_1, float(sample.x_2), float(sample.x_3)], sample.s, surfaces)

			print('SAMPLE : ', sample)
			print('MEASUREMENT : ', measurement)

			all_measurements = []
			for s in param_space[0].options:
				all_measurements.append(
					surface_general([sample.x_0, sample.x_1, float(sample.x_2), float(sample.x_3)], sample.s, surfaces)
				)
			true_measurements.append(np.mean(all_measurements))

			campaign.add_observation(sample, measurement)

	
	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET

	if is_constrained:
		meas_params = campaign.observations.get_params()
		kcs = [known_constraints[0](param) for param in meas_params]
		assert all(kcs)


def run_cat_general_mixed_cat_disc_func(
		  init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, is_constrained, num_init_design=5):
	""" single categorical general parameter
	"""
	param_space = ParameterSpace()

	# general parameter
	param_space.add(
		ParameterCategorical(
			name='s',
			options=[str(i) for i in range(3)],
			descriptors=[[float(i),float(i)] for i in range(3)],   
		)
	)
	# functional parameters
	param_space.add(
		ParameterCategorical(
			name='x_0',
			options=[f'x{i}' for i in range(5)],
			descriptors=[[float(i),float(i)] for i in range(5)],   
		)
	)
	param_space.add(
		ParameterCategorical(
			name='x_1',
			options=[f'x{i}' for i in range(5)],
			descriptors=[[float(i),float(i)] for i in range(5)],   
		)
	)
	param_space.add(ParameterDiscrete(
		name='x_2',
		low=0.,
		high=1.,
		options=list(np.linspace(0.,1.,8)),
	))
	param_space.add(ParameterDiscrete(
		name='x_3',
		low=0.,
		high=1.,
		options=list(np.linspace(0.,1.,8)),
	))

	surfaces = {}
	problem_gen = ProblemGenerator(problem_type='mixed_cat_disc')
	for general_param_option in param_space[0].options:
		surface_callable, _ = problem_gen.generate_instance()
		surfaces[general_param_option] = surface_callable

	if is_constrained:
		known_constraints = [KnownConstraintsGenerator(is_general=True).get_constraint('cat_disc')]
	else:
		known_constraints = None

	campaign = Campaign()
	campaign.set_param_space(param_space)

	planner = BoTorchPlanner(
		goal='minimize',
		init_design_strategy=init_design_strategy,
		num_init_design=num_init_design,
		batch_size=batch_size,
		use_descriptors=use_descriptors,
		acquisition_type=acquisition_type,
		acquisition_optimizer_kind=acquisition_optimizer,
		general_parameters=[0],
		known_constraints=known_constraints
	)
	planner.set_param_space(param_space)

	BUDGET = num_init_design + batch_size * 4
	true_measurements = []

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)
		for sample in samples:
			measurement = surface_general([sample.x_0, sample.x_1, float(sample.x_2), float(sample.x_3)], sample.s, surfaces)
			all_measurements = []
			for s in param_space[0].options:
				all_measurements.append(
					surface_general([sample.x_0, sample.x_1, float(sample.x_2), float(sample.x_3)], sample.s, surfaces)
				)
			true_measurements.append(np.mean(all_measurements))

			campaign.add_observation(sample, measurement)

	
	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET

	if is_constrained:
		meas_params = campaign.observations.get_params()
		kcs = [known_constraints[0](param) for param in meas_params]
		assert all(kcs)


def run_cat_general_mixed_disc_cont_func(
		  init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, is_constrained, num_init_design=5):
	""" single categorical general parameter
	"""
	param_space = ParameterSpace()

	# general parameter
	param_space.add(
		ParameterCategorical(
			name='s',
			options=[str(i) for i in range(3)],
			descriptors=[[float(i),float(i)] for i in range(3)],   
		)
	)
	# functional parameters
	param_space.add(ParameterDiscrete(
		name='x_0',
		low=0.,
		high=1.,
		options=list(np.linspace(0.,1.,8)),
	))
	param_space.add(ParameterDiscrete(
		name='x_1',
		low=0.,
		high=1.,
		options=list(np.linspace(0.,1.,8)),
	))
	param_space.add(ParameterContinuous(name='x_2'))
	param_space.add(ParameterContinuous(name='x_3'))

	surfaces = {}
	problem_gen = ProblemGenerator(problem_type='mixed_disc_cont')
	for general_param_option in param_space[0].options:
		surface_callable, _ = problem_gen.generate_instance()
		surfaces[general_param_option] = surface_callable

	if is_constrained:
		known_constraints = [KnownConstraintsGenerator(is_general=True).get_constraint('disc_cont')]
	else:
		known_constraints = None

	campaign = Campaign()
	campaign.set_param_space(param_space)

	planner = BoTorchPlanner(
		goal='minimize',
		init_design_strategy=init_design_strategy,
		num_init_design=num_init_design,
		batch_size=batch_size,
		use_descriptors=use_descriptors,
		acquisition_type=acquisition_type,
		acquisition_optimizer_kind=acquisition_optimizer,
		general_parameters=[0],
		known_constraints=known_constraints
	)
	planner.set_param_space(param_space)

	BUDGET = num_init_design + batch_size * 4
	true_measurements = []

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)
		for sample in samples:
			measurement = surface_general([float(sample.x_0), float(sample.x_1), float(sample.x_2), float(sample.x_3)], sample.s, surfaces)
			all_measurements = []
			for s in param_space[0].options:
				all_measurements.append(
					surface_general([float(sample.x_0), float(sample.x_1), float(sample.x_2), float(sample.x_3)], sample.s, surfaces)
				)
			true_measurements.append(np.mean(all_measurements))

			campaign.add_observation(sample, measurement)

	
	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET

	if is_constrained:
		meas_params = campaign.observations.get_params()
		kcs = [known_constraints[0](param) for param in meas_params]
		assert all(kcs)


def run_cat_general_mixed_cat_disc_cont_func(
		  init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, is_constrained, num_init_design=5):
	""" single categorical general parameter
	"""
	param_space = ParameterSpace()

	# general parameter
	param_space.add(
		ParameterCategorical(
			name='s',
			options=[str(i) for i in range(3)],
			descriptors=[[float(i),float(i)] for i in range(3)],   
		)
	)
	# functional parameters
	param_space.add(
		ParameterCategorical(
			name='x_0',
			options=[f'x{i}' for i in range(5)],
			descriptors=[[float(i),float(i)] for i in range(5)],   
		)
	)
	param_space.add(
		ParameterCategorical(
			name='x_1',
			options=[f'x{i}' for i in range(5)],
			descriptors=[[float(i),float(i)] for i in range(5)],   
		)
	)
	param_space.add(ParameterDiscrete(
		name='x_2',
		low=0.,
		high=1.,
		options=list(np.linspace(0.,1.,8)),
	))
	param_space.add(ParameterDiscrete(
		name='x_3',
		low=0.,
		high=1.,
		options=list(np.linspace(0.,1.,8)),
	))
	param_space.add(ParameterContinuous(name='x_4'))
	param_space.add(ParameterContinuous(name='x_5'))

	surfaces = {}
	problem_gen = ProblemGenerator(problem_type='mixed_cat_disc_cont')
	for general_param_option in param_space[0].options:
		surface_callable, _ = problem_gen.generate_instance()
		surfaces[general_param_option] = surface_callable

	if is_constrained:
		known_constraints = [KnownConstraintsGenerator(is_general=True).get_constraint('cat_disc_cont')]
	else:
		known_constraints = None

	campaign = Campaign()
	campaign.set_param_space(param_space)

	planner = BoTorchPlanner(
		goal='minimize',
		init_design_strategy=init_design_strategy,
		num_init_design=num_init_design,
		batch_size=batch_size,
		use_descriptors=use_descriptors,
		acquisition_type=acquisition_type,
		acquisition_optimizer_kind=acquisition_optimizer,
		general_parameters=[0],
		known_constraints=known_constraints
	)
	planner.set_param_space(param_space)

	BUDGET = num_init_design + batch_size * 4
	true_measurements = []

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)
		for sample in samples:
			measurement = surface_general([sample.x_0, sample.x_1, float(sample.x_2), float(sample.x_3), float(sample.x_4), float(sample.x_5) ], sample.s, surfaces)
			all_measurements = []
			for s in param_space[0].options:
				all_measurements.append(
					surface_general([sample.x_0, sample.x_1, float(sample.x_2), float(sample.x_3), float(sample.x_4), float(sample.x_5) ], sample.s, surfaces)
				)
			true_measurements.append(np.mean(all_measurements))

			campaign.add_observation(sample, measurement)

	
	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET

	if is_constrained:
		meas_params = campaign.observations.get_params()
		kcs = [known_constraints[0](param) for param in meas_params]
		assert all(kcs)



if __name__ == '__main__':

	run_cat_general_cont_func('random', 1, False, 'general', 'pymoo', True, 5) 
	run_cat_general_disc_func('random', 1, False, 'general', 'pymoo', True, 5) 
	run_cat_general_cat_func('random', 1, False, 'general', 'pymoo', True, 5)  
	run_cat_general_mixed_cat_cont_func('random', 1, False, 'general', 'pymoo', True, 5)  
	run_cat_general_mixed_cat_disc_func('random', 1, False, 'general', 'pymoo', True, 5)
	run_cat_general_mixed_disc_cont_func('random', 1, False, 'general', 'pymoo', True, 5)
	run_cat_general_mixed_cat_disc_cont_func('random', 1, False, 'general', 'pymoo', True, 5)

	# init_design_strategy, batch_size, use_descriptors, acquisition_type, acquisition_optimizer, is_constrained, num_init_design=5
	pass
