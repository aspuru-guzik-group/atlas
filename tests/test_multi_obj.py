#!/usr/bin/env python

import numpy as np
import pytest
from olympus.campaigns import Campaign, ParameterSpace
from olympus.datasets import Dataset
from olympus.emulators import Emulator
from olympus.objects import (
	ParameterCategorical,
	ParameterContinuous,
	ParameterDiscrete,
	ParameterVector,
)
from olympus.scalarizers import Scalarizer
from olympus.surfaces import Surface

from atlas.planners.gp.planner import BoTorchPlanner
from problem_generator import ProblemGenerator
from problem_generator import KnownConstraintsGenerator
from problem_generator import HybridSurface

#SCALARIZER_KINDS = ["WeightedSum", "Parego", "Hypervolume", "Chimera"]
SCALARIZER_KINDS = ['Hypervolume']
IS_CONSTRAINED = [True] # [False, True ]

CONT = {
	"init_design_strategy": ["random"],  # init design strategies
	"batch_size": [1],  # batch size
	"use_descriptors": [False],  # use descriptors
	"scalarizer_kind": SCALARIZER_KINDS,
	"acquisition_kind": ['ucb'],
	"acquisition_optimizer": ['pymoo'],
	"num_init_design": [5],
	"is_constrained": IS_CONSTRAINED, 
}


DISC = {
	"init_design_strategy": ["random"], 
	"batch_size": [1], 
	"use_descriptors": [False],  
	"scalarizer_kind": SCALARIZER_KINDS,
	"acquisition_kind": ['ucb'],
	"acquisition_optimizer": ['pymoo'],
	"num_init_design": [5],
	"is_constrained": IS_CONSTRAINED, 
}


CAT = {
	"init_design_strategy": ["random"], 
	"batch_size": [1], 
	"use_descriptors": [False, True],  
	"scalarizer_kind": SCALARIZER_KINDS,
	"acquisition_kind": ['ucb'],
	"acquisition_optimizer": ['pymoo'],
	"num_init_design": [5],
	"is_constrained": IS_CONSTRAINED,  
}


MIXED_CAT_CONT = {
	"init_design_strategy": ["random"], 
	"batch_size": [1], 
	"use_descriptors": [False, True],  
	"scalarizer_kind": SCALARIZER_KINDS,
	"acquisition_kind": ['ucb'],
	"acquisition_optimizer": ['pymoo'],
	"num_init_design": [5],
	"is_constrained": IS_CONSTRAINED, 
}


MIXED_DISC_CONT = {
	"init_design_strategy": ["random"], 
	"batch_size": [1], 
	"use_descriptors": [False],  
	"scalarizer_kind": SCALARIZER_KINDS,
	"acquisition_kind": ['ucb'],
	"acquisition_optimizer": ['pymoo'],
	"num_init_design": [5],
	"is_constrained": IS_CONSTRAINED, 
}


MIXED_CAT_DISC = {
	"init_design_strategy": ["random"], 
	"batch_size": [1], 
	"use_descriptors": [False, True],  
	"scalarizer_kind": SCALARIZER_KINDS,
	"acquisition_kind": ['ucb'],
	"acquisition_optimizer": ['pymoo'],
	"num_init_design": [5],
	"is_constrained": IS_CONSTRAINED,  
}


MIXED_CAT_DISC_CONT = {
	"init_design_strategy": ["random"], 
	"batch_size": [1], 
	"use_descriptors": [False, True],  
	"scalarizer_kind": SCALARIZER_KINDS,
	"acquisition_kind": ['ucb'],
	"acquisition_optimizer": ['pymoo'],
	"num_init_design": [5],
	"is_constrained": IS_CONSTRAINED,  
}


BATCHED = {
	"problem_type": [
		'cont', 'disc', 'cat', 'mixed_cat_cont',
		'mixed_disc_cont', 'mixed_cat_disc', 'mixed_cat_disc_cont'
	],
	"init_design_strategy": ["random"],
	"batch_size": [2, 4],
	"scalarizer_kind": SCALARIZER_KINDS,
	"acquisition_kind": ['ucb'],
	"acquisition_optimizer": ['pymoo'],
	"num_init_design": [4],
	"is_constrained": IS_CONSTRAINED,  
}



@pytest.mark.parametrize("problem_type", BATCHED["problem_type"])
@pytest.mark.parametrize("init_design_strategy", BATCHED["init_design_strategy"])
@pytest.mark.parametrize("batch_size", BATCHED["batch_size"])
@pytest.mark.parametrize("scalarizer_kind", BATCHED["scalarizer_kind"])
@pytest.mark.parametrize("acquisition_kind", BATCHED["acquisition_kind"])
@pytest.mark.parametrize("acquisition_optimizer", BATCHED["acquisition_optimizer"])
@pytest.mark.parametrize("num_init_design", BATCHED["num_init_design"])
@pytest.mark.parametrize("is_constrained", BATCHED["is_constrained"])
def test_batched(problem_type, init_design_strategy, batch_size, scalarizer_kind,
	acquisition_kind, acquisition_optimizer, num_init_design, is_constrained):
	# NOTE: always use Hypervolume here to limit the number of tests
	run_batched(problem_type, init_design_strategy, batch_size, scalarizer_kind,
	acquisition_kind, acquisition_optimizer, num_init_design, is_constrained)


@pytest.mark.parametrize("init_design_strategy", CONT["init_design_strategy"])
@pytest.mark.parametrize("batch_size", CONT["batch_size"])
@pytest.mark.parametrize("use_descriptors", CONT["use_descriptors"])
@pytest.mark.parametrize("scalarizer_kind", CONT["scalarizer_kind"])
@pytest.mark.parametrize("acquisition_kind", CONT["acquisition_kind"])
@pytest.mark.parametrize("acquisition_optimizer", CONT["acquisition_optimizer"])
@pytest.mark.parametrize("num_init_design", CONT["num_init_design"])
@pytest.mark.parametrize("is_constrained", CONT["is_constrained"])
def test_cont(
	init_design_strategy, batch_size, use_descriptors, scalarizer_kind,
	acquisition_kind, acquisition_optimizer, num_init_design, is_constrained
):
	run_continuous(
		init_design_strategy, batch_size, use_descriptors, scalarizer_kind,
		acquisition_kind, acquisition_optimizer, num_init_design, is_constrained
	)


@pytest.mark.parametrize("init_design_strategy", DISC["init_design_strategy"])
@pytest.mark.parametrize("batch_size", DISC["batch_size"])
@pytest.mark.parametrize("use_descriptors", DISC["use_descriptors"])
@pytest.mark.parametrize("scalarizer_kind", DISC["scalarizer_kind"])
@pytest.mark.parametrize("acquisition_kind", DISC["acquisition_kind"])
@pytest.mark.parametrize("acquisition_optimizer", DISC["acquisition_optimizer"])
@pytest.mark.parametrize("num_init_design", DISC["num_init_design"])
@pytest.mark.parametrize("is_constrained", DISC["is_constrained"])
def test_disc(
	init_design_strategy, batch_size, use_descriptors, scalarizer_kind,
	acquisition_kind, acquisition_optimizer, num_init_design, is_constrained
):
	run_discrete(
		init_design_strategy, batch_size, use_descriptors, scalarizer_kind,
		acquisition_kind, acquisition_optimizer, num_init_design, is_constrained
	)


@pytest.mark.parametrize("init_design_strategy", CAT["init_design_strategy"])
@pytest.mark.parametrize("batch_size", CAT["batch_size"])
@pytest.mark.parametrize("use_descriptors", CAT["use_descriptors"])
@pytest.mark.parametrize("scalarizer_kind", CAT["scalarizer_kind"])
@pytest.mark.parametrize("acquisition_kind", CAT["acquisition_kind"])
@pytest.mark.parametrize("acquisition_optimizer", CAT["acquisition_optimizer"])
@pytest.mark.parametrize("num_init_design", CAT["num_init_design"])
@pytest.mark.parametrize("is_constrained", CAT["is_constrained"])
def test_cat(
	init_design_strategy, batch_size, use_descriptors, scalarizer_kind,
	acquisition_kind, acquisition_optimizer, num_init_design, is_constrained
):
	run_categorical(
		init_design_strategy, batch_size, use_descriptors, scalarizer_kind,
		acquisition_kind, acquisition_optimizer, num_init_design, is_constrained
	)


@pytest.mark.parametrize("init_design_strategy", MIXED_CAT_CONT["init_design_strategy"])
@pytest.mark.parametrize("batch_size", MIXED_CAT_CONT["batch_size"])
@pytest.mark.parametrize("use_descriptors", MIXED_CAT_CONT["use_descriptors"])
@pytest.mark.parametrize("scalarizer_kind", MIXED_CAT_CONT["scalarizer_kind"])
@pytest.mark.parametrize("acquisition_kind", MIXED_CAT_CONT["acquisition_kind"])
@pytest.mark.parametrize("acquisition_optimizer", MIXED_CAT_CONT["acquisition_optimizer"])
@pytest.mark.parametrize("num_init_design", MIXED_CAT_CONT["num_init_design"])
@pytest.mark.parametrize("is_constrained", MIXED_CAT_CONT["is_constrained"])
def test_mixed_cat_cont(
	init_design_strategy, batch_size, use_descriptors, scalarizer_kind,
	acquisition_kind, acquisition_optimizer, num_init_design, is_constrained
):
	run_mixed_cat_cont(
		init_design_strategy, batch_size, use_descriptors, scalarizer_kind,
		acquisition_kind, acquisition_optimizer, num_init_design, is_constrained
	)


@pytest.mark.parametrize("init_design_strategy", MIXED_DISC_CONT["init_design_strategy"])
@pytest.mark.parametrize("batch_size", MIXED_DISC_CONT["batch_size"])
@pytest.mark.parametrize("use_descriptors", MIXED_DISC_CONT["use_descriptors"])
@pytest.mark.parametrize("scalarizer_kind", MIXED_DISC_CONT["scalarizer_kind"])
@pytest.mark.parametrize("acquisition_kind", MIXED_DISC_CONT["acquisition_kind"])
@pytest.mark.parametrize("acquisition_optimizer", MIXED_DISC_CONT["acquisition_optimizer"])
@pytest.mark.parametrize("num_init_design", MIXED_DISC_CONT["num_init_design"])
@pytest.mark.parametrize("is_constrained", MIXED_DISC_CONT["is_constrained"])
def test_mixed_disc_cont(
	init_design_strategy, batch_size, use_descriptors, scalarizer_kind,
	acquisition_kind, acquisition_optimizer, num_init_design, is_constrained
):
	run_mixed_disc_cont(
		init_design_strategy, batch_size, use_descriptors, scalarizer_kind,
		acquisition_kind, acquisition_optimizer, num_init_design, is_constrained
	)


@pytest.mark.parametrize("init_design_strategy", MIXED_CAT_DISC["init_design_strategy"])
@pytest.mark.parametrize("batch_size", MIXED_CAT_DISC["batch_size"])
@pytest.mark.parametrize("use_descriptors", MIXED_CAT_DISC["use_descriptors"])
@pytest.mark.parametrize("scalarizer_kind", MIXED_CAT_DISC["scalarizer_kind"])
@pytest.mark.parametrize("acquisition_kind", MIXED_CAT_DISC["acquisition_kind"])
@pytest.mark.parametrize("acquisition_optimizer", MIXED_CAT_DISC["acquisition_optimizer"])
@pytest.mark.parametrize("num_init_design", MIXED_CAT_DISC["num_init_design"])
@pytest.mark.parametrize("is_constrained", MIXED_CAT_DISC["is_constrained"])
def test_mixed_cat_disc(
	init_design_strategy, batch_size, use_descriptors, scalarizer_kind,
	acquisition_kind, acquisition_optimizer, num_init_design, is_constrained
):
	run_mixed_cat_disc(
		init_design_strategy, batch_size, use_descriptors, scalarizer_kind,
		acquisition_kind, acquisition_optimizer, num_init_design, is_constrained
	)

@pytest.mark.parametrize("init_design_strategy", MIXED_CAT_DISC_CONT["init_design_strategy"])
@pytest.mark.parametrize("batch_size", MIXED_CAT_DISC_CONT["batch_size"])
@pytest.mark.parametrize("use_descriptors", MIXED_CAT_DISC_CONT["use_descriptors"])
@pytest.mark.parametrize("scalarizer_kind", MIXED_CAT_DISC_CONT["scalarizer_kind"])
@pytest.mark.parametrize("acquisition_kind", MIXED_CAT_DISC_CONT["acquisition_kind"])
@pytest.mark.parametrize("acquisition_optimizer", MIXED_CAT_DISC_CONT["acquisition_optimizer"])
@pytest.mark.parametrize("num_init_design", MIXED_CAT_DISC_CONT["num_init_design"])
@pytest.mark.parametrize("is_constrained", MIXED_CAT_DISC_CONT["is_constrained"])
def test_mixed_cat_disc_cont(
	init_design_strategy, batch_size, use_descriptors, scalarizer_kind,
	acquisition_kind, acquisition_optimizer, num_init_design, is_constrained
):
	run_mixed_cat_disc_cont(
		init_design_strategy, batch_size, use_descriptors, scalarizer_kind,
		acquisition_kind, acquisition_optimizer, num_init_design, is_constrained
	)



#------------------
# HELPER FUNCTIONS
#------------------

def generate_scalarizer_object(scalarizer_kind, value_space):

	if scalarizer_kind == "WeightedSum":
		moo_params = {
			"weights": np.random.randint(1, 5, size=len(value_space))
		}
	elif scalarizer_kind == "Parego":
		moo_params = {}
	elif scalarizer_kind == "Hypervolume":
		moo_params = {}
	elif scalarizer_kind == "Chimera":
		moo_params = {
			"absolutes": [False for _ in range(len(value_space))],
			"tolerances": np.random.rand(len(value_space)),
		}

	goals = np.random.choice(["min", "max"], size=len(value_space))

	scalarizer = Scalarizer(
		kind=scalarizer_kind,
		value_space=value_space,
		goals=goals,
		**moo_params,
	)

	return scalarizer, moo_params, goals

#------------------

def run_batched(problem_type, init_design_strategy, batch_size, scalarizer_kind,
	acquisition_kind, acquisition_optimizer, num_init_design, is_constrained):

	if problem_type == 'cont':
		run_continuous(init_design_strategy, batch_size, False, scalarizer_kind,
			acquisition_kind, acquisition_optimizer, num_init_design, is_constrained)
	elif problem_type == 'disc':
		run_discrete(init_design_strategy, batch_size, False, scalarizer_kind,
			acquisition_kind, acquisition_optimizer, num_init_design, is_constrained)
	elif problem_type == 'cat':
		run_categorical(init_design_strategy, batch_size, True, scalarizer_kind,
			acquisition_kind, acquisition_optimizer, num_init_design, is_constrained)
	elif problem_type == 'mixed_cat_cont': 
		run_mixed_cat_cont(init_design_strategy, batch_size, True, scalarizer_kind,
			acquisition_kind, acquisition_optimizer, num_init_design, is_constrained)
	elif problem_type == 'mixed_disc_cont': 
		run_mixed_cat_cont(init_design_strategy, batch_size, False, scalarizer_kind,
			acquisition_kind, acquisition_optimizer, num_init_design, is_constrained)
	elif problem_type == 'mixed_cat_disc': 
		run_mixed_cat_cont(init_design_strategy, batch_size, True, scalarizer_kind,
			acquisition_kind, acquisition_optimizer, num_init_design, is_constrained)
	elif problem_type == 'mixed_cat_disc_cont': 
		run_mixed_cat_cont(init_design_strategy, batch_size, True, scalarizer_kind,
			acquisition_kind, acquisition_optimizer, num_init_design, is_constrained)
	else:
		pass


def run_continuous(
	init_design_strategy, 
	batch_size, 
	use_descriptors, 
	scalarizer_kind,
	acquisition_kind, 
	acquisition_optimizer, 
	num_init_design, 
	is_constrained,
	):

	problem_gen = ProblemGenerator(problem_type='continuous', is_moo=True)	
	surface_callable, param_space = problem_gen.generate_instance()

	if is_constrained:
		known_constraints = [KnownConstraintsGenerator().get_constraint('continuous')]
	else:
		known_constraints = None
	
	scalarizer, moo_params, goals = generate_scalarizer_object(
		scalarizer_kind, surface_callable.value_space
	)

	planner = BoTorchPlanner(
		goal="minimize",
		init_design_strategy=init_design_strategy,
		num_init_design=num_init_design,
		batch_size=batch_size,
		use_descriptors=use_descriptors,
		is_moo=True,
		acquisition_kind=acquisition_kind,
		acquisition_optimizer_kind=acquisition_optimizer,
		value_space=surface_callable.value_space,
		scalarizer_kind=scalarizer_kind,
		moo_params=moo_params,
		goals=goals,
		known_constraints=known_constraints,
	)

	planner.set_param_space(surface_callable.param_space)

	campaign = Campaign()
	campaign.set_param_space(surface_callable.param_space)
	campaign.set_value_space(surface_callable.value_space)

	BUDGET = num_init_design + batch_size * 4

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)

		for sample in samples:
			measurement = surface_callable.run(sample, return_paramvector=True)
			campaign.add_and_scalarize(sample, measurement, scalarizer)

	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET
	assert campaign.observations.get_values().shape[1] == len(
		surface_callable.value_space
	)
	
	if is_constrained:
		meas_params = campaign.observations.get_params()
		kcs = [known_constraints[0](param) for param in meas_params]
		assert all(kcs)


def run_discrete(	
	init_design_strategy, 
	batch_size, 
	use_descriptors, 
	scalarizer_kind,
	acquisition_kind, 
	acquisition_optimizer, 
	num_init_design, 
	is_constrained,
	):

	problem_gen = ProblemGenerator(problem_type='discrete', is_moo=True)	
	surface_callable, param_space = problem_gen.generate_instance()
	
	if is_constrained:
		known_constraints = [KnownConstraintsGenerator().get_constraint('discrete')]
	else:
		known_constraints = None

	scalarizer, moo_params, goals = generate_scalarizer_object(
		scalarizer_kind, surface_callable.value_space
	)

	planner = BoTorchPlanner(
		goal="minimize",
		init_design_strategy=init_design_strategy,
		batch_size=batch_size,
		use_descriptors=use_descriptors,
		num_init_design=num_init_design,
		is_moo=True,
		acquisition_kind=acquisition_kind,
		acquisition_optimizer_kind=acquisition_optimizer,
		value_space=surface_callable.value_space,
		scalarizer_kind=scalarizer_kind,
		moo_params=moo_params,
		goals=goals,
		known_constraints=known_constraints,
	)

	planner.set_param_space(surface_callable.param_space)

	campaign = Campaign()
	campaign.set_param_space(surface_callable.param_space)
	campaign.set_value_space(surface_callable.value_space)

	BUDGET = num_init_design + batch_size * 4

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)

		for sample in samples:
			measurement = surface_callable.run(sample, return_paramvector=True)
			campaign.add_and_scalarize(sample, measurement, scalarizer)

	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET
	assert campaign.observations.get_values().shape[1] == len(
		surface_callable.value_space
	)
	
	if is_constrained:
		meas_params = campaign.observations.get_params()
		kcs = [known_constraints[0](param) for param in meas_params]
		assert all(kcs)


def run_categorical(
	init_design_strategy, 
	batch_size, 
	use_descriptors, 
	scalarizer_kind,
	acquisition_kind, 
	acquisition_optimizer, 
	num_init_design, 
	is_constrained,
	):

	problem_gen = ProblemGenerator(problem_type='categorical', is_moo=True)	
	surface_callable, param_space = problem_gen.generate_instance()

	if is_constrained:
		known_constraints = [KnownConstraintsGenerator().get_constraint('categorical')]
	else:
		known_constraints = None

	scalarizer, moo_params, goals = generate_scalarizer_object(
		scalarizer_kind, surface_callable.value_space
	)

	planner = BoTorchPlanner(
		goal="minimize",
		init_design_strategy=init_design_strategy,
		num_init_design=num_init_design,
		batch_size=batch_size,
		use_descriptors=use_descriptors,
		is_moo=True,
		acquisition_kind=acquisition_kind,
		acquisition_optimizer_kind=acquisition_optimizer,
		value_space=surface_callable.value_space,
		scalarizer_kind=scalarizer_kind,
		moo_params=moo_params,
		goals=goals,
		known_constraints=known_constraints,
	)

	planner.set_param_space(surface_callable.param_space)

	campaign = Campaign()
	campaign.set_param_space(surface_callable.param_space)
	campaign.set_value_space(surface_callable.value_space)

	BUDGET = num_init_design + batch_size * 4

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)

		for sample in samples:
			measurement = surface_callable.run(sample)
			campaign.add_and_scalarize(sample, measurement, scalarizer)

	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET
	assert campaign.observations.get_values().shape[1] == len(
		surface_callable.value_space
	)
	
	if is_constrained:
		meas_params = campaign.observations.get_params()
		kcs = [known_constraints[0](param) for param in meas_params]
		assert all(kcs)


def run_mixed_cat_cont(
	init_design_strategy, 
	batch_size, 
	use_descriptors, 
	scalarizer_kind,
	acquisition_kind, 
	acquisition_optimizer, 
	num_init_design, 
	is_constrained,
	):

	problem_gen = ProblemGenerator(problem_type='mixed_cat_cont', is_moo=True)	
	surface_callable, param_space = problem_gen.generate_instance()

	if is_constrained:
		known_constraints = [KnownConstraintsGenerator().get_constraint('cat_cont')]
	else:
		known_constraints = None

	scalarizer, moo_params, goals = generate_scalarizer_object(
		scalarizer_kind, surface_callable.value_space
	)

	planner = BoTorchPlanner(
		goal="minimize",
		init_design_strategy=init_design_strategy,
		num_init_design=num_init_design,
		batch_size=batch_size,
		use_descriptors=use_descriptors,
		is_moo=True,
		acquisition_kind=acquisition_kind,
		acquisition_optimizer_kind=acquisition_optimizer,
		value_space=surface_callable.value_space,
		scalarizer_kind=scalarizer_kind,
		moo_params=moo_params,
		goals=goals,
		known_constraints=known_constraints,
	)

	planner.set_param_space(surface_callable.param_space)

	campaign = Campaign()
	campaign.set_param_space(surface_callable.param_space)
	campaign.set_value_space(surface_callable.value_space)

	BUDGET = num_init_design + batch_size * 4

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)

		for sample in samples:
			measurement = surface_callable.run(sample)
			campaign.add_and_scalarize(sample, measurement, scalarizer)

	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET
	assert campaign.observations.get_values().shape[1] == len(
		surface_callable.value_space
	)

	if is_constrained:
		meas_params = campaign.observations.get_params()
		kcs = [known_constraints[0](param) for param in meas_params]
		assert all(kcs)


def run_mixed_disc_cont(
	init_design_strategy, 
	batch_size, 
	use_descriptors, 
	scalarizer_kind,
	acquisition_kind, 
	acquisition_optimizer, 
	num_init_design, 
	is_constrained,

	):

	problem_gen = ProblemGenerator(problem_type='mixed_disc_cont', is_moo=True)	
	surface_callable, param_space = problem_gen.generate_instance()

	if is_constrained:
		known_constraints = [KnownConstraintsGenerator().get_constraint('disc_cont')]
	else:
		known_constraints = None

	scalarizer, moo_params, goals = generate_scalarizer_object(
		scalarizer_kind, surface_callable.value_space
	)

	planner = BoTorchPlanner(
		goal="minimize",
		init_design_strategy=init_design_strategy,
		num_init_design=num_init_design,
		batch_size=batch_size,
		use_descriptors=use_descriptors,
		is_moo=True,
		acquisition_kind=acquisition_kind,
		acquisition_optimizer_kind=acquisition_optimizer,
		value_space=surface_callable.value_space,
		scalarizer_kind=scalarizer_kind,
		moo_params=moo_params,
		goals=goals,
		known_constraints=known_constraints,
	)

	planner.set_param_space(surface_callable.param_space)

	campaign = Campaign()
	campaign.set_param_space(surface_callable.param_space)
	campaign.set_value_space(surface_callable.value_space)

	BUDGET = num_init_design + batch_size * 4

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)

		for sample in samples:
			measurement = surface_callable.run(sample)
			campaign.add_and_scalarize(sample, measurement, scalarizer)

	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET
	assert campaign.observations.get_values().shape[1] == len(
		surface_callable.value_space
	)
	
	if is_constrained:
		meas_params = campaign.observations.get_params()
		kcs = [known_constraints[0](param) for param in meas_params]
		assert all(kcs)


def run_mixed_cat_disc(
	init_design_strategy, 
	batch_size, 
	use_descriptors, 
	scalarizer_kind,
	acquisition_kind, 
	acquisition_optimizer, 
	num_init_design, 
	is_constrained,
	):

	problem_gen = ProblemGenerator(problem_type='mixed_cat_disc', is_moo=True)	
	surface_callable, param_space = problem_gen.generate_instance()

	if is_constrained:
		known_constraints = [KnownConstraintsGenerator().get_constraint('cat_disc')]
	else:
		known_constraints = None

	scalarizer, moo_params, goals = generate_scalarizer_object(
		scalarizer_kind, surface_callable.value_space
	)

	planner = BoTorchPlanner(
		goal="minimize",
		init_design_strategy=init_design_strategy,
		num_init_design=num_init_design,
		batch_size=batch_size,
		use_descriptors=use_descriptors,
		is_moo=True,
		acquisition_kind=acquisition_kind,
		acquisition_optimizer_kind=acquisition_optimizer,
		value_space=surface_callable.value_space,
		scalarizer_kind=scalarizer_kind,
		moo_params=moo_params,
		goals=goals,
		known_constraints=known_constraints,
	)

	planner.set_param_space(surface_callable.param_space)

	campaign = Campaign()
	campaign.set_param_space(surface_callable.param_space)
	campaign.set_value_space(surface_callable.value_space)

	BUDGET = num_init_design + batch_size * 4

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)

		for sample in samples:
			measurement = surface_callable.run(sample)
			campaign.add_and_scalarize(sample, measurement, scalarizer)

	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET
	assert campaign.observations.get_values().shape[1] == len(
		surface_callable.value_space
	)

	if is_constrained:
		meas_params = campaign.observations.get_params()
		kcs = [known_constraints[0](param) for param in meas_params]
		assert all(kcs)


def run_mixed_cat_disc_cont(
	init_design_strategy, 
	batch_size, 
	use_descriptors, 
	scalarizer_kind,
	acquisition_kind, 
	acquisition_optimizer, 
	num_init_design, 
	is_constrained,
	):

	problem_gen = ProblemGenerator(problem_type='mixed_cat_disc_cont', is_moo=True)	
	surface_callable, param_space = problem_gen.generate_instance()

	if is_constrained:
		known_constraints = [KnownConstraintsGenerator().get_constraint('cat_disc_cont')]
	else:
		known_constraints = None

	scalarizer, moo_params, goals = generate_scalarizer_object(
		scalarizer_kind, surface_callable.value_space
	)

	planner = BoTorchPlanner(
		goal="minimize",
		init_design_strategy=init_design_strategy,
		num_init_design=num_init_design,
		batch_size=batch_size,
		use_descriptors=use_descriptors,
		is_moo=True,
		acquisition_kind=acquisition_kind,
		acquisition_optimizer_kind=acquisition_optimizer,
		value_space=surface_callable.value_space,
		scalarizer_kind=scalarizer_kind,
		moo_params=moo_params,
		goals=goals,
		known_constraints=known_constraints,
	)

	planner.set_param_space(surface_callable.param_space)

	campaign = Campaign()
	campaign.set_param_space(surface_callable.param_space)
	campaign.set_value_space(surface_callable.value_space)

	BUDGET = num_init_design + batch_size * 4

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)

		for sample in samples:
			measurement = surface_callable.run(sample)
			campaign.add_and_scalarize(sample, measurement, scalarizer)

	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET
	assert campaign.observations.get_values().shape[1] == len(
		surface_callable.value_space
	)

	if is_constrained:
		meas_params = campaign.observations.get_params()
		kcs = [known_constraints[0](param) for param in meas_params]
		assert all(kcs)


if __name__ == "__main__":

	#run_continuous('random', 2, False, 'Hypervolume', 'ucb', 'pymoo', 4, True)
	run_discrete('random', 2, False, 'Hypervolume', 'ucb', 'pymoo', 4, True)
	# run_categorical('random', 1, False, 'Hypervolume', 'pymoo', 5, True)
	# run_mixed_disc_cont('random', 1, False, 'Hypervolume', 'pymoo', 5, True)
	# run_mixed_cat_disc('random', 1, False, 'Hypervolume', 'pymoo', 5, True)
	# run_mixed_cat_cont('random', 1, False, 'Hypervolume', 'pymoo', 5, True)
	run_mixed_cat_disc_cont('random', 1, False, 'Hypervolume', 'pymoo', 5, True)
	
