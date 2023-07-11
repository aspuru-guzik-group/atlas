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


CONT = {
	"init_design_strategy": [
		"random",
		"sobol",
		"lhs",
	],  # init design strategues
	"batch_size": [1],  # batch size
	"use_descriptors": [False],  # use descriptors
}

DISC = {
	"init_design_strategy": ["random"],
	"batch_size": [1],
	"use_descriptors": [False],
}

CAT = {
	"init_design_strategy": ["random"],
	"batch_size": [1],
	"use_descriptors": [False, True],
}

MIXED_CAT_CONT = {
	"init_design_strategy": ["random"],
	"batch_size": [1],
	"use_descriptors": [False, True],
}

MIXED_DISC_CONT = {
	"init_design_strategy": ["random"],
	"batch_size": [1],
	"use_descriptors": [False],
}


MIXED_CAT_DISC = {
	"init_design_strategy": ["random"],
	"batch_size": [1],
	"use_descriptors": [False, True],
}

MIXED_CAT_DISC_CONT = {
	"init_design_strategy": ["random"],
	"batch_size": [1],
	"use_descriptors": [False, True],
}


BATCHED = {
	"problem_type": ['cont', 'disc', 'cat', 'mixed_cat_cont'],
	"init_design_strategy": ["random"],
	"batch_size": [2, 5],
}

SCALARIZER_KINDS = ["WeightedSum", "Parego", "Hypervolume", "Chimera"]


@pytest.mark.parametrize("problem_type", BATCHED["problem_type"])
@pytest.mark.parametrize("init_design_strategy", BATCHED["init_design_strategy"])
@pytest.mark.parametrize("batch_size", BATCHED["batch_size"])
def test_batched(problem_type, init_design_strategy, batch_size):
	# NOTE: always use Hypervolume here to limit the number of tests
	run_batched(problem_type, init_design_strategy, batch_size)


@pytest.mark.parametrize("init_design_strategy", CONT["init_design_strategy"])
@pytest.mark.parametrize("batch_size", CONT["batch_size"])
@pytest.mark.parametrize("use_descriptors", CONT["use_descriptors"])
@pytest.mark.parametrize("scalarizer_kind", SCALARIZER_KINDS)
def test_cont(
	init_design_strategy, batch_size, use_descriptors, scalarizer_kind
):
	run_continuous(
		init_design_strategy, batch_size, use_descriptors, scalarizer_kind
	)


@pytest.mark.parametrize("init_design_strategy", DISC["init_design_strategy"])
@pytest.mark.parametrize("batch_size", DISC["batch_size"])
@pytest.mark.parametrize("use_descriptors", DISC["use_descriptors"])
@pytest.mark.parametrize("scalarizer_kind", SCALARIZER_KINDS)
def test_disc(
	init_design_strategy, batch_size, use_descriptors, scalarizer_kind
):
	run_discrete(
		init_design_strategy, batch_size, use_descriptors, scalarizer_kind
	)


@pytest.mark.parametrize("init_design_strategy", CAT["init_design_strategy"])
@pytest.mark.parametrize("batch_size", CAT["batch_size"])
@pytest.mark.parametrize("use_descriptors", CAT["use_descriptors"])
@pytest.mark.parametrize("scalarizer_kind", SCALARIZER_KINDS)
def test_cat(
	init_design_strategy, batch_size, use_descriptors, scalarizer_kind
):
	run_categorical(
		init_design_strategy, batch_size, use_descriptors, scalarizer_kind
	)


@pytest.mark.parametrize(
	"init_design_strategy", MIXED_CAT_CONT["init_design_strategy"]
)
@pytest.mark.parametrize("batch_size", MIXED_CAT_CONT["batch_size"])
@pytest.mark.parametrize("use_descriptors", MIXED_CAT_CONT["use_descriptors"])
@pytest.mark.parametrize("scalarizer_kind", SCALARIZER_KINDS)
def test_mixed_cat_cont(
	init_design_strategy, batch_size, use_descriptors, scalarizer_kind
):
	run_mixed_cat_cont(
		init_design_strategy, batch_size, use_descriptors, scalarizer_kind
	)


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


def run_batched(problem_type, init_design_strategy, batch_size, ):

	if problem_type == 'cont':
		run_continuous(init_design_strategy, batch_size, False, 'Hypervolume', num_init_design=5,acquisition_type='qei',)
		#pass
	elif problem_type == 'disc':
		run_discrete(init_design_strategy, batch_size, False, 'Hypervolume', num_init_design=5,acquisition_type='qei',)
		#pass
	elif problem_type == 'cat':
		run_categorical(init_design_strategy, batch_size, True, 'Hypervolume', num_init_design=5,acquisition_type='qei',)
		#pass
	elif problem_type == 'mixed_cat_cont': 
		#pass
		run_mixed_cat_cont(init_design_strategy, batch_size, True, 'Hypervolume', num_init_design=5,acquisition_type='qei',)
	else:
		pass


def run_continuous(
	init_design_strategy,
	batch_size,
	use_descriptors,
	scalarizer_kind,
	acquisition_optimizer,
	num_init_design=5,
):

	problem_gen = ProblemGenerator(problem_type='continuous', is_moo=True)	
	surface_callable, param_space = problem_gen.generate_instance()
	
	scalarizer, moo_params, goals = generate_scalarizer_object(
		scalarizer_kind, surface_callable.value_space
	)

	planner = BoTorchPlanner(
		goal="minimize",
		init_design_strategy=init_design_strategy,
		batch_size=batch_size,
		use_descriptors=use_descriptors,
		is_moo=True,
		acquisition_optimizer_kind=acquisition_optimizer,
		value_space=surface_callable.value_space,
		scalarizer_kind=scalarizer_kind,
		moo_params=moo_params,
		goals=goals,
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


def run_categorical(
	init_design_strategy,
	batch_size,
	use_descriptors,
	scalarizer_kind,
	num_init_design=5,
	acquisition_type='ei', 
):

	moo_surface = Dataset(kind="redoxmers")

	scalarizer, moo_params, goals = generate_scalarizer_object(
		scalarizer_kind, moo_surface.value_space
	)

	planner = BoTorchPlanner(
		goal="minimize",
		feas_strategy="naive-0",
		init_design_strategy=init_design_strategy,
		batch_size=batch_size,
		use_descriptors=use_descriptors,
		acquisition_type=acquisition_type, 
		is_moo=True,
		value_space=moo_surface.value_space,
		scalarizer_kind=scalarizer_kind,
		moo_params=moo_params,
		goals=goals,
	)

	planner.set_param_space(moo_surface.param_space)

	campaign = Campaign()
	campaign.set_param_space(moo_surface.param_space)
	campaign.set_value_space(moo_surface.value_space)

	BUDGET = num_init_design + batch_size * 4

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)

		for sample in samples:
			sample_arr = sample.to_array()
			measurement = moo_surface.run(sample_arr, return_paramvector=True)
			campaign.add_and_scalarize(sample_arr, measurement, scalarizer)

	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET
	assert campaign.observations.get_values().shape[1] == len(
		moo_surface.value_space
	)


def run_discrete(
	init_design_strategy,
	batch_size,
	use_descriptors,
	scalarizer_kind,
	acquisition_optimizer,
	num_init_design=5,
):

	problem_gen = ProblemGenerator(problem_type='discrete', is_moo=True)	
	surface_callable, param_space = problem_gen.generate_instance()
	
	scalarizer, moo_params, goals = generate_scalarizer_object(
		scalarizer_kind, surface_callable.value_space
	)

	planner = BoTorchPlanner(
		goal="minimize",
		init_design_strategy=init_design_strategy,
		batch_size=batch_size,
		use_descriptors=use_descriptors,
		is_moo=True,
		acquisition_optimizer_kind=acquisition_optimizer,
		value_space=surface_callable.value_space,
		scalarizer_kind=scalarizer_kind,
		moo_params=moo_params,
		goals=goals,
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


def run_mixed_cat_cont(
	init_design_strategy,
	batch_size,
	use_descriptors,
	scalarizer_kind,
	num_init_design=5,
	acquisition_type='ei', 
):

	moo_surface = Emulator(dataset="suzuki_i", model="BayesNeuralNet")

	scalarizer, moo_params, goals = generate_scalarizer_object(
		scalarizer_kind, moo_surface.value_space
	)

	planner = BoTorchPlanner(
		goal="minimize",
		feas_strategy="naive-0",
		init_design_strategy=init_design_strategy,
		batch_size=batch_size,
		use_descriptors=use_descriptors,
		acquisition_type=acquisition_type,
		is_moo=True,
		value_space=moo_surface.value_space,
		scalarizer_kind=scalarizer_kind,
		moo_params=moo_params,
		goals=goals,
	)

	planner.set_param_space(moo_surface.param_space)

	campaign = Campaign()
	campaign.set_param_space(moo_surface.param_space)
	campaign.set_value_space(moo_surface.value_space)

	BUDGET = num_init_design + batch_size * 4

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)

		for sample in samples:
			measurement, _, __ = moo_surface.run(sample, return_paramvector=True)
			campaign.add_and_scalarize(sample, measurement, scalarizer)

	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET
	assert campaign.observations.get_values().shape[1] == len(
		moo_surface.value_space
	)


def run_mixed_disc_cont(
	init_design_strategy,
	batch_size,
	use_descriptors,
	scalarizer_kind,
	num_init_design=5,
	acquisition_type='ei', 
):

	problem_gen = ProblemGenerator(problem_type='mixed_disc_cont', is_moo=True)	
	surface_callable, param_space = problem_gen.generate_instance()

	scalarizer, moo_params, goals = generate_scalarizer_object(
		scalarizer_kind, surface_callable.value_space
	)

	planner = BoTorchPlanner(
		goal="minimize",
		feas_strategy="naive-0",
		init_design_strategy=init_design_strategy,
		batch_size=batch_size,
		use_descriptors=use_descriptors,
		acquisition_type=acquisition_type,
		is_moo=True,
		value_space=surface_callable.value_space,
		scalarizer_kind=scalarizer_kind,
		moo_params=moo_params,
		goals=goals,
	)

	planner.set_param_space(surface_callable.param_space)

	campaign = Campaign()
	campaign.set_param_space(surface_callable.param_space)
	campaign.set_value_space(surface_callable.value_space)

	BUDGET = num_init_design + batch_size * 4

	while len(campaign.observations.get_values()) < BUDGET:

		samples = planner.recommend(campaign.observations)

		for sample in samples:
			measurement, _, __ = surface_callable.run(sample, return_paramvector=True)
			campaign.add_and_scalarize(sample, measurement, scalarizer)

	assert len(campaign.observations.get_params()) == BUDGET
	assert len(campaign.observations.get_values()) == BUDGET
	assert campaign.observations.get_values().shape[1] == len(
		surface_callable.value_space
	)


if __name__ == "__main__":
	# run_mixed_cat_cont(
	# 	init_design_strategy='random',
	# 	batch_size=1,
	# 	use_descriptors=False,
	# 	scalarizer_kind='Chimera',
	# )

	# run_continuous('random', 1, False, 'Hypervolume', 'pymoo', 5)
	# run_discrete('random', 1, False, 'Hypervolume', 'pymoo', 5)
	# run_categorical('random', 1, False, 'Hypervolume', 'pymoo', 5)
	# run_mixed_disc_cont('random', 1, False, 'Hypervolume', 'pymoo', 5)
	
