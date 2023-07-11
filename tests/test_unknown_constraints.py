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
from problem_generator import ProblemGenerator, KnownConstraintsGenerator
from problem_generator import HybridSurface

CONT = {
    "init_design_strategy": [
        "random",
    ],  # init design strategues
    "batch_size": [1, 2],  # batch size
    "feas_strategy_param": [
        "naive-0_0",
        "fia_1000",
        "fwa_0",
        "fca_0.2",
        "fca_0.8",
        "fia_0.5",
        "fia_2.0",
    ],
    "use_descriptors": [False],  # use descriptors
    "acquisition_optimizer": ['gradient', 'genetic'],
}

CAT = {
    "init_design_strategy": [
        "random",
    ],  # init design strategues
    "batch_size": [1],  # batch size
    "feas_strategy_param": [
        "naive-0_0",
        "fia_1000",
        "fwa_0",
        "fca_0.2",
        "fca_0.8",
        "fia_0.5",
        "fia_2.0",
    ],
    "use_descriptors": [False, True],  # use descriptors
    "acquisition_optimizer": ['gradient', 'genetic'],
}


@pytest.mark.parametrize("init_design_strategy", CONT["init_design_strategy"])
@pytest.mark.parametrize("batch_size", CONT["batch_size"])
@pytest.mark.parametrize("feas_strategy_param", CONT["feas_strategy_param"])
@pytest.mark.parametrize("use_descriptors", CONT["use_descriptors"])
@pytest.mark.parametrize("acquisition_optimizer", CONT["acquisition_optimizer"])
def test_unknown_cont(
    init_design_strategy, batch_size, feas_strategy_param, use_descriptors, acquisition_optimizer,
):
    run_continuous(
        init_design_strategy, batch_size, feas_strategy_param, use_descriptors, acquisition_optimizer,
    )


@pytest.mark.parametrize("init_design_strategy", CAT["init_design_strategy"])
@pytest.mark.parametrize("batch_size", CAT["batch_size"])
@pytest.mark.parametrize("feas_strategy_param", CAT["feas_strategy_param"])
@pytest.mark.parametrize("use_descriptors", CAT["use_descriptors"])
@pytest.mark.parametrize("acquisition_optimizer", CAT["acquisition_optimizer"])
def test_unknown_cat(
    init_design_strategy, batch_size, feas_strategy_param, use_descriptors, acquisition_optimizer,
):
    run_categorical(
        init_design_strategy, batch_size, feas_strategy_param, use_descriptors, acquisition_optimizer,
    )


def run_continuous(
    init_design_strategy,
    batch_size,
    feas_strategy_param,
    use_descriptors,
    acquisition_optimizer,
    num_init_design=5,
):
    problem_gen = ProblemGenerator(problem_type='continuous')
    surface_callable, param_space = problem_gen.generate_instance()
    known_constraints = KnownConstraintsGenerator().get_constraint('continuous')

    split = feas_strategy_param.split("_")
    feas_strategy, feas_param = split[0], float(split[1])

    planner = BoTorchPlanner(
        goal="minimize",
        feas_strategy=feas_strategy,
        feas_param=feas_param,
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        acquisition_optimizer_kind=acquisition_optimizer,
        use_descriptors=use_descriptors,
        batch_size=batch_size,
    )

    planner.set_param_space(param_space)

    campaign = Campaign()
    campaign.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 10
    
    while len(campaign.observations.get_values()) < BUDGET:

        samples = planner.recommend(campaign.observations)
        for sample in samples:
            sample = sample.to_array()
            if known_constraints(sample):
                measurement = surface_callable.run(sample)[0] # return float only
            else:
                measurement = np.array([np.nan])
            campaign.add_observation(sample, measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET


def run_discrete(    
    init_design_strategy,
    batch_size,
    feas_strategy_param,
    use_descriptors,
    acquisition_optimizer,
    num_init_design=5,
):
    problem_gen = ProblemGenerator(problem_type='discrete')
    surface_callable, param_space = problem_gen.generate_instance()
    known_constraints = KnownConstraintsGenerator().get_constraint('discrete')


    split = feas_strategy_param.split('_')
    feas_strategy, feas_param = split[0], float(split[1])

    planner = BoTorchPlanner(
        goal="minimize",
        feas_strategy=feas_strategy,
        feas_param=feas_param,
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        acquisition_optimizer_kind=acquisition_optimizer,
        use_descriptors=use_descriptors,
        batch_size=batch_size,
    )

    planner.set_param_space(param_space)

    campaign = Campaign()
    campaign.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 10

    while len(campaign.observations.get_values()) < BUDGET:

        samples = planner.recommend(campaign.observations)
        for sample in samples:
            sample = sample.to_array()
            if known_constraints(sample):
                measurement = surface_callable.run(sample)[0] # return float only
            else:
                measurement = np.array([np.nan])
            campaign.add_observation(sample, measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET


def run_categorical(
    init_design_strategy,
    batch_size,
    feas_strategy_param,
    use_descriptors,
    acquisition_optimizer,
    num_init_design=5,
):

    problem_gen = ProblemGenerator(problem_type='categorical')
    surface_callable, param_space = problem_gen.generate_instance()
    known_constraints = KnownConstraintsGenerator().get_constraint('categorical')

    split = feas_strategy_param.split("_")
    feas_strategy, feas_param = split[0], float(split[1])

    campaign = Campaign()
    campaign.set_param_space(param_space)

    planner = BoTorchPlanner(
        goal="minimize",
        feas_strategy=feas_strategy,
        feas_param=feas_param,
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        acquisition_optimizer_kind=acquisition_optimizer,
        use_descriptors=use_descriptors,
    )
    planner.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 10

    while len(campaign.observations.get_values()) < BUDGET:

        samples = planner.recommend(campaign.observations)
        for sample in samples:
            sample = sample.to_array()
            if known_constraints(sample):
                measurement = surface_callable.run(sample)[0] # return float only
            else:
                measurement = np.array([np.nan])
            campaign.add_observation(sample, measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET

def run_mixed_disc_cont(    
    init_design_strategy,
    batch_size,
    feas_strategy_param,
    use_descriptors,
    acquisition_optimizer,
    num_init_design=5,
):

    problem_gen = ProblemGenerator(problem_type='mixed_disc_cont')
    surface_callable, param_space = problem_gen.generate_instance()
    known_constraints = KnownConstraintsGenerator().get_constraint('disc_cont')

    split = feas_strategy_param.split("_")
    feas_strategy, feas_param = split[0], float(split[1])

    campaign = Campaign()
    campaign.set_param_space(param_space)

    planner = BoTorchPlanner(
        goal="minimize",
        feas_strategy=feas_strategy,
        feas_param=feas_param,
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        acquisition_optimizer_kind=acquisition_optimizer,
        use_descriptors=use_descriptors,
    )
    planner.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 10

    while len(campaign.observations.get_values()) < BUDGET:

        samples = planner.recommend(campaign.observations)
        for sample in samples:
            sample = sample.to_array()
            if known_constraints(sample):
                measurement = surface_callable.run(sample) 
            else:
                measurement = np.nan
            campaign.add_observation(sample, measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET


def run_mixed_cat_disc(    
    init_design_strategy,
    batch_size,
    feas_strategy_param,
    use_descriptors,
    acquisition_optimizer,
    num_init_design=5,
):

    problem_gen = ProblemGenerator(problem_type='mixed_cat_disc')
    surface_callable, param_space = problem_gen.generate_instance()
    known_constraints = KnownConstraintsGenerator().get_constraint('cat_disc')

    split = feas_strategy_param.split("_")
    feas_strategy, feas_param = split[0], float(split[1])

    campaign = Campaign()
    campaign.set_param_space(param_space)

    planner = BoTorchPlanner(
        goal="minimize",
        feas_strategy=feas_strategy,
        feas_param=feas_param,
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        acquisition_optimizer_kind=acquisition_optimizer,
        use_descriptors=use_descriptors,
    )
    planner.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 10

    while len(campaign.observations.get_values()) < BUDGET:

        samples = planner.recommend(campaign.observations)
        for sample in samples:
            sample = sample.to_array()
            if known_constraints(sample):
                measurement = surface_callable.run(sample) 
            else:
                measurement = np.nan
            campaign.add_observation(sample, measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET


def run_mixed_cat_cont(    
    init_design_strategy,
    batch_size,
    feas_strategy_param,
    use_descriptors,
    acquisition_optimizer,
    num_init_design=5,
):

    problem_gen = ProblemGenerator(problem_type='mixed_cat_cont')
    surface_callable, param_space = problem_gen.generate_instance()
    known_constraints = KnownConstraintsGenerator().get_constraint('cat_cont')

    split = feas_strategy_param.split("_")
    feas_strategy, feas_param = split[0], float(split[1])

    campaign = Campaign()
    campaign.set_param_space(param_space)

    planner = BoTorchPlanner(
        goal="minimize",
        feas_strategy=feas_strategy,
        feas_param=feas_param,
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        acquisition_optimizer_kind=acquisition_optimizer,
        use_descriptors=use_descriptors,
    )
    planner.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 10

    while len(campaign.observations.get_values()) < BUDGET:

        samples = planner.recommend(campaign.observations)
        for sample in samples:
            sample = sample.to_array()
            if known_constraints(sample):
                measurement = surface_callable.run(sample) 
            else:
                measurement = np.nan
            campaign.add_observation(sample, measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET


def run_mixed_cat_disc_cont(    
    init_design_strategy,
    batch_size,
    feas_strategy_param,
    use_descriptors,
    acquisition_optimizer,
    num_init_design=5,
):

    problem_gen = ProblemGenerator(problem_type='mixed_cat_disc_cont')
    surface_callable, param_space = problem_gen.generate_instance()
    known_constraints = KnownConstraintsGenerator().get_constraint('cat_disc_cont')

    split = feas_strategy_param.split("_")
    feas_strategy, feas_param = split[0], float(split[1])

    campaign = Campaign()
    campaign.set_param_space(param_space)

    planner = BoTorchPlanner(
        goal="minimize",
        feas_strategy=feas_strategy,
        feas_param=feas_param,
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        acquisition_optimizer_kind=acquisition_optimizer,
        use_descriptors=use_descriptors,
    )
    planner.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 10

    while len(campaign.observations.get_values()) < BUDGET:

        samples = planner.recommend(campaign.observations)
        for sample in samples:
            sample = sample.to_array()
            if known_constraints(sample):
                measurement = surface_callable.run(sample) 
            else:
                measurement = np.nan
            campaign.add_observation(sample, measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET


if __name__ == "__main__":
    pass

    # run_continuous('random', 1, 'fwa_0', False, 'pymoo')
    # run_discrete('random', 1, 'fwa_0', False, 'pymoo')
    # run_categorical('random', 1, 'fwa_0', False, 'pymoo')
    # run_mixed_disc_cont('random', 1, 'fwa_0', False, 'pymoo')
    # run_mixed_cat_disc('random', 1, 'fwa_0', False, 'pymoo')
    # run_mixed_cat_cont('random', 1, 'fwa_0', False, 'pymoo')
    # run_mixed_cat_disc_cont('random', 1, 'fwa_0', False, 'pymoo')
    
 
