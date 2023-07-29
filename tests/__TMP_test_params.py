#!/usr/bin/env python

import math

import numpy as np
import pytest
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import (
    ParameterCategorical,
    ParameterContinuous,
    ParameterDiscrete,
)
from olympus.planners import RandomSearch
from olympus.surfaces import Surface

from atlas.params.params import Parameters


def param_space_factory(problem_type, has_descriptors):
    num_params = np.random.randint(3, 8)

    num_observations = np.random.randint(3, 20)

    split = problem_type.split("_")
    unique_param_types = split[1:]
    num_tiles = math.ceil(num_params / len(unique_param_types))
    param_types = unique_param_types * num_tiles
    param_types = param_types[:num_params]

    param_space = ParameterSpace()

    for param_ix, param_type in enumerate(param_types):
        if param_type == "cont":
            param_space.add(
                ParameterContinuous(
                    name=f"param_{param_ix}",
                    low=np.random.uniform(-10, -0.1),
                    high=np.random.uniform(0.1, 10),
                )
            )
        elif param_type == "disc":
            options = np.random.uniform(
                -10,
                10,
                size=np.random.randint(3, 10),
            )
            options.sort()
            param_space.add(
                ParameterDiscrete(
                    name=f"param_{param_ix}",
                    options=list(options),
                )
            )
        elif param_type == "cat":
            num_options = np.random.randint(3, 10)
            options = [f"x_{i}" for i in range(num_options)]
            if has_descriptors:
                desc = [[i, i] for i in range(num_options)]
            else:
                desc = [None for _ in range(num_options)]
            param_space.add(
                ParameterCategorical(
                    name=f"param_{param_ix}",
                    options=options,
                    descriptors=desc,
                )
            )
    # generate some observations
    planner = RandomSearch()
    planner.set_param_space(param_space)
    campaign = Campaign()
    campaign.set_param_space(param_space)

    for _ in range(num_observations):
        sample = planner.recommend(campaign.observations)[0]
        campaign.add_observation(sample, np.random.uniform())

    return param_space, campaign.observations


NO_DESC_TESTS = {
    "problem_type": ["fully_cont", "fully_disc", "mixed_disc_cont"],
    "has_descriptors": [False],
}

DESC_TESTS = {
    "problem_type": [
        "fully_cat",
        "mixed_cat_disc",
        "mixed_cat_cont",
        "mixed_cat_disc_cont",
    ],
    "has_descriptors": [True, False],
}

GENERAL_TESTS = {
    "problem_type": [
        "fully_cat",
        "mixed_cat_disc",
        "mixed_cat_cont",
        "mixed_cat_disc_cont",
    ],
    "has_descriptors": [True, False],
}


@pytest.mark.parametrize("problem_type", NO_DESC_TESTS["problem_type"])
def test_params_init_no_desc(problem_type, has_descriptors=False):
    run_init_parameters(problem_type, has_descriptors)


@pytest.mark.parametrize("problem_type", DESC_TESTS["problem_type"])
@pytest.mark.parametrize("has_descriptors", DESC_TESTS["has_descriptors"])
def test_params_init_desc(problem_type, has_descriptors):
    run_init_parameters(problem_type, has_descriptors)


@pytest.mark.parametrize("problem_type", GENERAL_TESTS["problem_type"])
@pytest.mark.parametrize("has_descriptors", GENERAL_TESTS["has_descriptors"])
def test_params_init_general(problem_type, has_descriptors):
    run_init_parameters_general(problem_type, has_descriptors)


def run_init_parameters(
    problem_type,
    has_descriptors,
):
    param_space, observations = param_space_factory(
        problem_type, has_descriptors
    )

    params = Parameters(
        olympus_param_space=param_space,
        observations=observations,
        has_descriptors=has_descriptors,
    )

    assert params.num_params == len(param_space)
    assert params.expanded_dims >= params.num_params


def run_init_parameters_general(problem_type, has_descriptors):
    param_space, observations = param_space_factory(
        problem_type, has_descriptors
    )

    cat_dis_inds = []
    for ix, param in enumerate(param_space):
        if param.type in ["discrete", "categorical"]:
            cat_dis_inds.append(ix)

    # make one of the parameters a general parameter
    general_parameters = [np.random.choice(cat_dis_inds, size=None)]

    params = Parameters(
        olympus_param_space=param_space,
        observations=observations,
        has_descriptors=has_descriptors,
        general_parameters=general_parameters,
    )

    assert params.num_params == len(param_space)
    assert params.expanded_dims >= params.num_params

    assert params.general_dims == general_parameters
    assert len(params.general_mask) == len(param_space)
    assert len(params.exp_general_mask) == params.expanded_raw.shape[1]


if __name__ == "__main__":
    # run_init_parameters('fully_cont', has_descriptors=False)
    run_init_parameters_general("mixed_cat_cont", has_descriptors=False)
