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
from olympus.surfaces import Surface

from atlas.planners.gp.planner import GPPlanner

RANDOM = {
    "param_type": ["cont", "disc", "cat"],
    "batch_size": [1, 5],
    "num_init_design": [10],
}

LHS = {
    "param_type": ["cont"],
    "batch_size": [1, 5],
    "num_init_design": [10],
}

SOBOL = {
    "param_type": ["cont"],
    "batch_size": [1, 5],
    "num_init_design": [10],
}

# TODO implement Grid as inital design planner
GRID = {}


def set_cont(init_design_strategy, batch_size, num_init_design):
    param_space = ParameterSpace()
    param_0 = ParameterContinuous(name="param_0", low=0.0, high=1.0)
    param_1 = ParameterContinuous(name="param_1", low=0.0, high=1.0)
    param_2 = ParameterContinuous(name="param_2", low=0.0, high=1.0)
    param_space.add(param_0)
    param_space.add(param_1)
    param_space.add(param_2)

    planner = GPPlanner(
        goal="minimize",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
    )

    planner.set_param_space(param_space)

    campaign = Campaign()
    campaign.set_param_space(param_space)

    return planner, campaign


def set_disc(init_design_strategy, batch_size, num_init_design):
    param_space = ParameterSpace()
    param_0 = ParameterDiscrete(
        name="param_0",
        options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    )
    param_1 = ParameterDiscrete(
        name="param_1",
        options=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    param_2 = ParameterDiscrete(
        name="param_2",
        options=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    param_space.add(param_0)
    param_space.add(param_1)
    param_space.add(param_2)

    planner = GPPlanner(
        goal="minimize",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
    )

    planner.set_param_space(param_space)

    campaign = Campaign()
    campaign.set_param_space(param_space)

    return planner, campaign


def set_cat(init_design_strategy, batch_size, num_init_design):
    surface_kind = "CatDejong"
    surface = Surface(kind=surface_kind, param_dim=2, num_opts=21)

    campaign = Campaign()
    campaign.set_param_space(surface.param_space)

    planner = GPPlanner(
        goal="minimize",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        use_descriptors=False,
    )
    planner.set_param_space(surface.param_space)

    return planner, campaign


@pytest.mark.parametrize("param_type", RANDOM["param_type"])
@pytest.mark.parametrize("batch_size", RANDOM["batch_size"])
@pytest.mark.parametrize("num_init_design", RANDOM["num_init_design"])
def test_init_design_random(param_type, batch_size, num_init_design):
    run_random(param_type, batch_size, num_init_design)


@pytest.mark.parametrize("param_type", LHS["param_type"])
@pytest.mark.parametrize("batch_size", LHS["batch_size"])
@pytest.mark.parametrize("num_init_design", LHS["num_init_design"])
def test_init_design_lhs(param_type, batch_size, num_init_design):
    run_lhs(param_type, batch_size, num_init_design)


@pytest.mark.parametrize("param_type", SOBOL["param_type"])
@pytest.mark.parametrize("batch_size", SOBOL["batch_size"])
@pytest.mark.parametrize("num_init_design", SOBOL["num_init_design"])
def test_init_design_sobol(param_type, batch_size, num_init_design):
    run_sobol(param_type, batch_size, num_init_design)


def run_random(param_type, batch_size, num_init_design):
    if param_type == "cont":
        planner, campaign = set_cont("random", batch_size, num_init_design)
    elif param_type == "disc":
        planner, campaign = set_disc("random", batch_size, num_init_design)
    elif param_type == "cat":
        planner, campaign = set_cat("random", batch_size, num_init_design)

    BUDGET = num_init_design

    while len(campaign.observations.get_values()) < BUDGET:
        samples = planner.recommend(campaign.observations)
        for sample in samples:
            sample_arr = sample.to_array()
            measurement = np.random.uniform(size=None)
            campaign.add_observation(sample_arr, measurement)

    params_ = campaign.observations.get_params()
    values_ = campaign.observations.get_values()
    assert len(params_) == BUDGET
    assert len(values_) == BUDGET
    # check that no recommendations are duplicated
    assert np.unique(params_, axis=0).shape[0] == params_.shape[0]


def run_lhs(param_type, batch_size, num_init_design):
    if param_type == "cont":
        planner, campaign = set_cont("lhs", batch_size, num_init_design)

    BUDGET = num_init_design

    while len(campaign.observations.get_values()) < BUDGET:
        samples = planner.recommend(campaign.observations)
        for sample in samples:
            sample_arr = sample.to_array()
            measurement = np.random.uniform(size=None)
            campaign.add_observation(sample_arr, measurement)

    params_ = campaign.observations.get_params()
    values_ = campaign.observations.get_values()
    assert len(params_) == BUDGET
    assert len(values_) == BUDGET
    # check that no recommendations are duplicated
    assert np.unique(params_, axis=0).shape[0] == params_.shape[0]


def run_sobol(param_type, batch_size, num_init_design):
    if param_type == "cont":
        planner, campaign = set_cont("sobol", batch_size, num_init_design)

    BUDGET = num_init_design

    while len(campaign.observations.get_values()) < BUDGET:
        samples = planner.recommend(campaign.observations)
        for sample in samples:
            sample_arr = sample.to_array()
            measurement = np.random.uniform(size=None)
            campaign.add_observation(sample_arr, measurement)

    params_ = campaign.observations.get_params()
    values_ = campaign.observations.get_values()
    assert len(params_) == BUDGET
    assert len(values_) == BUDGET
    # check that no recommendations are duplicated
    assert np.unique(params_, axis=0).shape[0] == params_.shape[0]


if __name__ == "__main__":
    run_lhs("cont", 5, 10)
