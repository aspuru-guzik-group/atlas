#!/usr/bin/env python

import os

import matplotlib.pyplot as plt
import numpy as np
import pytest
import seaborn as sns
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import (
    ParameterCategorical,
    ParameterContinuous,
    ParameterDiscrete,
)
from olympus.scalarizers import Scalarizer
from olympus.surfaces import Surface

from atlas.planners.gp.planner import BoTorchPlanner
from atlas.utils.synthetic_data import trig_factory

def test_reg_surrogate_cont():
    def surface(x):
        return np.sin(8 * x[0]) - 2 * np.cos(6 * x[1]) + np.exp(-2.0 * x[2])

    param_space = ParameterSpace()
    param_0 = ParameterContinuous(name="param_0", low=0.0, high=1.0)
    param_1 = ParameterContinuous(name="param_1", low=0.0, high=1.0)
    param_2 = ParameterContinuous(name="param_2", low=0.0, high=1.0)
    param_space.add(param_0)
    param_space.add(param_1)
    param_space.add(param_2)

    planner = BoTorchPlanner(
        goal="minimize",
        feas_strategy="naive-0",
        init_design_strategy="random",
        num_init_design=4,
        batch_size=1,
    )

    planner.set_param_space(param_space)

    campaign = Campaign()
    campaign.set_param_space(param_space)

    BUDGET = 10

    while len(campaign.observations.get_values()) < BUDGET:

        samples = planner.recommend(campaign.observations)
        for sample in samples:
            sample_arr = sample.to_array()
            measurement = surface(sample_arr)
            campaign.add_observation(sample_arr, measurement)

    # make prediction
    X = np.random.uniform(size=(200, 3))

    pred_mu, pred_sigma = planner.reg_surrogate(X, return_np=True)

    assert isinstance(pred_mu, np.ndarray) and isinstance(
        pred_sigma, np.ndarray
    )
    assert np.shape(pred_mu) == np.shape(pred_sigma) == (X.shape[0], 1)


def test_cla_surrogate_cont():
    def surface(x):
        if np.random.uniform() > 0.5:
            return np.nan
        else:
            return (
                np.sin(8 * x[0]) - 2 * np.cos(6 * x[1]) + np.exp(-2.0 * x[2])
            )

    param_space = ParameterSpace()
    param_0 = ParameterContinuous(name="param_0", low=0.0, high=1.0)
    param_1 = ParameterContinuous(name="param_1", low=0.0, high=1.0)
    param_2 = ParameterContinuous(name="param_2", low=0.0, high=1.0)
    param_space.add(param_0)
    param_space.add(param_1)
    param_space.add(param_2)

    planner = BoTorchPlanner(
        goal="minimize",
        feas_strategy="fwa",
        init_design_strategy="random",
        num_init_design=4,
        batch_size=1,
    )

    planner.set_param_space(param_space)

    campaign = Campaign()
    campaign.set_param_space(param_space)

    BUDGET = 10

    while len(campaign.observations.get_values()) < BUDGET:

        samples = planner.recommend(campaign.observations)
        for sample in samples:
            sample_arr = sample.to_array()
            measurement = surface(sample_arr)
            campaign.add_observation(sample_arr, measurement)

    # make prediction
    X = np.random.uniform(size=(200, 3))

    pred = planner.cla_surrogate(X, return_np=True, normalize=True)

    assert isinstance(pred, np.ndarray)
    assert np.shape(pred) == (X.shape[0], 1)


def test_acquisition_function():
    def surface(x):
        return np.sin(8 * x[0]) - 2 * np.cos(6 * x[1]) + np.exp(-2.0 * x[2])

    param_space = ParameterSpace()
    param_0 = ParameterContinuous(name="param_0", low=0.0, high=1.0)
    param_1 = ParameterContinuous(name="param_1", low=0.0, high=1.0)
    param_2 = ParameterContinuous(name="param_2", low=0.0, high=1.0)
    param_space.add(param_0)
    param_space.add(param_1)
    param_space.add(param_2)

    planner = BoTorchPlanner(
        goal="minimize",
        feas_strategy="naive-0",
        init_design_strategy="random",
        num_init_design=4,
        batch_size=1,
    )

    planner.set_param_space(param_space)

    campaign = Campaign()
    campaign.set_param_space(param_space)

    BUDGET = 10

    while len(campaign.observations.get_values()) < BUDGET:

        samples = planner.recommend(campaign.observations)
        for sample in samples:
            sample_arr = sample.to_array()
            measurement = surface(sample_arr)
            campaign.add_observation(sample_arr, measurement)

    # get acquisition function
    X = np.random.uniform(size=(200, 3))

    acqf_vals = planner.acquisition_function(X, return_np=True, normalize=True)

    assert isinstance(acqf_vals, np.ndarray)
    assert np.shape(acqf_vals) == (X.shape[0], 1)


if __name__ == '__main__':

	test_acquisition_function()
