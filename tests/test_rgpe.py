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


def test_continuous_hypervolume():

    moo_surface = Surface(kind="MultFonseca")

    # create the source tasks
    train_tasks = []
    for i in range(10):
        params = np.random.uniform(size=(20, 2))
        values = np.array(moo_surface.run(params))
        train_tasks.append({"params": params, "values": values})

    planner = RGPEPlanner(
        goal="minimize",
        warm_start=False,
        train_tasks=train_tasks,
        valid_tasks=train_tasks,
        init_design_strategy="lhs",
        num_init_design=4,
        batch_size=1,
        is_moo=True,
        value_space=moo_surface.value_space,
        scalarizer_kind="Hypervolume",
        moo_params={},
        goals=["min", "max"],
    )

    scalarizer = Scalarizer(
        kind="Hypervolume",
        value_space=moo_surface.value_space,
        goals=["min", "max"],
    )

    planner.set_param_space(moo_surface.param_space)

    campaign = Campaign()
    campaign.set_param_space(moo_surface.param_space)
    campaign.set_value_space(moo_surface.value_space)

    BUDGET = 10

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


if __name__ == "__main__":
    #run_continuous('lhs')
    # run_discrete('random')
    test_continuous_hypervolume()
