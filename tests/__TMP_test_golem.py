#!/usr/bin/env python

import numpy as np
import pytest
from golem import *
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import (
    ParameterCategorical,
    ParameterContinuous,
    ParameterDiscrete,
    ParameterVector,
)
from olympus.surfaces import Surface

from atlas.planners.gp.planner import GPPlanner
from atlas.utils.golem_utils import get_golem_dists

CONT = {
    "golem_config": [
        # param config as dictionaries
        {
            "name": "dicts",
            "config": {
                "param0": {"dist_type": "Normal", "dist_params": {"std": 0.2}},
                "param1": {"dist_type": "Normal", "dist_params": {"std": 0.3}},
            },
        },
        # params as Golem distribution objects
        {
            "name": "objects",
            "config": {
                "param0": Normal(0.2),
                "param1": Normal(0.3),
            },
        },
        # missing parameters
        {
            "name": "missing_param",
            "config": {
                "param0": {"dist_type": "Normal", "dist_params": {"std": 0.2}},
            },
        },
        # all parameters = Delta(), should return None
        {
            "name": "all_delta",
            "config": {
                "param0": {"dist_type": "Delta", "dist_params": None},
                "param1": {"dist_type": "Delta", "dist_params": None},
            },
        },
    ]
}


@pytest.mark.parametrize("golem_config", CONT["golem_config"])
def test_get_golem_dists_cont(golem_config):
    test_name = golem_config["name"]
    config = golem_config["config"]

    param_space = ParameterSpace()
    param_space.add(ParameterContinuous(name="param0"))
    param_space.add(ParameterContinuous(name="param1"))

    dists = get_golem_dists(config, param_space)

    if test_name in ["dicts", "objects"]:
        assert all([isinstance(dist, Normal) for dist in dists])
    elif test_name == "missing_param":
        assert isinstance(dists[0], Normal)
        assert isinstance(dists[1], Delta)
    elif test_name == "all_delta":
        assert dists is None


# @pytest.mark.parametrize("golem_config", MIXED["golem_config"])
# def test_get_golem_dists_mixed(golem_config):
#
#     test_name = golem_config['name']
#     config = golem_config['config']
#
#     param_space = ParameterSpace()
#     param_0 = ParameterCategorical(
#         name="param_0",
#         options=["x0", "x1", "x2"],
#         descriptors=desc_param_0,
#     )
#     param_1 = ParameterDiscrete(
#         name="param_1",
#         options=[0.0, 0.25, 0.5, 0.75, 1.0],
#     )
#     param_2 = ParameterContinuous(
#         name="param_2",
#         low=0.0,
#         high=1.0,
#     )
#     param_3 = ParameterContinuous(
#         name="param_3",
#         low=0.0,
#         high=1.0,
#     )
#     param_space.add(param_0)
#     param_space.add(param_1)
#     param_space.add(param_2)
#     param_space.add(param_3)
#
#
#     dists = get_golem_dists(config, param_space)


def test_golem_opt_cont():
    def surface(x):
        return np.sin(8 * x[0]) - 2 * np.cos(6 * x[1]) + np.exp(-2.0 * x[2])

    param_space = ParameterSpace()
    param_0 = ParameterContinuous(name="param0", low=0.0, high=1.0)
    param_1 = ParameterContinuous(name="param1", low=0.0, high=1.0)
    param_2 = ParameterContinuous(name="param2", low=0.0, high=1.0)
    param_space.add(param_0)
    param_space.add(param_1)
    param_space.add(param_2)

    planner = GPPlanner(
        goal="minimize",
        feas_strategy="naive-0",
        init_design_strategy="lhs",
        num_init_design=5,
        batch_size=1,
        acquisition_type="ei",
        acquisition_optimizer="gradient",
        golem_config={
            "param0": Normal(0.2),
            "param1": Normal(0.3),
        },
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

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET


# def test_golem_opt_mixed(golem_config):
#     ...


if __name__ == "__main__":
    # test_get_golem_dists_cont(CONT['golem_config'][0])

    test_golem_opt_cont()
