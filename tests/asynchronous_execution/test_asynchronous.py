#!/usr/bin/env python
import os
import sys

sys.path.append("../")

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


def run_continuous(
    init_design_strategy,
    batch_size,
    use_descriptors,
    acquisition_type,
    acquisition_optimizer,
    num_init_design=5,
):
    problem_gen = ProblemGenerator(problem_type="continuous")
    surface_callable, param_space = problem_gen.generate_instance()

    planner = BoTorchPlanner(
        goal="minimize",
        feas_strategy="naive-0",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        acquisition_type=acquisition_type,
        acquisition_optimizer_kind=acquisition_optimizer,
    )

    planner.set_param_space(param_space)

    campaign = Campaign()
    campaign.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 4

    # TODO: add pending parameter stuff here...

    while len(campaign.observations.get_values()) < BUDGET:
        samples = planner.recommend(campaign.observations)
        for sample in samples:
            sample_arr = sample.to_array()
            measurement = surface_callable.run(sample_arr)
            campaign.add_observation(sample_arr, measurement)

            print("SAMPLE : ", sample)
            print("MEASUREMENT : ", measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET
