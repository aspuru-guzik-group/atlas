#!/usr/bin/env python

import pickle

from olympus import Campaign, Surface

from atlas.planners.gp.planner import BoTorchPlanner

surface = Surface(
    kind="Branin"
)  # instantiate 2d Branin-Hoo objective function

campaign = Campaign()  # define Olympus campaign object
campaign.set_param_space(surface.param_space)

planner = BoTorchPlanner(
    goal="minimize", num_init_design=5
)  # instantiate GPPlanner
planner.set_param_space(surface.param_space)

while len(campaign.observations.get_values()) < 30:
    samples = planner.recommend(
        campaign.observations
    )  # ask planner for parameters (list of ParameterVectors)
    for sample in samples:
        measurement = surface.run(sample)  # measure Branin-Hoo function
        campaign.add_observation(
            sample, measurement
        )  # tell planner about most recent observation


pickle.dump(campaign, open("results.pkl", "wb"))
