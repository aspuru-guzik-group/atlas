#!/usr/bin/env python

import pickle
import pandas as pd

from olympus import Campaign, Surface
from atlas.unknown_constraints.benchmark_functions import BraninConstr

from atlas.planners.gp.planner import GPPlanner

from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument(
        "--tag",
        action="store",
        type=str,
        help="Number of workers, defaults 1.",
        choices=["naive-0", "fwa", "fca-0.2", "fca-0.5", "fca-0.8", "fia-0.5", "fia-1", "fia-2"]
    )
FLAGS = parser.parse_args()

#------
# Initialize parameters
#------

plan_args = {
    "naive-0": {'feas_strategy': 'naive-0', 'feas_param': 0},
    "fwa": {'feas_strategy': 'fwa', 'feas_param': 0.2},
    "fca-0.2": {'feas_strategy': 'fca', 'feas_param': 0.2},
    "fca-0.5":{'feas_strategy': 'fca', 'feas_param': 0.5}, 
    "fca-0.8": {'feas_strategy': 'fca', 'feas_param': 0.8},
    "fia-0.5": {'feas_strategy': 'fia', 'feas_param': 0.5},
    "fia-1": {'feas_strategy': 'fia', 'feas_param': 1.0},
    "fia-2": {'feas_strategy': 'fia', 'feas_param': 2.0},
}
plan_args = plan_args[FLAGS.tag]

NUM_RUNS = 100


#------
# Start the optimization
#------

# instantiate 2d constrained Branin-Hoo objective function (available on GitHub repo)
surface = BraninConstr()
all_runs = []
for _ in range(NUM_RUNS):

    campaign = Campaign() # define Olympus campaign object
    campaign.set_param_space(surface.param_space)

    planner = GPPlanner(
        goal='minimize',
        feas_strategy=plan_args["feas_strategy"],
        feas_param=plan_args["feas_param"],
        acquisition_type='ucb',
    ) # instantiate Atlas planner
    planner.set_param_space(surface.param_space)


    while len(campaign.observations.get_values()) < 100:
        samples = planner.recommend(
            campaign.observations
        )  # ask planner for parameters (list of ParameterVectors)
        for sample in samples:
            measurement = surface.run_constr(sample)  # measure constrained Branin-Hoo function
            campaign.add_observation(
                sample, measurement
            )  # tell planner about most recent observation

    x0_col = campaign.observations.get_params()[:, 0]
    x1_col = campaign.observations.get_params()[:, 1]
    
    df = pd.DataFrame({
        'param_0': x0_col,
        'param_1': x1_col,
        'obj': campaign.observations.get_values()
    })

    all_runs.append(df)

pickle.dump(all_runs, open(f"results_{FLAGS.tag}.pkl", "wb"))
