#!/usr/bin/env python

import os
import pickle
import sys

import numpy as np
import pandas as pd
import rich
from olympus import Campaign, Emulator

from atlas import Logger
from atlas.planners.gp.planner import BoTorchPlanner


def execute(benchmark_config):
    """Run the benchmark experiment"""
    problem_id = benchmark_config["problem_id"]
    problem_type = benchmark_config["olympus_problem_type"]
    param_space = benchmark_config["olympus_problem_obj"].param_space
    is_emulated = benchmark_config["is_emulated"]
    measurement_name = benchmark_config["measurement_name"]
    num_repeats = benchmark_config["num_repeats"]
    budget = benchmark_config["budget"]

    # sort out the measurement callable
    if is_emulated:
        measurement_fn = Emulator(
            dataset=benchmark_config["problem"], model="BayesNeuralNet"
        ).run
    else:
        measurement_fn = benchmark_config["olympus_problem_obj"].run

    repeat_results = []
    for repeat in range(num_repeats):
        Logger.log(
            f"Commencing repeat {repeat+1}/{num_repeats} for problem id : {problem_id}",
            "INFO",
        )

        campaign = Campaign()
        campaign.set_param_space(param_space)

        planner = BoTorchPlanner(
            goal=benchmark_config["goal"], **benchmark_config["optimizer_args"]
        )
        planner.set_param_space(param_space)
        while len(campaign.observations.get_values()) < budget:
            samples = planner.recommend(campaign.observations)
            for sample in samples:
                if is_emulated:
                    measurement, _, __ = measurement_fn(sample)
                else:
                    measurement = measurement_fn(sample)
                campaign.add_observation(sample, measurement)

        # record repeat results
        params_results_dict = {
            param.name: campaign.observations.get_params(as_array=True)[
                :, param_ix
            ]
            for param_ix, param in enumerate(param_space)
        }
        values_results_dict = {
            measurement_name: campaign.observations.get_values(
                as_array=True
            ).flatten()
        }
        results_df = pd.DataFrame(
            {**params_results_dict, **values_results_dict}
        )
        repeat_results.append(results_df)

        pickle.dump(
            repeat_results,
            open(f"results_{problem_id}.pkl", "wb"),
        )


if __name__ == "__main__":
    benchmark_config_file = str(sys.argv[1])
    benchmark_idx = int(sys.argv[2])

    with open(benchmark_config_file, "rb") as f:
        benchmark_configs = pickle.load(f)

    execute(benchmark_config=benchmark_configs[benchmark_idx])
