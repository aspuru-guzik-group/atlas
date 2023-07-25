#!/usr/bin/env python

import binascii
import itertools
import os
import pickle
import sys

import numpy as np
import pandas as pd
import yaml
from olympus import Dataset, Surface
from olympus.datasets import list_datasets
from olympus.surfaces import list_surfaces

from atlas import Logger

ALLOWED_PROBLEM_SPEC = ["single_objective"]

ALLOWED_DATASETS = list_datasets()

ALLOWED_SURFACES = list_surfaces()


def get_optimizer_args(optimizer_config):
    """gets product space for all optimizer configs for a particular problem"""
    optimizer_args_lists = [
        optimizer_config[key] for key in optimizer_config.keys()
    ]
    cart_product = list(itertools.product(*optimizer_args_lists))
    cart_product = [list(elem) for elem in cart_product]

    # reintegrate into list of dictionaries
    optimizer_args = []
    for product in cart_product:
        optimizer_args.append(
            dict(zip(list(optimizer_config.keys()), product))
        )
    return optimizer_args


def parse_config(config_file):
    """unpack yaml benchmark config file"""
    if os.path.isfile(config_file):
        content = open(f"{config_file}", "r")
        config = yaml.full_load(content)
    else:
        Logger.log(f"File {config_file} does not exist", "FATAL")

    general = config["general"]
    # TODO: unpack and analyze the problem specifications to see if
    # they match with the specified problems

    problems = config["problems"]

    benchmark_config = []
    for problem, problem_config in problems.items():
        # check the problem name make sure its in Olympus
        if problem in ALLOWED_DATASETS:
            olympus_problem_type = "dataset"
            olympus_problem_obj = Dataset(kind=problem)
            # olympus_problem_emulator = Emulator(dataset=problem, model='BayesNeuralNet')
            is_emulated = True
            goal = olympus_problem_obj.goal
            measurement_name = olympus_problem_obj.measurement_name

        elif problem in ALLOWED_SURFACES:
            olympus_problem_type = "surface"
            olympus_problem_obj = Surface(kind=problem)
            olympus_problem_emulator = None
            is_emulated = False
            goal = "minimize"  # all surfaces have minimize goal
            measurement_name = "obj"  # constant for single obj
        else:
            raise ValueError(f"Problem {problem} not included in Olympus...")

        # generate all possible combinations of optimizer args
        optimizer_args = get_optimizer_args(problem_config["optimizer"])

        # add benchmark config to list
        for optimizer_args_ in optimizer_args:
            # generate unique problem id - kinda messy
            problem_id = str(binascii.hexlify(os.urandom(16)))[2:]
            problem_id = problem_id[:-1]

            benchmark_config.append(
                {
                    "problem_id": problem_id,
                    "problem": problem,
                    "olympus_problem_type": olympus_problem_type,
                    "olympus_problem_obj": olympus_problem_obj,
                    "is_emulated": is_emulated,
                    "goal": goal,
                    "measurement_name": measurement_name,
                    "num_repeats": problem_config["num_repeats"],
                    "budget": problem_config["budget"],
                    "optimizer_args": optimizer_args_,
                }
            )

    return benchmark_config


if __name__ == "__main__":
    config_file = str(sys.argv[1])
    benchmark_config_file = str(sys.argv[2])

    benchmark_config = parse_config(config_file=config_file)

    Logger.log(
        f"Generated {len(benchmark_config)} unique experiments from config file: {config_file}",
        "INFO",
    )
    Logger.log(f"Saving benchmark config to: {benchmark_config_file}", "INFO")
    pickle.dump(benchmark_config, open("benchmark_configs.pkl", "wb"))
