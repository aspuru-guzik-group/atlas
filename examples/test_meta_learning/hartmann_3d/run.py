#!/usr/bin/env python

import glob
import pickle

import numpy as np
import olympus
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import ParameterContinuous
from olympus.planners import Planner
from olympus.surfaces import Surface

from atlas import __datasets__


# ----------------------------
# HARTMANN 3D SURFACE
# ----------------------------
def hm3(x):
    # the hartmann3 function (3D)
    # https://www.sfu.ca/~ssurjano/hart3.html
    # parameters
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
    P = 1e-4 * np.array(
        [
            [3689, 1170, 2673],
            [4699, 4387, 7470],
            [1091, 8732, 5547],
            [381, 5743, 8828],
        ]
    )
    x = x.reshape(x.shape[0], 1, -1)
    B = x - P
    B = B**2
    exponent = A * B
    exponent = np.einsum("ijk->ij", exponent)
    C = np.exp(-exponent)
    hm3 = -np.einsum("i, ki", alpha, C)
    # normalize
    mean = -0.93
    std = 0.95
    hm3 = 1 / std * (hm3 - mean)
    # maximize
    # hm3 = -hm3
    return hm3


# -----------------------------
# Instantiate surface
# -----------------------------
SOURCE_NAME = "hartmann_3D"
SURFACE_KIND = "hartmann"
surface = hm3


# ---------------
# Configuration
# ---------------


MODELS = [
    "RandomSearch",
    #'Gpyopt',
    "Botorch",
    "RGPE",
    "DKT",
]

META_PLANNERS = [
    "RGPE",
    "DKT",
]


NUM_RUNS = 40
BUDGET = 50


# ------------------
# begin experiment
# ------------------


for model_kind in MODELS:

    run_ix = 0
    while run_ix < NUM_RUNS:

        try:

            if (model_kind not in META_PLANNERS) and (model_kind != "Botorch"):

                # ------------------
                # olympus planners
                # ------------------

                # make a campaign and add parameter space
                campaign = Campaign()

                param_space = ParameterSpace()
                # add 3 continuous Parameters
                param_0 = ParameterContinuous(
                    name="param_0", low=0.0, high=1.0
                )
                param_1 = ParameterContinuous(
                    name="param_1", low=0.0, high=1.0
                )
                param_2 = ParameterContinuous(
                    name="param_2", low=0.0, high=1.0
                )
                param_space.add(param_0)
                param_space.add(param_1)
                param_space.add(param_2)

                campaign.set_param_space(param_space)

                planner = Planner(kind=model_kind)

                planner.set_param_space(campaign.param_space)

                # start the optimization experiment
                iteration = 0
                # optimization loop
                while len(campaign.values) < BUDGET:

                    print(f"\nITERATION : {iteration}\n")

                    samples = planner.recommend(campaign.observations)
                    print(f"SAMPLES : {samples}")
                    for sample in samples:
                        sample_arr = sample.to_array()
                        measurement = surface(
                            sample_arr.reshape((1, sample_arr.shape[0]))
                        )

                        campaign.add_observation(sample_arr, measurement)

                    pickle.dump(
                        {
                            "params": campaign.params,
                            "values": campaign.values,
                        },
                        open(
                            f"runs/run_{model_kind}_{SURFACE_KIND}_{run_ix}.pkl",
                            "wb",
                        ),
                    )

                    iteration += 1

            elif model_kind == "Botorch":

                # ------------------------
                # custom botorch planner
                # ------------------------

                from atlas.optimizers.gp.planner import BoTorchPlanner

                # make a campaign and add parameter space
                campaign = Campaign()

                param_space = ParameterSpace()
                # add 3 continuous Parameters
                param_0 = ParameterContinuous(
                    name="param_0", low=0.0, high=1.0
                )
                param_1 = ParameterContinuous(
                    name="param_1", low=0.0, high=1.0
                )
                param_2 = ParameterContinuous(
                    name="param_2", low=0.0, high=1.0
                )
                param_space.add(param_0)
                param_space.add(param_1)
                param_space.add(param_2)

                campaign.set_param_space(param_space)

                planner = BoTorchPlanner(
                    goal="minimize",
                    feas_strategy="naive-0",
                    init_design_strategy="lhs",
                    num_init_design=5,
                    batch_size=1,
                )

                planner.set_param_space(campaign.param_space)

                # start the optimization experiment
                iteration = 0
                # optimization loop
                while len(campaign.values) < BUDGET:

                    print(f"\nITERATION : {iteration}\n")

                    samples = planner.recommend(campaign.observations)
                    print(f"SAMPLES : {samples}")
                    for sample in samples:
                        sample_arr = sample.to_array()
                        measurement = surface(
                            sample_arr.reshape((1, sample_arr.shape[0]))
                        )

                        campaign.add_observation(sample_arr, measurement)

                    pickle.dump(
                        {
                            "params": campaign.params,
                            "values": campaign.values,
                        },
                        open(
                            f"runs/run_{model_kind}_{SURFACE_KIND}_{run_ix}.pkl",
                            "wb",
                        ),
                    )

                    iteration += 1

            elif model_kind == "DKT":

                # ------------------------------
                # Deep kernel transfer planner
                # ------------------------------

                from atlas.optimizers.dkt.planner import DKTPlanner

                # load the source tasks from disk
                tasks = pickle.load(
                    open(f"{__datasets__}/{SOURCE_NAME}_tasks.pkl", "rb")
                )
                train_tasks = tasks
                valid_tasks = tasks  # this shouldnt be needed

                # make a campaign and add parameter space
                campaign = Campaign()

                param_space = ParameterSpace()
                # add 3 continuous Parameters
                param_0 = ParameterContinuous(
                    name="param_0", low=0.0, high=1.0
                )
                param_1 = ParameterContinuous(
                    name="param_1", low=0.0, high=1.0
                )
                param_2 = ParameterContinuous(
                    name="param_2", low=0.0, high=1.0
                )
                param_space.add(param_0)
                param_space.add(param_1)
                param_space.add(param_2)

                campaign.set_param_space(param_space)

                planner = DKTPlanner(
                    goal="minimize",
                    warm_start=False,
                    train_tasks=train_tasks,
                    valid_tasks=valid_tasks,
                    model_path="./tmp_models/",
                    init_design_strategy="lhs",
                    num_init_design=5,
                    batch_size=1,
                    from_disk=False,
                    hyperparams={
                        "model": {
                            "epochs": 10000,
                        }
                    },
                )

                planner.set_param_space(campaign.param_space)

                # start the optimization experiment
                iteration = 0
                # optimization loop
                while len(campaign.values) < BUDGET:

                    print(f"\nITERATION : {iteration}\n")

                    samples = planner.recommend(campaign.observations)
                    print(f"SAMPLES : {samples}")
                    for sample in samples:
                        sample_arr = sample.to_array()
                        measurement = surface(
                            sample_arr.reshape((1, sample_arr.shape[0]))
                        )

                        campaign.add_observation(sample_arr, measurement)

                    pickle.dump(
                        {
                            "params": campaign.params,
                            "values": campaign.values,
                        },
                        open(
                            f"runs/run_{model_kind}_{SURFACE_KIND}_{run_ix}.pkl",
                            "wb",
                        ),
                    )

                    iteration += 1

            elif model_kind == "RGPE":

                # ---------------
                # RGPE planner
                # ---------------

                from atlas.optimizers.rgpe.planner import RGPEPlanner

                # load the source tasks from disk
                tasks = pickle.load(
                    open(f"{__datasets__}/{SOURCE_NAME}_tasks.pkl", "rb")
                )
                train_tasks = tasks
                valid_tasks = tasks  # this shouldnt be needed

                # make a campaign and add parameter space
                campaign = Campaign()

                param_space = ParameterSpace()
                # add 3 continuous Parameters
                param_0 = ParameterContinuous(
                    name="param_0", low=0.0, high=1.0
                )
                param_1 = ParameterContinuous(
                    name="param_1", low=0.0, high=1.0
                )
                param_2 = ParameterContinuous(
                    name="param_2", low=0.0, high=1.0
                )
                param_space.add(param_0)
                param_space.add(param_1)
                param_space.add(param_2)

                campaign.set_param_space(param_space)

                planner = RGPEPlanner(
                    goal="minimize",
                    warm_start=False,
                    train_tasks=train_tasks,
                    valid_tasks=valid_tasks,
                    init_design_strategy="lhs",
                    num_init_design=5,
                    batch_size=1,
                )

                planner.set_param_space(campaign.param_space)

                # start the optimization experiment
                iteration = 0
                # optimization loop
                while len(campaign.values) < BUDGET:

                    print(f"\nITERATION : {iteration}\n")

                    samples = planner.recommend(campaign.observations)
                    print(f"SAMPLES : {samples}")
                    for sample in samples:
                        sample_arr = sample.to_array()
                        measurement = surface(
                            sample_arr.reshape((1, sample_arr.shape[0]))
                        )

                        campaign.add_observation(sample_arr, measurement)

                    pickle.dump(
                        {
                            "params": campaign.params,
                            "values": campaign.values,
                        },
                        open(
                            f"runs/run_{model_kind}_{SURFACE_KIND}_{run_ix}.pkl",
                            "wb",
                        ),
                    )

                    iteration += 1

            run_ix += 1

        except Exception as e:
            print(e)
