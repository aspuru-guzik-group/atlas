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
# GOLDSTIEN-PRICE 2D SURFACE
# ----------------------------
def gprice(x):
    # the goldstein price function (2D)
    # https://www.sfu.ca/~ssurjano/goldpr.html
    x1 = x[:, 0]
    x2 = x[:, 1]

    # scale x
    x1 = x1 * 4.0
    x1 = x1 - 2.0
    x2 = x2 * 4.0
    x2 = x2 - 2.0

    gprice = (
        1
        + (x1 + x2 + 1) ** 2
        * (19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2)
    ) * (
        30
        + (2 * x1 - 3 * x2) ** 2
        * (18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2)
    )

    # lognormalize
    mean = 8.693
    std = 2.427
    gprice = 1 / std * (np.log(gprice) - mean)

    # maximize
    gprice = -gprice

    return gprice


# -----------------------------
# Instantiate Branin surface
# -----------------------------
SOURCE_NAME = "gprice_2D"
SURFACE_KIND = "gprice"
surface = gprice

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
                # add 2 continuous Parameters
                param_0 = ParameterContinuous(
                    name="param_0", low=0.0, high=1.0
                )
                param_1 = ParameterContinuous(
                    name="param_1", low=0.0, high=1.0
                )
                param_space.add(param_0)
                param_space.add(param_1)

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
                # add 2 continuous Parameters
                param_0 = ParameterContinuous(
                    name="param_0", low=0.0, high=1.0
                )
                param_1 = ParameterContinuous(
                    name="param_1", low=0.0, high=1.0
                )
                param_space.add(param_0)
                param_space.add(param_1)

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
                # add 2 continuous Parameters
                param_0 = ParameterContinuous(
                    name="param_0", low=0.0, high=1.0
                )
                param_1 = ParameterContinuous(
                    name="param_1", low=0.0, high=1.0
                )
                param_space.add(param_0)
                param_space.add(param_1)

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
                # add 2 continuous Parameters
                param_0 = ParameterContinuous(
                    name="param_0", low=0.0, high=1.0
                )
                param_1 = ParameterContinuous(
                    name="param_1", low=0.0, high=1.0
                )
                param_space.add(param_0)
                param_space.add(param_1)

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
