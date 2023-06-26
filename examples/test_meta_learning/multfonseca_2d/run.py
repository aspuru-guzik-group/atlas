#!/usr/bin/env python

import glob
import pickle

import numpy as np
import olympus
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import ParameterContinuous
from olympus.planners import Planner
from olympus.scalarizers import Scalarizer
from olympus.surfaces import Surface

from atlas import __datasets__

# -----------------------------
# Instantiate surface
# -----------------------------
SOURCE_NAME = "mult_fonseca_2D"
SURFACE_KIND = "MultFonseca"
surface = Surface(kind=SURFACE_KIND)

# ---------------
# Configuration
# ---------------

MODELS = [
    # 	'RandomSearch',
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

                campaign.set_param_space(surface.param_space)

                planner = Planner(kind=model_kind)

                planner.set_param_space(campaign.param_space)

                scalarizer = Scalarizer(
                    kind="Hypervolume",
                    value_space=surface.value_space,
                    goals=["min", "min"],
                )

                # start the optimization experiment
                iteration = 0
                # optimization loop
                while len(campaign.values) < BUDGET:

                    print(f"\nITERATION : {iteration}\n")

                    samples = planner.recommend(campaign.observations)
                    print(f"SAMPLES : {samples}")
                    for sample in samples:
                        sample_arr = sample.to_array()
                        measurement = surface.run(
                            sample_arr, return_paramvector=True
                        )
                        campaign.add_and_scalarize(
                            sample_arr, measurement, scalarizer
                        )

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

                campaign.set_param_space(surface.param_space)

                planner = BoTorchPlanner(
                    goal="minimize",
                    feas_strategy="naive-0",
                    init_design_strategy="lhs",
                    is_moo=True,
                    value_space=surface.value_space,
                    scalarizer_kind="Hypervolume",
                    moo_params={},
                    goals=["min", "min"],
                )

                scalarizer = Scalarizer(
                    kind="Hypervolume",
                    value_space=surface.value_space,
                    goals=["min", "min"],
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
                        measurement = surface.run(
                            sample_arr, return_paramvector=True
                        )
                        campaign.add_and_scalarize(
                            sample_arr, measurement, scalarizer
                        )

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

                campaign.set_param_space(surface.param_space)

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
                    is_moo=True,
                    value_space=surface.value_space,
                    scalarizer_kind="Hypervolume",
                    moo_params={},
                    goals=["min", "min"],
                )

                scalarizer = Scalarizer(
                    kind="Hypervolume",
                    value_space=surface.value_space,
                    goals=["min", "min"],
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
                        measurement = surface.run(
                            sample_arr, return_paramvector=True
                        )
                        campaign.add_and_scalarize(
                            sample_arr, measurement, scalarizer
                        )

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

                campaign.set_param_space(surface.param_space)

                planner = RGPEPlanner(
                    goal="minimize",
                    warm_start=False,
                    train_tasks=train_tasks,
                    valid_tasks=valid_tasks,
                    init_design_strategy="lhs",
                    num_init_design=5,
                    batch_size=1,
                    is_moo=True,
                    value_space=surface.value_space,
                    scalarizer_kind="Hypervolume",
                    moo_params={},
                    goals=["min", "min"],
                )

                scalarizer = Scalarizer(
                    kind="Hypervolume",
                    value_space=surface.value_space,
                    goals=["min", "min"],
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
                        measurement = surface.run(
                            sample_arr, return_paramvector=True
                        )
                        campaign.add_and_scalarize(
                            sample_arr, measurement, scalarizer
                        )

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
