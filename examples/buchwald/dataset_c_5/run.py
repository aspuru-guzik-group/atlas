#!/usr/bin/env python

import glob
import pickle

import numpy as np
import olympus
from olympus.campaigns import Campaign
from olympus.datasets import Dataset
from olympus.planners import Planner
from olympus.surfaces import Surface

from atlas import __datasets__


def check_convergence(samples, stop_after, bests, target_task_ix):
    sample = samples[0]
    best_params = bests[target_task_ix]["params"][:stop_after]
    for best in best_params:
        if np.all(
            [
                sample["aryl_halide"] == best[0],
                sample["additive"] == best[1],
                sample["base"] == best[2],
                sample["ligand"] == best[3],
            ]
        ):
            return True
        else:
            pass
    return False


# --------------
# load dataset
# --------------

dataset = Dataset(kind="buchwald_c")

tasks = pickle.load(
    open(f"{__datasets__}/dataset_buchwald/buchwald_tasks.pkl", "rb")
)
bests = pickle.load(
    open(f"{__datasets__}/dataset_buchwald/buchwald_best.pkl", "rb")
)


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


stop_after = 5

GOAL = "maximize"
NUM_RUNS = 40
BUDGET = 792
SURFACE_KIND = "buchwald_c"
target_task_ix = 2


SOURCE_TASKS = tasks.copy()
print("TARGET TASK IX : ", target_task_ix)
del SOURCE_TASKS[target_task_ix]


# -----------------------
# begin the experiments
# -----------------------

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

                campaign.set_param_space(dataset.param_space)

                planner = Planner(kind=model_kind, goal="maximize")
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
                        measurement = dataset.run(sample_arr)[0][0]
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

                    if check_convergence(
                        samples, stop_after, bests, target_task_ix
                    ):
                        print("found satisfactory molecule")
                        break

                    iteration += 1

            elif model_kind == "Botorch":

                # ------------------------
                # custom botorch planner
                # ------------------------

                from atlas.optimizers.gp.planner import BoTorchPlanner

                # make a campaign and add parameter space
                campaign = Campaign()

                campaign.set_param_space(dataset.param_space)

                planner = BoTorchPlanner(
                    goal="maximize",
                    feas_strategy="naive-0",
                    init_design_strategy="random",
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
                        measurement = dataset.run(sample_arr)[0][0]

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

                    if check_convergence(
                        samples, stop_after, bests, target_task_ix
                    ):
                        print("found satisfactory molecule")
                        break

                    iteration += 1

            elif model_kind == "DKT":

                # ------------------------------
                # Deep kernel transfer planner
                # ------------------------------

                from atlas.optimizers.dkt.planner import DKTPlanner

                # make a campaign and add parameter space
                campaign = Campaign()

                campaign.set_param_space(dataset.param_space)

                planner = DKTPlanner(
                    goal="maximize",
                    warm_start=False,
                    train_tasks=SOURCE_TASKS,
                    valid_tasks=SOURCE_TASKS,
                    model_path="./tmp_models/",
                    init_design_strategy="random",
                    num_init_design=5,
                    batch_size=1,
                    from_disk=False,
                    hyperparams={
                        "model": {
                            "epochs": 25000,
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
                        measurement = dataset.run(sample_arr)[0][0]

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

                    if check_convergence(
                        samples, stop_after, bests, target_task_ix
                    ):
                        print("found satisfactory molecule")
                        break

                    iteration += 1

            elif model_kind == "RGPE":

                # ---------------
                # RGPE planner
                # ---------------

                from atlas.optimizers.rgpe.planner import RGPEPlanner

                # make a campaign and add parameter space
                campaign = Campaign()

                campaign.set_param_space(dataset.param_space)

                planner = RGPEPlanner(
                    goal="maximize",
                    warm_start=False,
                    train_tasks=SOURCE_TASKS,
                    valid_tasks=SOURCE_TASKS,
                    init_design_strategy="random",
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
                        measurement = dataset.run(sample_arr)[0][0]
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

                    if check_convergence(
                        samples, stop_after, bests, target_task_ix
                    ):
                        print("found satisfactory molecule")
                        break

                    iteration += 1

            run_ix += 1

        except Exception as e:
            print(e)
