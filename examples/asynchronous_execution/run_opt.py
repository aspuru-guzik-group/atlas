#!/usr/bin/env python

import binascii
import glob
import os
import pickle
import sys
import time

from olympus import Campaign, Surface

from atlas import Logger
from atlas.planners.gp.planner import GPPlanner

SURFACE_KIND = "Branin"
BUDGET = 20
PICKUP_DIR = "./pickup/"
DUMP_DIR = "./dump/"
PICKUP_FILE = f"{PICKUP_DIR}priority_queue.pkl"
MONITOR_INTERVAL = 5.0


def refill_priority_queue(planner, campaign):
    # (re)fill the priority queue
    samples = planner.recommend(campaign.observations)

    pickup_samples = []
    for sample in samples:
        exp_id = str(binascii.hexlify(os.urandom(16)))[2:]
        exp_id = exp_id[:-1]
        pickup_samples.append({"exp_id": exp_id, "params": sample})

    # overwrite the current priority queue
    with open(PICKUP_FILE, "wb") as f:
        pickle.dump(pickup_samples, f)
    f.close()


def check_set_pending_exps(planner) -> None:
    if os.path.exists(f"{DUMP_DIR}pending_exps.pkl"):
        with open(f"{DUMP_DIR}pending_exps.pkl", "rb") as f:
            pending_exps = pickle.load(f)
        f.close()
        # let atlas know about pending experiments
        Logger.log(f"Pending_exps : {pending_exps}", "WARNING")

        planner.set_pending_experiments(pending_experiments=pending_exps)


def execute_opt():
    surface = Surface(kind=SURFACE_KIND)

    campaign = Campaign()
    campaign.set_param_space(surface.param_space)

    planner = GPPlanner(
        goal="minimize",
        init_design_strategy="random",
        num_init_design=3,
        batch_size=3,
        acquisition_optimizer_kind="pymoo",
    )
    planner.set_param_space(surface.param_space)

    iter_ = 0
    while len(campaign.observations.get_values()) < BUDGET:
        if iter_ == 0:
            # commence initial design right away without measurement files
            refill_priority_queue(planner, campaign)
            iter_ += 1
        else:
            # wait for new measurements before re-training
            measurement_files = glob.glob(f"{DUMP_DIR}worker_result_*")
            while len(measurement_files) == 0:
                time.sleep(MONITOR_INTERVAL)
                measurement_files = glob.glob(f"{DUMP_DIR}worker_result_*")

            # we have some measurement files, unpack them and
            # update the olympus campaign
            for measurement_file in measurement_files:
                with open(measurement_file, "rb") as f:
                    measurement_dict = pickle.load(f)
                f.close()
                campaign.add_observation(
                    measurement_dict["params"],
                    measurement_dict["values"],
                )
                # remove the measurement file
                os.system(f"rm {measurement_file}")

                # (re)fill the priority queue if initial design has completed
                if len(campaign.observations.get_values()) >= 3:
                    # tell Atlas planner about pending experiments
                    check_set_pending_exps(planner)

                    # refill the priority queue with planner conditioned on
                    # pending experiments
                    refill_priority_queue(planner, campaign)
                    iter_ += 1


if __name__ == "__main__":
    execute_opt()
