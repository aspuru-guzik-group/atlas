#!/usr/bin/env python

import os, sys, binascii
import glob
import time
import pickle
import numpy as np
import pandas as pd

from olympus import Campaign, Surface
from atlas.planners.gp.planner import BoTorchPlanner


SURFACE_KIND = 'Branin'
BUDGET = 20
PICKUP_DIR = './pickup/'
DUMP_DIR = './dump/'
PICKUP_FILE = f'{PICKUP_DIR}priority_queue.pkl'
MONITOR_INTERVAL = 5.


def refill_priority_queue(planner, campaign):
    # (re)fill the priority queue        
    samples = planner.recommend(campaign.observations)

    pickup_samples = []
    for sample in samples:
        exp_id = str(binascii.hexlify(os.urandom(16)))[2:]
        exp_id = exp_id[:-1]
        pickup_samples.append(
            {'exp_id':exp_id, 'params':sample}
        )
    
    # overwrite the current priority queue
    with open(PICKUP_FILE, 'wb') as f:
        pickle.dump(pickup_samples, f)
    f.close()



def execute_opt():

    surface = Surface(kind=SURFACE_KIND)

    campaign = Campaign()
    campaign.set_param_space(surface.param_space)

    planner = BoTorchPlanner(
        goal='minimize', 
        init_design_strategy='random', 
        num_init_design=3, 
        batch_size=3,
        acquisition_optimizer_kind='pymoo',
    )
    planner.set_param_space(surface.param_space)

    iter_ = 0
    while len(campaign.observations.get_values()) < BUDGET:
        
        if iter_ == 0:
            # commence initial design right away without measurement files
            refill_priority_queue(planner, campaign)
            iter_+=1
        else:

            # wait for new measurements before re-training
            measurement_files = glob.glob(f'{DUMP_DIR}worker_result_*')
            while len(measurement_files) == 0:
                time.sleep(MONITOR_INTERVAL)
                measurement_files = glob.glob(f'{DUMP_DIR}worker_result_*')
            
            # we have some measurement files, unpack them and 
            # update the olympus campaign
            for measurement_file in measurement_files:
                with open(measurement_file, 'rb') as f:
                    measurement_dict = pickle.load(f)
                f.close()
                campaign.add_observation(
                    measurement_dict['params'],
                    measurement_dict['values'],
                )
                # remove the measurement file
                os.system(f'rm {measurement_file}')

                # TODO: need to update the planners pending experiments attribute

                # (re)fill the priority queue if initial design has completed
                if len(campaign.observations.get_values()) >= 3:
                    refill_priority_queue(planner, campaign)
                    iter_+=1


if __name__ == '__main__':
    execute_opt()

            


