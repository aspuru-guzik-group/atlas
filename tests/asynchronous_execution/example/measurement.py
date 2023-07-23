#!/usr/bin/env python

import os, sys, binascii
import time
import pickle
import glob

import multiprocessing

import numpy as np

from atlas import Logger

class Worker:
    """ individual worker, can perform a singular measurement at a time
    """
    def __init__(
            self, 
            worker_id, 
            measurement_fn, 
            dump_dir,
            measurement_time_range=[5., 30.],
        ):
        self.worker_id = worker_id
        self.measurement_fn = measurement_fn
        self.dump_dir = dump_dir
        self.measurement_time_range = measurement_time_range

        self.is_available = True
        self.pending_exp = None

    def measure(self, params):
        measurement_time = np.random.uniform(
            self.measurement_time_range[0],
            self.measurement_time_range[1],
            size=None,
        )
        time.sleep(measurement_time)
        measurement = self.measurement_fn(params)
        measurement_timestamp = time.time()
        with open(f'{self.dump_dir}worker_result_{self.worker_id}_{measurement_timestamp}.pkl', 'wb') as f:
            pickle.dump({'params': params, 'values': measurement}, f)
        f.close()

        with open(f'{self.dump_dir}worker_status_{self.worker_id}.pkl', 'wb') as f:
            pickle.dump({'is_avail': True}, f)
        f.close()


    def check_status(self):
        return self.is_available
    
    

class Workers:
    """ Ensemble of workers, can perform multiple tasks simultaneously
    """
    def __init__(
            self, 
            num_workers,
            measurement_fn,
            pickup_dir='./pickup/', 
            dump_dir='./dump/', 
            monitor_interval=2.,
    ):
        self.num_workers = num_workers
        self.measurement_fn = measurement_fn
        self.pickup_dir = pickup_dir # where the priority queue is
        self.dump_dir = dump_dir  # where workers put their measurements
        self.monitor_interval = monitor_interval

        self.pickup_file = f'{self.pickup_dir}priority_queue.pkl'

        self.used_exp_ids = []

        # spawn the workers
        self.workers = self._create_workers()

    def _create_workers(self):
        workers = []
        for _ in range(self.num_workers):
            workers.append(
                Worker(
                    worker_id=self._gen_worker_id(),
                    measurement_fn=self.measurement_fn,
                    dump_dir=self.dump_dir, 
                )
            )
        return workers

    def _gen_worker_id(self):
        worker_id = str(binascii.hexlify(os.urandom(16)))[2:]
        worker_id = worker_id[:-1]
        return worker_id
    
    def get_avail_workers(self):
        avail_workers = []
        # read check if there are status files in dump dir

        for worker in self.workers:
            if worker.is_available:
                avail_workers.append(worker)
            else:
                # see if there are status files
                status_file = f'{self.dump_dir}worker_status_{worker.worker_id}.pkl'
                if os.path.exists(status_file):
                    worker.is_available = True
                    worker.pending_exp = None
                    avail_workers.append(worker)
                    os.system(f'rm {status_file}')

        return avail_workers
    
    def get_new_experiments(self, priority_queue_dict):
        exp_ids = [d['exp_id'] for d in priority_queue_dict]
        return [priority_queue_dict[ix] for ix, k in enumerate(exp_ids) if not k in self.used_exp_ids]

    def write_pending_exps(self):
        """ If we have some pending experiments, write them to disk to be 
        parsed by run_opt execute function which in turn informs Atlas 
        about them
        """
        pending_exps = []
        for worker in self.workers:
            if worker.pending_exp is not None:
                pending_exps.append(worker.pending_exp)
        with open(f'{self.dump_dir}pending_exps.pkl', 'wb') as f:
            pickle.dump(pending_exps, f)
        f.close()


    def monitor_pickup(self):
        while True:
            time.sleep(self.monitor_interval)
            if not os.path.exists(self.pickup_file):
                Logger.log(
                    f'No priority queue file available, checking again in {self.monitor_interval} sec',
                    'INFO'
                )
                pass
            else:
                with open(self.pickup_file, 'rb') as f:
                    priority_queue_dict = pickle.load(f)
                f.close()

                # get avail workers
                avail_workers = self.get_avail_workers()
                Logger.log(f'Available workers : {[worker.worker_id for worker in avail_workers]}', 'INFO')

                if len(avail_workers) == 0:
                    Logger.log('No available workers at this time...', 'WARNING')
                else:
                    # check if we have some experiments in the queue we have yet to 
                    # assign to worker
                    new_exps = self.get_new_experiments(priority_queue_dict)
                    if len(new_exps) == 0:
                        Logger.log('All queued experiments have been completed or assigned to workers...', 'WARNING')
                    else:
                        # we have some workers available, and new experiments, assign and
                        # commence the measurement jobs
                        worker_ix = 0
                        while worker_ix < len(avail_workers) and worker_ix < len(new_exps):

                            # unpack the parameters for the experiment
                            # Olympus parameter vector
                        
                            params = new_exps[worker_ix]
                            param_vec = params['params']
                            param_exp_id = params['exp_id']
                            worker = avail_workers[worker_ix]

                            # create process
                            worker.is_available = False
                            worker.pending_exp = param_vec
                            self.write_pending_exps()
                            Logger.log(
                                f'Submitting parameters {param_vec} (exp id {param_exp_id}) to worker {worker.worker_id}',
                                'INFO',
                            )

                            self.used_exp_ids.append(param_exp_id)
                            process = multiprocessing.Process(target=worker.measure, args=(param_vec,))
                            # run the measurement process
                            process.start()

                            worker_ix+=1
                        


                            
if __name__ == '__main__':

    from olympus import Surface

    SURFACE_KIND = 'Branin'
    surface = Surface(kind=SURFACE_KIND)


    workers_obj = Workers(
        num_workers = 3,
        measurement_fn=surface.run,
    )   

    workers_obj.monitor_pickup()