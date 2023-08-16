# Asynchronous data-driven experiment execution with Atlas

This directory contains a simple demonstration of aynchronous experimental execution with data-driven
experiment planning using `atlas`. 


## Installation

You will need to install multiprocessing capabilities to use this demo. You can do so by running

```bash
pip install multiprocess
```

## Overview

In this demo, we optimize the 2d continuous parameter `Branin` surface from `olympus`. We perform batched Bayesian optimization with multiple measurement workers, each of which can perform an individual measurement with varying execution times. 

`Workers` is an object defined in `measurement.py` which handles the measurement of the surface. `Workers` consists of 3 `Worker` instances, each of which can be delegated measurement tasks for a single set of parameters. 

The asynchronous optimization loop is defined in the `execute_opt()` function in `run_opt.py`. The `atlas` `GPPlanner` recommeneds batches of 3 parameters at a time, and places these recommendations in a file called `pickup/priority_queue.pkl` for pickup by the `Workers`. Whenever a worker is available, `Workers` reads from `priority_queue.pkl` and executes the next measurement job, reading the queue from top to bottom. For each measurement job, we randomly sample an execution time from the interval $[5 - 30]$ sec, simulating variability in experimental execution time. When the `Worker` has finished the measurement, it writes the resulting measurement in a `.pkl` file begining with `worker_result_` in `dump/`. 

The `dump/worker_result_*.pkl` files are loaded by the `run_opt.py` script and inform the planner about completed experiments. Importantly, experiments which have been assigned to a `Worker` but not yet completed are exposed to the planner as `pending_experiments` through its `set_pending_experiments()` method. The result of this is two-fold:

* First, this creates a `pending_experiment` constraint, which assures that experiments that are pending will not be proposed by the planner in subsequent calls to `recommend()`. This is particularly relevant when dealing with discrete or categorical parameters. 
* Second, `pending_experiments` are also used to update the regression surrogate model of the planner such that ...

The figure below gives a conceptual overview of this demo and of asynchronous experimental execution more generally. 


![alt text](https://github.com/aspuru-guzik-group/atlas/blob/main/static/async_opt_demo_fig.png)


## Usage

Go ahead and open two terminals. In the first terminal, run the `measurement.py` script 

```bash
./measurement.py
```

This script will begin monitoring the `pickup/` directory for a priority queue of experimental parameters to be measured. Initially, you will find that there exists no such queue. To produce the queue, we need to run the `run_opt.py` script. Run this script in your second terminal.

```bash
./run_opt.py
```

This kicks off the asynchronous experiment. The `GPPlanner` will propose an inital design batch of 3 parameters which will be immediately assigned to the workers. Note that I've constrained the experiment to wait for all 3 intial design measurements to be completed before starting asynchronous execution. This is done to give the planner some initial data to get its bearings, but note that you dont _have_ to run things this way. After the inital design portion of the experiment has completed, `run_opt.py` looks for fresh measurement files in `dump/` and informs the planner about the completed observations. Each time a new experiment (single experiment, not a full batch!!) is completed, the planner "refills" the priority queue with fresh parameter recommendations, whose is selection conditioned on the newest avialable observations and pending experiments.

By default, this experiment will run until 20 measurements are completed.


Before restarting the experiment, clear all the residual files by running the `clear_files.py` script.

```bash
./clear_files.py
```

## Extensions

While this demo is very primitive, it lays a simple foundation for aysnchronous experimentaion in SDLs. Feel free to customize this demo (especially the `measurement.py` file) to fit your measurement apparatus. Below are some notes and suggestions concerning extensions to the code.

* As an alternative to writing the priority queue and measurements to disk as pickle files, I've found that using the [Google sheets API](https://developers.google.com/sheets/api/guides/concepts) to be a nice approach for implementing proof-of-concept asynchronous workflows. You can use a Google sheet to house the prority queue, pending experiments, _and_ the completed observations. I would recommend using the [`gspread` Python library](https://docs.gspread.org/en/v5.10.0/) to progammatically update the cells in the Google sheet. This approach also works nicely for hybrid human-robotic SDLs, in which, for example, a human must make the measurements while robotic laboratory equipment prepares samples recommended by an `atlas` planner. A Google sheet is human-readable and editable, and can be accessed from anywhere using any internet browser.

To use the Google doc feature, you must install the Google client library

```bash
pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

and `gspread`

```bash
pip install gspread
```


* ... 

