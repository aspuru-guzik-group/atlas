#!/usr/bin/env python

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from pymoo.core.population import Population
from pymoo.core.problem import Problem

from olympus.objects import ParameterVector


def batch_local_penalization_selector(
    final_pop: Population,
    pymoo_problem: Problem,
    dist_param: float, 
) -> List[ParameterVector]:
    """ batch local penalization for batch construction 
    Applies a proximity penalty to samples from a batch of
    candidate parameters and adds this value to thier raw
    acquisition function values. In Atlas, this is mainly designed
    for use with the GA-type acquisition optimizers (either
    PyMOO or DEAP), where the local penalization is applied to 
    the final population of proposed parameter sets.

    Args: 
        final_pop (Population): final population after acquisition
            function optimization
        pymoo_problem (Problem): object defining the GA acquisition
            function optimization problem
        dist_param (float): adjustable parameter influencing how the
            penalization scales with distance 

    Inspired by the sample selector of Gryffin: 
    https://github.com/aspuru-guzik-group/gryffin/blob/develop/src/gryffin/sample_selector/sample_selector.py

    """
    

    


    return None
