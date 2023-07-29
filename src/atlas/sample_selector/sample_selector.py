#!/usr/bin/env python

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from olympus.objects import ParameterVector
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.result import Result

from atlas import tkwargs


def get_olympus_param_bounds(params_obj):
    """ Get unscaled param bounds from Olympus

    """
    param_uppers = []
    param_lowers = []

    for param in params_obj.param_space:
        if param.type == 'continuous':
            param_uppers.append(param.high)
            param_lowers.append(param.low)
        # TODO: deal with other param types
        else:
            pass

    return (
        torch.tensor(param_uppers, **tkwargs),
        torch.tensor(param_lowers, **tkwargs)
    )   


def batch_local_penalization_selector(
    pymoo_results: List[Result],
    pymoo_problem: Problem,
    batch_size: int, 
    dist_param: float,
) -> List[ParameterVector]:
    """batch local penalization for batch construction
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
   
    param_uppers, param_lowers = get_olympus_param_bounds(pymoo_problem.params_obj) # (param_dim,)
    
    # TODO: may want to change this attribute name its confusing
    prev_obs_params = pymoo_problem.params_obj.olympus # (num_obs, param_dim)
    num_obs = float(len(prev_obs_params))
    param_ranges = param_uppers-param_lowers
    char_dists = torch.divide(
        param_ranges, 
        torch.pow(torch.tensor([num_obs],**tkwargs), dist_param),
    ) # (param_dim,)
    print(num_obs)
    print(param_ranges)
    print(char_dists)

    batch_samples = []
    batch_samples_arr = np.empty((batch_size, len(pymoo_problem.params_obj.param_space)))

    if len(pymoo_results)==1:
        # one single result population
        result = pymoo_results[0]
        samples_pvec = pymoo_problem._pymoo_to_olympus(
            [ind.X for ind in result.pop],
            forward_transform=False, 
            return_param_vec=True, 
        )
        for sample in samples_pvec:
            print(sample)
        samples_arr = pymoo_problem._pymoo_to_olympus(
            [ind.X for ind in result.pop],
            forward_transform=False, 
            return_param_vec=False, 
        )
        samples_torch = [pvec.to_array() for pvec in samples_pvec]
        samples_torch = torch.tensor(samples_torch, **tkwargs) # (num_samples, param_dim)
        # these are negative for some reason? multiply by -1.
        acqf_vals = torch.tensor([ind.F for ind in result.pop],**tkwargs) * -1.

        penalized_acqf_vals = acqf_vals.clone()

        # if batch_size==1, just take the proposal corresponding to the best
        # acqf value, as long as its not a duplicate 
        batch_counter = 0 
        while batch_counter < batch_size:

            # get the maximum of the penalized acqf vals
            # and the corresponding sample 
            best_idx = torch.argmax(penalized_acqf_vals)
            best_sample_pvec = samples_pvec[best_idx]
            print('batch_counter : ', batch_counter)
            print('best_sample_pvec : ', best_sample_pvec)

            best_sample_arr = best_sample_pvec.to_array()
            
        
            # check to see if the set of perviously measured params
            # contains this sample
            if not any((prev_obs_params[:] == best_sample_arr).all(1)):
                # check if this proposal has already been committed to in 
                # the batch
                if not(any((batch_samples_arr[:] == best_sample_arr).all(1))):
                    batch_samples.append(best_sample_pvec)
                    batch_samples_arr[batch_counter, :] = best_sample_arr
                    penalized_acqf_vals[best_idx] = 0.

                    batch_counter+=1

                    # TODO: update penalties based on batch samples 
                    # committed to this far
                    
                    batch_samples_torch = torch.tensor(batch_samples_arr, **tkwargs)[:batch_counter,:]

                    # compute minimum distances of proposals to committed to batch proposals
                    dists = np.array([
                        torch.abs(
                                sample-batch_samples_torch
                                ) for sample in samples_torch
                    ])
                    min_dists = torch.amin(torch.tensor(dists), dim=-2)# (num_samples, param_dim)
        
    
                    penalty_factor = torch.mean(
                        torch.exp(
                            2. * (min_dists - char_dists) / param_ranges
                        ),
                        dim=1,
                    )
                
                    div_crits = torch.clamp(penalty_factor, max=1.).view(
                        penalty_factor.shape[0],1
                    )

                    penalized_acqf_vals *= div_crits

                else:
                    # avoid batch duplicate sample
                    # set reward very small 
                    penalized_acqf_vals[best_idx] = 0.
            else:
                # avoid duplicated sample
                # set reward very small
                penalized_acqf_vals[best_idx] = 0.
    

    else:
        raise NotImplementedError

    return batch_samples


