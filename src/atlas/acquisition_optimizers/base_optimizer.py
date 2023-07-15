#!/usr/bin/env python

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from abc import abstractmethod

import time
from copy import deepcopy
import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction
from olympus.campaigns import ParameterSpace

from atlas import Logger
from atlas.acquisition_functions.acqfs import VarianceBased
from atlas.acquisition_functions.acqf_utils import get_batch_initial_conditions


from atlas.params.params import Parameters


class AcquisitionOptimizer:
    def __init__(
        self,
        params_obj: Parameters,
        acquisition_type: str,
        acqf: AcquisitionFunction,
        known_constraints: Callable,
        batch_size: int,
        feas_strategy: str,
        fca_constraint: Callable,
        params: torch.Tensor,
        timings_dict: Dict,
        fixed_params:Optional[List[Dict[int, float]]]=[],
        **kwargs: Any,

    ):
        self.params_obj = params_obj
        self.acquisition_type = acquisition_type
        self.acqf = acqf
        self.known_constraints = known_constraints
        self.batch_size = batch_size
        self.feas_strategy = feas_strategy
        self.fca_constraint = fca_constraint
        self._params = params
        self.timings_dict = timings_dict

        self.fixed_params = fixed_params


    @abstractmethod
    def _optimize(self):
        ...

    def optimize(self):

        start_time = time.time()
        # returns list of parameter vectors with recommendations
        results = self._optimize()
        self.timings_dict['acquisition_opt'] = time.time()-start_time


        # if we have a general parameter optimization, we use a
        # variance-based sampling procedure to select the next general parameter(s)
        if self.acquisition_type == 'general':

            X_sns_empty, general_raw = self.acqf.generate_X_sns()

            functional_dims = np.logical_not(self.params_obj.exp_general_mask)

            # convert results to expanded tensor
            X_star = torch.tensor(
                self.params_obj.param_vectors_to_expanded(results, return_scaled=True)
            )
            # TODO: careful of batch size
            X_star = torch.unsqueeze(X_star,1)

            X_sns = torch.empty( (X_star.shape[0],) + X_sns_empty.shape )
            for x_ix in range(X_star.shape[0]):
                X_sn  = torch.clone(X_sns_empty)
                X_sn[:, :, functional_dims] = X_star[x_ix, :, functional_dims]
                X_sns[x_ix,:,:,:] = X_sn

            acqf_sn = VarianceBased(reg_model=self.acqf.reg_model)

            # generates list of stdevs, one set for each batch
            for ix, X_sn in enumerate(X_sns):
                sigma = acqf_sn(X_sn)
                select_gen_params = general_raw[ torch.argmax(sigma) ]

                for gen_param_ix in self.params_obj.general_dims:
                    results[ix][self.params_obj.param_space[gen_param_ix].name] = select_gen_params[gen_param_ix]
    
        # apply compositional constraint if present
        if self.known_constraints.has_compositional_constraint:
            results = self._apply_compositional_constraint(results)


        # apply process-constrained batch constraint(s) if present
        if self.known_constraints.has_batch_constraint:
            results = self._apply_batch_constraint(results)

        return results
    

    def _apply_compositional_constraint(self, results):
        constrained_results = []
        for pvec in results:
            constrained_pvec = deepcopy(pvec)
            sum_params = np.sum(
                [float(pvec[name]) for name in self.known_constraints.compositional_constraint_param_names[:-1]]
            )
            assert sum_params <= 1.
            # update dependent parameter
            constrained_pvec[self.known_constraints.compositional_constraint_param_names[-1]] = 1. - sum_params 
            constrained_results.append(constrained_pvec)

        return constrained_results


    def _apply_batch_constraint(self, results):
        constrained_results = []
        for pvec in results:
            constrained_pvec = deepcopy(pvec)
            for constrained_param_name in self.known_constraints.batch_constrained_param_names:
                constrained_pvec[constrained_param_name] = results[0][constrained_param_name]
            constrained_results.append(constrained_pvec)

        return constrained_results




    def gen_initial_conditions(self, num_restarts:int=200, return_raw:bool=True):
        """ generates inital conditions, particularly for problems with
        known constraints, or if using the FCA feasibiity-aware method for
        problems with unknown constraints
        """

        # TODO: take care of the known constraints and/or inequality constraints
        # ....

        nonlinear_inequality_constraints = []
        return_nonlinear_inequality_constraints = []

        if isinstance(self.known_constraints, list):
            nonlinear_inequality_constraints.extend(self.known_constraints)
            return_nonlinear_inequality_constraints.extend(self.known_constraints)
        elif isinstance(self.known_constraints, Callable):
            nonlinear_inequality_constraints.append(self.known_constraints)
            return_nonlinear_inequality_constraints.append(self.known_constraints)


        if self.feas_strategy == 'fca' and not self.use_reg_only:

            if self.kind == 'genetic':
                # add wrapped fca constraint if genetic algorithm optimizer
                return_nonlinear_inequality_constraints.append(self._wrapped_fca_constraint)
            else:
                return_nonlinear_inequality_constraints.append(self.fca_constraint)

            nonlinear_inequality_constraints.append(self.fca_constraint)


        if nonlinear_inequality_constraints == []:
            # we dont have any constraints, generate inital conditions

            batch_initial_conditions, raw_conditions = get_batch_initial_conditions(
                num_restarts=num_restarts,
                batch_size=self.batch_size,
                param_space=self.params_obj.param_space,
                known_constraints=self.known_constraints,
                fca_constraint=self.known_constraints,
                # constraint_callable=[],
                has_descriptors=self.has_descriptors,
                mins_x=self.params_obj._mins_x,
                maxs_x=self.params_obj._maxs_x,
                return_raw=return_raw,
            )

            return (
                None, # nonlinear_inequality_constraints
                batch_initial_conditions, # initial conditions
                raw_conditions, # raw conditions (gradient doesnt need)
            )

        else:

        # TODO: it shouldnt matter if we are using FCA or not here.. .
        # if self.feas_strategy == 'fca':

            # attempt to get the batch initial conditions
            batch_initial_conditions, raw_conditions = get_batch_initial_conditions(
                num_restarts=num_restarts,
                batch_size=self.batch_size,
                param_space=self.params_obj.param_space,
                #constraint_callable=nonlinear_inequality_constraints,
                known_constraints=self.known_constraints,
                fca_constraint=self.known_constraints,
                has_descriptors=self.has_descriptors,
                mins_x=self.params_obj._mins_x,
                maxs_x=self.params_obj._maxs_x,
                return_raw=return_raw,
            )

            if type(batch_initial_conditions) == type(None):
                # if we cant find sufficient inital design points, resort to using the
                # acqusition function only (without the feasibility constraint)
                msg = "Insufficient starting points for constrianed acqf optimization, resorting to optimization of regression acqf only"
                Logger.log(msg, "WARNING")

                nonlinear_inequality_constraints = []
                return_nonlinear_inequality_constraints = []

                # try again
                batch_initial_conditions, raw_conditions = get_batch_initial_conditions(
                    num_restarts=num_restarts,
                    batch_size=self.batch_size,
                    param_space=self.params_obj.param_space,
                    known_constraints=self.known_constraints,
                    fca_constraint=self.known_constraints,
                    #constraint_callable=nonlinear_inequality_constraints,
                    has_descriptors=self.has_descriptors,
                    mins_x=self.params_obj._mins_x,
                    maxs_x=self.params_obj._maxs_x,
                    return_raw=return_raw,
                )

                if type(batch_initial_conditions) == type(None):
                    # if we still cannot find initial conditions, there is likey a problem, return to user
                    message = "Could not find inital conditions for constrianed optimization..."
                    Logger.log(message, "FATAL")
                elif type(batch_initial_conditions) == torch.Tensor:
                    # weve found sufficient conditions on the second try, nothing to do
                    return (
                        None, # nonlinear_inequality_constraints
                        batch_initial_conditions, # initial conditions
                        raw_conditions, # raw conditions (gradient doesnt need)
                    )
            elif type(batch_initial_conditions) == torch.Tensor:
                # we've found initial conditions on the first try, nothing to do
                return (
                    return_nonlinear_inequality_constraints, # nonlinear_inequality_constraints
                    batch_initial_conditions, # initial conditions
                    raw_conditions, # raw conditions (gradient doesnt need)
                )
