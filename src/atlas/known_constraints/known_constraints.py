#!/usr/bin/env python

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from olympus.campaigns import ParameterSpace
from olympus.objects import ParameterVector

from atlas.utils.planner_utils import propose_randomly

from atlas import Logger


class KnownConstraintsIter:
    """ iterator class for known constraints
    """
    def __init__(self, known_constraints_obj):
        self._known_constraints_obj = known_constraints_obj
        self._current_idx = 0
        self._num_known_constraints = len(
            self._known_constraints_obj.known_constraints
        )

    def __iter__(self):
        return self

    def __next__(self):
        if self._current_idx < self._num_known_constraints:
            known_constraint_callable = self._known_constraints_obj.known_constraints[self._current_idx]
            self._current_idx += 1
            return known_constraint_callable
        raise StopIteration
    

class CompositionalConstraint:
    """ 
    """

    def __init__(self, compositional_params: List[int], param_space: ParameterSpace) -> None:
        self.compositional_params = compositional_params
        self.param_space = param_space

        self._validate()
        self.valid_constraint = True

    def __call__(self, params) -> bool:
        """ evaluate the constraint
        """
        # sum of first n-1 constrained parameters must not exceed 1
        if np.sum([float(params[idx]) for idx in self.compositional_params[:-1]]) > 1.:
            return False
        
        return True
        
    def _validate(self):

        # check that we dont only have 1 param
        if len(self.compositional_params)==1:
            Logger.log(
                'You must provide more than 1 parameter for compositional constraints',
                'FATAL',
            )
        
        # check constrained parameter types (categorical params are not applicable here)
        if not all([self.param_space[idx].type in ['continuous', 'discrete'] for idx in self.compositional_params]):
            Logger.log(
                'Parameters must be of type "continuous" or "discrete" to be subject to compostional constraints', 
                'FATAL',
            )

        # check that the dependent parameter is continuous 
        if self.param_space[self.compositional_params[-1]].type == 'discrete': 
            Logger.log(
                'Dependent parameter is "discrete", this must be set to continuous', 
                'FATAL', 
            )



class PermutationConstraint:
    """
    """

    def __init__(self, permutation_params: List[int], param_space: ParameterSpace) -> None:
        self.permutation_params = permutation_params
        self.param_space = param_space

        self._validate()
        self.valid_constraint = True
        

    def __call__(self, params) -> bool:
        """ evaluate the constraint
        """
        values = []
        for idx in self.permutation_params:
            constr_param = self.param_space[idx]
            if constr_param.type == 'categorical':
                values.append(constr_param.options.index(params[idx]))
            else:
                values.append(float(params[idx]))

        return values == sorted(values)


    def _validate(self):

        # assert that we dont just have 1 parameter
        if len(self.permutation_params)==1:
            Logger.log(
                'You must provide more than 1 parameter for permutation constraints',
                'FATAL',
            )
        
        # parameters need to be either all numerical or all categorical
        if not np.logical_or(
            all([self.param_space[idx].type in ['continuous', 'discrete'] for idx in self.permutation_params]),
            all([self.param_space[idx].type=='categorical' for idx in self.permutation_params])
        ):
            Logger.log(
                'Permutation constraint parameters must either all numerical or all categorical',
                'FATAL'
            )
        

class PendingExperimentConstraint:
    """ Blacklist pending experiments for asynchronous execution setting
    """
    def __init__(self, pending_experiments: List[ParameterVector], param_space: ParameterSpace) -> None: 
        self.pending_experiments = pending_experiments
        self.param_space = param_space

    def __str__(self):

        str_ = f'{len(self.pending_experiments)} experiments are currently pending\n'
        for pending_exp in self.pending_experiments:
            str_ += f'{pending_exp}\n'

        return str_

    def __call__(self, params) -> bool:
        """ Evaluate constraint
        """
        params_arr = np.array(params).astype(str)
        # check to see if poposed param is in the pending experiments
        # NOTE: str dtype so we dont have to check parameter types? 
        for pending_exp in self.pending_experiments:
            pending_exp_arr = pending_exp.to_array().astype(str)
            if (params_arr == pending_exp_arr).all():
                return False
        
        return True


class KnownConstraints:
    """ user-level known constraints wrapper

    Args:
        known_constraints (list): list of user-defined Python callables
            representing known constraints
        param_space (ParameterSpace): Olympus parameter space for the problem
        has_descriptors (bool): wether or not problem has descriptors
        compositional_params (list): list of parameter indices subject to compositional 
            constraints, must be of type 'continuous' or 'discrete'
        permutation_params (list): list of parameter indices subject to 
            permutation constraints, can be any parameter type
    """

    def __init__(
            self, 
            known_constraints: List[Callable], 
            param_space: ParameterSpace, 
            has_descriptors: bool,
            compositional_params: Optional[List[int]] = None,
            permutation_params: Optional[List[int]] = None,
            batch_constrained_params: Optional[List[int]] = None,
        ) -> None:
        self.param_space = param_space
        self.known_constraints = known_constraints
        self.has_descriptors = has_descriptors
        self.compositional_params = compositional_params
        self.permutation_params = permutation_params
        self.batch_constrained_params = batch_constrained_params

        # deal with process-constrained batch constraints
        if self.batch_constrained_params:
            self.has_batch_constraint = True
        else:
            self.has_batch_constraint = False


        # deal with compositional constraint
        if self.compositional_params:
            self.compositional_constraint = CompositionalConstraint(
                self.compositional_params, self.param_space,     
            )
            # add constraint to list
            self.known_constraints.append(self.compositional_constraint) 
            self.has_compositional_constraint = True
        else:
            self.has_compositional_constraint = False

        # deal with permutation constraint 
        if self.permutation_params:
            self.permutation_constraint = PermutationConstraint(
                self.permutation_params, self.param_space
            )
            # add constraint to list
            self.known_constraints.append(self.permutation_constraint)
            self.has_permutation_constraint = True
        else:
            self.has_permutation_constraint = False

        # validate the known constraint function(s)
        self._validate_known_constraints()

        self.has_pending_experiment_constraint = False

    @property
    def is_empty(self):
        return self.known_constraints==[]
    
    @property
    def batch_constrained_param_names(self):
        return [self.param_space[idx].name for idx in self.batch_constrained_params]

    @property 
    def compositional_constraint_param_names(self):
        return [self.param_space[idx].name for idx in self.compositional_params]
    
    @property 
    def permutation_constraint_param_names(self):
        return [self.param_space[idx].name for idx in self.permutation_params]
    
    
    @property
    def num_known_constraints(self):
        return len(self.known_constraints)
    
    @property
    def compositional_dependent_param(self):
        # param whose value is dependent on the value of others
        if self.compositional_params:
            return self.compositional_params[-1]
        else:
            return None

    def __iter__(self):
        return KnownConstraintsIter(self)
    
    def add_pending_experiments(self, pending_experiments):
        """ add pending experiment constraint
        """


    def remove_pending_experiments(self):
        """ remove pending experiment constraint
        """
        # TODO: delete this potentially ...
        return None

    def _validate_known_constraints(self):
        """ check known constraints
        """
        if not type(self.known_constraints)==list:
            msg = 'You must pass a list of callable for known constraints'
            Logger.log(msg, 'FATAL')
        else:
            if self.is_empty:
                # no user-defined known constraints, this is OK
                return 

            if not all([callable(kc) for kc in self.known_constraints]):
                msg = 'All known constraints must be Python callables'
                Logger.log(msg, 'FATAL')

            # propose samples randomly and validate that the user-defined known constraint 
            # functions behave as required
            num_samples = 1000
            _, samples = propose_randomly(
                num_samples, 
                self.param_space,
                self.has_descriptors,
            )
            kc_vals = []
            for kc in self.known_constraints:
                for sample in samples:
                    if type(kc(sample))!=bool:
                        Logger.log(
                            f'Known constraint functions must return booleans. Not the case for parameter setting {sample}',
                            'FATAL'
                        )









