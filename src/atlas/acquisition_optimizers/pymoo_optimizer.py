#!/usr/bin/env python

# from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# import numpy as np
# import torch
# from botorch.acquisition import AcquisitionFunction
# from pymoo.core.variable import Real, Integer, Choice
# from pymoo.algorithms.moo.nsga2 import NSGA2
# from pymoo.core.mixed import (
#     MixedVariableMating, 
#     MixedVariableGA, 
#     MixedVariableSampling, 
#     MixedVariableDuplicateElimination
# )
# from pymoo.core.population import Population
# from pymoo.optimize import minimize
# from pymoo.core.problem import Problem
# from pymoo.config import Config
# Config.show_compile_hint = False

# from atlas import Logger
# from atlas.acquisition_functions.acqfs import create_available_options
# from atlas.acquisition_optimizers.base_optimizer import AcquisitionOptimizer
# from atlas.params.params import Parameters
# from atlas.utils.planner_utils import (cat_param_to_feat, forward_normalize,
#                                     forward_standardize, get_cat_dims,
#                                     get_fixed_features_list,
#                                     infer_problem_type, param_vector_to_dict,
#                                     propose_randomly, reverse_normalize,
#                                     reverse_standardize)



# class PymooGAOptimizer(AcquisitionOptimizer):

#     def __init__(
#             self, 
#             params_obj: Parameters,
#             acquisition_type: str, 
#             acqf: AcquisitionFunction,
#             known_constraints: Union[Callable, List[Callable]],
#             batch_size: int,
#             feas_strategy: str,
#             fca_constraint: Callable,
#             params: torch.Tensor,
#             timings_dict: Dict,
#             use_reg_only:bool=False,
#             **kwargs: Any,
#     ):
#         """
#         Genetic algorithm acquisition optimizer from pymoo 
#         """
#         local_args = {
#             key: val for key, val in locals().items() if key != "self"
#         }
#         super().__init__(**local_args)

#         self.params_obj = params_obj
#         self.param_space = self.params_obj.param_space
#         self.problem_type = infer_problem_type(self.param_space)
#         self.acquisition_type = acquisition_type
#         self.acqf = acqf
#         self.bounds = self.params_obj.bounds
#         self.batch_size = batch_size
#         self.feas_strategy = feas_strategy
#         self.fca_constraint = fca_constraint
#         self.known_constraints = known_constraints
#         self.use_reg_only = use_reg_only
#         self.has_descriptors = self.params_obj.has_descriptors
#         self._params = params
#         self._mins_x = self.params_obj._mins_x
#         self._maxs_x = self.params_obj._maxs_x

#         self.kind = 'pymoo'



#     def _set_pymoo_param_space(self):
#         """ convert Olympus parameter space to pymoo 
#         """
#         pymoo_space = {}
#         for param in self.param_space:
#             if param.type == 'continuous':
#                 pymoo_space[param.name] = Real(bounds=(param.low,param.high))
#             elif param.type == 'discrete': 
#                 # TODO: need to map the discrete params to an integer
#                 quit()
#             elif param.type == 'categorical':
#                 quit()
        
#         return pymoo_space
    
#     def _set_pymoo_problem(self):
#         return PymooProblemWrapper()
    
#     def gen_initial_population(self):
#         return None
    
#     def _optimize(self):
#         return None
    
    




# class PymooProblemWrapper(Problem):
#     """ Wraps pymoo problem object with abstract method _evaluate which 
#     """

#     def __init__(self,
#                  params_obj: Parameters,
#                  acqf: AcquisitionFunction,
#                  known_constraints: Union[Callable, List[Callable]],
#     ): 
#         self.params_obj = params_obj
#         self.acqf = acqf
#         self.known_constraints = known_constraints


#     def _evaluate(self, params):

#         # acqf evaluation

#         # known constraints evaluation 
#         # <= 0 are mapped to feasible points 
#         # > 0 are mapped to infeasible points


#         return None
    


