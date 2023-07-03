#!/usr/bin/env python

# from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# import torch

# from botorch.acquisition import AcquisitionFunction



# class AdamOptimizer:
#     """ Adam optimizer as reported in https://arxiv.org/abs/1412.6980
#     """

#     def  __init__(
#         self, 
#         acqf: AcquisitionFunction,
#         select_params:List[int] = None,
#         eta: float = None, 
#         beta_1: float = None,
#         beta_2: float = None,
#         epsilon: float = None,
#         decay: bool = False,
#     ) -> None: 

#         self.acqf = acqf
#         self.select_params = select_params
#         self.eta = eta 
#         self.beta_1 = beta_1 
#         self.beta_2 = beta_2
#         self.epsilon = epsilon
#         self.decay = decay
#         self.iterations = 0


#         def _init_params(self, select_params: List[int]) -> None:
#             self.select_bool = torch.tensor(select)
#             self.num_dims = len(self.select_bool)
#             self.select_idx = torch.arange(self.num_dims)[self.select_bool]
#             self.ms = torch.zeros(self.num_dims)  # moment vector (length is size of input vector, i.e. opt domain)
#             self.vs = torch.zeros(self.num_dims)  # exponentially weighted infinity norm


#         def reset(self) -> None:
#             self.iterations = 0
#             self.ms = torch.zeros(self.num_dims)
#             self.vs = torch.zeros(self.num_dims)


#         def set_acqf(self, acqf, select=None):
#             """
#             """
#             self.func = func
#             self.reset()
#             if select is not None:
#                 self.init_params(select)