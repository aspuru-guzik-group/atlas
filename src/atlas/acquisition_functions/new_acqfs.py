#!/usr/bin/env python

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from abc import abstractmethod

import itertools
from copy import deepcopy

import gpytorch
import numpy as np
import torch

from atlas import Logger
from atlas.objects.abstract_object import Object, ABCMeta, abstract_attribute
from atlas.utils.planner_utils import (
    cat_param_to_feat,
    forward_normalize,
    propose_randomly,
)


class Acqusition(Object, metaclass=ABCMeta):
    """ Base class for Atlas acqusition function """ 
    def __init__(self, reg_model, **acqf_args):
        self.reg_model = reg_model

    # @property
    # @abstract_attribute
    # def dummy(self):
    #     pass

    @abstractmethod
    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """ evaluate the acquisition function
        """
        pass

    def __call__(self, X: torch.Tensor):
        return self.evaluate(X)


class FeasibilityAwareAcquisition(Object, metaclass=ABCMeta):
    """ Base class for feasibility aware Atlas acquisition function
    for use with unknown constraints
    """
    def __init__(self, reg_model, cla_model, **acqf_args: Dict) -> None: 
        Object.__init__(self, reg_model, cla_model, **acqf_args)

    # @property
    # @abstract_attribute
    # def dummy(self):
    #     pass

    @abstractmethod
    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        """ evaluate the acquisition function
        """
        pass

    def __call__(self, X: torch.Tensor):
        acqf_val = self.evaluate(X)
        return self.compute_combined_acqf(acqf_val, X)


    def compute_feas_post(self, X: torch.Tensor):
        """computes the posterior P(infeasible|X)
        Args:
            X (torch.tensor): input tensor with shape (num_samples, q_batch_size, num_dims)
        """
        with gpytorch.settings.cholesky_jitter(1e-1):
            return self.cla_likelihood(
                self.cla_model(X.float().squeeze(1))
            ).mean

    def compute_combined_acqf(self, acqf_val: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """compute the combined acqusition function"""

        # approximately normalize the UCB acquisition function
        acqf_val = (acqf_val - self.acqf_min_max[0]) / (
            self.acqf_min_max[1] - self.acqf_min_max[0]
        )

        if self.use_reg_only:
            return acqf_val
        else:
            # p_feas should be 1 - P(infeasible|X) because EI is
            # maximized by default
            if not "naive-" in self.feas_strategy:
                p_feas = 1.0 - self.compute_feas_post(X)

            else:
                p_feas = 1.0

            if self.feas_strategy == "fwa":
                return acqf_val * self.p_feas_postprocess(p_feas)
            elif self.feas_strategy == "fca":
                return acqf_val
            elif self.feas_strategy == "fia":
                return (
                    (1.0 - self.infeas_ratio**self.feas_param) * acqf_val
                ) + (
                    (self.infeas_ratio**self.feas_param)
                    * (self.p_feas_postprocess(p_feas))
                )
            elif "naive-" in self.feas_strategy:
                if self.use_p_feas_only:
                    # we do not filter in this case
                    return p_feas
                else:
                    return acqf_val
            else:
                raise NotImplementedError


    # helper methods
    # def get_mean_sigma(posterior):
    #     return None
    




#--------------------------------
# ACQUISITION FUNCTION INSTANCES
#--------------------------------


class LCB(FeasibilityAwareAcquisition):
    """ Feasibility-aware lower confidence bound acquisition function 
    """
    def __init__(self, reg_model, cla_model, **acqf_args):
        super().__init__(reg_model, cla_model, **acqf_args)

    def evaluate(self, X: torch.Tensor):
        posterior = self.reg_model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        mean = posterior.mean.squeeze(-2).squeeze(-1)
        sigma = posterior.variance.clamp_min(1e-12).sqrt().view(mean.shape)
        acqf_val = (mean if self.maximize else -mean) - self.beta.sqrt() * sigma
        return acqf_val




