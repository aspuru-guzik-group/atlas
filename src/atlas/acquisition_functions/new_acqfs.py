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


# class Acqusition(Object, metaclass=ABCMeta):
#     """ Base class for Atlas acqusition function """ 
#     def __init__(self, reg_model, **acqf_args):
#         self.reg_model = reg_model

#     # @property
#     # @abstract_attribute
#     # def dummy(self):
#     #     pass

#     @abstractmethod
#     def evaluate(self, X: torch.Tensor) -> torch.Tensor:
#         """ evaluate the acquisition function
#         """
#         pass

#     def __call__(self, X: torch.Tensor):
#         return self.evaluate(X)


class FeasibilityAwareAcquisition(Object, metaclass=ABCMeta):
    """ Base class for feasibility aware Atlas acquisition function
    for use with unknown constraints
    """
    def __init__(self, reg_model, cla_model=None, fix_min_max=False, **acqf_args: Dict) -> None: 
        Object.__init__(self, reg_model, cla_model, **acqf_args)
        self.reg_model = reg_model
        self.cla_model = cla_model
        
        # estimate min max of acquisition function
        if not fix_min_max:
            self.acqf_min_max = self._estimate_acqf_min_max()
        else:
            self.acqf_min_max = 0., 1.


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
        
    def _evaluate_raw(self, X: torch.Tensor):
        return self.evaluate(X)


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

    def _estimate_acqf_min_max(self, num_samples:int=3000) -> Tuple[int, int]:
        """computes the min and max value of the acquisition function without
        the feasibility contribution. These values will be used to approximately
        normalize the acquisition function
        """

        samples, _ = propose_randomly(
            num_samples, self.params_obj.param_space, self.params_obj.has_descriptors,
        )
        if (
            self.problem_type == "fully_categorical"
            and not self.has_descriptors
        ):
            # we dont scale the parameters if we have a fully one-hot-encoded representation
            pass
        else:
            # scale the parameters
            samples = forward_normalize(
                samples, self.params_obj._mins_x, self.params_obj._maxs_x
            )

        acqf_val = self._evaluate_raw(
            torch.tensor(samples)
            .view(samples.shape[0], 1, samples.shape[-1])
            .double()
        )

        min_ = torch.amin(acqf_val).item()
        max_ = torch.amax(acqf_val).item()

        if np.abs(max_ - min_) < 1e-6:
            max_ = 1.0
            min_ = 0.0

        return min_, max_



    




#--------------------------------
# ACQUISITION FUNCTION INSTANCES
#--------------------------------

class VarianceBased(FeasibilityAwareAcquisition):
    """ Feasibility-aware variance-based utility function 
    """
    def __init__(self, reg_model, cla_model, **acqf_args):
        super().__init__(reg_model, cla_model, **acqf_args)
        self.reg_model = reg_model

    def evaluate(self, X: torch.Tensor):
        posterior = self.reg_model.posterior(X=X)
        mean = posterior.mean.squeeze(-2).squeeze(-1)
        acqf_val = posterior.variance.clamp_min(1e-12).sqrt().view(mean.shape)
        return acqf_val


class LCB(FeasibilityAwareAcquisition):
    """ Feasibility-aware lower confidence bound acquisition function 
    """
    def __init__(self, reg_model, cla_model, **acqf_args):
        super().__init__(reg_model, cla_model, **acqf_args)
        self.reg_model = reg_model

    def evaluate(self, X: torch.Tensor):
        posterior = self.reg_model.posterior(X=X)
        mean = posterior.mean.squeeze(-2).squeeze(-1)
        sigma = posterior.variance.clamp_min(1e-12).sqrt().view(mean.shape)
        acqf_val = mean - self.beta.sqrt()*sigma
        return acqf_val


class UCB(FeasibilityAwareAcquisition):
    """ Feasibility-aware upper confidence bound acquisition function 
    """
    def __init__(self, reg_model, cla_model, **acqf_args):
        super().__init__(reg_model, cla_model, **acqf_args)
        self.reg_model = reg_model

    def evaluate(self, X: torch.Tensor):
        posterior = self.reg_model.posterior(X=X)
        mean = posterior.mean.squeeze(-2).squeeze(-1)
        sigma = posterior.variance.clamp_min(1e-12).sqrt().view(mean.shape)
        acqf_val = mean + self.beta.sqrt()*sigma
        return acqf_val

class PI(FeasibilityAwareAcquisition):
    def __init__(self, reg_model, cla_model, **acqf_args):
        super().__init__(reg_model, cla_model, **acqf_args)
        self.reg_model = reg_model

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        posterior = self.reg_model.posterior(X=X)
        mean = posterior.mean.squeeze(-2).squeeze(-1)
        sigma = posterior.variance.clamp_min(1e-12).sqrt().view(mean.shape)
        u = - ( (mean - self.f_best_scaled.expand_as(mean)) / sigma )

        # TODO: complete this method
        return None


class EI(FeasibilityAwareAcquisition):
    def __init__(self, reg_model, cla_model, **acqf_args):
        super().__init__(reg_model, cla_model, **acqf_args)
        self.reg_model = reg_model

    def evaluate(self, X: torch.Tensor):
        posterior = self.reg_model.posterior(X=X)
        mean = posterior.mean.squeeze(-2).squeeze(-1)
        sigma = posterior.variance.clamp_min(1e-12).sqrt().view(mean.shape)
        u = - ( (mean - self.f_best_scaled.expand_as(mean)) / sigma )
        normal = torch.distributions.Normal(
            torch.zeros_like(u), torch.ones_like(u)
        )
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        acqf_val = sigma * (updf + u * ucdf)

        return acqf_val



class General(FeasibilityAwareAcquisition):
    """ Acqusition funciton for general parameters
    """
    def __init__(self, reg_model, cla_model, f_best_scaled, **acqf_args):
        super().__init__(reg_model, cla_model, fix_min_max=True, **acqf_args)
        # self.base_acqf = EI(reg_model, cla_model **acqf_args) # base acqf
        self.reg_model = reg_model
        self.f_best_scaled = f_best_scaled

        # deal with general parameter stuff
        self.X_sns_empty, _ = self.generate_X_sns()
        self.functional_dims = np.logical_not(self.params_obj.exp_general_mask)

        

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        X = X.double()
        #TODO: messy clean this up
        if X.shape[-1] == self.X_sns_empty.shape[-1]:
            X = X[:, :, self.functional_dims]
        self.f_best_scaled = self.f_best_scaled.to(X)
        # shape (# samples, # exp general dims, # batch size, # exp param dims)
        X_sns = torch.empty((X.shape[0],) + self.X_sns_empty.shape).double()

        for x_ix in range(X.shape[0]):
            X_sn = torch.clone(self.X_sns_empty)
            #X_sn[:, :, self.functional_dims] = X[x_ix, :, self.functional_dims]
            X_sn[:, :, self.functional_dims] = X[x_ix, :]
            X_sns[x_ix, :, :, :] = X_sn

        pred_mu_x, pred_sigma_x = [], []

        for X_sn in X_sns:
            posterior = self.reg_model.posterior(X_sn.double())
            mu = posterior.mean
            view_shape = mu.shape[:-2] if mu.shape[-2] == 1 else mu.shape[:-1]
            mu = mu.view(view_shape)
            sigma = posterior.variance.clamp_min(1e-9).sqrt().view(view_shape)
            pred_mu_x.append(mu)
            pred_sigma_x.append(sigma)

        pred_mu_x = torch.stack(pred_mu_x)
        pred_sigma_x = torch.stack(pred_sigma_x)
        mu_x = torch.mean(pred_mu_x, 0)
        sigma_x = torch.mean(pred_sigma_x, 0)

        u = - (mu_x - self.f_best_scaled.expand_as(mu_x)) / sigma_x
        normal = torch.distributions.Normal(
            torch.zeros_like(u), torch.ones_like(u)
        )
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        acqf_val = sigma * (updf + u * ucdf)

        return acqf_val


    def generate_X_sns(self):
        # generate Cartesian product space of the general parameter options
        param_options = []
        for ix in self.params_obj.general_dims:
            param_options.append(self.params_obj.param_space[ix].options)

        cart_product = list(itertools.product(*param_options))
        cart_product = [list(elem) for elem in cart_product]

        X_sns_empty = torch.empty(
            size=(len(cart_product), self.params_obj.expanded_dims)
        ).double()
        general_expanded = []
        general_raw = []
        for elem in cart_product:
            # convert to ohe and add to currently available options
            ohe, raw = [], []
            for val, obj in zip(elem, self.params_obj.param_space):
                if obj.type == "categorical":
                    ohe.append(
                        cat_param_to_feat(
                            obj, val, self.params_obj.has_descriptors
                        )
                    )
                    raw.append(val)
                else:
                    ohe.append([val])
            general_expanded.append(np.concatenate(ohe))
            general_raw.append(raw)

        general_expanded = torch.tensor(np.array(general_expanded))

        X_sns_empty[:, self.params_obj.exp_general_mask] = general_expanded
        X_sns_empty = forward_normalize(
            X_sns_empty,
            self.params_obj._mins_x,
            self.params_obj._maxs_x,
        )
        # TODO: careful of the batch size, will need to change this
        X_sns_empty = torch.unsqueeze(X_sns_empty, 1)

        return X_sns_empty, general_raw



def get_acqf_instance(acquisition_type, reg_model, cla_model, acqf_args:Dict[str,Any]):
    """ Convenience function to get acquisition function instance
    """
    if acquisition_type in ['ei','pi']:
        module = __import__(f'atlas.acquisition_functions.new_acqfs', fromlist=[acquisition_type.upper()])
        _class = getattr(module, acquisition_type.upper())
        return _class(reg_model=reg_model,cla_model=cla_model,**acqf_args)
    elif acquisition_type == 'general':
        return General(reg_model=reg_model,cla_model=cla_model,**acqf_args)
    elif acquisition_type == 'variance':
        return VarianceBased(reg_model=reg_model,cla_model=cla_model,**acqf_args)
    elif acquisition_type in ['lcb', 'ucb']:
        acqf_args['beta'] = torch.tensor([0.2]).repeat(acqf_args['batch_size']) # default value of beta
        module = __import__(f'atlas.acquisition_functions.new_acqfs', fromlist=[acquisition_type.upper()])
        _class = getattr(module, acquisition_type.upper())
        return _class(reg_model=reg_model,cla_model=cla_model,**acqf_args)
    else:
        msg = f"Acquisition function type {acquisition_type} not understood!"
        Logger.log(msg, "FATAL")
        
        

        

