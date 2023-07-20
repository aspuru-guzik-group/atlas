#!/usr/bin/env python

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from abc import abstractmethod

import itertools
import math
from functools import partial

import gpytorch
import numpy as np
import torch
from botorch.acquisition.acquisition import MCSamplerMixin
from botorch.acquisition.objective import IdentityMCObjective
from botorch.utils.transforms import _verify_output_shape

from atlas import Logger, tkwargs
from atlas.objects.abstract_object import Object, ABCMeta
from atlas.acquisition_functions.acqf_utils import (
    concatenate_pending_params,
    t_batch_mode_transform,
    match_batch_shape
)
from atlas.utils.planner_utils import (
    cat_param_to_feat,
    forward_normalize,
    propose_randomly
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
    def __init__(self, reg_model, cla_model, cla_likelihood, fix_min_max=False, **acqf_args: Dict) -> None: 
        Object.__init__(self, reg_model, cla_model, **acqf_args)
        self.reg_model = reg_model
        self.cla_model = cla_model
        self.cla_likelihood = cla_likelihood

        self.use_min_filter = acqf_args['use_min_filter']
    
        
        # estimate min max of acquisition function
        if not fix_min_max:
            self.set_p_feas_postprocess()
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


    def _p_feas_filter(self, p_feas, filter_val: float = 0.5):
        return torch.minimum(p_feas, torch.ones_like(p_feas) * filter_val)

    def _p_feas_nofilter(self, p_feas):
        return p_feas

    def set_p_feas_postprocess(self):
        if self.use_min_filter:
            self.p_feas_postprocess = self._p_feas_filter
        else:
            self.p_feas_postprocess = self._p_feas_nofilter

    def forward_unconstrained(self, X):
        """evaluates the acquisition function without the
        feasibility portion, i.e. $\alpha(x)$ in the paper
        """
        acqf = super().forward(X)
        return acqf


    # helper methods
    def compute_mean_sigma(self, posterior) -> Tuple[torch.Tensor]:
        """ Takes in posterior of fitted model and returns 
        the mean and covariance """

        mean = posterior.mean.squeeze(-2).squeeze(-1)
        sigma = posterior.variance.clamp_min(1e-12).sqrt().view(mean.shape)

        return mean, sigma

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
            and not self.params_obj.has_descriptors
        ):
            # we dont scale the parameters if we have a fully one-hot-encoded representation
            pass
        else:
            # scale the parameters
            samples = forward_normalize(
                samples, self.params_obj._mins_x, self.params_obj._maxs_x
            )

        acqf_val = self._evaluate_raw(
            torch.tensor(samples, **tkwargs)
            .view(samples.shape[0], 1, samples.shape[-1])
        )

        min_ = torch.amin(acqf_val).item()
        max_ = torch.amax(acqf_val).item()

        if np.abs(max_ - min_) < 1e-6:
            max_ = 1.0
            min_ = 0.0

        return min_, max_
    


class MonteCarloAcquisition(FeasibilityAwareAcquisition, MCSamplerMixin):

    def __init__(self, reg_model, cla_model, cla_likelihood, **acqf_args: Dict[str,Any]) -> None: 
        # TODO: eventually set fix_min_max to False
        super().__init__(reg_model, cla_model, cla_likelihood, fix_min_max=True, **acqf_args)
        MCSamplerMixin.__init__(self, sampler=None) # instantiated by get_sampler()
        self.reg_model = reg_model
        self.cla_model = cla_model
        self.objective = IdentityMCObjective() # default

        # sample shape is property of MCSamplerMixin
        sample_shape = torch.Size([512]) # hardcoded default for MCSamplerMixin
        sample_red_dim = tuple(range(len(sample_shape))) 
        
        self._sample_red_op = partial(torch.mean, dim=sample_red_dim)
        self._batch_red_op = partial(torch.amax, dim=-1) # max reduction over last dimension

        # initially there are no pending_params by convention
        self.pending_params = self.set_pending_params(pending_params=None)


    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        return super(MonteCarloAcquisition, self).evaluate(X)
    
    @abstractmethod
    def _sample(self, X: torch.Tensor) -> torch.Tensor:
        """ ...
        """
        pass

    @concatenate_pending_params
    #@t_batch_mode_transform()
    def __call__(self, X: torch.Tensor) -> torch.Tensor:
        """ Full forward pass of MC acquisition function - should overwrite
        __call__ method of FeasibilityAwareAcquisition superclass
        """
        samples, obj = self.get_samples_obj(X)
        acqf_val_raw = self._sample(obj) # (sample_shape x batch_shape x q)
        acqf_val_red = self._sample_red_op(self._batch_red_op(acqf_val_raw))

        return acqf_val_red


    def get_samples_obj(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ ...
        """
        # this effectively does the conditioning on pending points?? 
        posterior = self.reg_model.posterior(X=X)
        samples = self.get_posterior_samples(posterior)
        obj = self.objective(samples=samples, X=X)

        return samples, obj


    def set_pending_params(self, pending_params: Optional[torch.Tensor]=None) -> None:
        """ Sets pending params attribute to inform the MC acquisition function
        about parameters that have been committed to but not yet measured

        Args:
            pendings_params (torch.Tensor): `n x d` tensor of pending parameter 
                values
        """
        if pending_params is not None:
            if pending_params.requires_grad:
                Logger.log('No gradients provided by acqf for pending params', 'WARNING')
            self.pending_params = pending_params.detach().clone()
        else:
            self.pending_params = pending_params
    
    
    
    # METHODS
    #  _sample: evaluate the monte carlo acquisition at the points X
    #   takes in `batch_shape x q x d` tensor 
    #   batch_shape = 
    #   q = 
    #   d = param dimension
    #   returns the reduced acquisition values, a tensor of batch_shape, 
    #   where batch_shape is the broadcasted batch shape of model and 
    #   input `X`

    # set_pending_params: set the parameter sets that have been committed to 
    #   but not yet evaluated

    # reduce_samples (sample_reduction_protocol): reduces the samples along some axis with
    #   one of three operations (I think this is either mean, min or max...), in Atlas 
    #   just choose one and stick with in --> needs to have sample_reduction and q_reduction 
    #   arguments, which are the dimensions to be reduced along in each case...
    
    # add_pending_params (concatenate_pending_points - decorator): concatenates on the 
    #   pending_params to the the set X? Why do we need this exactly?? 

    # get_samples_objs: computes posterior samples and objective(acqusiition function?)
    #   values at input locations X takes in `batch_shape x q x d` tensor
    #   returns a two tuple - first is tensor of posterior samples with shape
    #   `sample_shape x batch_shape x q x m`, and obj are MC objective values with shape
    #   `sample_shape x batch_shape x q`

    # get_posterior_samples: this is a method of BoTorch's MCSamplerMixin class - use this
    #   off the shelf probably


#--------------------------------
# ACQUISITION FUNCTION INSTANCES
#--------------------------------

class VarianceBased(FeasibilityAwareAcquisition):
    """ Feasibility-aware variance-based utility function 
    """
    def __init__(self, reg_model, cla_model, cla_likelihood, fix_min_max=False,**acqf_args):
        super().__init__(reg_model, cla_model, cla_likelihood, fix_min_max, **acqf_args)
        self.reg_model = reg_model

    def evaluate(self, X: torch.Tensor):
        posterior = self.reg_model.posterior(X=X)
        mean, sigma = self.compute_mean_sigma(posterior)
        return sigma


class LCB(FeasibilityAwareAcquisition):
    """ Feasibility-aware lower confidence bound acquisition function 
    """
    def __init__(self, reg_model, cla_model, cla_likelihood=None, **acqf_args):
        super().__init__(reg_model, cla_model, cla_likelihood, **acqf_args)
        self.reg_model = reg_model

    def evaluate(self, X: torch.Tensor):
        posterior = self.reg_model.posterior(X=X)
        mean, sigma = self.compute_mean_sigma(posterior)
        acqf_val = mean - self.beta.sqrt()*sigma
        return acqf_val


class UCB(FeasibilityAwareAcquisition):
    """ Feasibility-aware upper confidence bound acquisition function 
    """
    def __init__(self, reg_model, cla_model, cla_likelihood, **acqf_args):
        super().__init__(reg_model, cla_model, cla_likelihood, **acqf_args)
        self.reg_model = reg_model

    def evaluate(self, X: torch.Tensor):
        posterior = self.reg_model.posterior(X=X)
        mean, sigma = self.compute_mean_sigma(posterior)
        acqf_val = mean + self.beta.sqrt()*sigma
        return acqf_val


class PI(FeasibilityAwareAcquisition):
    def __init__(self, reg_model, cla_model, cla_likelihood, **acqf_args):
        super().__init__(reg_model, cla_model, cla_likelihood, **acqf_args)
        self.reg_model = reg_model

    def evaluate(self, X: torch.Tensor) -> torch.Tensor:
        posterior = self.reg_model.posterior(X=X)
        mean, sigma = self.compute_mean_sigma(posterior)
        u = - ( (mean - self.f_best_scaled.expand_as(mean)) / sigma )

        # TODO: complete this method
        return None


class EI(FeasibilityAwareAcquisition):
    def __init__(self, reg_model, cla_model, cla_likelihood, **acqf_args):
        super().__init__(reg_model, cla_model, cla_likelihood, **acqf_args)
        self.reg_model = reg_model

    def evaluate(self, X: torch.Tensor):
        posterior = self.reg_model.posterior(X=X)
        mean, sigma = self.compute_mean_sigma(posterior)
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
    def __init__(self, reg_model, cla_model, cla_likelihood, **acqf_args):
        super().__init__(reg_model, cla_model, cla_likelihood, fix_min_max=True, **acqf_args)
        # self.base_acqf = EI(reg_model, cla_model **acqf_args) # base acqf
        self.reg_model = reg_model
        self.f_best_scaled = acqf_args['f_best_scaled']

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



class qUCB(MonteCarloAcquisition):
    """ q-Upper Confidence Bound
    """

    def __init__(self, reg_model, cla_model, cla_likelihood, **acqf_args):
        super().__init__( reg_model, cla_model, cla_likelihood, **acqf_args)
        self.reg_model = reg_model
        self.beta_prime = torch.sqrt(acqf_args['beta'] * torch.Tensor([3.141])/ 2)

    def _sample(self, X: torch.Tensor) -> torch.Tensor:
        """ evaluate per sample Monte Carlo acquisition function on 
        set of candidates X

        Args:
            X (torch.Tensor): `sample_shape x batch_shape x q` Tensor of 
                Monte Carlo objective values
        """
        
        mean = X.mean(dim=0)
        return mean + self.beta_prime * (X - mean).abs()



class qLCB(MonteCarloAcquisition):
    """ q-Lower Confidence Bound
    """
    def __init__(self, reg_model, cla_model, cla_likelihood=None, **acqf_args):
        pass


def get_acqf_instance(acquisition_type, reg_model, cla_model, cla_likelihood, acqf_args:Dict[str,Any]):
    """ Convenience function to get acquisition function instance
    """
    # determine if we should use q-acqf
    batch_size = acqf_args['batch_size']
    acquisition_optimizer_kind = acqf_args['acquisition_optimizer_kind']
    use_q_acqf = (batch_size > 1 and acquisition_optimizer_kind == 'gradient')


    if acquisition_type in ['ei','pi']:
        module = __import__(f'atlas.acquisition_functions.acqfs', fromlist=[acquisition_type.upper()])
        _class = getattr(module, acquisition_type.upper())
        return _class(reg_model=reg_model,cla_model=cla_model, cla_likelihood=cla_likelihood,**acqf_args)
    elif acquisition_type == 'general':
        return General(reg_model=reg_model,cla_model=cla_model,cla_likelihood=cla_likelihood,**acqf_args)
    elif acquisition_type == 'variance':
        return VarianceBased(reg_model=reg_model,cla_model=cla_model,cla_likelihood=cla_likelihood,**acqf_args)
    elif acquisition_type in ['lcb', 'ucb']:
        if not use_q_acqf:
            acqf_args['beta'] = torch.tensor([0.2], **tkwargs) # default value of beta
            module = __import__(f'atlas.acquisition_functions.acqfs', fromlist=[acquisition_type.upper()])
            _class = getattr(module, acquisition_type.upper())
        else:
            acqf_args['beta'] = torch.tensor([0.2], **tkwargs).repeat(acqf_args['batch_size']) # default value of beta
            module = __import__(f'atlas.acquisition_functions.acqfs', fromlist=[
                ''.join(['q', acquisition_type.upper()])
            ])
            _class = getattr(module, ''.join(['q', acquisition_type.upper()]))
        return _class(reg_model=reg_model,cla_model=cla_model,cla_likelihood=cla_likelihood,**acqf_args)
    else:
        msg = f"Acquisition function type {acquisition_type} not understood!"
        Logger.log(msg, "FATAL")
        
        
if __name__ == '__main__':
    pass
