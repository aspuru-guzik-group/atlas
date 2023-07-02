#!/usr/bin/env python

import itertools
from copy import deepcopy

import gpytorch
import numpy as np
import pandas as pd
import torch
from botorch.acquisition import (
    AcquisitionFunction,
    AnalyticAcquisitionFunction,
    ExpectedImprovement,
    UpperConfidenceBound,
)
from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
)
from botorch.acquisition.multi_objective.monte_carlo import (
    qNoisyExpectedHypervolumeImprovement,
)
from botorch.acquisition.multi_objective.objective import (
    IdentityMCMultiOutputObjective,
)

from atlas import Logger
from atlas.utils.planner_utils import (
    cat_param_to_feat,
    forward_normalize,
    propose_randomly,
)


class FeasibilityAwareAcquisition:
    def compute_feas_post(self, X: torch.Tensor):
        """computes the posterior P(infeasible|X)
        Args:
            X (torch.tensor): input tensor with shape (num_samples, q_batch_size, num_dims)
        """
        with gpytorch.settings.cholesky_jitter(1e-1):
            return self.cla_likelihood(
                self.cla_model(X.float().squeeze(1))
            ).mean

    def compute_combined_acqf(self, acqf, X):
        """compute the combined acqusition function"""

        # approximately normalize the UCB acquisition function
        acqf = (acqf - self.acqf_min_max[0]) / (
            self.acqf_min_max[1] - self.acqf_min_max[0]
        )

        if self.use_reg_only:
            return acqf
        else:
            # p_feas should be 1 - P(infeasible|X) because EI is
            # maximized by default
            if not "naive-" in self.feas_strategy:
                p_feas = 1.0 - self.compute_feas_post(X)

            else:
                p_feas = 1.0

            if self.feas_strategy == "fwa":
                return acqf * self.p_feas_postprocess(p_feas)
            elif self.feas_strategy == "fca":
                return acqf
            elif self.feas_strategy == "fia":
                return (
                    (1.0 - self.infeas_ratio**self.feas_param) * acqf
                ) + (
                    (self.infeas_ratio**self.feas_param)
                    * (self.p_feas_postprocess(p_feas))
                )
            elif "naive-" in self.feas_strategy:
                if self.use_p_feas_only:
                    # we do not filter in this case
                    return p_feas
                else:
                    return acqf
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


class VarianceBased(AcquisitionFunction):

    """Variance-based sampling (active learning)"""

    def __init__(self, reg_model, **kwargs):
        super().__init__(reg_model, **kwargs)
        self.reg_model = reg_model

    def forward(self, X):
        posterior = self.reg_model.posterior(X.double())
        sigma = posterior.variance.clamp_min(1e-9).sqrt()
        view_shape = (
            sigma.shape[:-2] if sigma.shape[-2] == 1 else sigma.shape[:-1]
        )

        return sigma.view(view_shape)


class LowerConfidenceBound(AnalyticAcquisitionFunction):
    def __init__(
        self, model, beta, maximize=False, posterior_transform=None, **kwargs
    ) -> None:
        super().__init__(
            model=model, posterior_transform=posterior_transform, **kwargs
        )
        self.reg_model = model
        self.posterior_transform = posterior_transform
        self.maximize = maximize
        self.beta = beta

    def forward(self, X):
        # mean, sigma = self._mean_and_sigma(X)
        posterior = self.reg_model.posterior(
            X=X, posterior_transform=self.posterior_transform
        )
        mean = posterior.mean.squeeze(-2).squeeze(-1)
        sigma = posterior.variance.clamp_min(1e-12).sqrt().view(mean.shape)
        acqf = (mean if self.maximize else -mean) - self.beta.sqrt() * sigma
        return acqf


class FeasibilityAwareVarainceBased(
    VarianceBased, FeasibilityAwareAcquisition
):
    """Feasibility aware variance-based sampling (active learning)"""

    def __init__(
        self,
        reg_model,
        cla_model,
        cla_likelihood,
        param_space,
        best_f,
        feas_strategy,
        feas_param,
        infeas_ratio,
        acqf_min_max,
        use_p_feas_only=False,
        use_reg_only=False,
        use_min_filter=True,
        objective=None,
        maximize=False,
        **kwargs,
    ) -> None:
        super().__init__(reg_model, **kwargs)
        self.best_f = best_f
        self.reg_model = reg_model
        self.cla_model = cla_model
        self.cla_likelihood = cla_likelihood
        self.param_space = param_space
        self.feas_strategy = feas_strategy
        self.feas_param = feas_param
        self.infeas_ratio = infeas_ratio
        self.acqf_min_max = acqf_min_max
        self.use_p_feas_only = use_p_feas_only
        self.maximize = maximize
        self.use_min_filter = use_min_filter
        self.use_reg_only = use_reg_only
        # set the p_feas postprocessing step
        self.set_p_feas_postprocess()

    def forward(self, X):
        acqf = super().forward(X)
        return self.compute_combined_acqf(acqf, X)


class FeasibilityAwareGeneral(
    AcquisitionFunction, FeasibilityAwareAcquisition
):
    """Abstract feasibilty aware general purpose optimization acquisition function."""

    def __init__(
        self,
        reg_model,
        cla_model,
        cla_likelihood,
        params_obj,
        # general_parameters,
        param_space,
        best_f,
        feas_strategy,
        feas_param,
        infeas_ratio,
        acqf_min_max,
        use_p_feas_only=False,
        use_reg_only=False,
        use_min_filter=True,
        objective=None,
        maximize=False,
        **kwargs,
    ) -> None:
        super().__init__(reg_model, **kwargs)
        self.best_f = best_f
        self.reg_model = reg_model
        self.cla_model = cla_model
        self.cla_likelihood = cla_likelihood
        self.params_obj = params_obj
        self.param_space = param_space
        self.feas_strategy = feas_strategy
        self.feas_param = feas_param
        self.infeas_ratio = infeas_ratio
        self.acqf_min_max = acqf_min_max
        self.use_p_feas_only = use_p_feas_only
        self.maximize = maximize
        self.use_min_filter = use_min_filter
        self.use_reg_only = use_reg_only

        # set the p_feas postprocessing step
        self.set_p_feas_postprocess()

        self.X_sns_empty, _ = self.generate_X_sns()
        self.functional_dims = np.logical_not(self.params_obj.exp_general_mask)
        

    def forward(self, X):

        X = X.double()
        best_f = self.best_f.to(X)

        # shape (# samples, # exp general dims, # batch size, # exp param dims)
        X_sns = torch.empty((X.shape[0],) + self.X_sns_empty.shape).double()

        for x_ix in range(X.shape[0]):
            X_sn = torch.clone(self.X_sns_empty)
            #X_sn[:, :, self.functional_dims] = X[x_ix, :] #X[x_ix, :, self.functional_dims]
            X_sn[:, :, self.functional_dims] = X[x_ix, :, self.functional_dims]
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

        u = (mu_x - best_f.expand_as(mu_x)) / sigma_x
        if not self.maximize:
            u = -u
        normal = torch.distributions.Normal(
            torch.zeros_like(u), torch.ones_like(u)
        )
        ucdf = normal.cdf(u)
        updf = torch.exp(normal.log_prob(u))
        ei = sigma * (updf + u * ucdf)

        return ei

    def generate_X_sns(self):
        # generate Cartesian product space of the general parameter options
        param_options = []
        for ix in self.params_obj.general_dims:
            param_options.append(self.param_space[ix].options)

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
            for val, obj in zip(elem, self.param_space):
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
        # forward normalize
        X_sns_empty = forward_normalize(
            X_sns_empty,
            self.params_obj._mins_x,
            self.params_obj._maxs_x,
        )
        # TODO: careful of the batch size, will need to change this
        X_sns_empty = torch.unsqueeze(X_sns_empty, 1)

        return X_sns_empty, general_raw


class FeasibilityAwareQEI(qExpectedImprovement, FeasibilityAwareAcquisition):
    """Abstract feasibility aware expected improvement acquisition function. Compatible
    with the FIA, FCA and FWA strategies, as well as any of the naive strategies.
    Args:
                    reg_model (gpytorch.models.ExactGP): gpytorch regression surrogate GP model
                    cla_model (gpytorch.models.ApproximateGP): gpytorch variational GP for fesibility surrogate
                    cla_likelihood (gpytorch.likelihoods.BernoulliLikelihood): gpytorch Bernoulli likelihood
                    best_f (torch.tensor): incumbent point
                    feas_strategy (str): feasibility acqusition function name
                    feas_param (float): feasibilty parameter (called "t" in the paper)
                    infeas_ratio (float): the quotient of number of infeasible points with total points
                    objective ():
                    maximize (bool): whether the problem is maximization
    """

    def __init__(
        self,
        reg_model,
        cla_model,
        cla_likelihood,
        param_space,
        best_f,
        feas_strategy,
        feas_param,
        infeas_ratio,
        acqf_min_max,
        use_p_feas_only=False,
        use_reg_only=False,
        use_min_filter=True,
        objective=None,
        maximize=False,
        **kwargs,
    ) -> None:
        super().__init__(
            reg_model, best_f, objective=objective, maximize=maximize, **kwargs
        )
        self.reg_model = reg_model
        self.cla_model = cla_model
        self.cla_likelihood = cla_likelihood
        self.param_space = param_space
        self.feas_strategy = feas_strategy
        self.feas_param = feas_param
        self.infeas_ratio = infeas_ratio
        self.acqf_min_max = acqf_min_max
        self.use_p_feas_only = use_p_feas_only
        self.use_min_filter = use_min_filter
        self.use_reg_only = use_reg_only
        # set the p_feas postprocessing step
        self.set_p_feas_postprocess()

    def forward(self, X):
        acqf = super().forward(X)
        return -self.compute_combined_acqf(acqf, X)


class FeasibilityAwareEI(ExpectedImprovement, FeasibilityAwareAcquisition):
    """Abstract feasibility aware expected improvement acquisition function. Compatible
    with the FIA, FCA and FWA strategies, as well as any of the naive strategies.
    Args:
                    reg_model (gpytorch.models.ExactGP): gpytorch regression surrogate GP model
                    cla_model (gpytorch.models.ApproximateGP): gpytorch variational GP for fesibility surrogate
                    cla_likelihood (gpytorch.likelihoods.BernoulliLikelihood): gpytorch Bernoulli likelihood
                    best_f (torch.tensor): incumbent point
                    feas_strategy (str): feasibility acqusition function name
                    feas_param (float): feasibilty parameter (called "t" in the paper)
                    infeas_ratio (float): the quotient of number of infeasible points with total points
                    objective ():
                    maximize (bool): whether the problem is maximization
    """

    def __init__(
        self,
        reg_model,
        cla_model,
        cla_likelihood,
        param_space,
        best_f,
        feas_strategy,
        feas_param,
        infeas_ratio,
        acqf_min_max,
        use_p_feas_only=False,
        use_reg_only=False,
        use_min_filter=True,
        objective=None,
        maximize=False,
        **kwargs,
    ) -> None:
        super().__init__(reg_model, best_f, objective, maximize, **kwargs)
        self.reg_model = reg_model
        self.cla_model = cla_model
        self.cla_likelihood = cla_likelihood
        self.param_space = param_space
        self.feas_strategy = feas_strategy
        self.feas_param = feas_param
        self.infeas_ratio = infeas_ratio
        self.acqf_min_max = acqf_min_max
        self.use_p_feas_only = use_p_feas_only
        self.use_min_filter = use_min_filter
        self.use_reg_only = use_reg_only
        # set the p_feas postprocessing step
        self.set_p_feas_postprocess()

    def forward(self, X):
        acqf = super().forward(X) # get the EI acquisition
        return self.compute_combined_acqf(acqf, X)


class FeasibilityAwareUCB(UpperConfidenceBound, FeasibilityAwareAcquisition):
    def __init__(
        self,
        reg_model,
        cla_model,
        cla_likelihood,
        param_space,
        best_f,
        feas_strategy,
        feas_param,
        infeas_ratio,
        acqf_min_max,
        use_p_feas_only=False,
        use_reg_only=False,
        beta=torch.tensor([0.2]),
        use_min_filter=True,
        objective=None,
        maximize=False,
        **kwargs,
    ) -> None:
        super().__init__(reg_model, beta, maximize=maximize, **kwargs)
        self.reg_model = reg_model
        self.cla_model = cla_model
        self.cla_likelihood = cla_likelihood
        self.param_space = param_space
        self.best_f = best_f
        self.feas_strategy = feas_strategy
        self.feas_param = feas_param
        self.feas_strategy = feas_strategy
        self.feas_param = feas_param
        self.infeas_ratio = infeas_ratio
        self.acqf_min_max = acqf_min_max
        self.use_p_feas_only = use_p_feas_only
        self.use_reg_only = use_reg_only
        self.beta = beta
        self.use_min_filter = use_min_filter
        self.objective = objective
        self.maximize = maximize
        # set the p_feas postprocessing step
        self.set_p_feas_postprocess()

    def forward(self, X):
        acqf = super().forward(X)
        return self.compute_combined_acqf(acqf, X)


class FeasibilityAwareLCB(LowerConfidenceBound, FeasibilityAwareAcquisition):
    def __init__(
        self,
        reg_model,
        cla_model,
        cla_likelihood,
        param_space,
        best_f,
        feas_strategy,
        feas_param,
        infeas_ratio,
        acqf_min_max,
        use_p_feas_only=False,
        use_reg_only=False,
        beta=torch.tensor([0.2]),
        use_min_filter=True,
        objective=None,
        maximize=False,
        **kwargs,
    ) -> None:
        super().__init__(
            model=reg_model, beta=beta, posterior_transform=None, **kwargs
        )
        self.reg_model = reg_model
        self.cla_model = cla_model
        self.cla_likelihood = cla_likelihood
        self.param_space = param_space
        self.best_f = best_f
        self.feas_strategy = feas_strategy
        self.feas_param = feas_param
        self.feas_strategy = feas_strategy
        self.feas_param = feas_param
        self.infeas_ratio = infeas_ratio
        self.acqf_min_max = acqf_min_max
        self.use_p_feas_only = use_p_feas_only
        self.use_reg_only = use_reg_only
        self.beta = beta
        self.use_min_filter = use_min_filter
        self.objective = objective
        self.maximize = maximize
        # set the p_feas postprocessing step
        self.set_p_feas_postprocess()

    def forward(self, X):
        acqf = super().forward(X)
        return self.compute_combined_acqf(acqf, X)


class FeasibilityAwareqNEHVI(
    qNoisyExpectedHypervolumeImprovement, FeasibilityAwareAcquisition
):
    def __init__(
        self,
        reg_model,
        cla_model,
        cla_likelihood,
        param_space,
        feas_strategy,
        feas_param,
        infeas_ratio,
        acqf_min_max,
        # qNEHVI-specific parameters ----
        ref_point,
        sampler,
        X_baseline,
        prune_baseline=False,
        # --------------------------------
        use_p_feas_only=False,
        use_reg_only=False,
        use_min_filter=True,
        objective=IdentityMCMultiOutputObjective(),
        **kwargs,
    ) -> None:
        super().__init__(
            model=reg_model,
            ref_point=ref_point,  # reference point --> make this the worst measured objective value in each dim (list)
            X_baseline=X_baseline,  # normalized x values (observed parameters)
            prune_baseline=prune_baseline,  # prune baseline points that have estimated zero probability
            sampler=sampler,  # instance of SobolQMCNormalSampler for continuous variables
            objective=objective,
            # **kwargs,
        )
        self.reg_model = reg_model
        self.cla_model = cla_model
        self.cla_likelihood = cla_likelihood
        self.param_space = param_space
        self.feas_strategy = feas_strategy
        self.feas_param = feas_param
        self.feas_strategy = feas_strategy
        self.feas_param = feas_param
        self.infeas_ratio = infeas_ratio
        self.acqf_min_max = acqf_min_max
        self.use_p_feas_only = use_p_feas_only
        self.use_reg_only = use_reg_only
        self.use_min_filter = use_min_filter
        self.objective = objective
        # set the p_feas postprocessing step
        self.set_p_feas_postprocess()

    def forward(self, X):
        acqf = -super().forward(X)  # why do we seem to need negative sign??
        return self.compute_combined_acqf(acqf, X)


class MedusaAcquisition():
    """ Acquisition function for Medsua

    Args: 
        reg_model: gpytorch model 
        params_obj: atlas Parameter object
        X_sns_empty (torch.Tensor): torch tensor of all general parameter reps with 
            empty values for functional parameters
        funcitonal_dims (np.ndarray): ... 
    """
    def __init__(
            self, 
            reg_model,
            params_obj,
            X_sns_empty,
            functional_dims,
            maximize=False,
            beta=torch.tensor(1.)
        ):
        self.reg_model = reg_model
        self.params_obj = params_obj
        self.param_space = self.params_obj.param_space
        self.X_sns_empty = X_sns_empty
        self.functional_dims = functional_dims
        self.maximize = maximize
        self.beta = beta


    def __call__(self, X_func, G):
        """ Evaluate the acquisition function

        Args:
            X_func (): functional parameter settings, this should be 
            a list with 'shape' (# samples, Ng, param_dim)

            G (): list of subsets of the non-functional parameters
            with shape (# samples, Ng, |Sg| )

        """
        # generate a tensor of all options
        all_options = []
        all_options_raw = []
        for X, S in zip(X_func, G):
            for si in S:
                all_options_raw.append([X, si])
                opt = deepcopy(self.X_sns_empty[si])
                opt[:,self.functional_dims] = torch.tensor(X)
                all_options.append(opt)
        
        all_options = torch.cat(all_options) # (num options, num dims)

        # make prediction with regression surrogate
        posterior = self.reg_model.posterior(all_options.double())
        mu = posterior.mean   # (num options, 1)
        sigma = posterior.variance.clamp_min(1e-9).sqrt() # (num options, 1)

        # sum the mean and sigmas
        mu_sum  = torch.sum(mu)  # scalar
        sigma_sum = torch.sum(sigma) # scalar

        # compute the UCB acquisition with these values
        return (mu_sum if self.maximize else -mu_sum) + self.beta.sqrt() * sigma_sum


    def run_mu_only(self, X_func, G):
        """ mean prediction with regression surorgate only
        """
        # generate a tensor of all options
        all_options = []
        all_options_raw = []
        for X, S in zip(X_func, G):
            for si in S:
                all_options_raw.append([X, si])
                opt = deepcopy(self.X_sns_empty[si])
                opt[:,self.functional_dims] = torch.tensor(X)
                all_options.append(opt)
        
        all_options = torch.cat(all_options) # (num options, num dims)

        # make prediction with regression surrogate
        posterior = self.reg_model.posterior(all_options.double())
        mu = posterior.mean   # (num options, 1)
        mu_sum = torch.sum(mu)

        return mu_sum if self.maximize else -mu_sum

    # def __call__(self, X_funcs, Gs):
    #     """ dummy acquisition evaluation """
    #     return np.random.uniform(size=None)

    # TODO: need to extend this to batched case
    def acqf_var(self, X_funcs_deindex, G, X_funcs_cat):
        """ variance-based sampling over all potential options
        """
        sigmas = []
        all_options_raw = []
        if X_funcs_cat is not None: 
            # have some categorical functional parameters
            all_options_cat = []
            for X, S, X_cat in zip(X_funcs_deindex, G, X_funcs_cat):
                for si in S:
                    # print('si :', si)
                    # print('X :', X)
                    # print('G :', G)
                    # print('X_cat :', X_cat)
    
                    all_options_raw.append([X, si])
                    all_options_cat.append([X_cat, si])
                    # produce the option
                    opt = deepcopy(self.X_sns_empty[si])
                    opt[:,self.functional_dims] = torch.tensor(X)

                    #print('opt :', opt)
                    
                    # make prediction with regression surrogate model
                    posterior = self.reg_model.posterior(opt.double())
                    sigma = posterior.variance.clamp_min(1e-9).sqrt()
                    sigmas.append(sigma.detach().numpy().item())

            # get the index of the largest sigma --> most uncertain
            select_idx = np.argmax(sigmas)
            select_option = all_options_cat[select_idx]

        else: 
            # continuous and/or discrete functional parameters

            for X, S in zip(X_funcs_deindex, G):
                for si in S:
                    all_options_raw.append([X, si])
                    # produce the option
                    opt = deepcopy(self.X_sns_empty[si])
                    opt[:,self.functional_dims] = torch.tensor(X)
                    # make prediction with regression surrogate model
                    posterior = self.reg_model.posterior(opt.double())
                    sigma = posterior.variance.clamp_min(1e-9).sqrt()
                    sigmas.append(sigma.detach().numpy().item())

            # get the index of the largest sigma --> most uncertain
            select_idx = np.argmax(sigmas)
            select_option = all_options_raw[select_idx]


    
        return select_option[0], select_option[1] # X_func, si



def get_batch_initial_conditions(
    num_restarts,
    batch_size,
    param_space,
    known_constraints,
    fca_constraint, 
    mins_x,
    maxs_x,
    has_descriptors,
    num_chances=15,
    return_raw=False,
):
    """generate batches of initial conditions for a
    random restart optimization subject to some constraints. This uses
    rejection sampling, and might get very inefficient for parameter spaces with
    a large infeasible fraction.
    Args:
                    num_restarts (int): number of optimization restarts
                    batch_size (int): number of samples to recommend per ask/tell call (fixed to 1)
                    param_space (obj): Olympus parameter space object for the given problem
                    known_constriants (list): list of callables which specifies the user-level constraint function(s)
                    fca_constraint (callable): fca constraint callable
                    mins_x (np.array): minimum values of each parameter space dimension
                    maxs_x (np.array): maximum values of each parameter
                    num_chances (int):
    Returns:
                    a torch.tensor with shape (num_restarts, batch_size, num_dims)
                    of initial optimization conditions
    """
    # take 20*num_restarts points randomly and evaluate the constraint function on all of
    # them, if we have enough, proceed, if not proceed to sequential rejection sampling
    num_raw_samples = 250 * num_restarts
    raw_samples, raw_proposals = propose_randomly(
        num_proposals=num_raw_samples * batch_size,
        param_space=param_space,
        has_descriptors=has_descriptors,
    )

    # forward normalize the randomly generated samples (expanded rep only)
    raw_samples = forward_normalize(raw_samples, mins_x, maxs_x)

    # reshape to botorch format
    raw_samples = torch.tensor(raw_samples).view(
        raw_samples.shape[0] // batch_size, batch_size, raw_samples.shape[1]
    )

    if known_constraints.is_empty and fca_constraint==[]:
        # no constraints, return samples
        batch_initial_conditions = raw_samples
        batch_initial_conditions_raw = raw_proposals

    else:
        #----------------
        # fca constraint
        #----------------
        if type(fca_constraint)==callable:
            # we have an fca constraint 
            # evaluate using expanded torch representation
            constraint_val = fca_constraint(raw_samples)
            if len(constraint_val.shape) == 1:
                constraint_val = constraint_val.view(
                    constraint_val.shape[0], 1
                )
            constraint_vals.append(constraint_val)

            fca_feas_idx = torch.where(torch.all(constraint_vals>=0, dim=1))[0]
        else:
            # no fca constraint
            fca_feas_idx = torch.arange(raw_samples.shape[0])

        #------------------------------
        # user-level known constraints
        #------------------------------
        if not known_constraints.is_empty:
            # we have some user-level known constraints
            # use raw_propsals here, user-level known constraints 
            # evaluated on compressed representation of parameters
            constraint_vals = []
            #loop through all known constriaint callables
            for constraint_callable in known_constraints:
                # returns True if feasible, False if infeasible
                kc_res = [constraint_callable(params) for params in raw_proposals] 
                constraint_vals.append(kc_res)

            constraint_vals = torch.tensor(constraint_vals)
    
            # get indices for which known constraints are satisfied
            kc_feas_idx = torch.where(torch.all(constraint_vals, dim=0))[0]
        else:
            # no user-level known constraints
            kc_feas_idx = torch.arange(raw_samples.shape[0])

        # find the union of the two sets of feasible indices
        feas_idx = np.intersect1d(kc_feas_idx, fca_feas_idx)



        # project onto original proposals
        batch_initial_conditions = raw_samples[feas_idx, :, :]
        batch_initial_conditions_raw = raw_proposals[feas_idx, :]

    if batch_initial_conditions.shape[0] >= num_restarts:
        # if we have more samples than we need, truncate
        if return_raw:
            return (
                batch_initial_conditions[:num_restarts, :, :],
                batch_initial_conditions_raw[:num_restarts, :],
            )
        return batch_initial_conditions[:num_restarts, :, :]
    
    elif 0 < batch_initial_conditions.shape[0] < num_restarts:
        # we dont have enough proposals, sample around the feasible ones we have...
        Logger.log(
            f"Insufficient initial samples, resorting to local sampling for {num_chances} iterations...",
            "WARNING",
        )
        for chance in range(num_chances):
            batch_initial_conditions = sample_around_x(
                batch_initial_conditions, constraint_callable
            )
            Logger.log(
                f"Chance : {chance+1}/{num_chances}\t# samples : {batch_initial_conditions.shape[0]}",
                "INFO",
            )

            if batch_initial_conditions.shape[0] >= num_restarts:
                if return_raw:
                    return (
                        batch_initial_conditions[:num_restarts, :, :],
                        batch_initial_conditions_raw[:num_restarts, :],
                    )
                return batch_initial_conditions[:num_restarts, :, :]
        return None, None
    else:
        Logger.log(
            f"Insufficient initial samples after {num_chances} sampling iterations. Resorting to unconstrained acquisition",
            "WARNING",
        )
        return None, None

    assert len(batch_initial_conditions.size()) == 3

    if return_raw:
        return batch_initial_conditions, batch_initial_conditions_raw
    return batch_initial_conditions


def sample_around_x(raw_samples, constraint_callable):
    """draw samples around points which we already know are feasible by adding
    some Gaussian noise to them
    """
    tiled_raw_samples = raw_samples.tile((20, 1, 1))
    means = deepcopy(tiled_raw_samples)
    stds = torch.ones_like(means) * 0.05
    perturb_samples = tiled_raw_samples + torch.normal(means, stds)
    # # project the values
    # perturb_samples = torch.where(perturb_samples>1., 1., perturb_samples)
    # perturb_samples = torch.where(perturb_samples<0., 0., perturb_samples)
    inputs = torch.cat((raw_samples, perturb_samples))

    constraint_vals = []
    for constraint in constraint_callable:
        constraint_val = constraint(inputs)
        if len(constraint_val.shape) == 1:
            constraint_val = constraint_val.view(constraint_val.shape[0], 1)
        constraint_vals.append(constraint_val)

    if len(constraint_vals) == 2:
        constraint_vals = torch.cat(constraint_vals, dim=1)
        feas_ix = torch.where(torch.all(constraint_vals >= 0))[0]
    elif len(constraint_vals) == 1:
        constraint_vals = torch.tensor(constraint_vals[0])
        feas_ix = torch.where(constraint_vals >= 0)[0]

    batch_initial_conditions = inputs[feas_ix, :, :]

    return batch_initial_conditions


def create_available_options(
    param_space,
    params,
    fca_constraint_callable,
    known_constraint_callables,
    normalize,
    mins_x,
    maxs_x,
    has_descriptors,
    max_options=int(1.5e10),
):
    """build cartesian product space of options, then remove options
    which have already been measured. Returns an (num_options, num_dims)
    torch tensor with all possible options

    If the parameter space is mixed, build and return the Cartesian product space of
    only the categorical and discrete parameters.

    Args:
                    param_space (obj): Olympus parameter space object
                    params (list): parameters from the current Campaign
                    fca_constraint_callable (Callable): FCA constraint callable
                    known_constraint_callables (List[Callable]): list of known constraints
                    mins_x (np.array): minimum values of each parameter space dimension
                    maxs_x (np.array): maximum values of each parameter
                    has_descriptors (bool): wether or not cat options have descriptors
                    max_options (int): max number of allowed options to return. If CP space
                        exceeds this value, return random subset
    """

    # make sure params are proper data type
    real_params = []
    for param in params:
        p = []
        for ix, elem in enumerate(param):
            if param_space[ix].type == "categorical":
                p.append(elem)
            else:
                p.append(float(elem))
        real_params.append(p)

    # params = [list(elem) for elem in params]
    # get the relevant parmeters -> only categorical and discrete
    relevant_params = [
        p for p in param_space if p.type in ["categorical", "discrete"]
    ]
    param_names = [p.name for p in relevant_params]
    param_options = [p.options for p in relevant_params]

    cart_product = list(itertools.product(*param_options))
    cart_product = [list(elem) for elem in cart_product]

    if len(cart_product) > max_options:
        Logger.log(
            f'CP space of cardnality {len(cart_product)} exceeds max allowed options. Proceeding with random subset..',
            'WARNING',
        )
        select_idx = np.random.choice(
            np.arange(len(cart_product)),
            size=(max_options,),
            replace=False,
            p=None,
        )
        cart_product = [cart_product[idx] for idx in select_idx]


    if len(param_names) == len(param_space):
        # no continuous parameters
        # remove options that we have measured already
        current_avail_feat = []
        current_avail_cat = []

        for elem in cart_product:
            if elem not in real_params:
                # convert to ohe and add to currently available options
                ohe = []
                for val, obj in zip(elem, param_space):
                    if obj.type == "categorical":
                        ohe.append(
                            cat_param_to_feat(obj, val, has_descriptors)
                        )
                    else:
                        ohe.append([val])
                current_avail_feat.append(np.concatenate(ohe))
                current_avail_cat.append(elem)

        current_avail_feat_unconst = np.array(current_avail_feat)
        current_avail_cat_unconst = np.array(current_avail_cat)

        # # check known constraints not associated with FCA (if any)
        if known_constraint_callables is not None:
            # known constraints
            kc_results = []
            for cat_unconst in current_avail_cat_unconst:
                if all([kc(cat_unconst) for kc in known_constraint_callables]):
                    # feasible
                    kc_results.append(True)
                else:
                    kc_results.append(False)
            feas_mask = np.where(kc_results)[0]
            current_avail_feat_kc = current_avail_feat_unconst[feas_mask]
            current_avail_cat_kc = current_avail_cat_unconst[feas_mask]
        else:
            current_avail_feat_kc = current_avail_feat_unconst
            current_avail_cat_kc = current_avail_cat_unconst

        # forward normalize the options before evaluating the fca constraint
        if normalize:
            current_avail_feat_kc = forward_normalize(
                current_avail_feat_kc, mins_x, maxs_x
            )

        current_avail_feat_kc = torch.tensor(current_avail_feat_kc)

        # remove options which are infeasible given the feasibility surrogate model
        # and the threshold
        if fca_constraint_callable is not None:
            # FCA approach, apply feasibility constraint
            constraint_input = current_avail_feat_kc.view(
                current_avail_feat_kc.shape[0],
                1,
                current_avail_feat_kc.shape[1],
            )
            constraint_vals = fca_constraint_callable(constraint_input)
            feas_mask = torch.where(constraint_vals >= 0.0)[0]
            print(
                f"{feas_mask.shape[0]}/{len(current_avail_feat)} options are feasible"
            )
            if feas_mask.shape[0] == 0:
                msg = "No feasible samples after FCA constraint, resorting back to full space"
                Logger.log(msg, "WARNING")
                # if we have zero feasible samples
                # resort back to the full set of unobserved options
                current_avail_feat = current_avail_feat_kc
                current_avail_cat = current_avail_cat_kc
            else:
                current_avail_feat = current_avail_feat_kc[feas_mask]
                current_avail_cat = current_avail_cat_kc[
                    feas_mask.detach().numpy()
                ]
        else:
            current_avail_feat = current_avail_feat_kc
            current_avail_cat = current_avail_cat_kc

        return current_avail_feat, current_avail_cat

    else:
        # at least one continuous parameter, no need to remove any options
        current_avail_feat = []
        current_avail_cat = []

        for elem in cart_product:
            # convert to ohe and add to currently available options
            ohe = []
            for val, obj in zip(elem, param_space):
                if obj.type == "categorical":
                    ohe.append(cat_param_to_feat(obj, val, has_descriptors))
                else:
                    ohe.append([val])
            current_avail_feat.append(np.concatenate(ohe))
            current_avail_cat.append(elem)

        current_avail_feat_unconst = np.array(current_avail_feat)
        current_avail_cat_unconst = np.array(current_avail_cat)

        # TODO: may need to add the constraint checking here too...
        # forward normalize the options before evaluating the constaints
        # if normalize:
        # 	current_avail_feat_unconst = forward_normalize(current_avail_feat_unconst, mins_x, maxs_x)

        current_avail_feat_unconst = torch.tensor(current_avail_feat_unconst)

        return current_avail_feat_unconst, current_avail_cat_unconst
