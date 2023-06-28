#!/usr/bin/env python

import os
import pickle
import sys
import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gpytorch
import numpy as np
import torch
from botorch.acquisition import (
    ExpectedImprovement,
    UpperConfidenceBound,
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qUpperConfidenceBound,
)
from botorch.fit import fit_gpytorch_model
from botorch.models import MixedSingleTaskGP, SingleTaskGP

from gpytorch.mlls import ExactMarginalLogLikelihood
from olympus import ParameterVector
from olympus.campaigns import ParameterSpace

from atlas import Logger
from atlas.acquisition_functions.acqfs import (
    FeasibilityAwareEI,
    FeasibilityAwareGeneral,
    FeasibilityAwareLCB,
    FeasibilityAwareQEI,
    FeasibilityAwareUCB,
    FeasibilityAwareVarainceBased,
    LowerConfidenceBound,
    VarianceBased,
    create_available_options,
)
from atlas.acquisition_optimizers import (
    GeneticOptimizer,
    GradientOptimizer,
)
from atlas.base.base import BasePlanner
from atlas.gps.gps import (
    CategoricalSingleTaskGP,
    ClassificationGPMatern,
)
from atlas.params.params import Parameters
from atlas.utils.planner_utils import (
    cat_param_to_feat,
    forward_normalize,
    forward_standardize,
    get_cat_dims,
    get_fixed_features_list,
    infer_problem_type,
    propose_randomly,
    reverse_normalize,
    reverse_standardize,
)


class BoTorchPlanner(BasePlanner):
    """Wrapper for GP-based Bayesiam optimization with BoTorch
    Args:
            goal (str): the optimization goal, "maximize" or "minimize"
            feas_strategy (str): feasibility acqusition function name
            feas_param (float): feasibilty parameter
            batch_size (int): number of samples to measure per batch (will be fixed at 1 for now)
            random_seed (int): the random seed to use
            num_initial_design (int): number of points to sample using the initial
                    design strategy
            init_design_strategy (str): the inital design strategy, "random" or "sobol"
            vgp_iters (int): number of training iterations for the variational GP
            vgp_lr (float): learning rate for the variational optimization procedure
            max_jitter (float):
            cla_threshold (float): classification threshold for the predictions of the
                    feasibilty surrogate
            known_constraints (callable): callable which takes parameters and returns boolean
                    corresponding to the feaibility of that experiment (True-->feasible, False-->infeasible)
            general_parameters (list): list of parameter indices for which we average the objective
                    function over
            is_moo (bool): whether or not we have a multiobjective optimization problem
    """

    def __init__(
        self,
        goal: str,
        feas_strategy: Optional[str] = "naive-0",
        feas_param: Optional[float] = 0.2,
        use_min_filter: bool = True,
        batch_size: int = 1,
        batched_strategy: str = "sequential",  # sequential or greedy
        random_seed: Optional[int] = None,
        use_descriptors: bool = False,
        num_init_design: int = 5,
        init_design_strategy: str = "random",
        acquisition_type: str = "ei",  # qei, ei, ucb, variance, general
        acquisition_optimizer_kind: str = "gradient",  # gradient, genetic
        vgp_iters: int = 2000,
        vgp_lr: float = 0.1,
        max_jitter: float = 1e-1,
        cla_threshold: float = 0.5,
        known_constraints: Optional[List[Callable]] = None,
        compositional_params: Optional[List[int]] = None,
        permutation_params: Optional[List[int]] = None,
        general_parameters: Optional[List[int]] = None,
        is_moo: bool = False,
        value_space: Optional[ParameterSpace] = None,
        scalarizer_kind: Optional[str] = "Hypervolume",
        moo_params: Dict[str, Union[str, float, int, bool, List]] = {},
        goals: Optional[List[str]] = None,
        golem_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        local_args = {
            key: val for key, val in locals().items() if key != "self"
        }
        super().__init__(**local_args)

        # check that we are using the 'general' parameter acquisition
        if self.general_parameters is not None:
            if not self.acquisition_type == 'general':
                msg = f'Acquisition type {self.acquisition_type} requested, but general parameters specified. Overriding to "general"...'
                Logger.log(msg, 'WARNING')

                self.acquisition_type = 'general'

    def build_train_regression_gp(
        self, train_x: torch.Tensor, train_y: torch.Tensor
    ) -> gpytorch.models.ExactGP:
        """Build the regression GP model and likelihood"""
        # infer the model based on the parameter types
        if self.problem_type in [
            "fully_continuous",
            "fully_discrete",
            "mixed_disc_cont",
        ]:
            model = SingleTaskGP(train_x, train_y)
        elif self.problem_type == "fully_categorical":
            if self.has_descriptors:
                # we have some descriptors, use the Matern kernel
                model = SingleTaskGP(train_x, train_y)
            else:
                # if we have no descriptors, use a Categorical kernel
                # based on the HammingDistance
                model = CategoricalSingleTaskGP(train_x, train_y)
        elif "mixed_cat_" in self.problem_type:
            if self.has_descriptors:
                # we have some descriptors, use the Matern kernel
                model = SingleTaskGP(train_x, train_y)
            else:
                cat_dims = get_cat_dims(self.param_space)
                model = MixedSingleTaskGP(train_x, train_y, cat_dims=cat_dims)

        else:
            raise NotImplementedError

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        # fit the GP
        start_time = time.time()
        with gpytorch.settings.cholesky_jitter(self.max_jitter):
            fit_gpytorch_model(mll)
        gp_train_time = time.time() - start_time
        Logger.log(
            f"Regression surrogate GP trained in {round(gp_train_time,3)} sec",
            "INFO",
        )

        return model

    def _ask(self) -> List[ParameterVector]:
        """query the planner for a batch of new parameter points to measure"""
        # if we have all nan values, just continue with initial design
        if np.logical_or(
            len(self._values) < self.num_init_design,
            np.all(np.isnan(self._values)),
        ):
            return_params = self.initial_design()

        else:
            # timings dictionary for analysis
            self.timings_dict = {}

            # use GP surrogate to propose the samples
            # get the scaled parameters and values for both the regression and classification data
            (
                self.train_x_scaled_cla,
                self.train_y_scaled_cla,
                self.train_x_scaled_reg,
                self.train_y_scaled_reg,
            ) = self.build_train_data()

            use_p_feas_only = False
            # check to see if we are using the naive approaches
            if "naive-" in self.feas_strategy:
                infeas_ix = torch.where(self.train_y_scaled_cla == 1.0)[0]
                feas_ix = torch.where(self.train_y_scaled_cla == 0.0)[0]
                # checking if we have at least one objective function measurement
                #  and at least one infeasible point (i.e. at least one point to replace)
                if np.logical_and(
                    self.train_y_scaled_reg.size(0) >= 1,
                    infeas_ix.shape[0] >= 1,
                ):
                    if self.feas_strategy == "naive-replace":
                        # NOTE: check to see if we have a trained regression surrogate model
                        # if not, wait for the following iteration to make replacements
                        if hasattr(self, "reg_model"):
                            # if we have a trained regression model, go ahead and make replacement
                            new_train_y_scaled_reg = deepcopy(
                                self.train_y_scaled_cla
                            ).double()

                            input = self.train_x_scaled_cla[infeas_ix].double()

                            posterior = self.reg_model.posterior(X=input)
                            pred_mu = posterior.mean.detach()

                            new_train_y_scaled_reg[
                                infeas_ix
                            ] = pred_mu.squeeze(-1)
                            new_train_y_scaled_reg[
                                feas_ix
                            ] = self.train_y_scaled_reg.squeeze(-1)

                            self.train_x_scaled_reg = deepcopy(
                                self.train_x_scaled_cla
                            ).double()
                            self.train_y_scaled_reg = (
                                new_train_y_scaled_reg.view(
                                    self.train_y_scaled_cla.size(0), 1
                                ).double()
                            )

                        else:
                            use_p_feas_only = True

                    elif self.feas_strategy == "naive-0":
                        new_train_y_scaled_reg = deepcopy(
                            self.train_y_scaled_cla
                        ).double()

                        worst_obj = torch.amax(
                            self.train_y_scaled_reg[
                                ~self.train_y_scaled_reg.isnan()
                            ]
                        )

                        to_replace = torch.ones(infeas_ix.size()) * worst_obj

                        new_train_y_scaled_reg[infeas_ix] = to_replace.double()
                        new_train_y_scaled_reg[
                            feas_ix
                        ] = self.train_y_scaled_reg.squeeze()

                        self.train_x_scaled_reg = (
                            self.train_x_scaled_cla.double()
                        )
                        self.train_y_scaled_reg = new_train_y_scaled_reg.view(
                            self.train_y_scaled_cla.size(0), 1
                        )

                    else:
                        raise NotImplementedError
                else:
                    # if we are not able to use the naive strategies, propose randomly
                    # do nothing at all and use the feasibilty surrogate as the acquisition
                    use_p_feas_only = True

            # builds and fits the regression surrogate model
            self.reg_model = self.build_train_regression_gp(
                self.train_x_scaled_reg, self.train_y_scaled_reg
            )

            if (
                not "naive-" in self.feas_strategy
                and torch.sum(self.train_y_scaled_cla).item() != 0.0
            ):
                # build and train the classification surrogate model
                (
                    self.cla_model,
                    self.cla_likelihood,
                ) = self.build_train_classification_gp(
                    self.train_x_scaled_cla, self.train_y_scaled_cla
                )

                self.cla_model.eval()
                self.cla_likelihood.eval()

                use_reg_only = False

                # estimate the max and min of the cla surrogate
                (
                    self.cla_surr_min_,
                    self.cla_surr_max_,
                ) = self.get_cla_surr_min_max(num_samples=5000)
                self.fca_cutoff = (
                    self.cla_surr_max_ - self.cla_surr_min_
                ) * self.feas_param + self.cla_surr_min_

            else:
                use_reg_only = True
                self.cla_model, self.cla_likelihood = None, None
                self.cla_surr_min_, self.cla_surr_max_ = None, None

            # get the incumbent point
            f_best_argmin = torch.argmin(self.train_y_scaled_reg)

            f_best_scaled = self.train_y_scaled_reg[f_best_argmin][0].float()

            # compute the ratio of infeasible to total points
            infeas_ratio = (
                torch.sum(self.train_y_scaled_cla)
                / self.train_x_scaled_cla.size(0)
            ).item()
            # get the approximate max and min of the acquisition function without the feasibility contribution
            acqf_min_max = self.get_aqcf_min_max(self.reg_model, f_best_scaled)

            if self.acquisition_type == "ei":
                if (
                    self.batch_size > 1
                    and self.batched_strategy == "sequential"
                ):
                    Logger.log(
                        'Cannot use "sequential" batched strategy with EI acquisition function',
                        "FATAL",
                    )
                self.acqf = FeasibilityAwareEI(
                    self.reg_model,
                    self.cla_model,
                    self.cla_likelihood,
                    self.param_space,
                    f_best_scaled,
                    self.feas_strategy,
                    self.feas_param,
                    infeas_ratio,
                    acqf_min_max,
                    use_min_filter=self.use_min_filter,
                    use_reg_only=use_reg_only,
                )

            elif self.acquisition_type == "qei":
                if not self.batch_size > 1:
                    Logger.log(
                        "QEI acquisition function can only be used if batch size > 1",
                        "FATAL",
                    )

                self.acqf = FeasibilityAwareQEI(
                    self.reg_model,
                    self.cla_model,
                    self.cla_likelihood,
                    self.param_space,
                    f_best_scaled,
                    self.feas_strategy,
                    self.feas_param,
                    infeas_ratio,
                    acqf_min_max,
                    use_min_filter=self.use_min_filter,
                    use_reg_only=use_reg_only,
                )

            elif self.acquisition_type == "ucb":
                self.acqf = FeasibilityAwareUCB(
                    self.reg_model,
                    self.cla_model,
                    self.cla_likelihood,
                    self.param_space,
                    f_best_scaled,
                    self.feas_strategy,
                    self.feas_param,
                    infeas_ratio,
                    acqf_min_max,
                    use_reg_only=use_reg_only,
                    # beta=torch.tensor([0.2]).repeat(self.batch_size),
                    beta=torch.tensor([1.0]).repeat(self.batch_size),
                    use_min_filter=self.use_min_filter,
                )

            elif self.acquisition_type == "lcb":
                self.acqf = FeasibilityAwareLCB(
                    self.reg_model,
                    self.cla_model,
                    self.cla_likelihood,
                    self.param_space,
                    f_best_scaled,
                    self.feas_strategy,
                    self.feas_param,
                    infeas_ratio,
                    acqf_min_max,
                    use_reg_only=use_reg_only,
                    # beta=torch.tensor([0.2]).repeat(self.batch_size),
                    beta=torch.tensor([1.0]).repeat(self.batch_size),
                    use_min_filter=self.use_min_filter,
                )

            elif self.acquisition_type == "variance":
                self.acqf = FeasibilityAwareVarainceBased(
                    self.reg_model,
                    self.cla_model,
                    self.cla_likelihood,
                    self.param_space,
                    f_best_scaled,
                    self.feas_strategy,
                    self.feas_param,
                    infeas_ratio,
                    acqf_min_max,
                    use_min_filter=self.use_min_filter,
                    use_reg_only=use_reg_only,
                )

            elif self.acquisition_type == "general":
                self.acqf = FeasibilityAwareGeneral(
                    self.reg_model,
                    self.cla_model,
                    self.cla_likelihood,
                    self.params_obj,
                    # self.general_parameters,
                    self.param_space,
                    f_best_scaled,
                    self.feas_strategy,
                    self.feas_param,
                    infeas_ratio,
                    acqf_min_max,
                    use_min_filter=self.use_min_filter,
                    use_reg_only=use_reg_only,
                )

            else:
                msg = f"Acquisition function type {self.acquisition_type} not understood!"
                Logger.log(msg, "FATAL")

            if self.acquisition_optimizer_kind == "gradient":
                acquisition_optimizer = GradientOptimizer(
                    self.params_obj,
                    self.acquisition_type,
                    self.acqf,
                    self.known_constraints,
                    self.batch_size,
                    self.feas_strategy,
                    self.fca_constraint,
                    self._params,
                    self.batched_strategy,
                    self.timings_dict,
                    use_reg_only=use_reg_only,
                )
            elif self.acquisition_optimizer_kind == "genetic":
                acquisition_optimizer = GeneticOptimizer(
                    self.params_obj,
                    self.acquisition_type,
                    self.acqf,
                    self.known_constraints,
                    self.batch_size,
                    self.feas_strategy,
                    self.fca_constraint,
                    self._params,
                    self.timings_dict,
                    use_reg_only=use_reg_only,
                )

            return_params = acquisition_optimizer.optimize()

        return return_params

    def get_aqcf_min_max(
        self,
        reg_model: gpytorch.models.ExactGP,
        f_best_scaled: torch.Tensor,
        num_samples: int = 3000,
    ) -> Tuple[int, int]:
        """computes the min and max value of the acquisition function without
        the feasibility contribution. These values will be used to approximately
        normalize the acquisition function
        """
        if self.acquisition_type == "ei":
            acqf = ExpectedImprovement(
                reg_model, f_best_scaled, objective=None, maximize=False
            )

        if self.acquisition_type == "qei":
            acqf = qExpectedImprovement(
                reg_model, f_best_scaled, objective=None, maximize=False
            )

        elif self.acquisition_type == "ucb":
            acqf = UpperConfidenceBound(
                reg_model,
                # beta=torch.tensor([0.2]).repeat(self.batch_size),
                beta=torch.tensor([1.0]).repeat(self.batch_size),
                objective=None,
                maximize=False,
            )

        elif self.acquisition_type == "lcb":
            acqf = LowerConfidenceBound(
                reg_model,
                # beta=torch.tensor([0.2]).repeat(self.batch_size),
                beta=torch.tensor([1.0]).repeat(self.batch_size),
                objective=None,
                maximize=False,
            )
        elif self.acquisition_type == "ucbv2":

            def acqf(X):
                posterior = self.reg_model.posterior(
                    X=X,
                )
                mean = posterior.mean.squeeze(-2).squeeze(
                    -1
                )  # removing redundant dimensions
                sigma = (
                    posterior.variance.clamp_min(1e-12).sqrt().view(mean.shape)
                )
                return -mean, sigma

        elif self.acquisition_type == "variance":
            acqf = VarianceBased(reg_model)

        elif self.acquisition_type == "general":
            # do not scale the acqf in this case
            # TODO is this OK?
            return 0.0, 1.0

        samples, _ = propose_randomly(
            num_samples,
            self.param_space,
            self.has_descriptors,
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

        acqf_vals = acqf(
            torch.tensor(samples)
            .view(samples.shape[0], 1, samples.shape[-1])
            .double()
        )

        if not self.acquisition_type == "ucbv2":
            min_ = torch.amin(acqf_vals).item()
            max_ = torch.amax(acqf_vals).item()

            if np.abs(max_ - min_) < 1e-6:
                max_ = 1.0
                min_ = 0.0

            return min_, max_
        else:
            min_mean_ = torch.amin(acqf_vals[0]).item()
            max_mean_ = torch.amax(acqf_vals[0]).item()

            min_sigma_ = torch.amin(acqf_vals[1]).item()
            max_sigma_ = torch.amax(acqf_vals[1]).item()

            if np.abs(max_mean_ - min_mean_) < 1e-6:
                max_mean_ = 1.0
                min_mean_ = 0.0

            if np.abs(max_sigma_ - min_sigma_) < 1e-6:
                max_sigma_ = 1.0
                min_sigma_ = 0.0

            return (min_mean_, max_mean_), (min_sigma_, max_sigma_)
