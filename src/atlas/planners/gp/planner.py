#!/usr/bin/env python

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import warnings

import gpytorch
import numpy as np
import torch

from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP, SingleTaskGP

from gpytorch.mlls import ExactMarginalLogLikelihood
from olympus import ParameterVector
from olympus.campaigns import ParameterSpace

from atlas import Logger, tkwargs

from atlas.acquisition_functions.acqfs import get_acqf_instance

from atlas.acquisition_optimizers import (
    GeneticOptimizer,
    GradientOptimizer,
    PymooGAOptimizer
)
from atlas.base.base import BasePlanner
from atlas.gps.gps import CategoricalSingleTaskGP, TanimotoGP

from atlas.utils.planner_utils import get_cat_dims

warnings.filterwarnings("ignore", "^.*jitter.*", category=RuntimeWarning)
torch.set_default_dtype(torch.double)



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
        acquisition_optimizer_kind: str = "gradient",  # gradient, genetic, pymoo
        vgp_iters: int = 2000,
        vgp_lr: float = 0.1,
        max_jitter: float = 1e-1,
        cla_threshold: float = 0.5,
        known_constraints: Optional[List[Callable]] = None,
        compositional_params: Optional[List[int]] = None,
        permutation_params: Optional[List[int]] = None,
        batch_constrained_params: Optional[List[int]] = None,
        general_parameters: Optional[List[int]] = None,
        molecular_params: Optional[List[int]] = None,
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
            model = SingleTaskGP(train_x, train_y).to(tkwargs['device'])
        elif self.problem_type == "fully_categorical":
            if self.has_descriptors:
                # we have some descriptors, use the Matern kernel or Tanimoto if we have all molecular dims
                if self.molecular_params == [0]:
                    # use TanimotoGP
                    # NOTE: here we assume we are given Morgan FPs as descriptors, might want to validate
                    # NOTE: this is only implemented for single molecular dimension
                    model = TanimotoGP(train_x, train_y).to(tkwargs['device'])
                else:
                    # no molecular parameters, use Matern GP
                    model = SingleTaskGP(train_x, train_y).to(tkwargs['device'])
            else:
                # if we have no descriptors, use a Categorical kernel
                # based on the HammingDistance
                model = CategoricalSingleTaskGP(train_x, train_y).to(tkwargs['device'])
        elif "mixed_cat_" in self.problem_type:
            if self.has_descriptors:
                # we have some descriptors, use the Matern kernel
                model = SingleTaskGP(train_x, train_y).to(tkwargs['device'])
            else:
                cat_dims = get_cat_dims(self.param_space)
                model = MixedSingleTaskGP(train_x, train_y, cat_dims=cat_dims).to(tkwargs['device'])

        else:
            raise NotImplementedError

        mll = ExactMarginalLogLikelihood(model.likelihood, model).to(tkwargs['device'])
        # fit the GP
        start_time = time.time()
        with gpytorch.settings.cholesky_jitter(self.max_jitter):
            fit_gpytorch_mll(mll)
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

            # handle naive unknown constriants strategies if relevant
            # TODO: put this in build_train_data method
            (
                self.train_x_scaled_reg,
                self.train_y_scaled_reg,
                self.train_x_scaled_cla,
                self.train_y_scaled_cla,
                use_p_feas_only
            ) = self.unknown_constraints.handle_naive_feas_strategies(
                self.train_x_scaled_reg,
                self.train_y_scaled_reg,
                self.train_x_scaled_cla,
                self.train_y_scaled_cla,
            )


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

            f_best_scaled = self.train_y_scaled_reg[f_best_argmin][0].double()

            # compute the ratio of infeasible to total points
            infeas_ratio = (
                torch.sum(self.train_y_scaled_cla)
                / self.train_x_scaled_cla.size(0)
            ).item()

        
     
            # get compile the basic feas-aware acquisition function arguments
            acqf_args = dict(
                acquisition_optimizer_kind=self.acquisition_optimizer_kind,
                params_obj=self.params_obj,
                problem_type=self.problem_type,
                feas_strategy=self.feas_strategy,
                feas_param=self.feas_param,
                infeas_ratio=infeas_ratio,
                use_reg_only=use_reg_only,
                f_best_scaled=f_best_scaled,
                batch_size=self.batch_size,
                use_min_filter=self.use_min_filter,

            )
            self.acqf = get_acqf_instance(
                acquisition_type=self.acquisition_type, 
                reg_model=self.reg_model,
                cla_model=self.cla_model,
                cla_likelihood=self.cla_likelihood,
                acqf_args=acqf_args,
            )


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
                    acqf_args=acqf_args,
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
                    acqf_args=acqf_args,
                )

            elif self.acquisition_optimizer_kind == 'pymoo':
                acquisition_optimizer = PymooGAOptimizer(
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
                    acqf_args=acqf_args,
                )

            return_params = acquisition_optimizer.optimize()

        return return_params
    

