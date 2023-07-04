#!/usr/bin/env python

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os
import pickle
import sys
import time

# remove warnings about adding jitter
import warnings

import gpytorch
import numpy as np
import olympus
import torch
from botorch.acquisition import (
    ExpectedImprovement,
    qExpectedImprovement,
    qNoisyExpectedImprovement,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.gpytorch import GPyTorchModel

from botorch.sampling.normal import SobolQMCNormalSampler
from gpytorch.distributions import MultivariateNormal
from gpytorch.lazy import PsdSumLazyTensor
from gpytorch.likelihoods import LikelihoodList
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import GP
from olympus import ParameterVector, ParameterSpace
from olympus.planners import AbstractPlanner, CustomPlanner, Planner
from olympus.scalarizers import Scalarizer
from torch.nn import ModuleList

from atlas import Logger
from atlas.acquisition_functions.acqfs import (
    FeasibilityAwareEI,
    FeasibilityAwareGeneral,
    FeasibilityAwareQEI,
    create_available_options,
    get_batch_initial_conditions,
)
from atlas.acquisition_optimizers import (
    GeneticOptimizer,
    GradientOptimizer,
    PymooGAOptimizer
)
from atlas.params.params import Parameters
from atlas.unknown_constraints.unknown_constraints import UnknownConstraints

from atlas.base.base import BasePlanner

from atlas.utils.planner_utils import (
    Scaler,
    cat_param_to_feat,
    flip_source_tasks,
    forward_normalize,
    forward_standardize,
    get_cat_dims,
    get_fixed_features_list,
    infer_problem_type,
    propose_randomly,
    reverse_normalize,
    reverse_standardize,
)

warnings.filterwarnings("ignore", "^.*jitter.*", category=RuntimeWarning)


class RGPE(GP, GPyTorchModel):
    """Rank-weighted GP ensemble. This class inherits from GPyTorchModel which
    provides an interface for GPyTorch models in botorch
    Args:
            models (List[SingleTaskGP]): list of GP models
            weights (torch.Tensor): weights
    """

    # meta-data for botorch
    _num_outputs = 1

    def __init__(self, models, weights):
        super().__init__()
        self.models = ModuleList(models)
        for m in models:
            if not hasattr(m, "likelihood"):
                raise ValueError(
                    "RGPE currently only supports models that have a likelihood (e.g. ExactGPs)"
                )
        self.likelihood = LikelihoodList(*[m.likelihood for m in models])
        self.weights = weights
        # self.to(weights)

    def forward(self, x):
        x = x.float()
        weighted_means = []
        weighted_covars = []
        # filter model with zero weights
        # weights on covariance matrices are weight**2
        non_zero_weight_indices = (self.weights**2 > 0).nonzero()
        non_zero_weights = self.weights[non_zero_weight_indices]
        # re-normalize
        non_zero_weights /= non_zero_weights.sum()

        for non_zero_weight_idx in range(non_zero_weight_indices.shape[0]):
            raw_idx = non_zero_weight_indices[non_zero_weight_idx].item()
            model = self.models[raw_idx]
            posterior = model.posterior(x)
            # unstandardize predictions
            # posterior_mean = posterior.mean.squeeze(-1)*model.Y_std + model.Y_mean
            # posterior_cov = posterior.mvn.lazy_covariance_matrix * model.Y_std.pow(2)
            posterior_mean = posterior.mean.squeeze(-1)
            posterior_cov = posterior.mvn.lazy_covariance_matrix
            # apply weight
            weight = non_zero_weights[non_zero_weight_idx]
            weighted_means.append(weight * posterior_mean)
            weighted_covars.append(posterior_cov * weight**2)
        # set mean and covariance to be the rank-weighted sum the means and covariances of the
        # base models and target model
        mean_x = torch.stack(weighted_means).sum(dim=0)
        covar_x = PsdSumLazyTensor(*weighted_covars)
        return MultivariateNormal(mean_x, covar_x)


class RGPEPlanner(BasePlanner):
    """Wrapper for the Rank-Weighted GP Ensemble (RGPE)
    https://arxiv.org/pdf/1802.02219.pdf
    code adapted from: https://botorch.org/v/0.1.0/tutorials/meta_learning_with_rgpe
    Args:
            cache_weights (bool): save the weights of the RGPE procedure to disk for
                    a posteriori analysis
            weights_path (str): the directory in which to save the weights, if cache_weights=True
    """

    def __init__(
        self,
		goal: str,
		feas_strategy: Optional[str] = "naive-0",
		feas_param: Optional[float] = 0.2,
		use_min_filter: bool = True,
		batch_size: int = 1,
		batched_strategy: str = 'sequential', # sequential or greedy
		random_seed: Optional[int] = None,
		use_descriptors: bool = False,
		num_init_design: int = 5,
		init_design_strategy: str = "random",
		acquisition_type: str = "ei",  # ei, ucb
		acquisition_optimizer_kind: str = "gradient",  # gradient, genetic
		vgp_iters: int = 2000,
		vgp_lr: float = 0.1,
		max_jitter: float = 1e-1,
		cla_threshold: float = 0.5,
		known_constraints: Optional[List[Callable]] = None,
		compositional_params: Optional[List[int]] = None,
        permutation_params: Optional[List[int]] = None,
		batch_constrained_params: Optional[List[int]] = None,
		general_parameters: Optional[List[int]] = None,
		is_moo: bool = False,
		value_space: Optional[ParameterSpace] = None,
		scalarizer_kind: Optional[str] = "Hypervolume",
		moo_params: Dict[str, Union[str, float, int, bool, List]] = {},
		goals: Optional[List[str]] = None,
		golem_config: Optional[Dict[str, Any]] = None,
        # meta-learning stuff
        cache_weights: bool = False,
        weights_path: str ="./weights/",
        train_tasks: List = [],
        valid_tasks: Optional[List] = None,
        hyperparams: Optional[Dict] = {},
        **kwargs,
    ):

        local_args = {
            key: val for key, val in locals().items() if key != "self"
        }
        super().__init__(**local_args)

        # meta learning stuff
        self.cache_weights = cache_weights
        self.weights_path = weights_path
        self.hyperparams = hyperparams
        self._train_tasks = train_tasks
        self._valid_tasks = valid_tasks

        self.all_rank_weights = []
        self.all_ranking_losses = []

        self.device = "cpu"

        # # NOTE: for maximization, we must flip the signs of the
        # source task values before scaling them
        if self.goal == "maximize":
            self._train_tasks = flip_source_tasks(self._train_tasks)
            self._valid_tasks = flip_source_tasks(self._valid_tasks)

        # if we have a multi-objective problem, scalarize the values
        # for the source tasks individually
        if self.is_moo:
            for task in self._train_tasks:
                scal_values = self.scalarizer.scalarize(task["values"])
                task["values"] = scal_values.reshape(-1, 1)

            for task in self._valid_tasks:
                scal_values = self.scalarizer.scalarize(task["values"])
                task["values"] = scal_values.reshape(-1, 1)

        # instantiate the scaler
        self.scaler = Scaler(
            param_type="normalization",
            value_type="standardization",
        )
        self._train_tasks = self.scaler.fit_transform_tasks(self._train_tasks)
        self._valid_tasks = self.scaler.transform_tasks(self._valid_tasks)

    def _get_fitted_model(self, train_X, train_Y, state_dict=None):
        """Get a fixed noise single task GP. The GP model will be fit unless
        a state_dict containing model hyperparameters is passed
        """
        with gpytorch.settings.cholesky_jitter(1e-1):
            model = SingleTaskGP(train_X, train_Y)
            if state_dict is None:
                mll = ExactMarginalLogLikelihood(model.likelihood, model).to(
                    train_X
                )
                fit_gpytorch_mll(mll)
            else:
                model.load_state_dict(state_dict)

        return model

    def _get_source_models(self):
        source_models = []
        for task_ix, task in enumerate(self._train_tasks):
            Logger.log(f"Fitting source model {task_ix}", "INFO")
            source_models.append(
                self._get_fitted_model(
                    torch.tensor(task["params"]),
                    torch.tensor(task["values"]),
                )
            )
        return source_models

    @staticmethod
    def roll_col(X, shift):
        """roll columns to the right by amount shift"""
        return torch.cat((X[..., -shift:], X[..., :-shift]), dim=-1)

    def compute_ranking_loss(self, f_samps, target_y):
        """Compute the ranking loss for each sample from the posterior
        over the target points
        Args:
                f_samps (torch.Tensor): samples of shape (num_samples, n, num_dim)
                target_y (torch.Tensor): tensor containing targets of shape (num_dim, 1)
        Returns:
                rank_loss (torch.Tensor): tensor containing the ranking loss for each
                        sample shape (num_samples)
        """
        n = target_y.shape[0]
        if f_samps.ndim == 3:
            # Compute ranking loss for target model
            # take cartesian product of target_y
            cartesian_y = torch.cartesian_prod(
                target_y.squeeze(-1),
                target_y.squeeze(-1),
            ).view(n, n, 2)
            # the diagonal of f_samps are the out-of-sample predictions
            # for each LOO model, compare the out of sample predictions to each in-sample prediction
            rank_loss = (
                (
                    (f_samps.diagonal(dim1=1, dim2=2).unsqueeze(-1) < f_samps)
                    ^ (cartesian_y[..., 0] < cartesian_y[..., 1])
                )
                .sum(dim=-1)
                .sum(dim=-1)
            )
        else:
            rank_loss = torch.zeros(
                f_samps.shape[0], dtype=torch.long, device=target_y.device
            )
            y_stack = target_y.squeeze(-1).expand(f_samps.shape)
            for i in range(1, target_y.shape[0]):
                rank_loss += (
                    (self.roll_col(f_samps, i) < f_samps)
                    ^ (self.roll_col(y_stack, i) < y_stack)
                ).sum(dim=-1)
        return rank_loss

    def get_target_model_loocv_sample_preds(
        self, train_x, train_y, target_model, num_samples
    ):
        """
        Create a batch-mode LOOCV GP and draw a joint sample across all points from the target task.
        Args:
                train_x: `n x d` tensor of training points
                train_y: `n x 1` tensor of training targets
                target_model: fitted target model
                num_samples: number of mc samples to draw
        Return: `num_samples x n x n`-dim tensor of samples, where dim=1 represents the `n` LOO models,
                and dim=2 represents the `n` training points.
        """
        batch_size = len(train_x)
        masks = torch.eye(
            len(train_x), dtype=torch.uint8, device=self.device
        ).bool()
        train_x_cv = torch.stack([train_x[~m] for m in masks])
        train_y_cv = torch.stack([train_y[~m] for m in masks])
        # train_yvar_cv = torch.stack([train_yvar[~m] for m in masks])
        state_dict = target_model.state_dict()
        # expand to batch size of batch_mode LOOCV model
        state_dict_expanded = {
            name: t.expand(batch_size, *[-1 for _ in range(t.ndim)])
            for name, t in state_dict.items()
        }
        model = self._get_fitted_model(
            train_x_cv, train_y_cv, state_dict=state_dict_expanded
        )
        with torch.no_grad():
            posterior = model.posterior(train_x)
            # Since we have a batch mode gp and model.posterior always returns an output dimension,
            # the output from `posterior.sample()` here `num_samples x n x n x 1`, so let's squeeze
            # the last dimension.
            sampler = SobolQMCNormalSampler(sample_shape=torch.tensor([num_samples]).size())
            return sampler(posterior).squeeze(-1)

    def compute_rank_weights(
        self, train_x, train_y, base_models, target_model, num_samples
    ):
        """Compute ranking weights for each base model and the target model (using
        LOOCV for the target model). Note: This implementation does not currently
        address weight dilution, since we only have a small number of base models.
        Args:
                train_x: `n x d` tensor of training points (for target task)
                train_y: `n` tensor of training targets (for target task)
                base_models: list of `n_t` base models
                num_samples: number of mc samples
        Returns:
                Tensor: `n_t`-dim tensor with the ranking weight for each model
        """
        ranking_losses = []
        # compute ranking loss for each base model
        for task_ix in range(len(base_models)):
            model = base_models[task_ix]
            # compute posterior over training points for target task
            posterior = model.posterior(train_x)
            sampler = SobolQMCNormalSampler(sample_shape=torch.tensor([num_samples]).size())
            base_f_samps = sampler(posterior).squeeze(-1).squeeze(-1)
            # compute and save ranking loss
            ranking_losses.append(
                self.compute_ranking_loss(base_f_samps, train_y)
            )
        # compute ranking loss for target model using LOOCV
        # f_samps
        target_f_samps = self.get_target_model_loocv_sample_preds(
            train_x,
            train_y,
            target_model,
            num_samples,
        )
        ranking_losses.append(
            self.compute_ranking_loss(target_f_samps, train_y)
        )
        ranking_loss_tensor = torch.stack(ranking_losses)
        # compute best model (minimum ranking loss) for each sample
        best_models = torch.argmin(ranking_loss_tensor, dim=0)
        # compute proportion of samples for which each model is best
        rank_weights = (
            best_models.bincount(minlength=len(ranking_losses)).type_as(
                train_x
            )
            / num_samples
        )
        return rank_weights, ranking_loss_tensor

    def _ask(self) -> List[ParameterVector]:
        """query the planner for a batch of new parameter points to measure"""

        # fit the source models
        if not hasattr(self, "source_models"):
            self.source_models = self._get_source_models()

        # if we have all nan values, just keep randomly sampling
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

            # builds the regression model
            target_model = self._get_fitted_model(
                self.train_x_scaled_reg, self.train_y_scaled_reg
            )
            model_list = self.source_models + [target_model]
            rank_weights, ranking_loss_tensor = self.compute_rank_weights(
                self.train_x_scaled_reg.float(),
                self.train_y_scaled_reg.float(),
                self.source_models,
                target_model,
                num_samples=10,
            )

            self.reg_model = RGPE(model_list, rank_weights)

            # check to see if we cache the weights and save them to disk
            if self.cache_weights:
                self.all_rank_weights.append(rank_weights.detach().numpy())
                self.all_ranking_losses.append(
                    ranking_loss_tensor.detach().numpy()
                )
                os.makedirs(self.weights_path, exist_ok=True)
                with open(
                    os.path.join(self.weights_path, "rank_weights.pkl"), "wb"
                ) as f:
                    pickle.dump(
                        {
                            "weights": self.all_rank_weights,
                            "losses": self.all_ranking_losses,
                        },
                        f,
                    )
            
            # TODO: can probably put this bit in the unknown constraints module
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
            else:
                Logger.log(
                    'RPGE planner requires using either EI or QEI acquisiton function', 'FATAL'
                )

            # set acquisition optimizer
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
                )

            return_params = acquisition_optimizer.optimize()

        return return_params

    def get_aqcf_min_max(self, reg_model, f_best_scaled, num_samples=2000):
        """computes the min and max value of the acquisition function without
        the feasibility contribution. These values will be used to approximately
        normalize the acquisition function
        """
        if self.batch_size == 1:
            acqf = ExpectedImprovement(
                reg_model, f_best_scaled, objective=None, maximize=False
            )
        elif self.batch_size > 1:
            acqf = qExpectedImprovement(
                reg_model, f_best_scaled, objective=None, maximize=False
            )
        samples, _ = propose_randomly(num_samples, self.param_space,  has_descriptors=self.has_descriptors)
        if (
            not self.problem_type == "fully_categorical"
            and not self.has_descriptors
        ):
            # we dont scale the parameters if we have a one-hot-encoded representation
            samples = forward_normalize(samples, self.params_obj._mins_x, self.params_obj._maxs_x)

        acqf_vals = acqf(
            torch.tensor(samples)
            .view(samples.shape[0], 1, samples.shape[-1])
            .double()
        )
        min_ = torch.amin(acqf_vals).item()
        max_ = torch.amax(acqf_vals).item()

        if np.abs(max_ - min_) < 1e-6:
            max_ = 1.0
            min_ = 0.0

        return min_, max_
