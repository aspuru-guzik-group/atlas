#!/usr/bin/env python

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import botorch
import gpytorch
import torch 
from torch.nn import ModuleList
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import Likelihood, GaussianLikelihood, LikelihoodList
from botorch.models.kernels.downsampling import DownsamplingKernel
from botorch.models.kernels.exponential_decay import ExponentialDecayKernel
from botorch.utils.datasets import SupervisedDataset
from gpytorch.models import ApproximateGP, ExactGP, GP
from gpytorch.priors import GammaPrior
from gpytorch.lazy import PsdSumLazyTensor
from gpytorch.constraints import GreaterThan
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    UnwhitenedVariationalStrategy,
    VariationalStrategy,
)
from botorch.models import SingleTaskGP

from gpytorch.priors import NormalPrior

from atlas.gps.kernels import TanimotoKernel


class ClassificationGPMatern(ApproximateGP):
    """Variational GP for binary classification. Produces a latent distribution,
    which is multivariate Gaussian, which is intended to be transfromed to a Bernoulli
    likelihood to give the binary class probabilities,
    i.e. prob = BernoulliLikelihood(model(inputs))
    Args:
            train_x (torch.tensor): 2D tensor with training inputs
    """

    def __init__(self, train_x, train_y):
        self.train_y = train_y
        variational_distribution = CholeskyVariationalDistribution(
            train_x.size(0)
        )
        # using this variational strategy because we use directly the training points
        # as inducing points for the GP
        variational_strategy = UnwhitenedVariationalStrategy(
            self,
            train_x,
            variational_distribution,
            learn_inducing_locations=False,
        )
        super(ClassificationGPMatern, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        lambda_=100.
        #scale=1.*np.exp(-train_y.shape[0]/lambda_)
        # self.covar_module = ScaleKernel(MaternKernel(
        #     lengthscale_prior=NormalPrior(loc=0., scale=scale)
        #     ))  # RBFKernel())
        self.covar_module = ScaleKernel(MaternKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        latent_output = MultivariateNormal(mean_x, covar_x)
        return latent_output


class CategoricalSingleTaskGP(ExactGP, GPyTorchModel):

    # meta-data for BoTorch
    _num_outputs = 1

    def __init__(self, train_x, train_y):
        """Single task GP with a categorical kernel based on the Hamming distance.
        Kernel computes  k(x, x') = -exp(dist(x, x')/l), where dist(x, x') = 0 if
        x == x' and 1 if x != x', and l is a hyperparameter. We use automatic relevance
        detection for each categorical dimension
        Note: this kernel is NOT differentiable with respect to the inputs (we dont
        really care, as we do not optimize the acqf for categorical spaces using gradients)
        Args:
                train_x (torch.tensor): 2D tensor with training inputs
                train_y (torch.tensor): 2D tensor with training targets
        """
        super().__init__(train_x, train_y.squeeze(-1), GaussianLikelihood())
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            base_kernel=botorch.models.kernels.categorical.CategoricalKernel(
                ard_num_dims=train_y.size(
                    -1
                )  # ARD for all categorical dimensions
            )
        )
        self.to(train_x)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


# class MultiFidelityGP(ExactGP):
#     """ Single task multi-fidelity GP model
#     """

#     _num_outputs = 1 

#     def __init__(
#         self,
#         train_x: torch.Tensor,
#         train_y: torch.Tensor,
#         data_fidelity: Optional[int] = None, 
#         nu: float = 0.2,
#         likelihood: Optional[Likelihood] = GaussianLikelihood,

#     ):
        
#         self.train_x = train_x
#         self.train_y = train_y
#         self.data_fidelity = data_fidelity
#         self.nu = nu
#         self.likelihood = likelihood

#         if not self.data_fidelity:
#             Logger.log('You must use at least one data fidelity parameter to use the MultiFidelityGP', 'FATAL')

#         # create the covariance module and subset batch dict


class DKTGP(GP, GPyTorchModel):

    # meta-data for botorch
    _num_outputs = 1

    def __init__(self, model, context_x, context_y):
        super().__init__()
        self.model = model
        self.context_x = context_x.float()
        self.context_y = context_y.float()

    def forward(self, x):
        """
        x shape  (# proposals, q_batch_size, # params)
        mean shape (# proposals, # params)
        covar shape (# proposals, q_batch_size, # params)
        """
        x = x.float()
        _, __, likelihood = self.model.forward(
            self.context_x, self.context_y, x
        )
        mean = likelihood.mean
        covar = likelihood.lazy_covariance_matrix

        return gpytorch.distributions.MultivariateNormal(mean, covar)



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


class TanimotoGP(ExactGP, GPyTorchModel):

    _num_outputs = 1

    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor):
        """ Single task GP for molecular fingerprint inputs
        Args:
                train_x (torch.tensor): 2D tensor with training inputs
                train_y (torch.tensor): 2D tensor with training targets
        """
        super().__init__(train_x, train_y.squeeze(-1), GaussianLikelihood())
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(base_kernel=TanimotoKernel())
        self.to(train_x)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)




class MixedTanimotoSingleTaskGP(SingleTaskGP):
    """ Supports mixed molecular-continuous/discrete parameter spaces
    where the molecular parameter options are represented using 
    Morgan fingerprints

    Similar kernel to MixedSingleTaskGP from BoTorch, but instead of 
    using CategoricalKernel based on Hamming distance it uses TanimotoKernel
    producing a regular kernel of the form

    K((x1, c1), (x2, c2)) =
            K_cont_1(x1, x2) + K_cat_1(c1, c2) +
            K_cont_2(x1, x2) * K_cat_2(c1, c2)


    inspired by:
    https://botorch.org/api/_modules/botorch/models/gp_regression_mixed.html#MixedSingleTaskGP
    """

    _num_outputs = 1

    def __init__(
        self, 
        train_x: torch.Tensor, 
        train_y: torch.Tensor,
        molecular_dims: List[int],  
    ):
        if len(molecular_dims) == 0:
            msg = 'You must define at least one molecular dimension to use the MixedTanimotoSingleTaskGP'
            Logger.log(msg, 'FATAL')

        _, aug_batch_shape = self.get_batch_dimensions(train_X=train_X, train_Y=train_Y)

        def cont_kernel_factory(
            batch_shape: torch.Size,
            ard_num_dims: int,
            active_dims: List[int],
        ) -> MaternKernel:
            return MaternKernel(
                nu=2.5,
                batch_shape=batch_shape,
                ard_num_dims=ard_num_dims,
                active_dims=active_dims,
                lengthscale_constraint=GreaterThan(1e-04),
            )

        # generate likelihood
        min_noise = 1e-5 if train_X.dtype == torch.float else 1e-6
        likelihood = GaussianLikelihood(
            batch_shape=aug_batch_shape,
            noise_constraint=GreaterThan(
                min_noise, transform=None, initial_value=1e-3
            ),
            noise_prior=GammaPrior(0.9, 10.0),
        )

        d = train_X.shape[-1]
        cat_dims = normalize_indices(indices=cat_dims, d=d)
        ord_dims = sorted(set(range(d)) - set(cat_dims))
        if len(ord_dims) == 0:
            covar_module = ScaleKernel(
                CategoricalKernel(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(cat_dims),
                    lengthscale_constraint=GreaterThan(1e-06),
                )
            )
        else:
            sum_kernel = ScaleKernel(
                cont_kernel_factory(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(ord_dims),
                    active_dims=ord_dims,
                )
                + ScaleKernel(
                    CategoricalKernel(
                        batch_shape=aug_batch_shape,
                        ard_num_dims=len(cat_dims),
                        active_dims=cat_dims,
                        lengthscale_constraint=GreaterThan(1e-06),
                    )
                )
            )
            prod_kernel = ScaleKernel(
                cont_kernel_factory(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(ord_dims),
                    active_dims=ord_dims,
                )
                * CategoricalKernel(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(cat_dims),
                    active_dims=cat_dims,
                    lengthscale_constraint=GreaterThan(1e-06),
                )
            )
            covar_module = sum_kernel + prod_kernel
        super().__init__(
            train_X=train_X,
            train_Y=train_Y,
            likelihood=likelihood,
            covar_module=covar_module,
            outcome_transform=None,
            input_transform=None,
        )


    @classmethod
    def construct_inputs(
        cls,
        training_data: SupervisedDataset,
        categorical_features: List[int],
        likelihood: Optional[Likelihood] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        r"""Construct `Model` keyword arguments from a dict of `BotorchDataset`.

        Args:
            training_data: A `SupervisedDataset` containing the training data.
            categorical_features: Column indices of categorical features.
            likelihood: Optional likelihood used to constuct the model.
        """
        return {
            **super().construct_inputs(training_data=training_data, **kwargs),
            "cat_dims": categorical_features,
            "likelihood": likelihood,
        }


        
