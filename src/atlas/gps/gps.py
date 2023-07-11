#!/usr/bin/env python

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import botorch
import gpytorch
import torch
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import MaternKernel, RBFKernel, ScaleKernel
from gpytorch.likelihoods import Likelihood, GaussianLikelihood
from botorch.models.kernels.downsampling import DownsamplingKernel
from botorch.models.kernels.exponential_decay import ExponentialDecayKernel
from gpytorch.models import ApproximateGP, ExactGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    UnwhitenedVariationalStrategy,
    VariationalStrategy,
)
from gpytorch.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel

from gpytorch.priors import NormalPrior


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



class TanimotoGP(ExactGP):

    _num_outputs = 1

    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor):
        """ Single task GP for molecular fingerprint inputs
        Args:
                train_x (torch.tensor): 2D tensor with training inputs
                train_y (torch.tensor): 2D tensor with training targets
        """
        super().__init__(train_x, train_y.squeeze(-1), GaussianLikelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = ScaleKernel(base_kernel=TanimotoKernel())
        self.to(train_x)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)



