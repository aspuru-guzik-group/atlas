#!/usr/bin/env python

import os
import pickle
import math
import sys
import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gpytorch
import numpy as np
import olympus
import torch

from golem import *
from golem import Golem
from olympus import ParameterVector
from olympus.campaigns import ParameterSpace
from olympus.planners import AbstractPlanner, CustomPlanner
from olympus.scalarizers import Scalarizer
from rich.progress import track

from atlas import Logger, tkwargs

from atlas.gps.gps import (
	ClassificationGPMatern,
)
from atlas.params.params import Parameters
from atlas.unknown_constraints.unknown_constraints import UnknownConstraints
from atlas.utils.planner_utils import (
	cat_param_to_feat,
	forward_normalize,
	forward_standardize,
	infer_problem_type,
	propose_randomly,
	reverse_standardize,
)
from atlas.utils.golem_utils import get_golem_dists
from atlas.known_constraints.known_constraints import (
    KnownConstraints,
	PendingExperimentConstraint,
)



class BasePlanner(CustomPlanner):
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
		molecular_params: Optional[List[int]] = None,
		is_moo: bool = False,
		value_space: Optional[ParameterSpace] = None,
		scalarizer_kind: Optional[str] = "Hypervolume",
		moo_params: Dict[str, Union[str, float, int, bool, List]] = {},
		goals: Optional[List[str]] = None,
		golem_config: Optional[Dict[str, Any]] = None,
		fidelity_params: int = None, 
        fidelities: List[float] = None,
        fixed_cost: Optional[float] = None,
		**kwargs: Any,
	):
		"""Base optimizer class containing higher-level operations.

		The golem_config argument is a dictionary with the following keys
			distributions -

		Args:

		"""
		AbstractPlanner.__init__(**locals())
		self.goal = goal
		self.feas_strategy = feas_strategy
		self.feas_param = feas_param
		self.use_min_filter = use_min_filter
		self.batch_size = batch_size
		if random_seed is None:
			self.random_seed = np.random.randint(0, int(10e6))
		else:
			self.random_seed = random_seed
		np.random.seed(self.random_seed)
		self.use_descriptors = use_descriptors
		self.num_init_design = num_init_design
		self.init_design_strategy = init_design_strategy
		self.acquisition_type = acquisition_type
		self.acquisition_optimizer_kind = acquisition_optimizer_kind
		self.vgp_iters = vgp_iters
		self.vgp_lr = vgp_lr
		self.max_jitter = max_jitter
		self.cla_threshold = cla_threshold
		
		if not known_constraints:
			self.user_known_constraints = []
		else:
			self.user_known_constraints = known_constraints

		self.general_parameters = general_parameters
		self.molecular_params = molecular_params
		self.is_moo = is_moo
		self.value_space = value_space
		self.scalarizer_kind = scalarizer_kind
		self.moo_params = moo_params
		self.goals = goals
		self.golem_config = golem_config

		self.fidelity_params = fidelity_params
		self.fidelities = fidelities
		self.fixed_cost = fixed_cost

		# initial design point trackers
		self.num_init_design_attempted = 0
		self.num_init_design_completed = 0

		# check multiobjective stuff
		if self.is_moo:
			if self.goals is None:
				message = f"You must provide individual goals for multi-objective optimization"
				Logger.log(message, "FATAL")

			if self.goal == "maximize":
				message = "Overall goal must be set to minimization for multi-objective optimization. Updating ..."
				Logger.log(message, "WARNING")
				self.goal = "minimize"

			self.scalarizer = Scalarizer(
				kind=self.scalarizer_kind,
				value_space=self.value_space,
				goals=self.goals,
				**self.moo_params,
			)

		# treat the inital design arguments
		if self.init_design_strategy == "random":
			self.init_design_planner = olympus.planners.RandomSearch(
				goal=self.goal
			)
		elif self.init_design_strategy == "sobol":
			self.init_design_planner = olympus.planners.Sobol(
				goal=self.goal, budget=self.num_init_design
			)
		elif self.init_design_strategy == "lhs":
			self.init_design_planner = olympus.planners.LatinHypercube(
				goal=self.goal, budget=self.num_init_design
			)
		else:
			message = f"Initial design strategy {self.init_design_strategy} not implemented"
			Logger.log(message, "FATAL")

		self.num_init_design_completed = 0



	def _set_param_space(self, param_space: ParameterSpace):
		"""set the Olympus parameter space (not actually really needed)"""

		# infer the problem type
		self.problem_type = infer_problem_type(self.param_space)

		# make attribute that indicates wether or not we are using descriptors for
		# categorical variables
		if self.problem_type == "fully_categorical":
			descriptors = []
			for p in self.param_space:
				if not self.use_descriptors:
					descriptors.extend([None for _ in range(len(p.options))])
				else:
					descriptors.extend(p.descriptors)
			if all(d is None for d in descriptors):
				self.has_descriptors = False
			else:
				self.has_descriptors = True

		elif self.problem_type in ["mixed_cat_cont", "mixed_cat_dis"]:
			descriptors = []
			for p in self.param_space:
				if p.type == "categorical":
					if not self.use_descriptors:
						descriptors.extend(
							[None for _ in range(len(p.options))]
						)
					else:
						descriptors.extend(p.descriptors)
			if all(d is None for d in descriptors):
				self.has_descriptors = False
			else:
				self.has_descriptors = True

		else:
			self.has_descriptors = False

		# check general parameter config
		if self.general_parameters is not None:
			# check types of general parameters
			if not all(
				[
					self.param_space[ix].type in ["discrete", "categorical"]
					for ix in self.general_parameters
				]
			):
				msg = "Only discrete- and categorical-type general parameters are currently supported"
				Logger.log(msg, "FATAL")

		# initialize golem
		if self.golem_config is not None:
			self.golem_dists = get_golem_dists(
				self.golem_config, self.param_space
			)
			if not self.golem_dists == None:
				self.golem = Golem(
					forest_type="dt",
					ntrees=50,
					goal="min",
					verbose=True,
				)  # always minimization goal
			else:
				self.golem = None
		else:
			self.golem_dists = None
			self.golem = None

		# deal with user-level known constraints
		self.known_constraints = KnownConstraints(
			self.user_known_constraints,
			param_space, 
			self.has_descriptors,
			self.compositional_params,
			self.permutation_params,
			self.batch_constrained_params,

		)

	def build_train_classification_gp(
			self, train_x: torch.Tensor, train_y: torch.Tensor
		) -> Tuple[
			gpytorch.models.ApproximateGP, gpytorch.likelihoods.BernoulliLikelihood
		]:
			"""build the GP classification model and likelihood
			and train the model
			"""
		
			model = ClassificationGPMatern(train_x, train_y)
			likelihood = gpytorch.likelihoods.BernoulliLikelihood()

			model, likelihood = self.train_vgp(model, likelihood, train_x, train_y)

			return model, likelihood

	def train_vgp(
		self,
		model: gpytorch.models.ApproximateGP,
		likelihood: gpytorch.likelihoods.BernoulliLikelihood,
		train_x: torch.Tensor,
		train_y: torch.Tensor,
		cross_validate: str = True,
	) -> Tuple[
		gpytorch.models.ApproximateGP, gpytorch.likelihoods.BernoulliLikelihood
	]:


		model.train()
		likelihood.train()
		optimizer = torch.optim.Adam(model.parameters(), lr=self.vgp_lr)

		mll = gpytorch.mlls.VariationalELBO(likelihood, model, train_y.numel())

		# cross-validation parameters
		num_folds = 3
		min_obs = 10
		es_patience = 300
		count_after_iter = 50


		if cross_validate and train_y.shape[0] >= min_obs:

			idx_0 = torch.where(train_y==0.)[0]
			idx_1 = torch.where(train_y==1.)[0]
			num_0, num_1 = len(idx_0), len(idx_1)

			if num_0 >= num_1:
				num_tiles = num_0//num_1
				y_infl = torch.tile(train_y[idx_1], dims=(num_tiles,))
				X_infl = torch.tile(train_x[idx_1], dims=(num_tiles,1))
			else:
				num_tiles = num_1//num_0
				y_infl = torch.tile(train_y[idx_0], dims=(num_tiles,))
				X_infl = torch.tile(train_x[idx_0], dims=(num_tiles,1))

			train_y_infl = torch.cat([train_y, y_infl])
			train_x_infl = torch.cat([train_x, X_infl])


			folds = []

			fold_size = math.ceil(train_y_infl.shape[0]/num_folds)
			indices = torch.randperm(train_y_infl.shape[0])#torch.arange(train_y_infl.shape[0])

			# cross validation procedure
			for fold_ix in range(num_folds):

				# create fold data
				train_x_fold, train_y_fold = train_x_infl[indices[fold_size:], :], train_y_infl[indices[fold_size:]]
				valid_x_fold, valid_y_fold = train_x_infl[indices[:fold_size], :], train_y_infl[indices[:fold_size]]

				# create new model and likelihood for fold
				model_fold = ClassificationGPMatern(train_x_fold, train_y_fold)
				likelihood_fold = gpytorch.likelihoods.BernoulliLikelihood()
				optimizer_fold = torch.optim.Adam(model_fold.parameters(), lr=self.vgp_lr)
				mll_fold = gpytorch.mlls.VariationalELBO(likelihood_fold, model_fold, train_y_fold.numel())

				model_fold.train()
				likelihood_fold.train()

				folds.append({
					'model': model_fold, 'likelihood': likelihood_fold,
					'optimizer': optimizer_fold, 'mll': mll_fold,
					'train_x': train_x_fold, 'train_y': train_y_fold,
					'valid_x': valid_x_fold, 'valid_y': valid_y_fold,
				})

				indices = torch.roll(indices, fold_size)

			num_epochs_fold = []

			# train model on folds with early stopping
			for fold_ix, fold in enumerate(folds):
				train_losses, valid_losses = [], []
				start_time = time.time()
				model_fold, likelihood_fold = fold['model'], fold['likelihood']
				optimizer_fold, mll_fold = fold['optimizer'], fold['mll']
				with gpytorch.settings.cholesky_jitter(self.max_jitter):

					best_loss = 1.e8
					patience_iter_ = 0
					for iter_ in track(
						range(self.vgp_iters), description=f'Training variational GP on fold {fold_ix+1}/{num_folds}'
					):
						optimizer_fold.zero_grad()
						train_pred = model_fold(fold['train_x'])
						valid_pred = model_fold(fold['valid_x'])
						train_loss = -mll_fold(train_pred, fold['train_y'])
						valid_loss = -mll_fold(valid_pred, fold['valid_y'])

						if iter_ > count_after_iter:
							if valid_loss < best_loss:
								best_loss = valid_loss
								patience_iter_ = 0  # reset patience
							else:
								patience_iter_ += 1 # increment patience

							if patience_iter_ > es_patience:
								break  # early stopping criteria met

						train_losses.append(train_loss)
						valid_losses.append(valid_loss)

						train_loss.backward()
						optimizer_fold.step()


				vgp_train_time = time.time() - start_time
				msg = f" Classification surrogate VGP trained in {round(vgp_train_time,3)} sec ({iter_} epochs)\t Loss : {round(train_loss.item(), 3)} "
				Logger.log(msg, "INFO")
				# TODO: update this
				num_epochs_fold.append(iter_)

			# train model on all observations
			num_iters_full = int(np.mean(num_epochs_fold))
			with gpytorch.settings.cholesky_jitter(self.max_jitter):
				for iter_ in track(
					range(num_iters_full), description=f"Training variational GP on all observations ..."
				):
					optimizer.zero_grad()
					output = model(train_x)
					loss = -mll(output, train_y)
					loss.backward()
					optimizer.step()
			vgp_train_time = time.time() - start_time
			msg = f" Classification surrogate VGP trained in {round(vgp_train_time,3)} sec ({num_iters_full} epochs)\t Loss : {round(loss.item(), 3)} "
			Logger.log(msg, "INFO")

		return model, likelihood

	def build_train_data(
			self, 
			return_scaled_input:bool=True,
		) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
		"""build the training dataset at each iteration"""
		if self.is_moo:
			# parameters should be the same for each objective
			# nans should be in the same locations for each objective
			feas_ix = np.where(~np.isnan(self._values[:, 0]))[0]
			# generate the classification dataset
			params_cla = self._params.copy()
			values_cla = np.where(
				~np.isnan(self._values[:, 0]), 0.0, self._values[:, 0]
			)
			train_y_cla = np.where(np.isnan(values_cla), 1.0, values_cla)
			# generate the regression dataset
			params_reg = self._params[feas_ix].reshape(-1, 1)
			train_y_reg = self._values[
				feas_ix, :
			]  # (num_feas_observations, num_objectives)
			# scalarize the data
			train_y_reg = self.scalarizer.scalarize(train_y_reg).reshape(
				-1, 1
			)  # (num_feas_observations, 1)

		else:
			feas_ix = np.where(~np.isnan(self._values))[0]
			# generate the classification dataset
			params_cla = self._params.copy()
			values_cla = np.where(~np.isnan(self._values), 0.0, self._values)
			train_y_cla = np.where(np.isnan(values_cla), 1.0, values_cla)

			# generate the regression dataset
			params_reg = self._params[feas_ix].reshape(-1, 1)
			train_y_reg = self._values[feas_ix].reshape(-1, 1)

		train_x_cla, train_x_reg = [], []

		# adapt the data from olympus form to torch tensors
		for ix in range(self._values.shape[0]):
			sample_x = []
			for param_ix, (space_true, element) in enumerate(
				zip(self.param_space, params_cla[ix])
			):
				if self.param_space[param_ix].type == "categorical":
					feat = cat_param_to_feat(
						space_true,
						element,
						has_descriptors=self.has_descriptors,
					)
					sample_x.extend(feat)
				else:
					sample_x.append(float(element))
			train_x_cla.append(sample_x)
			if ix in feas_ix:
				train_x_reg.append(sample_x)

		train_x_cla, train_x_reg = np.array(train_x_cla), np.array(train_x_reg)

		# if we are using Golem, fit Golem to current regression training data,
		# and replace data with its predictions
		if self.golem is not None:
			self.golem.fit(X=train_x_reg, y=train_y_reg.flatten())
			train_y_reg = self.golem.predict(
				X=train_x_reg,
				distributions=self.golem_dists,
			).reshape(-1, 1)

		# scale the training data - normalize inputs and standardize outputs
		self._means_y, self._stds_y = np.mean(train_y_reg, axis=0), np.std(
			train_y_reg, axis=0
		)
		self._stds_y = np.where(self._stds_y == 0.0, 1.0, self._stds_y)

		if (
			self.problem_type == "fully_categorical"
			and not self.has_descriptors
		):
			# we dont scale the parameters if we have a fully one-hot-encoded representation
			pass
		else:
			# scale the parameters
			train_x_cla = forward_normalize(
				train_x_cla, self.params_obj._mins_x, self.params_obj._maxs_x
			)
			train_x_reg = forward_normalize(
				train_x_reg, self.params_obj._mins_x, self.params_obj._maxs_x
			)

		# always forward transform the objectives for the regression problem
		train_y_reg = forward_standardize(
			train_y_reg, self._means_y, self._stds_y
		)

		# convert to torch tensors and return
		return (
			torch.tensor(train_x_cla, **tkwargs),
			torch.tensor(train_y_cla, **tkwargs).squeeze(),
			torch.tensor(train_x_reg, **tkwargs),
			torch.tensor(train_y_reg, **tkwargs),
		)

	def reg_surrogate(
		self,
		X: torch.Tensor,
		return_np: bool = False,
	) -> Tuple[
		Union[torch.Tensor, np.ndarray], Union[torch.Tensor, np.ndarray]
	]:
		"""make prediction using regression surrogate model

		Args:
				X (np.ndarray or list): 2d numpy array or nested list with input parameters
		"""

		if not hasattr(self, "reg_model"):
			msg = "Optimizer does not yet have regression surrogate model"
			Logger.log(msg, "FATAL")

		X_proc = []
		# adapt the data from olympus form to torch tensors
		for ix in range(len(X)):
			sample_x = []
			for param_ix, (space_true, element) in enumerate(
				zip(self.param_space, X[ix])
			):
				if self.param_space[param_ix].type == "categorical":
					feat = cat_param_to_feat(
						space_true,
						element,
						has_descriptors=self.has_descriptors,
					)
					sample_x.extend(feat)
				else:
					sample_x.append(float(element))
			X_proc.append(sample_x)

		X_proc = torch.tensor(np.array(X_proc), **tkwargs)

		if (
			self.problem_type == "fully_categorical"
			and not self.has_descriptors
		):
			# we dont scale the parameters if we have a fully one-hot-encoded representation
			pass
		else:
			# scale the parameters
			X_proc = forward_normalize(
				X_proc, self.params_obj._mins_x, self.params_obj._maxs_x
			)

		posterior = self.reg_model.posterior(X=X_proc)
		pred_mu, pred_sigma = posterior.mean.detach(), torch.sqrt(
			posterior.variance.detach()
		)

		# reverse scale the predictions
		pred_mu = reverse_standardize(pred_mu, self._means_y, self._stds_y)

		if self.goal == "maximize":
			pred_mu = -pred_mu

		if return_np:
			pred_mu, pred_sigma = pred_mu.numpy(), pred_sigma.numpy()

		return pred_mu, pred_sigma

	def cla_surrogate(
		self,
		X: torch.Tensor,
		return_np: bool = False,
		normalize: bool = True,
	) -> Union[torch.Tensor, np.ndarray]:

		if not hasattr(self, "cla_model"):
			msg = "Optimizer does not yet have classification surrogate model"
			Logger.log(msg, "FATAL")

		X_proc = []
		# adapt the data from olympus form to torch tensors
		for ix in range(len(X)):
			sample_x = []
			for param_ix, (space_true, element) in enumerate(
				zip(self.param_space, X[ix])
			):
				if self.param_space[param_ix].type == "categorical":
					feat = cat_param_to_feat(
						space_true,
						element,
						has_descriptors=self.has_descriptors,
					)
					sample_x.extend(feat)
				else:
					sample_x.append(float(element))
			X_proc.append(sample_x)

		X_proc = torch.tensor(np.array(X_proc, **tkwargs))

		if (
			self.problem_type == "fully_categorical"
			and not self.has_descriptors
		):
			# we dont scale the parameters if we have a fully one-hot-encoded representation
			pass
		else:
			# scale the parameters
			X_proc = forward_normalize(
				X_proc, self.params_obj._mins_x, self.params_obj._maxs_x
			)

		likelihood = self.cla_likelihood(self.cla_model(X_proc.float()))
		mean = likelihood.mean.detach()
		mean = mean.view(mean.shape[0], 1)
		# mean = 1.-mean.view(mean.shape[0],1) # switch from p_feas to p_infeas
		if normalize:
			_max = torch.amax(mean, axis=0)
			_min = torch.amin(mean, axis=0)
			mean = (mean - _min) / (_max - _min)

		if return_np:
			mean = mean.numpy()

		return mean

	def acquisition_function(
		self,
		X: torch.Tensor,
		return_np: bool = True,
		normalize: bool = True,
		unconstrained: bool = False,
	) -> Union[torch.Tensor, np.ndarray]:

		X_proc = []
		# adapt the data from olympus form to torch tensors
		for ix in range(len(X)):
			sample_x = []
			for param_ix, (space_true, element) in enumerate(
				zip(self.param_space, X[ix])
			):
				if self.param_space[param_ix].type == "categorical":
					feat = cat_param_to_feat(
						space_true,
						element,
						has_descriptors=self.has_descriptors,
					)
					sample_x.extend(feat)
				else:
					sample_x.append(float(element))
			X_proc.append(sample_x)

		X_proc = torch.tensor(np.array(X_proc), **tkwargs)

		if (
			self.problem_type == "fully_categorical"
			and not self.has_descriptors
		):
			# we dont scale the parameters if we have a fully one-hot-encoded representation
			pass
		else:
			# scale the parameters
			X_proc = forward_normalize(
				X_proc, self.params_obj._mins_x, self.params_obj._maxs_x
			)

		X_proc = X_proc.view(X_proc.shape[0], 1, X_proc.shape[-1])
		if unconstrained:
			acqf_vals = self.acqf.forward_unconstrained(X_proc).detach()
		else:
			acqf_vals = self.acqf(X_proc).detach()

		acqf_vals = acqf_vals.view(acqf_vals.shape[0], 1)

		if normalize:
			_max = torch.amax(acqf_vals, axis=0)
			_min = torch.amin(acqf_vals, axis=0)
			acqf_vals = (acqf_vals - _min) / (_max - _min)

		if return_np:
			acqf_vals = acqf_vals.numpy()

		return acqf_vals

	def _tell(self, observations: olympus.campaigns.observations.Observations):
		"""unpack the current observations from Olympus
		Args:
				observations (obj): Olympus campaign observations object
		"""

		# elif type(observations) == olympus.campaigns.observations.Observations:
		self._params = observations.get_params(
			as_array=True
		)  # string encodings of categorical params
		self._values = observations.get_values(
			as_array=True, opposite=self.flip_measurements
		)

		# make values 2d if they are not already
		if len(np.array(self._values).shape) == 1:
			self._values = np.array(self._values).reshape(-1, 1)

		# generate Parameters object
		self.params_obj = Parameters(
			olympus_param_space=self.param_space,
			observations=observations,
			has_descriptors=self.has_descriptors,
			general_parameters=self.general_parameters,
		)

		# generate unknown constraints object
		self.unknown_constraints = UnknownConstraints(
			params_obj=self.params_obj,
			feas_strategy=self.feas_strategy,
			feas_param=self.feas_param,
		)


	def fca_constraint(self, X: torch.Tensor) -> torch.Tensor:
		"""Each callable is expected to take a `(num_restarts) x q x d`-dim tensor as an
				input and return a `(num_restarts) x q`-dim tensor with the constraint
				values. The constraints will later be passed to SLSQP. You need to pass in
				`batch_initial_conditions` in this case. Using non-linear inequality
				constraints also requires that `batch_limit` is set to 1, which will be
				done automatically if not specified in `options`.
				>= 0 is a feasible point
				<  0 is an infeasible point
		Args:
				X (torch.tensor): 2d torch tensor with constraint values
		"""
		# handle the various potential input tensor sizes (this function can be called from
		# several places, including inside botorch)
		# TODO: this is pretty messy, consider cleaning up
		if len(X.size()) == 3:
			X = X.squeeze(1)
		if len(X.size()) == 1:
			X = X.view(1, X.shape[0])
		# squeeze the middle q dimension
		# this expression is >= 0 for a feasible point, < 0 for an infeasible point
		# p_feas should be 1 - P(infeasible|X) which is returned by the classifier
		with gpytorch.settings.cholesky_jitter(1e-1):
			p_infeas = (
				self.cla_likelihood(self.cla_model(X.float()))
				.mean.unsqueeze(-1)
				.double()
			)
			# convert to range of values expected by botorch/gpytorch acqusition optimizer
			constraint_val = (1. - p_infeas) - self.fca_cutoff

		return constraint_val
	

	def initial_design(self) -> List[ParameterVector]:
		''' Acquire initial design samples using one of several supported strategues
		'''
		num_init_remain = self.num_init_design - len(self._values)
		num_init_batches = math.ceil(self.num_init_design/self.batch_size)
		init_batch_num = int((len(self._values)/self.batch_size)+1)

		if num_init_remain > 0: 
			if num_init_remain % self.batch_size == 0:
				num_gen = self.batch_size
				Logger.log(f'Generating {num_gen} initial design points (batch {init_batch_num}/{num_init_batches})', 'INFO')
			else:
				num_gen = num_init_remain
				Logger.log(
					f'Remaining initial design points do not match batch size. Generating {num_gen} points (batch {init_batch_num}/{num_init_batches})',
					'WARNING',
				)
		elif num_init_remain <= 0: 
			# we have all nan values
			assert np.all(np.isnan(self._values))
			num_gen = self.batch_size
			Logger.log(f'Found all NaN observations after initial design. Generating {num_gen} additional points', 'WARNING')
		else:
			Logger.log('Something is wrong', 'FATAL')

		# set parameter space for the initial design planner
		self.init_design_planner.set_param_space(self.param_space)
		return_params = []
		while len(return_params) < num_gen:
			# TODO: this is pretty sloppy - consider standardizing this
			if self.init_design_strategy == "random":
				self.init_design_planner._tell(iteration=self.num_init_design_attempted)
			else:
				self.init_design_planner.tell()
			rec_params = self.init_design_planner.ask()
			if isinstance(rec_params, list):
				rec_params = rec_params[0]
			elif isinstance(rec_params, ParameterVector):
				pass
			else:
				raise TypeError
			
			# check to see if the recommended parameters satisfy the 
			# known constraints, if there are any
			if self.known_constraints is not None:
				# we have some known constraints 
				kc_res = [kc(rec_params.to_array()) for kc in self.known_constraints]
				if all(kc_res):
					return_params.append(rec_params)
					self.num_init_design_completed += 1  # batch_size always 1 for init design planner
			else:
				return_params.append(rec_params)
				self.num_init_design_completed += 1

			self.num_init_design_attempted += 1

		# apply compositional constraint to dependent param is required
		# TODO: make this methods in the known constraints class
		if self.known_constraints.has_compositional_constraint:
			contrained_results = []
			for pvec in return_params:
				constrained_pvec = deepcopy(pvec)
				sum_params = np.sum(
					[float(pvec[name]) for name in self.known_constraints.compositional_constraint_param_names[:-1]]
				)
				assert sum_params <= 1.
				# update dependent parameter
				constrained_pvec[self.known_constraints.compositional_constraint_param_names[-1]] = 1. - sum_params 
				contrained_results.append(constrained_pvec)

			return_params = contrained_results

		if self.known_constraints.has_batch_constraint:
			constrained_results = []
			for pvec in return_params:
				constrained_pvec = deepcopy(pvec)
				for constrained_param_name in self.known_constraints.batch_constrained_param_names:
					constrained_pvec[constrained_param_name] = return_params[0][constrained_param_name]
				constrained_results.append(constrained_pvec)
			
			return_params = constrained_results

		return return_params
	

	def get_cla_surr_min_max(self, num_samples:int=5000) -> Tuple[int, int]:
		""" estimate the max and min of the classification surrogate
		"""

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

		X = torch.tensor(samples, **tkwargs)

		likelihood = self.cla_likelihood(self.cla_model(X.float()))
		mean = 1.-likelihood.mean.detach() # convert p_infeas to p_feas
		mean = mean.view(mean.shape[0], 1)

		min_  = torch.amin(mean).item()
		max_ = torch.amax(mean).item()

		return min_, max_
	
	def set_pending_experiments(self, pending_experiments):
		""" set pending experiments by generating an additional known constraint
		callable
		"""
		# remove old pending experiment constraint (would be last list element)
		if self.known_constraints.has_pending_experiment_constraint:
			_ = self.known_constraints.known_constraints.pop()
			self.known_constraints.has_pending_experiment_constraint = False

		# instantiate new pending experiment constraint and add to list
		self.known_constraints.known_constraints.append(
			PendingExperimentConstraint(
				pending_experiments=pending_experiments, param_space=self.param_space,
			)
		)
		self.known_constraints.has_pending_experiment_constraint = True


	def remove_pending_experiments(self):
		""" remove pending experiments from known constraints
		"""
		# remove old pending experiment constraint (would be last list element)
		if self.known_constraints.has_pending_experiment_constraint:
			Logger.log('Removing pending experiment constriant', 'WARNING')
			_ = self.known_constraints.known_constraints.pop()
			self.known_constraints.has_pending_experiment_constraint = False
		else:
			Logger.log('No pending experiments found to be removed', 'WARNING')
