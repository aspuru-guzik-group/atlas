#!/usr/bin/env python

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import botorch
import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.optim import (
	optimize_acqf,
	optimize_acqf_discrete,
	optimize_acqf_mixed,
)
from olympus import ParameterVector
from olympus.campaigns import ParameterSpace

from atlas import Logger
from atlas.acquisition_functions.acqfs import (
	create_available_options,
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
from atlas.acquisition_optimizers.base_optimizer import AcquisitionOptimizer



class GradientOptimizer(AcquisitionOptimizer):
	def __init__(
		self,
		params_obj: Parameters,
		acquisition_type: str,
		acqf: AcquisitionFunction,
		known_constraints: Union[Callable, List[Callable]],
		batch_size: int,
		feas_strategy: str,
		fca_constraint: Callable,
		params: torch.Tensor,
		batched_strategy: str,
		timings_dict: Dict,
		use_reg_only=False,
		**kwargs: Any,
	):
		local_args = {
			key: val for key, val in locals().items() if key != "self"
		}
		super().__init__(**local_args)

		self.params_obj = params_obj
		self.param_space = self.params_obj.param_space
		self.problem_type = infer_problem_type(self.param_space)
		self.acquisition_type = acquisition_type
		self.acqf = acqf
		self.bounds = self.params_obj.bounds
		self.known_constraints = known_constraints
		self.batch_size = batch_size
		self.feas_strategy = feas_strategy
		self.batched_strategy=batched_strategy
		self.fca_constraint = fca_constraint
		self.use_reg_only = use_reg_only
		self.has_descriptors = self.params_obj.has_descriptors
		self._params = params
		self._mins_x = self.params_obj._mins_x
		self._maxs_x = self.params_obj._maxs_x

		self.choices_feat, self.choices_cat = None, None

		self.kind = 'gradient'

	def _optimize(self):

		best_idx = None  # only needed for the fully categorical case

		if self.acquisition_type == 'general': 
			func_dims = self.params_obj.functional_dims
			exp_func_dims = self.params_obj.exp_functional_dims

			# check to see if all functional parameters are continuous
			if all([self.param_space[ix].type=='continuous' for ix in func_dims]):
				results = self._optimize_mixed_general()

			else:
				msg = 'This is not yet implemented. Try again later!'
				Logger.log(msg, 'FATAL')

		else:   

			if self.problem_type == "fully_continuous":
				results = self._optimize_fully_continuous()
			elif self.problem_type in [
				"mixed_cat_cont",
				"mixed_disc_cont",
				"mixed_cat_disc_cont",
			]:
				results, best_idx = self._optimize_mixed()
			elif self.problem_type in [
				"fully_categorical",
				"fully_discrete",
				"mixed_cat_disc",
			]:
				results, best_idx = self._optimize_fully_categorical()

		return self.postprocess_results(results, best_idx)

	def _optimize_fully_continuous(self):

		(
			nonlinear_inequality_constraints,
			batch_initial_conditions,
			_
		) = self.gen_initial_conditions()

		results, _ = optimize_acqf(
			acq_function=self.acqf,
			bounds=self.bounds,
			num_restarts=20,
			q=self.batch_size,
			raw_samples=1000,
			nonlinear_inequality_constraints=nonlinear_inequality_constraints,
			batch_initial_conditions=batch_initial_conditions,
		)

		return results

	def _optimize_mixed(self):

		fixed_features_list = get_fixed_features_list(
			self.param_space,
			self.has_descriptors,
		)
		# TODO: add in fca constraint callable here...
		if self.feas_strategy == "fca" and not self.use_reg_only:
			# if we have feasibilty constrained acquisition, prepare only
			# the feasible options as availble choices
			fca_constraint_callable = self.fca_constraint
		else:
			fca_constraint_callable = None

		# generate initial samples
		(
			nonlinear_inequality_constraints,
			batch_initial_conditions,
			_
		) = self.gen_initial_conditions(num_restarts=30)

		self.choices_feat, self.choices_cat = create_available_options(
			self.param_space,
			self._params,
			fca_constraint_callable=fca_constraint_callable,
			known_constraint_callables=self.known_constraints,
			normalize=self.has_descriptors,
			has_descriptors=self.has_descriptors,
			mins_x=self._mins_x,
			maxs_x=self._maxs_x,
		)

		results, best_idx = self._optimize_acqf_mixed(
			acq_function=self.acqf,
			bounds=self.bounds,
			num_restarts=30,
			q=self.batch_size,
			fixed_features_list=fixed_features_list,
			cart_prod_choices=self.choices_feat.float(),
			raw_samples=800,
			batch_initial_conditions=batch_initial_conditions,
		)

		return results, best_idx


	def _optimize_mixed_general(self):
		""" function to optimize general acquisition function if we have all 
		continuous non-general/functional parameters
		"""
		functional_mask = np.logical_not(self.params_obj.exp_general_mask)
		
		func_bounds = self.bounds[:, functional_mask]

		(
			nonlinear_inequality_constraints,
			batch_initial_conditions,
			_
		) = self.gen_initial_conditions(num_restarts=30)

		func_batch_initial_conditions = batch_initial_conditions[:,:,functional_mask]

		# optimize using gradients only over the functional parameter dimensions
		results, _ = optimize_acqf(
			acq_function=self.acqf,
			num_restarts=10,
			bounds=func_bounds,
			q=self.batch_size,
			nonlinear_inequality_constraints=nonlinear_inequality_constraints,
			batch_initial_conditions=func_batch_initial_conditions,
		)

		# add back on the general dimension(s) - always use the first option (this will later be 
		# replaced and does not matter)
		X_sns = torch.empty((self.batch_size, self.params_obj.expanded_dims)).double()
		for ix, result in enumerate(results):
			X_sns[ix, functional_mask] = result
			X_sns[ix, self.params_obj.exp_general_mask] = torch.tensor(batch_initial_conditions[0, 0, self.params_obj.exp_general_mask])

		return X_sns
		

	def _optimize_fully_categorical(self):
		# need to implement the choices input, which is a
		# (num_choices * d) torch.Tensor of the possible choices
		# need to generate fully cartesian product space of possible
		# choices
		if self.feas_strategy == "fca" and not self.use_reg_only:
			# if we have feasibilty constrained acquisition, prepare only
			# the feasible options as availble choices
			fca_constraint_callable = self.fca_constraint
		else:
			fca_constraint_callable = None
		

		self.choices_feat, self.choices_cat = create_available_options(
			self.param_space,
			self._params,
			fca_constraint_callable=fca_constraint_callable,
			known_constraint_callables=self.known_constraints,
			normalize=self.has_descriptors,
			has_descriptors=self.has_descriptors,
			mins_x=self._mins_x,
			maxs_x=self._maxs_x,
		)

		results, best_idx = self._optimize_acqf_discrete(
			acq_function=self.acqf,
			q=self.batch_size,
			max_batch_size=1000,
			choices=self.choices_feat.float(),
			unique=True,
		)
		return results, best_idx

	def _optimize_acqf_discrete(
		self,
		acq_function,
		q,
		max_batch_size,
		choices,
		unique,
	):
		# this function assumes 'unique' argument is always set to True
		# strategy can be set to 'greedy' or 'sequential'
		original_choices_batched = torch.clone(choices)        
		choices_batched = choices.unsqueeze(-2)

		if q > 1:
			if self.batched_strategy == "sequential":
				candidate_list, acq_value_list = [], []
				base_X_pending = acq_function.X_pending
				for _ in range(q):
					with torch.no_grad():
						acq_values = torch.cat(
							[
								acq_function(X_)
								for X_ in choices_batched.split(max_batch_size)
							]
						)
					#best_idx = torch.argmax(acq_values)
					best_idxs_ =  torch.argsort(acq_values, descending=True)
					# print('num best idxs : ', len(best_idxs_))
					# print('num to sample : ', int(len(best_idxs_)*0.015))
					best_idx = best_idxs_[torch.randint(low=0, high=int(len(best_idxs_)*0.015), size=(1,))]
					candidate_list.append(choices_batched[best_idx])
					acq_value_list.append(acq_values[best_idx])
					# set pending points
					candidates = torch.cat(candidate_list, dim=-2)
					acq_function.set_X_pending(
						torch.cat([base_X_pending, candidates], dim=-2)
						if base_X_pending is not None
						else candidates
					)
					# need to remove choice from choice set if enforcing uniqueness
					if unique:
						choices_batched = torch.cat(
							[
								choices_batched[:best_idx],
								choices_batched[best_idx + 1 :],
							]
						)
				# Reset acq_func to previous X_pending state
				acq_function.set_X_pending(base_X_pending)
				# need to get and return the original indices of the selected candidates
				best_idxs = []
				for (
					candidate
				) in (
					candidate_list
				):  # each candidate is shape (1, num_features)
					bools = [
						torch.all(
							candidate[0] == original_choices_batched[i, :]
						)
						for i in range(original_choices_batched.shape[0])
					]
					assert bools.count(True) == 1
					best_idxs.append(np.where(bools)[0][0])

				return candidate_list, best_idxs

			elif self.batched_strategy == "greedy":
				with torch.no_grad():
					acq_values = torch.cat(
						[
							acq_function(X_)
							for X_ in choices_batched.split(max_batch_size)
						]
					)
				best_idxs = list(
					torch.argsort(acq_values, descending=True).detach().numpy()
				)[:q]

				return [choices[best_idx] for best_idx in best_idxs], best_idxs

		# otherwise we have q=1, just take the argmax acqusition value
		with torch.no_grad():
			acq_values = torch.cat(
				[
					acq_function(X_)
					for X_ in choices_batched.split(max_batch_size)
				]
			)
		best_idx = [torch.argmax(acq_values).detach()]

		return [choices[best_idx]], best_idx

	def _optimize_acqf_mixed(
		self,
		acq_function,
		bounds,
		num_restarts,
		q,
		fixed_features_list,
		cart_prod_choices,
		raw_samples=None,
		options=None,
		inequality_constraints=None,
		equality_constraints=None,
		post_processing_func=None,
		batch_initial_conditions=None,
		**kwargs,
	):
		# function inspired by botorch code
		if not fixed_features_list:
			raise ValueError("fixed_features_list must be non-empty.")

		if isinstance(
			acq_function,
			botorch.acquisition.acquisition.OneShotAcquisitionFunction,
		):
			if not hasattr(acq_function, "evaluate") and q > 1:
				raise ValueError(
					"`OneShotAcquisitionFunction`s that do not implement `evaluate` "
					"are currently not supported when `q > 1`. This is needed to "
					"compute the joint acquisition value."
				)

		# batch size of 1
		if q == 1:
			ff_candidate_list, ff_acq_value_list = [], []
			# iterate through all the fixed featutes and optimize the continuous
			# part of the parameter space
			# fixed features and cart_prod choices have the same ordering
			for fixed_features in fixed_features_list:
				candidate, acq_value = optimize_acqf(
					acq_function=acq_function,
					bounds=bounds,
					q=q,
					num_restarts=num_restarts,
					raw_samples=raw_samples,
					options=options or {},
					inequality_constraints=inequality_constraints,
					equality_constraints=equality_constraints,
					fixed_features=fixed_features,
					post_processing_func=post_processing_func,
					batch_initial_conditions=batch_initial_conditions,
					return_best_only=True,
				)
				ff_candidate_list.append(candidate)
				ff_acq_value_list.append(acq_value)

			ff_acq_values = torch.stack(ff_acq_value_list)
			best_idx = torch.argmax(ff_acq_values)

			return ff_candidate_list[best_idx], [best_idx.detach()]

		# For batch optimization with q > 1 we do not want to enumerate all n_combos^n
		# possible combinations of discrete choices. Instead, we use sequential greedy
		# optimization.
		base_X_pending = acq_function.X_pending
		candidates = torch.tensor([], device=bounds.device, dtype=bounds.dtype)

		for _ in range(q):
			candidate, acq_value = optimize_acqf_mixed(
				acq_function=acq_function,
				bounds=bounds,
				q=1,
				num_restarts=num_restarts,
				raw_samples=raw_samples,
				fixed_features_list=fixed_features_list,
				options=options or {},
				inequality_constraints=inequality_constraints,
				equality_constraints=equality_constraints,
				post_processing_func=post_processing_func,
				batch_initial_conditions=batch_initial_conditions,
			)
			candidates = torch.cat([candidates, candidate], dim=-2)
			acq_function.set_X_pending(
				torch.cat([base_X_pending, candidates], dim=-2)
				if base_X_pending is not None
				else candidates
			)

		acq_function.set_X_pending(base_X_pending)

		# compute joint acquisition value
		if isinstance(
			acq_function,
			botorch.acquisition.acquisition.OneShotAcquisitionFunction,
		):
			acq_value = acq_function.evaluate(X=candidates, bounds=bounds)
		else:
			acq_value = acq_function(candidates)

		return candidates, acq_value

	def postprocess_results(self, results, best_idx=None):
		# expects list as results

		# convert the results form torch tensor to numpy
		# results_np = np.squeeze(results.detach().numpy())
		if isinstance(results, list):
			results_torch = [torch.squeeze(res) for res in results]
		else:
			# TODO: update this
			results_torch = results

		if self.problem_type in [
			"fully_categorical",
			"fully_discrete",
			"mixed_cat_disc",
		]:
			# simple lookup
			return_params = []
			for sample_idx in range(len(results_torch)):
				sample = self.choices_cat[best_idx[sample_idx]]
				olymp_sample = {}
				for elem, param in zip(sample, [p for p in self.param_space]):
					# convert discrete parameter types to floats
					if param.type == "discrete":
						olymp_sample[param.name] = float(elem)
					else:
						olymp_sample[param.name] = elem
				return_params.append(
					ParameterVector().from_dict(olymp_sample, self.param_space)
				)

		else:
			# ['fully_continuous', 'mixed_cat_cont', 'mixed_dis_cont', 'mixed_cat_dis_cont']
			# reverse transform the inputs
			results_np = results_torch.detach().numpy()
			results_np = reverse_normalize(
				results_np, self._mins_x, self._maxs_x
			)

			return_params = []
			for sample_idx in range(results_np.shape[0]):
				# project the sample back to Olympus format
				if self.problem_type == "fully_continuous":
					cat_choice = None
				else:
					if self.acquisition_type=='general':
						cat_choice = self.param_space[self.params_obj.general_dims[0]].options[0]
					else:
						cat_choice = self.choices_cat[best_idx[sample_idx]]

				olymp_sample = {}
				idx_counter = 0
				cat_dis_idx_counter = 0
				for param_idx, param in enumerate(self.param_space):
					if param.type == "continuous":
						# if continuous, check to see if the proposed param is
						# within bounds, if not, project in
						val = results_np[sample_idx, idx_counter]
						if val > param.high:
							val = param.high
						elif val < param.low:
							val = param.low
						else:
							pass
						idx_counter += 1
					elif param.type == "categorical":
						val = cat_choice[cat_dis_idx_counter]
						if self.has_descriptors:
							idx_counter += len(param.descriptors[0])
						else:
							idx_counter += len(param.options)
						cat_dis_idx_counter += 1
					elif param.type == "discrete":
						val = float(cat_choice[cat_dis_idx_counter])
						idx_counter += 1
						cat_dis_idx_counter += 1

					olymp_sample[param.name] = val

				return_params.append(
					ParameterVector().from_dict(olymp_sample, self.param_space)
				)

		return return_params

	def dummy_constraint(self, X):
		""" dummy constraint that always returns value >= 0., i.e.
		evaluates any parameter space point as feasible
		"""
		return torch.ones(X.shape[0]).unsqueeze(-1)
