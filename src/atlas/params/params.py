#!/usr/bin/env python

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from olympus.objects import (
	ParameterCategorical,
	ParameterContinuous,
	ParameterDiscrete,
	ParameterVector,
)
from olympus.campaigns import Campaign, ParameterSpace, Observations

from atlas import Logger
from atlas.utils.planner_utils import (
	cat_param_to_feat,
	forward_normalize,
)



class Parameters():

	def __init__(
		self,
		olympus_param_space: ParameterSpace,
		observations: Observations,
		has_descriptors: bool,
		general_parameters: Optional[List[int]] = None,
	) -> None:

		self.param_space = olympus_param_space
		self.has_descriptors = has_descriptors
		# Olympus observations
		self.olympus = observations.get_params()
		observations._construct_param_vectors()
		self.param_vectors = list(observations._params_as_vectors)

		if general_parameters is not None:
			self.general_dims = general_parameters
		else:
			self.general_dims = []

		# dimensions of expanded representations
		(
			self.exp_cont_dims, self.exp_disc_dims, self.exp_cat_dims, self.exp_general_dims,
			) = self._get_expanded_dims()

		if len(observations.get_params())>0:
			# get expanded and indexed raw representations
			self.expanded_raw, self.indexed_raw = self._get_expanded_indexed()

			# get min max of the expanded parameters
			self._mins_x = np.amin(self.expanded_raw, axis=0)
			self._maxs_x = np.amax(self.expanded_raw, axis=0)

			# scale the expanded representation
			self.expanded_scaled = forward_normalize(self.expanded_raw,self._mins_x,self._maxs_x)

			# scale the indexed representation (only the continuous dims)
			disc_cat_indexed = self.indexed_raw[:,self.disc_dims+self.cat_dims]
			indexed_masked = self.indexed_raw.copy()
			indexed_masked[:, self.disc_dims+self.cat_dims] = 1.

			self.indexed_scaled = forward_normalize(
				indexed_masked , np.amin(indexed_masked,axis=0), np.amax(indexed_masked,axis=0)
			)
			self.indexed_scaled[:, self.disc_dims+self.cat_dims] = disc_cat_indexed

			self.bounds = self.get_bounds()

		else:
			pass


	@property
	def num_params(self):
		return len(self.param_space)

	@property
	def expanded_dims(self):
		return self.expanded_raw.shape[1]


	@property
	def general_mask(self):
		return [True if ix in self.general_dims else False for ix in range(len(self.param_space))]

	@property
	def exp_general_mask(self):
		return [True if ix in self.exp_general_dims else False for ix in range(self.expanded_raw.shape[1])]
	
	@property
	def functional_dims(self):
		return [ix for ix in range(len(self.param_space)) if ix not in self.general_dims]
	
	@property
	def exp_functional_dims(self):
		return [ix for ix in range(self.expanded_raw.shape[1]) if ix not in self.exp_general_dims]
	


	@property
	def cont_dims(self):
		return [
			ix for ix, param in enumerate(self.param_space) if param.type=='continuous'
		]

	@property
	def disc_dims(self):
		return [
			ix for ix, param in enumerate(self.param_space) if param.type=='discrete'
		]

	@property
	def cat_dims(self):
		return [
			ix for ix, param in enumerate(self.param_space) if param.type=='categorical'
		]

	@property
	def cont_mask(self):
		return [True if ix in self.cont_dims else False for ix in range(len(self.param_space))]

	@property
	def disc_mask(self):
		return [True if ix in self.disc_dims else False for ix in range(len(self.param_space))]

	@property
	def cat_mask(self):
		return [True if ix in self.cat_dims else False for ix in range(len(self.param_space))]

	@property
	def exp_cont_mask(self):
		return [True if ix in self.exp_cont_dims else False for ix in range(self.expanded_raw.shape[1])]

	@property
	def exp_disc_mask(self):
		return [True if ix in self.exp_disc_dims else False for ix in range(self.expanded_raw.shape[1])]

	@property
	def exp_cat_mask(self):
		return [True if ix in self.exp_cat_dims else False for ix in range(self.expanded_raw.shape[1])]



	def _get_expanded_indexed(self):
		expanded, indexed = [], []
		for sample_ix, sample in enumerate(self.olympus):
			exp, ind = [], []
			counter = 0
			for elem, param in zip(sample, self.param_space):
				if param.type == 'continuous':
					exp.append(float(elem))
					ind.append(float(elem))
					counter+=1
				elif param.type == 'discrete':
					exp.append(float(elem))
					ind.append(
						param.options.index(float(elem))
					)
					counter+=1
				elif param.type == 'categorical':
					exp.extend(
						cat_param_to_feat(
							param, elem, self.has_descriptors,
						)
					)
					ind.append(param.options.index(elem))
					if self.has_descriptors:
						counter+=len(param.descriptors[0])
					else:
						counter+=len(param.options)
			expanded.append(exp)
			indexed.append(ind)

		return np.array(expanded), np.array(indexed)


	def _get_expanded_dims(self) -> Tuple[Any]:
		dim = 0
		exp_cont_dims, exp_disc_dims, exp_cat_dims = [],[],[]
		exp_general_dims = []
		for param_ix, param in enumerate(self.param_space):
			if param.type == 'continuous':
				exp_cont_dims.append(dim)
				if param_ix in self.general_dims:
					exp_general_dims.append(dim)
				dim+=1
			elif param.type == 'discrete':
				exp_disc_dims.append(dim)
				if param_ix in self.general_dims:
					exp_general_dims.append(dim)
				dim+=1
			elif param.type == 'categorical':
				if self.has_descriptors:
					dims = np.arange(dim, dim + len(param.descriptors[0]))
				else:
					dims = np.arange(dim, dim + len(param.options))
				if param_ix in self.general_dims:
					exp_general_dims.extend(dims)
				dim+=len(dims)
		return (
			exp_cont_dims, exp_disc_dims, exp_cat_dims, exp_general_dims
		)


	def get_bounds(self) -> torch.Tensor:
		"""returns scaled bounds of the parameter space
		torch tensor of shape (# dims, 2) (low and upper bounds)
		"""
		bounds = []
		idx_counter = 0
		for param_ix, param in enumerate(self.param_space):
			if param.type == "continuous":
				b = np.array([param.low, param.high])
				b = (b - self._mins_x[idx_counter]) / (
					self._maxs_x[idx_counter] - self._mins_x[idx_counter]
				)
				bounds.append(b)
				idx_counter += 1
			elif param.type == "discrete":
				b = np.array([np.amin(param.options), np.amax(param.options)])
				b = (b - self._mins_x[idx_counter]) / (
					self._maxs_x[idx_counter] - self._mins_x[idx_counter]
				)
				bounds.append(b)
				idx_counter += 1
			elif param.type == "categorical":
				if self.has_descriptors:
					for desc_ix in range(len(param.descriptors[0])):
						min_ = np.amin(
							[
								param.descriptors[opt_ix][desc_ix]
								for opt_ix in range(len(param.options))
							]
						)
						max_ = np.amax(
							[
								param.descriptors[opt_ix][desc_ix]
								for opt_ix in range(len(param.options))
							]
						)
						bounds += [[min_, max_]]
					idx_counter += len(param.descriptors[0])
				else:
					bounds += [[0, 1] for _ in param.options]
					idx_counter += len(param.options)

		return torch.tensor(np.array(bounds)).T.float()


	def param_vectors_to_expanded(
		self, param_vectors: List[ParameterVector], is_scaled: bool=False,return_scaled:bool=False
		) -> np.ndarray:
		''' Convert from list of ParameterVectors to 2d array of scaled/raw expanded params

		TODO: need to consider the scaling of the ParameterVector inputs, but usually
		these are not scaled
		'''
		if isinstance(param_vectors, ParameterVector):
			param_vectors = [param_vectors]
		else:
			pass
		expanded = []
		for sample in param_vectors:
			expand = []
			for elem, param in zip(sample, self.param_space):
				elem = elem[1]
				if param.type == 'continuous':
					expand.append(elem)
				elif param.type == 'discrete':
					expand.append(elem)
				elif param.type == 'categorical':
					expand.extend(
						cat_param_to_feat(
							param, elem, self.has_descriptors,
						)
					)
			expanded.append(expand)
		expanded = np.array(expanded)
		if return_scaled:
			expanded = forward_normalize(expanded, self._mins_x, self._maxs_x)

		return expanded


	def param_vectors_to_indexed(
		self, param_vectors: List[ParameterVector], is_scaled:bool=False, return_scaled:bool=False
		) -> np.ndarray:
		''' Convert from list of ParameterVectors to 2d array of scaled/raw indexed params
		'''
		if isinstance(param_vectors, ParameterVector):
			param_vectors = [param_vectors]
		else:
			pass
		indexed = []
		for sample in param_vectors:
			index_ = []
			for elem, param in zip(sample, self.param_space):
				elem = elem[1]
				if param.type == 'continuous':
					index_.append(elem)
				elif param.type == 'discrete':
					index_.append(param.options.index(float(elem)))
				elif param.type == 'categorical':
					index_.append(param.options.index(elem))

			indexed.append(index_)

		indexed = np.array(indexed)
		if return_scaled:
			# scale the indexed representation (only the continuous dims)
			disc_cat_indexed = indexed[:,self.disc_dims+self.cat_dims]
			indexed_masked = indexed.copy()
			indexed_masked[:, self.disc_dims+self.cat_dims] = 1.

			_mins_x, _maxs_x = np.amin(indexed_masked,axis=0), np.amax(indexed_masked,axis=0)

			_mins_x[self.cont_dims] = self._mins_x[self.exp_cont_dims]
			_maxs_x[self.cont_dims] = self._maxs_x[self.exp_cont_dims]

			indexed_scaled = forward_normalize(
				indexed_masked , _mins_x, _maxs_x,
			)
			indexed_scaled[:, self.disc_dims+self.cat_dims] = disc_cat_indexed

			return indexed_scaled
		else:
			return indexed



	def indexed_to_param_vectors(
		self, indexed: np.ndarray, is_scaled:bool=False, return_scaled:bool=False
	) -> List[ParameterVector]:
		''' Convert from 2d array of scaled/raw indexed params to list of ParameterVectors

		TODO: need to deal with scaling for this example ...
		'''
		param_vectors = []

		for sample in indexed:
			param_dict = {}
			for elem, param in zip(sample, self.param_space):
				if param.type == 'continuous':
					param_dict[param.name] = elem
				elif param.type == 'discrete':
					param_dict[param.name] = param.options[int(elem)]
				elif param.type == 'categorical':
					param_dict[param.name] = param.options[int(elem)]

			param_vectors.append(
				ParameterVector().from_dict(param_dict, self.param_space)
			)

		return param_vectors


	def indexed_to_expanded(
		self, indexed: np.ndarray, is_scaled:bool=False, return_scaled:bool=False
		) -> List[ParameterVector]:
		''' Convert from 2d array of indexed parameters to 2d array of expanded parameters
		'''
		expanded = []
		for sample in indexed:
			expand = []
			for elem, param in zip(sample, self.param_space):
				if param.type == 'continuous':
					expand.append(elem)
				elif param.type == 'discrete':
					expand.append(param.options[int(elem)])
				elif param.type == 'categorical':
					option = param.options[int(elem)]
					expand.extend(
						cat_param_to_feat(
							param, option, self.has_descriptors
						)
					)
			expanded.append(expand)
		expanded = np.array(expanded)
		if return_scaled:
			expanded = forward_normalize(expanded, self._mins_x, self._maxs_x)

		return expanded
	







if __name__ == '__main__':

	use_descriptors = False

	def surface(x):
		if x["param_0"] == "x0":
			factor = 0.1
		elif x["param_0"] == "x1":
			factor = 1.0
		elif x["param_0"] == "x2":
			factor = 10.0

		return (
			np.sin(8.0 * x["param_1"])
			- 2.0 * np.cos(6.0 * x["param_1"])
			+ np.exp(-2.0 * x["param_2"])
			+ 2.0 * (1.0 / factor)
			+ x["param_3"]
		)

	if use_descriptors:
		desc_param_0 = [[float(i), float(i)] for i in range(3)]
	else:
		desc_param_0 = [None for i in range(3)]

	param_space = ParameterSpace()
	param_0 = ParameterCategorical(
		name="param_0",
		options=["x0", "x1", "x2"],
		descriptors=desc_param_0,
	)
	param_1 = ParameterDiscrete(
		name="param_1",
		options=[0.0, 0.25, 0.5, 0.75, 1.0],
	)
	param_2 = ParameterContinuous(
		name="param_2",
		low=5.0,
		high=10.0,
	)
	param_3 = ParameterContinuous(
		name="param_3",
		low=-2.,
		high=2.,
	)
	param_space.add(param_0)
	param_space.add(param_1)
	param_space.add(param_2)
	param_space.add(param_3)

	campaign = Campaign()
	campaign.set_param_space(param_space)

	samples = [
		ParameterVector().from_dict(
			{'param_0': 'x1', 'param_1': 0.25, 'param_2': 6.7, 'param_3': -0.8}
		),
		ParameterVector().from_dict(
			{'param_0': 'x2', 'param_1': 0.75, 'param_2': 9.8, 'param_3': -2.}
		),
		ParameterVector().from_dict(
			{'param_0': 'x0', 'param_1': 0.25, 'param_2': 7.2, 'param_3': 0.01}
		),
		ParameterVector().from_dict(
			{'param_0': 'x1', 'param_1': 0.0, 'param_2': 9.3, 'param_3': 1.04}
		),
	]
	for sample in samples:
		measurement = surface(sample)
		campaign.add_observation(sample, measurement)



	# initialize Parameters class

	params = Parameters(
		olympus_param_space=param_space,
		observations=campaign.observations,
		has_descriptors=use_descriptors,
	)

	print('\nolympus parameters : ')
	print(params.olympus)

	print('\nraw/indexed dims : ')
	print('cont : ' , params.cont_dims)
	print('disc : ', params.disc_dims)
	print('cat : ', params.cat_dims)

	print('\nexpanded dims : ')
	print('cont : ' , params.exp_cont_dims)
	print('disc : ', params.exp_disc_dims)
	print('cat : ', params.exp_cat_dims)


	print('\nindexed raw params : ')
	print(params.indexed_raw.shape)
	print(params.indexed_raw)

	print('\nexpanded raw params : ')
	print(params.expanded_raw.shape)
	print(params.expanded_raw)


	print('\n_mins_x, _maxs_x : ')
	print(params._mins_x)
	print(params._maxs_x)

	print('\nindexed scaled params : ')
	print(params.indexed_scaled.shape)
	print(params.indexed_scaled)

	print('\nexpanded scaled params : ')
	print(params.expanded_scaled.shape)
	print(params.expanded_scaled)


	print('\nparam vectors raw : ')
	print(len(params.param_vectors))
	print(params.param_vectors)

	print('\nbounds : ')
	print(params.bounds)


	# test the coversions
	print('\nTESTING CONVERSIONS : \n', )

	test_param_vector = [
		ParameterVector().from_dict(
			{'param_0': 'x2', 'param_1': 0.0, 'param_2': 7.3, 'param_3': -1.9}
		),
		ParameterVector().from_dict(
			{'param_0': 'x0', 'param_1': 0.5, 'param_2': 6.4, 'param_3': -1.0}
		),
	]

	test_indexed = np.array([
		[2., 0., 7.3, -1.9],
		[0., 2., 6.4, -1.0],
	])

	test_expanded = np.array([
		[0., 0., 1., 0., 7.3, -1.9],
		[1., 0., 0., 0.5, 6.4, -1.0],
	])

	print('\n param_vectors_to_expanded')
	# param vectors to expanded
	res = params.param_vectors_to_expanded(test_param_vector, return_scaled=False)
	print(res)

	res = params.param_vectors_to_expanded(test_param_vector, return_scaled=True)
	print(res)

	print('\n param_vectors_to_indexed')
	# param vectors to indexed
	res = params.param_vectors_to_indexed(test_param_vector, return_scaled=False)
	print(res)

	res = params.param_vectors_to_indexed(test_param_vector, return_scaled=True)
	print(res)


	print('\n indexed_to_param_vectors')

	res = params.indexed_to_param_vectors(test_indexed, return_scaled=False)
	print(res)


	print('\n indexed_to_expanded')

	res = params.indexed_to_expanded(test_indexed, return_scaled=False)
	print(res)
	res = params.indexed_to_expanded(test_indexed, return_scaled=True)
	print(res)
