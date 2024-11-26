#!/usr/bin/env python

import os
import pickle
import shutil
import numpy as np
import itertools
from copy import deepcopy
from olympus.surfaces import Surface
from olympus.surfaces import Dejong, Branin, HyperEllipsoid, AckleyPath, Levy, Michalewicz, Rastrigin, Schwefel, StyblinskiTang
from olympus.surfaces import DiscreteAckley, DiscreteDoubleWell, DiscreteMichalewicz, LinearFunnel, NarrowFunnel

from olympus.datasets import Dataset
from olympus.emulators import Emulator


# general function to replace nan in observations with worst
#  observed merit so far
def replace_nan_with_worst(observations):

	new_observations = []
	if len(observations) == 0:
		return new_observations

	worst_obj = np.nanmax([o['obj'] for o in observations])
	for obs in observations:
		if np.isnan(obs['obj']):
			obs2 = deepcopy(obs)
			obs2['obj'] = worst_obj
			new_observations.append(obs2)
		else:
			new_observations.append(obs)
	return new_observations


def replace_nan_with_surrogate(observations, gryffin_obj):
	''' replace the nan values with the current surrogate model predictions
	at that location

	Args:
		observations (list): list of dictionaries used by Gryffin including the
			nan values
		gryffin_obj (obj): Gryffin object for the current iteration
	'''

	new_observations = []
	if len(observations) == 0:
		return new_observations

	if np.isnan([o['obj'] for o in observations[:-1]]).all():
		# all values are nan, do not replace yet, need at least one
		# real-valued objective measurement to train regression surrogate
		return observations

	for obs in observations:
		if np.isnan(obs['obj']):
			obs2 = deepcopy(obs)
			# get the Gryffin surrogate prediction
			pred = gryffin_obj.get_regression_surrogate([obs])
			obs2['obj'] = pred[0]
			new_observations.append(obs2)
		else:
			new_observations.append(obs)
	return new_observations


def ignore_nan(observations):
	''' remove the nan measurements from the current observations list

	Args:
		observations (list): list of dictionaries used by Gryffin including
			nan values
	'''
	new_observations = []
	if len(observations) == 0:
		return new_observations

	num_nan = 0
	for obs in observations:
		if np.isnan(obs['obj']):
			num_nan+=1
		else:
			new_observations.append(obs)
	assert len(new_observations)==len(observations)-num_nan

	return new_observations


def save_pkl_file(data_all_repeats):
	"""save pickle file with results so far"""

	if os.path.isfile('results.pkl'):
		shutil.move('results.pkl', 'bkp-results.pkl')  # overrides existing files

	# store run results to disk
	with open("results.pkl", "wb") as content:
		pickle.dump(data_all_repeats, content)


def load_data_from_pkl_and_continue(N):
	"""load results from pickle file"""

	data_all_repeats = []
	# if no file, then we start from scratch/beginning
	if not os.path.isfile('results.pkl'):
		return data_all_repeats, N

	# else, we load previous results and continue
	with open("results.pkl", "rb") as content:
		data_all_repeats = pickle.load(content)

	missing_N = N - len(data_all_repeats)

	return data_all_repeats, missing_N


# function to write category details/descriptors
def write_categories(num_dims=2, num_opts=21, home_dir='.', num_descs=1, with_descriptors=True):

	for dim in range(num_dims):
		cat_details = {}
		for opt in range(num_opts):
			if with_descriptors is True:
				cat_details[f'x_{opt}'] = [opt] * num_descs
			else:
				cat_details[f'x_{opt}'] = None

		# create cat_details dir if necessary
		if not os.path.isdir('%s/CatDetails' % home_dir):
			os.mkdir('%s/CatDetails' % home_dir)

		cat_details_file = f'{home_dir}/CatDetails/cat_details_x{dim}.pkl'
		pickle.dump(cat_details, open(cat_details_file, 'wb'))


# ==============
# Parent classes
# ==============

# --------------------------------------
# Parent classes for Constrained surface
# --------------------------------------
class Constr:
	def eval_constr(self, X):
		"""Evaluate constraints, return True for feasible locations
		and False for infeasible ones.
		"""
		if isinstance(X, dict):
			X_arr = [X[k] for k in list(X.keys())]
			return self.is_feasible(X_arr)
		else:
			Y = []
			for Xi in X:
				Y.append(self.is_feasible(Xi))
			return np.array(Y)

		# elif len(np.shape(X)) == 1:
		# 	return self.is_feasible(X)

	def run_constr(self, X):
		"Evaluate surface and return nan for infeasible locations"
		X = np.array(list(X.to_dict().values()))
		if len(np.shape(X)) == 2:
			Y = []
			for Xi in X:
				if self.is_feasible(Xi) is True:
					Yi = self.run(Xi)
					Y.append(np.squeeze(Yi))
				else:
					Y.append(np.nan)
			return np.array(Y)
		elif len(np.shape(X)) == 1:
			if self.is_feasible(X) is True:
				return np.squeeze(self.run(X))[()]
			else:
				return np.nan

	def eval_merit(self, param):
		''' Evaluate merit of Gryffin's param object.
		'''
		param_arr = [param[k] for k in list(param.keys())]
		param['obj'] = self.run_constr(param_arr)
		return param



# -------------------------------------
# Parent class for Categorical surfaces
# -------------------------------------
class CategoricalEvaluator:

	def __init__(self, num_dims=2, num_opts=21):
		self.num_dims = num_dims
		self.num_opts = num_opts

	@staticmethod
	def str2array(sample):
		if type(sample[0]) == str or type(sample[0]) == np.str_:
			return np.array([round(float(entry[2:])) for entry in np.squeeze(sample)])
		else:
			return sample

	def run(self, sample):
		vector = self.str2array(sample)
		return self.evaluate(sample=vector)




class AbstractCont(Constr):

	def __init__(self, surface_type='Dejong', n_dims=2, constraint_type='circles', num_constraints=5,
				constraint_params=[0.1, 0.3], seed=100700,
		):
		self.surface_type = surface_type
		self.n_dims = n_dims
		self.constraint_type = constraint_type
		self.num_constraints = num_constraints
		self.constraint_params = constraint_params
		self.seed = seed

		self.surface = Surface(kind=self.surface_type)

		if np.all([param.type=='continuous' for param in self.surface.param_space]):
			self.problem_type = 'fully_continuous'
		elif np.all([param.type=='categorical' for param in self.surface.param_space]):
			self.problem_type = 'fully_categorical'
		else:
			raise ValueError

		if self.constraint_type == 'circles':
			# TODO: assert we have continous surface
			assert self.problem_type == 'fully_continuous'
			self.constraint_func = self.circles_constraint
		elif self.constraint_type == 'rectangles':
			# TODO: assert we have a continuous surface
			assert self.problem_type == 'fully_continuous'
			self.constraint_func = self.rectangles_constraint
		elif self.constraint_type == 'squares':
			# TODO: assert that we have a categorical surface here
			assert self.problem_type == 'fully_categorical'
			self.constraint_func = self.squares_constraint
		else:
			raise NotImplementedError


	def circles_constraint(self, Xi):
		np.random.seed(self.seed)
		centers = [np.random.uniform(low=0.0, high=1.0, size=2) for i in range(self.num_constraints)]
		radii = [
			np.random.uniform(low=self.constraint_params[0], high=self.constraint_params[1], size=1) for i in range(self.num_constraints)
		]
		for c, r in zip(centers, radii):
			if np.linalg.norm(c - Xi) < r:
				return False
		return True


	def rectangles_constraint(self, Xi):
		np.random.seed(self.seed)
		ws = []
		coords = []
		for _ in range(self.num_constraints):
			# min_width, max_width
			w = np.random.uniform(
				self.constraint_params[0], self.constraint_params[1], size=(self.n_dims,)
			)
			coord_1 = np.random.uniform(size=(self.n_dims,))
			coord_2 = coord_1.copy()
			ix = np.random.randint(self.n_dims)
			w[ix] = 0.
			coord_1[ix] = 0.
			coord_2[ix] = 1.

			ws.append(w)
			coords.append([coord_1, coord_2])

		for w, coord in zip(ws, coords):
			bools = []
			for param_ix in range(self.n_dims):
				bool_ = np.logical_and(
					Xi[param_ix] > coord[0][param_ix]-(w[param_ix]/2.),
					Xi[param_ix] < coord[1][param_ix]+(w[param_ix]/2.)
				)
				bools.append(bool_)
			if all(bools):
				return False

		return True


	def squares_constraint(self, Xi):
		# choose infeasible points at random
		options = [f'x_{i}' for i in range(0, self.surface.num_opts, 1)]
		np.random.seed(self.seed)
		infeas_arrays = np.array([np.random.choice(options, size=self.num_constraints, replace=True),
								  np.random.choice(options, size=self.num_constraints, replace=True)]).T
		infeas_tuples = [tuple(x) for x in infeas_arrays]

		sample_tuple = tuple(Xi)
		if sample_tuple in infeas_tuples:
			return False
		return True

	def run(self, Xi):
		if self.problem_type == 'fully_continuous':
			return self.surface.run(Xi)
		elif self.problem_type == 'fully_categorical':
			if type(Xi[0]) == str or type(Xi[0]) == np.str_:
				vector = [elem.replace('_','') for elem in Xi]
			return float(self.surface.run(vector)[0][0])



	@property
	def minima(self):
		return self.surface.minima

	@property
	def num_opts(self):
		return self.surface.num_opts



	def is_feasible(self, Xi):
		if isinstance(Xi, dict):
			Xi = [val for _, val in Xi.items()]

		elif np.logical_or(
			isinstance(Xi, list),
			isinstance(Xi, np.ndarray)
		):
			pass

		return self.constraint_func(Xi)


	def estimate_feas_frac(self, num_points=10000):
		if self.problem_type == 'fully_continuous':
			samples = np.random.uniform(size=(num_points, self.n_dims))
			bools = []
			for sample in samples:
				bools.append(self.is_feasible(sample))

		elif self.problem_type == 'fully_categorical':
			# make cartesian product space
			opts = [f'x_{i}' for i in range(0, self.surface.num_opts, 1)]
			tiled_opts = [opts for _ in range(self.n_dims)]
			cp = list(itertools.product(*tiled_opts))

			bools = []
			for sample in cp:
				bools.append(self.is_feasible(sample))

		return bools.count(True)/len(bools)











# ==============================
# Constrained benchmark surfaces
# ==============================
class DejongConstr(Dejong, Constr):
	def is_feasible(self, Xi):
		if isinstance(Xi, dict):
			Xi = [val for _, val in Xi.items()]

		elif np.logical_or(
			isinstance(Xi, list),
			isinstance(Xi, np.ndarray)
		):
			pass
		x0 = Xi[0]
		x1 = Xi[1]
		y = (x0-0.5)**2 + (x1-0.5)**2

		if np.abs(x0-x1) < 0.1:
			return False

		if 0.05 < y < 0.15:
			return False
		else:
			return True

class BraninConstr(Branin, Constr):
	def is_feasible(self, Xi):
		if isinstance(Xi, dict):
			Xi = [val for _, val in Xi.items()]

		elif np.logical_or(
			isinstance(Xi, list),
			isinstance(Xi, np.ndarray)
		):
			pass
		x0 = Xi[0]
		x1 = Xi[1]

		y0 = (x0-0.12389382)**2 + (x1-0.81833333)**2
		y1 = (x0-0.961652)**2 + (x1-0.165)**2

		if y0 < 0.2**2 or y1 < 0.35**2:
			return False
		else:
			return True

class HyperEllipsoidConstr(HyperEllipsoid, Constr):
	def __init__(self):
		HyperEllipsoid.__init__(self)

		np.random.seed(42)
		N = 20
		self.centers = [np.random.uniform(low=0.0, high=1.0, size=2) for i in range(N)]
		self.radii = [np.random.uniform(low=0.05, high=0.15, size=1) for i in range(N)]

	def is_feasible(self, Xi):
		if isinstance(Xi, dict):
			Xi = [val for _, val in Xi.items()]

		elif np.logical_or(
			isinstance(Xi, list),
			isinstance(Xi, np.ndarray)
		):
			pass
		for c, r in zip(self.centers, self.radii):
			if np.linalg.norm(c - Xi) < r:
				return False

		return True

class AckleyPathConstr(AckleyPath, Constr):
	def is_feasible(self, Xi):
		if isinstance(Xi, dict):
			Xi = [val for _, val in Xi.items()]

		elif np.logical_or(
			isinstance(Xi, list),
			isinstance(Xi, np.ndarray)
		):
			pass
		x0 = Xi[0]
		x1 = Xi[1]

		if 0.3 < x0 < 0.7 and 0.3 < x1 < 0.7:
			if 0.4 < x0 < 0.6 and 0.4 < x1 < 0.6:
				return True
			else:
				return False
		return True

class LevyConstr(Levy, Constr):
	def __init__(self):
		Levy.__init__(self)

		np.random.seed(0)
		N = 30
		self.centers = [np.random.uniform(low=0.0, high=1.0, size=2) for i in range(N)]
		self.radii = [np.random.uniform(low=0.025, high=0.1, size=1) for i in range(N)]

	def is_feasible(self, Xi):
		if isinstance(Xi, dict):
			Xi = [val for _, val in Xi.items()]

		elif np.logical_or(
			isinstance(Xi, list),
			isinstance(Xi, np.ndarray)
		):
			pass
		for c, r in zip(self.centers, self.radii):
			if np.abs(c[0] - Xi[0]) < r and np.abs(c[1] - Xi[1]) < r:
				return False
		return True


class MichalewiczConstr(Michalewicz, Constr):
	def is_feasible(self, Xi):
		if isinstance(Xi, dict):
			Xi = [val for _, val in Xi.items()]

		elif np.logical_or(
			isinstance(Xi, list),
			isinstance(Xi, np.ndarray)
		):
			pass
		x0 = Xi[0]
		x1 = Xi[1]
		y = ((x0-0.5)/1.5)**2 + (x1-0.5)**2
		if 0.05 < y < 0.1:
			return False
		if np.abs(x0-x1) < 0.1:
			return False
		return True


class RastriginConstr(Rastrigin, Constr):
	def is_feasible(self, Xi):
		if isinstance(Xi, dict):
			Xi = [val for _, val in Xi.items()]

		elif np.logical_or(
			isinstance(Xi, list),
			isinstance(Xi, np.ndarray)
		):
			pass
		x0 = Xi[0]
		x1 = Xi[1]
		if 0.15 < x0 < 0.45 and 0.15 < x1 < 0.45:
			return False
		if 0.2 < x0 < 0.4 and 0.6 < x1 < 0.8:
			return False
		if 0.55 < x0 < 0.85 and 0.55 < x1 < 0.85:
			return False
		if 0.6 < x0 < 0.8 and 0.2 < x1 < 0.4:
			return False
		return True


class SchwefelConstr(Schwefel, Constr):
	def is_feasible(self, Xi):
		if isinstance(Xi, dict):
			Xi = [val for _, val in Xi.items()]

		elif np.logical_or(
			isinstance(Xi, list),
			isinstance(Xi, np.ndarray)
		):
			pass
		x0 = Xi[0]
		x1 = Xi[1]
		y = (x0-1)**2 + (x1-1)**2
		if 0.1 < y < 0.25:
			return False


class SchwefelConstr(Schwefel, Constr):
	def __init__(self):
		Schwefel.__init__(self)

		np.random.seed(42)
		N = 20
		self.centers = [np.random.uniform(low=0.0, high=1.0, size=2) for i in range(N)]
		self.radii = [np.random.uniform(low=0.05, high=0.15, size=1) for i in range(N)]

	def is_feasible(self, Xi):
		if isinstance(Xi, dict):
			Xi = [val for _, val in Xi.items()]

		elif np.logical_or(
			isinstance(Xi, list),
			isinstance(Xi, np.ndarray)
		):
			pass
		x0 = Xi[0]
		x1 = Xi[1]
		Xi = np.array([x0, x1])
		for c, r in zip(self.centers, self.radii):
			if np.linalg.norm(c - Xi) < r:
				return False
		return True



class StyblinskiTangConstr(StyblinskiTang, Constr):
	def is_feasible(self, Xi):
		if isinstance(Xi, dict):
			Xi = [val for _, val in Xi.items()]

		elif np.logical_or(
			isinstance(Xi, list),
			isinstance(Xi, np.ndarray)
		):
			pass
		x0 = Xi[0]
		x1 = Xi[1]

		if x0+x1 < 0.4:
			return False
		if x0 > 0.6 and x1 > 0.6:
			return False
		if x0 < 0.4 and x1 > 0.6:
			return False
		if x0 > 0.6 and x1 < 0.4:
			return False
		return True


class DiscreteAckleyConstr(DiscreteAckley, Constr):
	def is_feasible(self, Xi):
		if isinstance(Xi, dict):
			Xi = [val for _, val in Xi.items()]

		elif np.logical_or(
			isinstance(Xi, list),
			isinstance(Xi, np.ndarray)
		):
			pass
		x0 = Xi[0]
		x1 = Xi[1]
		if x0+x1 < 0.3:
			return False
		return True


class DiscreteDoubleWellConstr(DiscreteDoubleWell, Constr):
	def is_feasible(self, Xi):
		if isinstance(Xi, dict):
			Xi = [val for _, val in Xi.items()]

		elif np.logical_or(
			isinstance(Xi, list),
			isinstance(Xi, np.ndarray)
		):
			pass
		x0 = Xi[0]
		x1 = Xi[1]
		if x0+x1 < 0.3:
			return False
		return True


class DiscreteMichalewiczConstr(DiscreteMichalewicz, Constr):
	def is_feasible(self, Xi):
		if isinstance(Xi, dict):
			Xi = [val for _, val in Xi.items()]

		elif np.logical_or(
			isinstance(Xi, list),
			isinstance(Xi, np.ndarray)
		):
			pass
		x0 = Xi[0]
		x1 = Xi[1]
		if x0+x1 < 0.3:
			return False
		return True


class LinearFunnelConstr(LinearFunnel, Constr):
	def is_feasible(self, Xi):
		if isinstance(Xi, dict):
			Xi = [val for _, val in Xi.items()]

		elif np.logical_or(
			isinstance(Xi, list),
			isinstance(Xi, np.ndarray)
		):
			pass
		x0 = Xi[0]
		x1 = Xi[1]
		if x0+x1 < 0.3:
			return False
		return True


class NarrowFunnelConstr(NarrowFunnel, Constr):
	def is_feasible(self, Xi):
		if isinstance(Xi, dict):
			Xi = [val for _, val in Xi.items()]

		elif np.logical_or(
			isinstance(Xi, list),
			isinstance(Xi, np.ndarray)
		):
			pass
		x0 = Xi[0]
		x1 = Xi[1]
		if x0+x1 < 0.3:
			return False
		return True

# ====================
# Categorical surfaces
# ====================
class CatDejongConstr(CategoricalEvaluator, Constr):
	"""
		Dejong is to be evaluated on the hypercube
		x_i in [-5.12, 5.12] for i = 1, ..., d
	"""
	def dejong(self, vector):
		result = np.sum(vector**2)
		return result

	def evaluate(self, sample):
		# map sample onto hypercube
		vector = np.zeros(self.num_dims)
		for index, element in enumerate(sample):
			vector[index] = 10.24 * ( element / float(self.num_opts - 1) ) - 5.12
		return self.dejong(vector)

	def is_feasible(self, sample):
		Xi = self.str2array(sample)
		x0 = Xi[0]
		x1 = Xi[1]
		if x0 in [9, 11]:
			return False
		if x1 in [9, 11]:
			return False
		return True

	@property
	def best(self):
		return ('x_10', 'x_10')


class CatAckleyConstr(CategoricalEvaluator, Constr):
	"""
		Ackley is to be evaluated on the hypercube
		x_i in [-32.768, 32.768] for i = 1, ..., d
	"""
	def ackley(self, vector, a=20., b=0.2, c=2.0* np.pi):
		result = - a * np.exp(- b*np.sqrt( np.sum(vector**2)/self.num_dims)) - np.exp(np.sum(np.cos(c*vector))) + a + np.exp(1)
		return result

	def evaluate(self, sample):
		# map sample onto hypercube
		vector = np.zeros(self.num_dims)
		for index, element in enumerate(sample):
			vector[index] = 65.536 * ( element / float(self.num_opts - 1) ) - 32.768
		return self.ackley(vector)

	def is_feasible(self, sample):
		Xi = self.str2array(sample)
		x0 = Xi[0]
		x1 = Xi[1]
		if 4 < x0 < 16 and 4 < x1 < 16:
			if 7 < x0 < 13 and 7 < x1 < 13:
				return True
			else:
				return False
		return True

	@property
	def best(self):
		return ('x_10', 'x_10')


class CatMichalewiczConstr(CategoricalEvaluator, Constr):
	"""
		Michalewicz is to be evaluated on the hypercube
		x_i in [0, pi] for i = 1, ..., d
	"""

	def michalewicz(self, vector, m=10.):
		result = 0.
		for index, element in enumerate(vector):
			result += - np.sin(element) * np.sin( (index + 1) * element**2 / np.pi)**(2 * m)
		return result

	def evaluate(self, sample):
		# map sample onto hypercube
		vector = np.zeros(self.num_dims)
		for index, element in enumerate(sample):
			vector[index] = np.pi * element / float(self.num_opts - 1)
		return self.michalewicz(vector)

	def is_feasible(self, sample):

		Xi = self.str2array(sample)
		x0 = Xi[0]
		x1 = Xi[1]


		y = ((x0-14))**2 + (x1-10)**2
		if 5 < y < 30:
			return False
		if 12.5 < x0 < 15.5:
			if x1 < 5.5:
				return False
		if 8.5 < x1 < 11.5:
			if x0 < 9.5:
				return False
		return True

	@property
	def best(self):
		return ('x_14', 'x_10')


class CatSlopeConstr(CategoricalEvaluator, Constr):
	"""
		Response sampled from standard normal distribution
		with correlation
	"""
	def random_correlated(self, vector):
		seed   = 0
		vector = np.array(vector)
		for index, element in enumerate(vector):
			seed += self.num_opts**index * element
		result = np.sum(vector / self.num_opts)
		return result

	def evaluate(self, sample):
		return self.random_correlated(sample)

	def is_feasible(self, sample):
		Xi = self.str2array(sample)
		x0 = Xi[0]
		x1 = Xi[1]

		y = x0**2 + x1**2
		if 5 < y < 25:
			return False
		if 70 < y < 110:
			return False
		if 200 < y < 300:
			return False
		return True

	@property
	def best(self):
		return ('x_0', 'x_0')
	
	@property
	def minima(self):
		return [('x_0', 'x_0')]


class CatCamelConstr(CategoricalEvaluator, Constr):
	"""
		Camel is to be evaluated on the hypercube
		x_i in [-3, 3] for i = 1, ..., d
	"""
	def __init__(self, num_dims=2, num_opts=21):
		CategoricalEvaluator.__init__(self, num_dims, num_opts)

		# choose infeasible points at random
		options = [f'x_{i}' for i in range(0, num_opts, 1)]
		num_infeas = 100
		np.random.seed(42)
		infeas_arrays = np.array([np.random.choice(options, size=num_infeas, replace=True),
								  np.random.choice(options, size=num_infeas, replace=True)]).T
		self.infeas_tuples = [tuple(x) for x in infeas_arrays]
		# always exclude the other minima
		self.infeas_tuples.append(('x_7', 'x_11'))
		self.infeas_tuples.append(('x_7', 'x_15'))
		self.infeas_tuples.append(('x_13', 'x_5'))

	def camel(self, vector):
		result = 0.

		# global minima
		loc_0 = np.array([-1., 0.])
		loc_1 = np.array([ 1., 0.])
		weight_0 = np.array([4., 1.])
		weight_1 = np.array([4., 1.])

		# local minima
		loc_2  = np.array([-1., 1.5])
		loc_3  = np.array([ 1., -1.5])
		loc_5  = np.array([-0.5, -1.0])
		loc_6  = np.array([ 0.5,  1.0])
		loss_0 = np.sum(weight_0 * (vector - loc_0)**2) + 0.01 + np.prod(vector - loc_0)
		loss_1 = np.sum(weight_1 * (vector - loc_1)**2) + 0.01 + np.prod(vector - loc_1)
		loss_2 = np.sum((vector - loc_2)**2) + 0.075
		loss_3 = np.sum((vector - loc_3)**2) + 0.075
		loss_5 = 3000. * np.exp( - np.sum((vector - loc_5)**2) / 0.25)
		loss_6 = 3000. * np.exp( - np.sum((vector - loc_6)**2) / 0.25)
		result = loss_0 * loss_1 * loss_2 * loss_3 + loss_5 + loss_6
		return result

	def evaluate(self, sample):
		# map sample onto hypercube
		vector = np.zeros(self.num_dims)
		for index, element in enumerate(sample):
			vector[index] = 6 * ( element / float(self.num_opts - 1) ) - 3
		return self.camel(vector)

	def is_feasible(self, sample):
		sample_tuple = tuple(sample)
		if sample_tuple in self.infeas_tuples:
			return False
		return True

	@property
	def best(self):
		return ('x_13', 'x_9')




#--------------------------------------------
# Olympus emulated datasets with constraints
#--------------------------------------------


class DoubleWellConstr(Constr):
	def __init__(self):
		pass

	def is_feasible(self, Xi):


		if isinstance(Xi, dict):
			Xi = [val for _, val in Xi.items()]

		elif np.logical_or(
			isinstance(Xi, list),
			isinstance(Xi, np.ndarray)
		):
			pass

		y0 = (Xi[0]-0.31843715617394186)**2 + (Xi[1]-0.647954318515612)**2

		if y0 <0.0015:
			return False

		y1 = (Xi[0]-0.6741341747478864)**2 + (Xi[1]-0.6503197706249563)**2
		if 0.01 < y1 < 0.02:
			return False

		y2 = (Xi[0]-0.647954318515612)**2 + (Xi[1]-0.31843715617394186)**2
		if y2 <0.01:
			return False

		return True

	def run(self, Xi):
		x = Xi[0]*(2+2) - 2
		y = Xi[1]*(2+2) - 2
		return (15*x + y + 100*x*y) / (np.exp(x**2 + y**2))

	def estimate_feas_frac(self, num_points):

		return None

	def minima(self):
		return [{'params': [0.5, 0.5], 'values': 0.0}]

	def estimate_minima(self, num_samples):
		samples = np.random.uniform(size=(num_samples, 2))
		meas = self.run(samples)
		ix = np.argmin(meas)
		return [{'params': [samples[ix, 0], samples[ix, 1]], 'values': meas[ix]}]

	def estimate_maxima(self, num_samples):
		samples = np.random.uniform(size=(num_samples, 2))
		meas = self.run(samples)
		ix = np.argmax(meas)
		return [{'params': [samples[ix, 0], samples[ix, 1]], 'values': meas[ix]}]


class DoubleWell2Constr(Constr):
	def __init__(self, factor=0.2):
		self.factor=factor

	def is_feasible(self, Xi):

		if isinstance(Xi, dict):
			Xi = [val for _, val in Xi.items()]

		elif np.logical_or(
			isinstance(Xi, list),
			isinstance(Xi, np.ndarray)
		):
			pass

		if Xi[0] + Xi[1] > 1.+self.factor:
			return False
		if Xi[0] + Xi[1] < 1.-self.factor:
			return False

		return True

	def run(self, Xi):
		x = Xi[:,0]*(2+2) - 2
		y = Xi[:,1]*(2+2) - 2
		return (15*x + y + 100*x*y) / (np.exp(x**2 + y**2))

	def estimate_feas_frac(self, num_points):

		samples = np.random.uniform(size=(num_points, 2))
		num_feas = 0
		for sample in samples:
			if self.is_feasible(sample):
				num_feas+=1

		return num_feas/num_points

	def estimate_minima(self, num_samples):
		samples = np.random.uniform(size=(num_samples, 2))
		meas = self.run(samples)
		ix = np.argmin(meas)
		return [{'params': [samples[ix, 0], samples[ix, 1]], 'values': meas[ix]}]


	def estimate_maxima(self, num_samples):
		samples = np.random.uniform(size=(num_samples, 2))
		meas = self.run(samples)
		ix = np.argmax(meas)
		return [{'params': [samples[ix, 0], samples[ix, 1]], 'values': meas[ix]}]

	# def minima(self):
	# 	return [{'params': [0.5, 0.5], 'values': 0.0}]


class DejongN5Constr(Constr):
	def __init__(self):

		self.a = np.array([
			[-32, -16, 0, 16, 32,-32, -16, 0, 16, 32,-32, -16, 0, 16, 32,-32, -16, 0, 16, 32,-32, -16, 0, 16, 32],
			[-32,-32,-32,-32,-32,-16,-16,-16,-16,-16,0,0,0,0,0,16,16,16,16,16,32,32,32,32,32],
		])


	def is_feasible(self, Xi):

		if isinstance(Xi, dict):
			Xi = [val for _, val in Xi.items()]

		elif np.logical_or(
			isinstance(Xi, list),
			isinstance(Xi, np.ndarray)
		):
			pass

		if 0.7<Xi[0]<0.8: #and Xi[1]>0.99:
			return False

		if 0.2<Xi[0]<0.3: #and Xi[1]>0.99:
			return False

		if 0.45<Xi[0]<0.55: #and Xi[1]>0.99:
			return False

		if 0.61<Xi[0]<0.63: #and Xi[1]>0.99:
			return False

		if 0.37<Xi[0]<0.39: #and Xi[1]>0.99:
			return False

		if 0.24<Xi[1]<0.26: #and Xi[1]>0.99:
			return False

		if 0.73<Xi[1]<0.75: #and Xi[1]>0.99:
			return False


		return True

	def run(self, Xi):
		x = Xi[:,0]*(65.536+65.536) - 65.536
		y = Xi[:,1]*(65.536+65.536) - 65.536

		f = 0.002
		for i in range(self.a.shape[1]):
			f += 1 / (i+1 + (x-self.a[0,i])**6 + (y-self.a[1,i])**6 )
		f = f**(-1)
		return f


class HPLCConstr(Constr):
	def __init__(self):
		self.dataset = Dataset(kind='hplc')
		self.emulator = Emulator(dataset='hplc', model='BayesNeuralNet')


	def is_feasible(self, Xi):
		if isinstance(Xi, dict):
			Xi = [val for _, val in Xi.items()]

		elif np.logical_or(
			isinstance(Xi, list),
			isinstance(Xi, np.ndarray)
		):
			pass

		sample_loop = Xi[0]
		additional_volume = Xi[1]
		tubing_volume = Xi[2]
		sample_flow = Xi[3]
		push_speed = Xi[4]
		wait_time = Xi[5]

		if np.logical_and(
			sample_loop < 0.02,
			additional_volume > 0.05
		):
			return False

		if additional_volume*10 +tubing_volume > 12:
			return False

		if np.logical_and(
			sample_flow < 1.2,
			push_speed > 130
		):
			return False

		if wait_time > 7:
			if sample_flow > 2.1:
				return False
			if tubing_volume > 0.8:
				return False

		return True

	def sample_hplc_params(self, num_samples, as_array=False):
		samples = []
		for _ in range(num_samples):
			samples.append(
				{
					'sample_loop': np.random.uniform(0.00, 0.08),
					'additional_volume': np.random.uniform(0.00, 0.06),
					'tubing_volume': np.random.uniform(0.1, 0.9),
					'sample_flow': np.random.uniform(0.5, 2.5),
					'push_speed': np.random.uniform(80, 150),
					'wait_time': np.random.uniform(1, 10),
				}
			)
		if as_array:
			samples = [np.array(list(d.values())) for d in samples]
			samples = np.array(samples)
		return samples


	def run(self, Xi):
		return self.emulator.run(Xi)

	def estimate_feas_frac(self, num_points):
		samples = self.sample_hplc_params(num_samples=num_points, as_array=True)
		bools = []
		for sample in samples:
			bools.append(self.is_feasible(sample))

		return bools.count(True)/len(bools)



class AckleyPathConstrNDim(AckleyPath, Constr):
	''' Continuous AckleyPath function extended to arbitrary number of
	dimensions with randomly generated hyper-rectangular constraints

	'''
	def __init__(self, param_dim=2):
		AckleyPath.__init__(self, param_dim=param_dim)
		self.param_dim = param_dim
		np.random.seed(43)
		num_rect = int(param_dim*10)
		max_width = 0.025*(param_dim**1.8)

		self.ws = []
		self.coords = []

		for _ in range(num_rect):
			w = np.random.uniform(0.05, max_width, size=(param_dim,))
			coord_1 = np.random.uniform(size=(param_dim,))
			coord_2 = coord_1.copy()
			ix = np.random.randint(param_dim)
			w[ix] = 0.
			coord_1[ix] = 0.
			coord_2[ix] = 1.

			self.ws.append(w)
			self.coords.append([coord_1, coord_2])


	def is_feasible(self, Xi):
		if isinstance(Xi, dict):
			Xi = [val for _, val in Xi.items()]

		elif np.logical_or(
			isinstance(Xi, list),
			isinstance(Xi, np.ndarray)
		):
			pass

		for w, coord in zip(self.ws, self.coords):
			bools = []
			for param_ix in range(self.param_dim):
				bool_ = np.logical_and(
					Xi[param_ix] > coord[0][param_ix]-(w[param_ix]/2.),
					Xi[param_ix] < coord[1][param_ix]+(w[param_ix]/2.)
				)
				bools.append(bool_)
			if all(bools):
				return False

		return True


	def estimate_feas_frac(self, num_points):
		samples = np.random.uniform(size=(num_points, self.param_dim))
		bools = []
		for sample in samples:
			bools.append(self.is_feasible(sample))

		return bools.count(True)/len(bools)



class SchwefelConstrNDim(Schwefel, Constr):
	''' Continuous Schwefel surface extended to arbitray number of dimensions
	with randomly generated n-sphere constraints
	'''
	def __init__(self, param_dim=2):
		Schwefel.__init__(self, param_dim=param_dim)
		self.param_dim = param_dim
		np.random.seed(42)
		num_spheres = int(param_dim*10)
		max_radius = np.sqrt(param_dim*1.**2) / 4.6
		# generate n-sphere centres and radii
		self.centers = [np.random.uniform(size=(param_dim,)) for _ in range(num_spheres)]
		self.radii = [np.random.uniform(0.1, max_radius) for _ in range(num_spheres)]

	def is_feasible(self, Xi):
		if isinstance(Xi, dict):
			Xi = [val for _, val in Xi.items()]

		elif np.logical_or(
			isinstance(Xi, list),
			isinstance(Xi, np.ndarray)
		):
			pass

		for c, r in zip(self.centers, self.radii):
			if np.linalg.norm(c-Xi) < r:
				return False

		return True


	def estimate_feas_frac(self, num_points):
		samples = np.random.uniform(size=(num_points, self.param_dim))
		bools = []
		for sample in samples:
			bools.append(self.is_feasible(sample))

		return bools.count(True)/len(bools)


	def estimate_minima(self, num_samples):
		samples = np.random.uniform(size=(num_samples, 5))
		meas = self.run(samples)
		ix = np.argmin(meas)
		return [{'params': [samples[ix, i] for i in range(5)], 'values': meas[ix]}]




class CatMichalewiczConstrNDim(CategoricalEvaluator, Constr):
	"""
		Michalewicz is to be evaluated on the hypercube
		x_i in [0, pi] for i = 1, ..., d

	"""
	def __init__(
			self,
			num_dims,
			num_opts,
			num_constr_options=[2, 6, 3],
			random_seed=42,
	):
		self.num_dims = num_dims
		self.num_opts = num_opts
		self.num_constr_options = num_constr_options
		assert len(self.num_constr_options) == self.num_dims
		self.random_seed = random_seed

		np.random.seed(self.random_seed)

		self.constr_options = []
		for dim in range(self.num_dims):
			constr_opts = np.random.choice(
				np.arange(self.num_opts),
				size=(self.num_constr_options[dim]),
				replace=False
			)
			self.constr_options.append(constr_opts)


	def michalewicz(self, vector, m=10.):
		result = 0.
		for index, element in enumerate(vector):
			result += - np.sin(element) * np.sin( (index + 1) * element**2 / np.pi)**(2 * m)
		return result

	def evaluate(self, sample):
		# map sample onto hypercube
		vector = np.zeros(self.num_dims)
		for index, element in enumerate(sample):
			vector[index] = np.pi * element / float(self.num_opts - 1)
		return self.michalewicz(vector)

	def is_feasible(self, sample):

		if isinstance(sample, dict):
			sample = [val for _, val in sample.items()]

		elif np.logical_or(
			isinstance(sample, list),
			isinstance(sample, np.ndarray)
		):
			pass


		for dim in range(self.num_dims):
			constr_opts = self.constr_options[dim]
			if sample[dim] in constr_opts:
				return False

		return True

	# @property
	# def best(self):
	#     # in 5d these parameters lead to a value of -4.570112695431905
	#     return ('x_14', 'x_10', 'x_8', 'x_7', 'x_11')



class CatDejongConstrNDim(CategoricalEvaluator, Constr):
	"""
		Dejong is to be evaluated on the hypercube
		x_i in [-5.12, 5.12] for i = 1, ..., d
	"""
	def dejong(self, vector):
		result = np.sum(vector**2)
		return result

	def evaluate(self, sample):
		# map sample onto hypercube
		vector = np.zeros(self.num_dims)
		for index, element in enumerate(sample):
			vector[index] = 10.24 * ( element / float(self.num_opts - 1) ) - 5.12
		return self.dejong(vector)

	def is_feasible(self, sample):
		if isinstance(sample, dict):
			sample = [val for _, val in sample.items()]

		elif np.logical_or(
			isinstance(sample, list),
			isinstance(sample, np.ndarray)
		):
			pass

		Xi = self.str2array(sample)

		if np.logical_and(
			np.sum(Xi) > 10,
			np.sum(Xi) < 25
		):
			return False

		if np.logical_and(
			np.sum(Xi) > 45,
			np.sum(Xi) < 60
		):
			return False

		if np.prod(Xi[-2:]) > 350:
			return False

		if np.prod(Xi[1:3]) <  150:
			return False


		return True