#!/usr/bin/env python

import itertools
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from olympus.campaigns import ParameterSpace
from olympus.objects import (
    ParameterCategorical,
    ParameterContinuous,
    ParameterDiscrete,
    ParameterVector,
)

torch.set_default_dtype(torch.double)


def infer_problem_type(param_space: ParameterSpace) -> str:
    """infer the parameter space from Olympus. The three possibilities are
    "fully_continuous", "mixed" or "fully_categorical"
    Args:
            param_space (obj): Olympus parameter space object
    """
    param_types = [p.type for p in param_space]
    if param_types.count("continuous") == len(param_types):
        problem_type = "fully_continuous"
    elif param_types.count("categorical") == len(param_types):
        problem_type = "fully_categorical"
    elif param_types.count("discrete") == len(param_types):
        problem_type = "fully_discrete"
    elif all(
        [
            "continuous" in param_types,
            "categorical" in param_types,
            "discrete" not in param_types,
        ]
    ):
        problem_type = "mixed_cat_cont"
    elif all(
        [
            "continuous" not in param_types,
            "categorical" in param_types,
            "discrete" in param_types,
        ]
    ):
        problem_type = "mixed_cat_disc"
    elif all(
        [
            "continuous" in param_types,
            "categorical" not in param_types,
            "discrete" in param_types,
        ]
    ):
        problem_type = "mixed_disc_cont"
    elif all(
        [
            "continuous" in param_types,
            "categorical" in param_types,
            "discrete" in param_types,
        ]
    ):
        problem_type = "mixed_cat_disc_cont"

    return problem_type


def get_cat_dims(param_space: ParameterSpace) -> List[int]:
    dim = 0
    cat_dims = []
    for p in param_space:
        if p.type == "categorical":
            # this will only work for OHE variables
            d = np.arange(dim, dim + len(p.options))
            cat_dims.extend(list(d))
        else:
            dim += 1

    return cat_dims


def get_fixed_features_list(
    param_space: ParameterSpace, has_descriptors: bool
):
    dim = 0
    fixed_features_list = []
    cat_dims = []
    cat_params = []
    for p in param_space:
        if p.type == "categorical":
            if has_descriptors:
                dims = np.arange(dim, dim + len(p.descriptors[0]))
            else:
                dims = np.arange(dim, dim + len(p.options))
            cat_dims.extend(dims)
            cat_params.append(p)
            dim += len(dims)
        elif p.type == "discrete":
            cat_dims.append(dim)
            cat_params.append(p)
            dim += 1
        else:
            dim += 1

    param_options = [p.options for p in cat_params]
    cart_product = list(itertools.product(*param_options))
    cart_product = [list(elem) for elem in cart_product]

    current_avail_feat = []
    current_avail_cat = []
    for elem in cart_product:
        ohe = []
        for val, obj in zip(elem, cat_params):
            if obj.type == "categorical":
                ohe.extend(cat_param_to_feat(obj, val, has_descriptors))
            else:
                ohe.append(val)
        current_avail_feat.append(np.array(ohe))
        current_avail_cat.append(elem)

    # make list
    for feat in current_avail_feat:
        fixed_features_list.append(
            {dim_ix: feat[ix] for ix, dim_ix in enumerate(cat_dims)}
        )

    return fixed_features_list


def cat_param_to_feat(
    param: Union[ParameterContinuous, ParameterDiscrete, ParameterCategorical],
    val: str,
    has_descriptors: bool,
) -> Union[List, np.ndarray]:
    """convert the option selection of a categorical variable (usually encoded
    as a string) to a machine readable feature vector
    Args:
            param (object): the categorical olympus parameter
            val (str): the value of the chosen categorical option
    """
    # get the index of the selected value amongst the options

    arg_val = param.options.index(val)
    if not has_descriptors:
        # no provided descriptors, resort to one-hot encoding
        feat = np.zeros(len(param.options))
        feat[arg_val] += 1.0
    else:
        # we have descriptors, use them as the features
        feat = param.descriptors[arg_val]
    return feat


def propose_randomly(
    num_proposals: int,
    param_space: ParameterSpace,
    has_descriptors: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Randomly generate num_proposals proposals. Returns the numerical
    representation of the proposals as well as the string based representation
    for the categorical variables
    Args:
            num_proposals (int): the number of random proposals to generate
    """
    proposals = []
    raw_proposals = []
    for propsal_ix in range(num_proposals):
        sample = []
        raw_sample = []
        for param_ix, param in enumerate(param_space):
            if param.type == "continuous":
                p = np.random.uniform(param.low, param.high, size=None)
                sample.append(p)
                raw_sample.append(p)
            elif param.type == "discrete":
                options = param.options
                p = np.random.choice(options, size=None, replace=False)
                sample.append(p)
                raw_sample.append(p)
            elif param.type == "categorical":
                options = param.options
                p = np.random.choice(options, size=None, replace=False)
                feat = cat_param_to_feat(param, p, has_descriptors)
                sample.extend(feat)  # extend because feat is vector
                raw_sample.append(p)
        proposals.append(sample)
        raw_proposals.append(raw_sample)
    proposals = np.array(proposals)
    raw_proposals = np.array(raw_proposals)

    return proposals, raw_proposals


def forward_standardize(
    data: Union[torch.Tensor, np.ndarray],
    means: Union[torch.Tensor, np.ndarray],
    stds: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    """forward standardize the data"""
    return (data - means) / stds


def reverse_standardize(
    data: Union[torch.Tensor, np.ndarray],
    means: Union[torch.Tensor, np.ndarray],
    stds: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    """un-standardize the data"""
    return (data * stds) + means


def forward_normalize(
    data: Union[torch.Tensor, np.ndarray],
    min_: Union[torch.Tensor, np.ndarray],
    max_: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    """forward normalize the data"""
    ixs = np.where(np.abs(max_ - min_) < 1e-10)[0]
    if not ixs.size == 0:
        max_[ixs] = np.ones_like(ixs)
        min_[ixs] = np.zeros_like(ixs)
    return (data - min_) / (max_ - min_)


def reverse_normalize(
    data: Union[torch.Tensor, np.ndarray],
    min_: Union[torch.Tensor, np.ndarray],
    max_: Union[torch.Tensor, np.ndarray],
) -> Union[torch.Tensor, np.ndarray]:
    """un-normlaize the data"""
    ixs = np.where(np.abs(max_ - min_) < 1e-10)[0]
    if not ixs.size == 0:
        max_[ixs] = np.ones_like(ixs)
        min_[ixs] = np.zeros_like(ixs)
    return data * (max_ - min_) + min_


def param_vector_to_dict(
    sample: np.ndarray,
    param_space: ParameterSpace,
) -> Dict[str, Union[float, int, str]]:
    """parse single sample and return a dict"""
    param_dict = {}
    for param_index, param in enumerate(param_space):
        param_type = param.type
        if param_type == "continuous":
            param_dict[param.name] = sample[param_index]

        elif param_type == "categorical":
            options = param.options
            selected_option_idx = int(sample[param_index])
            selected_option = options[selected_option_idx]
            param_dict[param.name] = selected_option

        elif param_type == "discrete":
            options = param.options
            selected_option_idx = int(sample[param_index])
            selected_option = options[selected_option_idx]
            param_dict[param.name] = selected_option
    return param_dict


def flip_source_tasks(source_tasks):
    """flip the sign of the source tasks if the
    optimization goal is maximization
    """
    flipped_source_tasks = []
    for task in source_tasks:
        flipped_source_tasks.append(
            {
                "params": task["params"],
                "values": -1 * task["values"],
            }
        )

    return flipped_source_tasks


def partition(S):
    """..."""
    if len(S) == 1:
        yield [S]
        return

    first = S[0]
    for smaller in partition(S[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1 :]
        yield [[first]] + smaller


def gen_partitions(S):
    """
    generate all possible partitions of Ns-element set S

    Args:
        S (list): list of non-functional parameters S
    """
    return [p for _, p in enumerate(partition(S), 1)]


class Scaler:
    SUPP_TYPES = ["standardization", "normalization", "identity"]

    """ scaler for source data
    Args:
        type (str): scaling type, supported are standardization or
                    normalization
        data (str): data type, either params or values
    """

    def __init__(self, param_type, value_type):
        if not param_type in self.SUPP_TYPES:
            raise NotImplementedError
        else:
            self.param_type = param_type

        if not value_type in self.SUPP_TYPES:
            raise NotImplementedError
        else:
            self.value_type = value_type

        self.is_fit = False

    def _compute_stats(self, source_tasks):
        """computes the stats for an entire set of source tasks"""
        # join source tasks params
        all_source_params = []
        all_source_values = []
        for task in source_tasks:
            all_source_params.append(task["params"])
            all_source_values.append(task["values"])
        all_source_params = np.concatenate(np.array(all_source_params), axis=0)
        all_source_values = np.concatenate(np.array(all_source_values), axis=0)

        # make sure these are 2d
        assert len(all_source_params.shape) == 2
        assert len(all_source_values.shape) == 2

        # compute stats for parameters
        param_stats = {}
        if self.param_type == "normalization":
            param_stats["max"] = np.amax(all_source_params, axis=0)
            param_stats["min"] = np.amin(all_source_params, axis=0)
        elif self.param_type == "standardization":
            # need the mean and the standard deviation
            param_stats["mean"] = np.mean(all_source_params, axis=0)
            std = np.std(all_source_params, axis=0)
            param_stats["std"] = np.where(std == 0.0, 1.0, std)
        self.param_stats = param_stats

        # compute stats for values
        value_stats = {}
        if self.value_type == "normalization":
            value_stats["max"] = np.amax(all_source_values, axis=0)
            value_stats["min"] = np.amin(all_source_values, axis=0)
        elif self.value_type == "standardization":
            # need the mean and the standard deviation
            value_stats["mean"] = np.mean(all_source_values, axis=0)
            std = np.std(all_source_values, axis=0)
            value_stats["std"] = np.where(std == 0.0, 1.0, std)
        self.value_stats = value_stats

    def fit_transform_tasks(self, source_tasks):
        """compute stats for a set of source tasks"""
        # register the stats
        self._compute_stats(source_tasks)

        transformed_source_tasks = []

        for task in source_tasks:
            trans_task = {}
            # params
            if self.param_type == "normalization":
                trans_task["params"] = self.normalize(
                    task["params"],
                    self.param_stats["min"],
                    self.param_stats["max"],
                    "forward",
                )
            elif self.param_type == "standardization":
                trans_task["params"] = self.standardize(
                    task["params"],
                    self.param_stats["mean"],
                    self.param_stats["std"],
                    "forward",
                )
            elif self.param_type == "identity":
                trans_task["params"] = self.identity(task["params"], "forward")
            # values
            if self.value_type == "normalization":
                trans_task["values"] = self.normalize(
                    task["values"],
                    self.value_stats["min"],
                    self.value_stats["max"],
                    "forward",
                )
            elif self.value_type == "standardization":
                trans_task["values"] = self.standardize(
                    task["values"],
                    self.value_stats["mean"],
                    self.value_stats["std"],
                    "forward",
                )
            elif self.value_type == "identity":
                trans_task["values"] = self.identity(task["values"], "forward")

            transformed_source_tasks.append(trans_task)

        return transformed_source_tasks

    def identity(self, x, direction):
        """identity transformation"""
        return x

    def standardize(self, x, mean, std, direction):
        """standardize the data given parameters"""
        if direction == "forward":
            return (x - mean) / std
        elif direction == "reverse":
            return x * std + mean

    def normalize(self, x, min, max, direction):
        """normalize the data given parameters"""
        if direction == "forward":
            return (x - min) / (max - min)
        elif direction == "reverse":
            return x * (max - min) + min

    def transform_tasks(self, tasks):
        """transform a set of tasks"""
        transformed_source_tasks = []
        for task in tasks:
            trans_task = {}
            # params
            trans_task["params"] = self.transform(
                task["params"], type="params"
            )
            # values
            trans_task["values"] = self.transform(
                task["values"], type="values"
            )
        transformed_source_tasks.append(trans_task)

        return transformed_source_tasks

    def transform(self, sample, type):
        """transforms a sample"""
        # make sure this sample is 2d array
        assert len(sample.shape) == 2

        if type == "params":
            if self.param_type == "normalization":
                return self.normalize(
                    sample,
                    self.param_stats["min"],
                    self.param_stats["max"],
                    "forward",
                )
            elif self.param_type == "standardization":
                return self.standardize(
                    sample,
                    self.param_stats["mean"],
                    self.param_stats["std"],
                    "forward",
                )
            elif self.param_type == "identity":
                return self.identity(sample, "forward")
        elif type == "values":
            if self.value_type == "normalization":
                return self.normalize(
                    sample,
                    self.value_stats["min"],
                    self.value_stats["max"],
                    "forward",
                )
            elif self.value_type == "standardization":
                return self.standardize(
                    sample,
                    self.value_stats["mean"],
                    self.value_stats["std"],
                    "forward",
                )
            elif self.value_type == "identity":
                return self.identity(sample, "forward")

    def inverse_transform(self, sample, type):
        """perform inverse transformation"""
        # make sure this sample is 2d array
        assert len(sample.shape) == 2

        if type == "params":
            if self.param_type == "normalization":
                return self.normalize(
                    sample,
                    self.param_stats["min"],
                    self.param_stats["max"],
                    "forward",
                )
            elif self.param_type == "standardization":
                return self.standardize(
                    sample,
                    self.param_stats["mean"],
                    self.param_stats["std"],
                    "forward",
                )
            elif self.param_type == "identity":
                return self.identity(sample, "reverse")
        elif type == "values":
            if self.value_type == "normalization":
                return self.normalize(
                    sample,
                    self.value_stats["min"],
                    self.value_stats["max"],
                    "forward",
                )
            elif self.value_type == "standardization":
                return self.standardize(
                    sample,
                    self.value_stats["mean"],
                    self.value_stats["std"],
                    "forward",
                )
            elif self.value_type == "identity":
                return self.identity(sample, "reverse")
