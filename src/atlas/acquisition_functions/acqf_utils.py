#!/usr/bin/env python

import itertools
from copy import deepcopy
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from botorch.acquisition.acquisition import MCSamplerMixin
from botorch.acquisition.objective import IdentityMCObjective
from botorch.utils.transforms import _verify_output_shape

from atlas import Logger
from atlas.utils.planner_utils import (
    cat_param_to_feat,
    forward_normalize,
    propose_randomly,
)


def concatenate_pending_params(
    method: Callable[[Any, torch.Tensor], Any],
) -> Callable[[Any, torch.Tensor], Any]:
    """Decorator to add pending parameters to MC acqf argument
    Works if the MonteCarloAcquisition instance has attribute `pending_params`
    that is not None
    """

    @wraps(method)
    def decorated(mc_acqf: Any, X: torch.Tensor, **kwargs: Any) -> Any:
        if mc_acqf.pending_params is not None:
            X = torch.cat(
                [X, match_batch_shape(mc_acqf.pending_params, X)], dim=-2
            )
        return method(mc_acqf, X, **kwargs)

    return decorated


def t_batch_mode_transform(
    expected_q: Optional[int] = None,
    assert_output_shape: bool = True,
) -> Callable[[Callable[[Any, Any], Any]], Callable[[Any, Any], Any],]:
    r"""Factory for decorators enabling consistent t-batch behavior.

    This method creates decorators for instance methods to transform an input tensor
    `X` to t-batch mode (i.e. with at least 3 dimensions). This assumes the tensor
    has a q-batch dimension. The decorator also checks the q-batch size if `expected_q`
    is provided, and the output shape if `assert_output_shape` is `True`.

    Args:
        expected_q: The expected q-batch size of `X`. If specified, this will raise an
            AssertionError if `X`'s q-batch size does not equal expected_q.
        assert_output_shape: If `True`, this will raise an AssertionError if the
            output shape does not match either the t-batch shape of `X`,
            or the `acqf.model.batch_shape` for acquisition functions using
            batched models.

    Returns:
        The decorated instance method.

    Example:
        >>> class ExampleClass:
        >>>     @t_batch_mode_transform(expected_q=1)
        >>>     def single_q_method(self, X):
        >>>         ...
        >>>
        >>>     @t_batch_mode_transform()
        >>>     def arbitrary_q_method(self, X):
        >>>         ...

    Code taken and modified from:
        https://github.com/pytorch/botorch/blob/main/botorch/utils/transforms.py#L298

    """

    def decorator(
        method: Callable[[Any, Any], Any],
    ) -> Callable[[Any, Any], Any]:
        @wraps(method)
        def decorated(acqf: Any, X: Any, *args: Any, **kwargs: Any) -> Any:
            # Allow using acquisition functions for other inputs (e.g. lists of strings)
            if not isinstance(X, torch.Tensor):
                return method(acqf, X, *args, **kwargs)

            if X.dim() < 2:
                raise ValueError(
                    f"{type(acqf).__name__} requires X to have at least 2 dimensions,"
                    f" but received X with only {X.dim()} dimensions."
                )
            elif expected_q is not None and X.shape[-2] != expected_q:
                raise AssertionError(
                    f"Expected X to be `batch_shape x q={expected_q} x d`, but"
                    f" got X with shape {X.shape}."
                )
            # add t-batch dim
            X = X if X.dim() > 2 else X.unsqueeze(0)
            output = method(acqf, X, *args, **kwargs)
            if hasattr(
                acqf, "reg_model"
            ):  # and is_fully_bayesian(acqf.reg_model): -> NOTE: Not relevant in Atlas
                output = output.mean(dim=-1)
            if assert_output_shape and not _verify_output_shape(
                acqf=acqf,
                X=X,
                output=output,
            ):
                raise AssertionError(
                    "Expected the output shape to match either the t-batch shape of "
                    "X, or the `model.batch_shape` in the case of acquisition "
                    "functions using batch models; but got output with shape "
                    f"{output.shape} for X with shape {X.shape}."
                )
            return output

        return decorated

    return decorator


def match_batch_shape(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Expand shape of tensor `X` to match that of `Y`"""
    return X.expand(X.shape[: -(Y.dim())] + Y.shape[:-2] + X.shape[-2:])


def get_batch_initial_conditions(
    num_restarts,
    batch_size,
    param_space,
    known_constraints,
    fca_constraint,
    mins_x,
    maxs_x,
    has_descriptors,
    num_chances=15,
    return_raw=False,
):
    """generate batches of initial conditions for a
    random restart optimization subject to some constraints. This uses
    rejection sampling, and might get very inefficient for parameter spaces with
    a large infeasible fraction.
    Args:
                    num_restarts (int): number of optimization restarts
                    batch_size (int): number of samples to recommend per ask/tell call (fixed to 1)
                    param_space (obj): Olympus parameter space object for the given problem
                    known_constriants (list): list of callables which specifies the user-level constraint function(s)
                    fca_constraint (callable): fca constraint callable
                    mins_x (np.array): minimum values of each parameter space dimension
                    maxs_x (np.array): maximum values of each parameter
                    num_chances (int):
    Returns:
                    a torch.tensor with shape (num_restarts, batch_size, num_dims)
                    of initial optimization conditions
    """
    # take 20*num_restarts points randomly and evaluate the constraint function on all of
    # them, if we have enough, proceed, if not proceed to sequential rejection sampling
    num_raw_samples = 250 * num_restarts
    raw_samples, raw_proposals = propose_randomly(
        num_proposals=num_raw_samples * batch_size,
        param_space=param_space,
        has_descriptors=has_descriptors,
    )

    # forward normalize the randomly generated samples (expanded rep only)
    raw_samples = forward_normalize(raw_samples, mins_x, maxs_x)

    # reshape to botorch format
    raw_samples = torch.tensor(raw_samples).view(
        raw_samples.shape[0] // batch_size, batch_size, raw_samples.shape[1]
    )

    if known_constraints.is_empty and fca_constraint == []:
        # no constraints, return samples
        batch_initial_conditions = raw_samples
        batch_initial_conditions_raw = raw_proposals

    else:
        # ----------------
        # fca constraint
        # ----------------
        if type(fca_constraint) == callable:
            # we have an fca constraint
            # evaluate using expanded torch representation
            constraint_val = fca_constraint(raw_samples)
            if len(constraint_val.shape) == 1:
                constraint_val = constraint_val.view(
                    constraint_val.shape[0], 1
                )
            constraint_vals.append(constraint_val)

            fca_feas_idx = torch.where(torch.all(constraint_vals >= 0, dim=1))[
                0
            ]
        else:
            # no fca constraint
            fca_feas_idx = torch.arange(raw_samples.shape[0])

        # ------------------------------
        # user-level known constraints
        # ------------------------------
        if not known_constraints.is_empty:
            # we have some user-level known constraints
            # use raw_propsals here, user-level known constraints
            # evaluated on compressed representation of parameters
            constraint_vals = []
            # loop through all known constriaint callables
            for constraint_callable in known_constraints:
                # returns True if feasible, False if infeasible
                kc_res = [
                    constraint_callable(params) for params in raw_proposals
                ]
                constraint_vals.append(kc_res)

            constraint_vals = torch.tensor(constraint_vals)

            # get indices for which known constraints are satisfied
            kc_feas_idx = torch.where(torch.all(constraint_vals, dim=0))[0]
        else:
            # no user-level known constraints
            kc_feas_idx = torch.arange(raw_samples.shape[0])

        # find the union of the two sets of feasible indices
        feas_idx = np.intersect1d(kc_feas_idx, fca_feas_idx)

        # project onto original proposals
        batch_initial_conditions = raw_samples[feas_idx, :, :]
        batch_initial_conditions_raw = raw_proposals[feas_idx, :]

    if batch_initial_conditions.shape[0] >= num_restarts:
        # if we have more samples than we need, truncate
        if return_raw:
            return (
                batch_initial_conditions[:num_restarts, :, :],
                batch_initial_conditions_raw[:num_restarts, :],
            )
        return batch_initial_conditions[:num_restarts, :, :]

    elif 0 < batch_initial_conditions.shape[0] < num_restarts:
        # we dont have enough proposals, sample around the feasible ones we have...
        Logger.log(
            f"Insufficient initial samples, resorting to local sampling for {num_chances} iterations...",
            "WARNING",
        )
        for chance in range(num_chances):
            batch_initial_conditions = sample_around_x(
                batch_initial_conditions, constraint_callable
            )
            Logger.log(
                f"Chance : {chance+1}/{num_chances}\t# samples : {batch_initial_conditions.shape[0]}",
                "INFO",
            )

            if batch_initial_conditions.shape[0] >= num_restarts:
                if return_raw:
                    return (
                        batch_initial_conditions[:num_restarts, :, :],
                        batch_initial_conditions_raw[:num_restarts, :],
                    )
                return batch_initial_conditions[:num_restarts, :, :]
        return None, None
    else:
        Logger.log(
            f"Insufficient initial samples after {num_chances} sampling iterations. Resorting to unconstrained acquisition",
            "WARNING",
        )
        return None, None

    assert len(batch_initial_conditions.size()) == 3

    if return_raw:
        return batch_initial_conditions, batch_initial_conditions_raw
    return batch_initial_conditions


def sample_around_x(raw_samples, constraint_callable):
    """draw samples around points which we already know are feasible by adding
    some Gaussian noise to them
    """
    tiled_raw_samples = raw_samples.tile((20, 1, 1))
    means = deepcopy(tiled_raw_samples)
    stds = torch.ones_like(means) * 0.05
    perturb_samples = tiled_raw_samples + torch.normal(means, stds)
    # # project the values
    # perturb_samples = torch.where(perturb_samples>1., 1., perturb_samples)
    # perturb_samples = torch.where(perturb_samples<0., 0., perturb_samples)
    inputs = torch.cat((raw_samples, perturb_samples))

    constraint_vals = []
    for constraint in constraint_callable:
        constraint_val = constraint(inputs)
        if len(constraint_val.shape) == 1:
            constraint_val = constraint_val.view(constraint_val.shape[0], 1)
        constraint_vals.append(constraint_val)

    if len(constraint_vals) == 2:
        constraint_vals = torch.cat(constraint_vals, dim=1)
        feas_ix = torch.where(torch.all(constraint_vals >= 0))[0]
    elif len(constraint_vals) == 1:
        constraint_vals = torch.tensor(constraint_vals[0])
        feas_ix = torch.where(constraint_vals >= 0)[0]

    batch_initial_conditions = inputs[feas_ix, :, :]

    return batch_initial_conditions


def create_available_options(
    param_space,
    params,
    fca_constraint_callable,
    known_constraint_callables,
    normalize,
    mins_x,
    maxs_x,
    has_descriptors,
    max_options=int(1.5e10),
):
    """build cartesian product space of options, then remove options
    which have already been measured. Returns an (num_options, num_dims)
    torch tensor with all possible options

    If the parameter space is mixed, build and return the Cartesian product space of
    only the categorical and discrete parameters.

    Args:
                    param_space (obj): Olympus parameter space object
                    params (list): parameters from the current Campaign
                    fca_constraint_callable (Callable): FCA constraint callable
                    known_constraint_callables (List[Callable]): list of known constraints
                    mins_x (np.array): minimum values of each parameter space dimension
                    maxs_x (np.array): maximum values of each parameter
                    has_descriptors (bool): wether or not cat options have descriptors
                    max_options (int): max number of allowed options to return. If CP space
                        exceeds this value, return random subset
    """

    # make sure params are proper data type
    real_params = []
    for param in params:
        p = []
        for ix, elem in enumerate(param):
            if param_space[ix].type == "categorical":
                p.append(elem)
            else:
                p.append(float(elem))
        real_params.append(p)

    # params = [list(elem) for elem in params]
    # get the relevant parmeters -> only categorical and discrete
    relevant_params = [
        p for p in param_space if p.type in ["categorical", "discrete"]
    ]
    param_names = [p.name for p in relevant_params]
    param_options = [p.options for p in relevant_params]

    cart_product = list(itertools.product(*param_options))
    cart_product = [list(elem) for elem in cart_product]

    if len(cart_product) > max_options:
        Logger.log(
            f"CP space of cardnality {len(cart_product)} exceeds max allowed options. Proceeding with random subset..",
            "WARNING",
        )
        select_idx = np.random.choice(
            np.arange(len(cart_product)),
            size=(max_options,),
            replace=False,
            p=None,
        )
        cart_product = [cart_product[idx] for idx in select_idx]

    if len(param_names) == len(param_space):
        # no continuous parameters
        # remove options that we have measured already
        current_avail_feat = []
        current_avail_cat = []

        for elem in cart_product:
            if elem not in real_params:
                # convert to ohe and add to currently available options
                ohe = []
                for val, obj in zip(elem, param_space):
                    if obj.type == "categorical":
                        ohe.append(
                            cat_param_to_feat(obj, val, has_descriptors)
                        )
                    else:
                        ohe.append([val])
                current_avail_feat.append(np.concatenate(ohe))
                current_avail_cat.append(elem)

        current_avail_feat_unconst = np.array(current_avail_feat)
        current_avail_cat_unconst = np.array(current_avail_cat)

        # # check known constraints not associated with FCA (if any)
        if known_constraint_callables is not None:
            # known constraints
            kc_results = []
            for cat_unconst in current_avail_cat_unconst:
                if all([kc(cat_unconst) for kc in known_constraint_callables]):
                    # feasible
                    kc_results.append(True)
                else:
                    kc_results.append(False)
            feas_mask = np.where(kc_results)[0]
            current_avail_feat_kc = current_avail_feat_unconst[feas_mask]
            current_avail_cat_kc = current_avail_cat_unconst[feas_mask]
        else:
            current_avail_feat_kc = current_avail_feat_unconst
            current_avail_cat_kc = current_avail_cat_unconst

        # forward normalize the options before evaluating the fca constraint
        if normalize:
            current_avail_feat_kc = forward_normalize(
                current_avail_feat_kc, mins_x, maxs_x
            )

        current_avail_feat_kc = torch.tensor(current_avail_feat_kc)

        # remove options which are infeasible given the feasibility surrogate model
        # and the threshold
        if fca_constraint_callable is not None:
            # FCA approach, apply feasibility constraint
            constraint_input = current_avail_feat_kc.view(
                current_avail_feat_kc.shape[0],
                1,
                current_avail_feat_kc.shape[1],
            )
            constraint_vals = fca_constraint_callable(constraint_input)
            feas_mask = torch.where(constraint_vals >= 0.0)[0]
            print(
                f"{feas_mask.shape[0]}/{len(current_avail_feat)} options are feasible"
            )
            if feas_mask.shape[0] == 0:
                msg = "No feasible samples after FCA constraint, resorting back to full space"
                Logger.log(msg, "WARNING")
                # if we have zero feasible samples
                # resort back to the full set of unobserved options
                current_avail_feat = current_avail_feat_kc
                current_avail_cat = current_avail_cat_kc
            else:
                current_avail_feat = current_avail_feat_kc[feas_mask]
                current_avail_cat = current_avail_cat_kc[
                    feas_mask.detach().numpy()
                ]
        else:
            current_avail_feat = current_avail_feat_kc
            current_avail_cat = current_avail_cat_kc

        return current_avail_feat, current_avail_cat

    else:
        # at least one continuous parameter, no need to remove any options
        current_avail_feat = []
        current_avail_cat = []

        for elem in cart_product:
            # convert to ohe and add to currently available options
            ohe = []
            for val, obj in zip(elem, param_space):
                if obj.type == "categorical":
                    ohe.append(cat_param_to_feat(obj, val, has_descriptors))
                else:
                    ohe.append([val])
            current_avail_feat.append(np.concatenate(ohe))
            current_avail_cat.append(elem)

        current_avail_feat_unconst = np.array(current_avail_feat)
        current_avail_cat_unconst = np.array(current_avail_cat)

        # TODO: may need to add the constraint checking here too...
        # forward normalize the options before evaluating the constaints
        # if normalize:
        # 	current_avail_feat_unconst = forward_normalize(current_avail_feat_unconst, mins_x, maxs_x)

        current_avail_feat_unconst = torch.tensor(current_avail_feat_unconst)

        return current_avail_feat_unconst, current_avail_cat_unconst
