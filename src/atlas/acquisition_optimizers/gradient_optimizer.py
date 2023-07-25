#!/usr/bin/env python

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import botorch
import numpy as np
import rich
import torch
from botorch.acquisition import AcquisitionFunction
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from olympus import ParameterVector

from atlas import Logger, tkwargs
from atlas.acquisition_functions.acqf_utils import (
    create_available_options,
    get_batch_initial_conditions,
)
from atlas.acquisition_functions.acqfs import FeasibilityAwareAcquisition
from atlas.acquisition_optimizers.base_optimizer import AcquisitionOptimizer
from atlas.params.params import Parameters
from atlas.utils.planner_utils import (
    get_fixed_features_list,
    infer_problem_type,
    reverse_normalize,
)


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
        acqf_args=None,
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
        self.batched_strategy = batched_strategy
        self.fca_constraint = fca_constraint
        self.use_reg_only = use_reg_only
        self.has_descriptors = self.params_obj.has_descriptors
        self._params = params
        self._mins_x = self.params_obj._mins_x
        self._maxs_x = self.params_obj._maxs_x

        self.choices_feat, self.choices_cat = None, None

        self.kind = "gradient"

    def _optimize(self):
        best_idx = None  # only needed for the fully categorical case

        if self.acquisition_type == "general":
            func_dims = self.params_obj.functional_dims
            exp_func_dims = self.params_obj.exp_functional_dims

            # check to see if all functional parameters are continuous
            if all(
                [self.param_space[ix].type == "continuous" for ix in func_dims]
            ):
                results = self._optimize_mixed_general()
            elif all(
                [
                    self.param_space[ix].type == "categorical"
                    for ix in func_dims
                ]
            ):
                results, best_idx = self._optimize_fully_categorical()

            else:
                # TODO: this is broken for now...
                results, _ = self._optimize_acqf_mixed()
                # msg = 'This is not yet implemented. Try again later!'
                # Logger.log(msg, 'FATAL')

        else:
            if self.problem_type == "fully_continuous":
                results = self._optimize_fully_continuous()
            elif self.problem_type in [
                "mixed_cat_cont",
                "mixed_disc_cont",
                "mixed_cat_disc_cont",
            ]:
                results, best_idx = self._optimize_mixed(
                    acqf=self.acqf,
                    bounds=self.bounds,
                    num_restarts=30,
                    batch_size=self.batch_size,
                    raw_samples=800,
                    inequality_constraints=None,
                    equality_constraints=None,
                )
            elif self.problem_type in [
                "fully_categorical",
                "fully_discrete",
                "mixed_cat_disc",
            ]:
                results, best_idx = self._optimize_fully_categorical()

        return self.postprocess_results(results, best_idx)

    def _optimize_fully_continuous(self):
        """SLSQP optimzer strategy for fully-continuous parameter spaces"""

        (
            nonlinear_inequality_constraints,
            batch_initial_conditions,
            _,
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

    def _optimize_fully_categorical(self):
        """Special case of _optimize_cartesian_product where we first need to
        generate the expended choices
        """
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

        results, best_idx = self._optimize_cartesian_product(
            acqf=self.acqf,
            batch_size=self.batch_size,
            max_batch_size=512,
            choices=self.choices_feat.float(),
            unique=True,
        )

        return results, best_idx

    def _optimize_cartesian_product(
        self,
        acqf: FeasibilityAwareAcquisition,
        batch_size: int,
        max_batch_size: int,
        choices: torch.Tensor,  # (`num_choices` x `param_dim`)
        unique: bool = True,
    ):
        """Optimize acquisition function for Cartesian product
        parameter spaces, i.e. fully-categorical, fully-discrete,
        and mixed categorical-discrete problems

        For batch_size > 1, this strategy uses sequential conditioning based
        on the kridging believer strategy. This effectively fixes the
        posterior mean for the entire sequentual batch selection
        procedure and updates the variance when pending recommendations
        are selected.

        Args:
                acqf (FeasibilityAwareAcquisition): acqusition function instance

        Returns:
                Tuple containing (`batch_size` x `param_dim`) tensor of candidates and
                the corresponding acqf values
        """

        original_choices_batched = torch.clone(choices)
        choices_batched = choices.unsqueeze(-2)

        if batch_size > 1:
            # batch selection by sequential conditioning
            candidate_list, acqf_val_list = [], []
            init_pending_params = acqf.set_pending_params(pending_params=None)

            for batch_idx in range(batch_size):
                with torch.no_grad():
                    acqf_vals = [
                        acqf(X_)
                        for X_ in choices_batched.split(max_batch_size)
                    ]
                    acqf_vals = torch.cat(acqf_vals)

                best_idx = torch.argmax(acqf_vals)
                candidate_list.append(choices_batched[best_idx])
                acqf_val_list.append(acqf_vals[best_idx])

                # set pending parameters
                candidates = torch.cat(candidate_list, dim=-2)
                acqf.set_pending_params(
                    torch.cat([init_pending_params, candidates], dim=-2)
                    if init_pending_params is not None
                    else candidates
                )
                # remove most recent selected candidate from available options
                if unique:
                    choices_batched = torch.cat(
                        [
                            choices_batched[:best_idx],
                            choices_batched[best_idx + 1 :],
                        ]
                    )

            # reset acqf to initial state
            _ = acqf.set_pending_params(pending_params=None)
            # need to return the original indices of the selected candidates
            best_idxs = []
            for (
                candidate
            ) in candidate_list:  # each candidate is shape (1, num_features)
                bools = [
                    torch.all(candidate[0] == original_choices_batched[i, :])
                    for i in range(original_choices_batched.shape[0])
                ]
                assert bools.count(True) == 1
                best_idxs.append(np.where(bools)[0][0])
            return candidates, best_idxs

        # otherwise we have batch_size=1, just take argmax over all available options
        with torch.no_grad():
            acq_vals = torch.cat(
                [acqf(X_) for X_ in choices_batched.split(max_batch_size)]
            )
        best_idx = [torch.argmax(acq_vals).detach()]

        return [choices[best_idx]], best_idx

    def _optimize_mixed_general(self):
        """Special case where we have discrete/categorical general params
        and fully-continuous functional/non-general problem. Breaks up the
        problem and uses SLSQP to optimize the continuous functional params
        """
        functional_mask = np.logical_not(self.params_obj.exp_general_mask)

        func_bounds = self.bounds[:, functional_mask]

        (
            nonlinear_inequality_constraints,
            batch_initial_conditions,
            _,
        ) = self.gen_initial_conditions(num_restarts=30)

        func_batch_initial_conditions = batch_initial_conditions[
            :, :, functional_mask
        ]

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
        X_sns = torch.empty(
            (self.batch_size, self.params_obj.expanded_dims)
        ).double()
        for ix, result in enumerate(results):
            X_sns[ix, functional_mask] = result
            X_sns[ix, self.params_obj.exp_general_mask] = torch.tensor(
                batch_initial_conditions[
                    0, 0, self.params_obj.exp_general_mask
                ]
            )

        return X_sns

    def _optimize_mixed(
        self,
        acqf,
        bounds,
        num_restarts,
        batch_size,
        raw_samples=None,
        inequality_constraints=None,
        equality_constraints=None,
        **kwargs,
    ):
        """Optimize acquisition functions that contain at least one continuous
        parameter and at least one categorical/discrete parameter, i.e. the
        `mixed_cat_cont`, `mixed_disc_cont`, and `mixed_cat_disc_cont` problem types.

        If `batch_size` > 1, this strategy uses seqential batch selection with
        conditioning on pending parameters


        This method is inspired by `_optimize_acqf_mixed` from:
        https://github.com/pytorch/botorch/blob/main/botorch/optim/optimize.py

        Args:


        """
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
            _,
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

        if batch_size == 1:
            ff_candidate_list, ff_acqf_val_list = [], []
            # iterate through all the fixed featutes and optimize the continuous
            # part of the parameter space
            # fixed features and cart_prod choices have the same ordering
            for fixed_features in fixed_features_list:
                candidate, acq_value = optimize_acqf(
                    acq_function=acqf,
                    bounds=bounds,
                    q=batch_size,
                    num_restarts=30,
                    raw_samples=800,
                    options={},
                    inequality_constraints=inequality_constraints,
                    equality_constraints=equality_constraints,
                    fixed_features=fixed_features,
                    batch_initial_conditions=batch_initial_conditions,
                    return_best_only=True,
                )
                ff_candidate_list.append(candidate)
                ff_acqf_val_list.append(acq_value)

            ff_acqf_val = torch.stack(ff_acqf_val_list)
            best_idx = torch.argmax(ff_acqf_val)

            return ff_candidate_list[best_idx], [best_idx.detach()]

        # batch_size > 1, batch selection via sequential conditioning
        init_pending_params = acqf.pending_params
        candidates = torch.tensor([], **tkwargs)

        for batch_ix in range(self.batch_size):
            (
                candidate,
                acqf_val,
            ) = self._optimize_mixed(  # recursive call to this method
                acqf=acqf,
                bounds=bounds,
                num_restarts=30,
                batch_size=1,
                raw_samples=800,
                inequality_constraints=None,
                equality_constraints=None,
            )

            candidates = torch.cat([candidates, candidate], dim=-2)

            acqf.set_pending_params(
                torch.cat([init_pending_params, candidates], dim=-2)
                if init_pending_params is not None
                else candidates
            )

        acqf.set_pending_params(pending_params=None)

        acqf_vals = acqf(candidates)

        return candidates, acq_vals

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
                    if self.acquisition_type == "general":
                        cat_choice = self.param_space[
                            self.params_obj.general_dims[0]
                        ].options[0]
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

    # TODO: to be deleted??
    def dummy_constraint(self, X):
        """dummy constraint that always returns value >= 0., i.e.
        evaluates any parameter space point as feasible
        """
        return torch.ones(X.shape[0]).unsqueeze(-1)
