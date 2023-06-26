#!/usr/bin/env python

import os
import pickle
import sys
import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gpytorch
import numpy as np
import itertools
import olympus
import torch
from botorch.acquisition import (
    ExpectedImprovement,
    UpperConfidenceBound,
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qUpperConfidenceBound,
)
from botorch.fit import fit_gpytorch_model
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.optim import (
    optimize_acqf,
    optimize_acqf_discrete,
    optimize_acqf_mixed,
)
from gpytorch.mlls import ExactMarginalLogLikelihood
from olympus import ParameterVector
from olympus.campaigns import ParameterSpace
from olympus.planners import AbstractPlanner, CustomPlanner, Planner
from olympus.scalarizers import Scalarizer

from golem import *
from golem import Golem

from atlas import Logger
from atlas.optimizers.acqfs import (
    MedusaAcquisition,
    create_available_options,
)
from atlas.optimizers.acquisition_optimizers import GeneticGeneralOptimizer
from atlas.optimizers.base import BasePlanner
from atlas.optimizers.gps import (
    CategoricalSingleTaskGP,
    ClassificationGPMatern,
)
from atlas.optimizers.params import Parameters
from atlas.optimizers.utils import (
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

from atlas.utils.golem_utils import get_golem_dists


class MedusaPlanner(BasePlanner):
    """..."""

    def __init__(
        self,
        goal: str,
        feas_strategy: Optional[str] = "naive-0",
        feas_param: Optional[float] = 0.2,
        use_min_filter: bool = True,
        batch_size: int = 1,
        batched_strategy: str = "sequential",  # sequential or greedy
        random_seed: Optional[int] = None,
        use_descriptors: bool = False,
        num_init_design: int = 5,
        init_design_strategy: str = "random",
        vgp_iters: int = 2000,
        vgp_lr: float = 0.1,
        max_jitter: float = 1e-1,
        cla_threshold: float = 0.5,
        known_constraints: Optional[List[Callable]] = None,
        is_moo: bool = False,
        value_space: Optional[ParameterSpace] = None,
        scalarizer_kind: Optional[str] = "Hypervolume",
        moo_params: Dict[str, Union[str, float, int, bool, List]] = {},
        goals: Optional[List[str]] = None,
        golem_config: Optional[Dict[str, Any]] = None,
        # MEDUSA-SPECIFIC ARGUMENTS
        # -----------------------------
        general_parameters: List[int] = None, # indices of general parameters in param space
        max_Ng: Optional[int] = None,
        use_random_acqf: bool = False, # random sampling acquisition function (for baseline)
        **kwargs: Any,
    ):
        local_args = {
            key: val for key, val in locals().items() if key != "self"
        }
        super().__init__(**local_args)
        
        self.acquisition_type = 'medusa'
        self.max_Ng = max_Ng
        self.use_random_acqf = use_random_acqf


        # check that we have some general parameters
        if not self.general_parameters:
            msg = 'No general parameters define. MEDUSA must have at least one general parameter'
            Logger.log(msg, 'FATAL')

        # TODO: things to validate about general parameters
        # num general parameters is less than the total num parameters
        # all defined general parameters are either categorical or discrete
        # tmp check that num general parameters = 1 (to be extended)
        # ...

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

        # set functional parameter space object
        self.func_param_space = ParameterSpace()
        for param_ix, param in enumerate(self.param_space):
            if not param_ix in self.general_parameters:
                self.func_param_space.add(param)
        

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

        # if using random acqusition function for baseline
        # construct random sampling planner from Olympus
        if self.use_random_acqf:
            self.random_acqf = olympus.planners.RandomSearch(goal='minimize')
            self.random_acqf.set_param_space(self.param_space)



    def build_train_regression_gp(
        self, train_x: torch.Tensor, train_y: torch.Tensor
    ) -> gpytorch.models.ExactGP:
        """Build the regression GP model and likelihood"""
        # infer the model based on the parameter types
        if self.problem_type in [
            "fully_continuous",
            "fully_discrete",
            "mixed_disc_cont",
        ]:
            model = SingleTaskGP(train_x, train_y)
        elif self.problem_type == "fully_categorical":
            if self.has_descriptors:
                # we have some descriptors, use the Matern kernel
                model = SingleTaskGP(train_x, train_y)
            else:
                # if we have no descriptors, use a Categorical kernel
                # based on the HammingDistance
                model = CategoricalSingleTaskGP(train_x, train_y)
        elif "mixed_cat_" in self.problem_type:
            if self.has_descriptors:
                # we have some descriptors, use the Matern kernel
                model = SingleTaskGP(train_x, train_y)
            else:
                cat_dims = get_cat_dims(self.param_space)
                model = MixedSingleTaskGP(train_x, train_y, cat_dims=cat_dims)

        else:
            raise NotImplementedError

        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        # fit the GP
        start_time = time.time()
        with gpytorch.settings.cholesky_jitter(self.max_jitter):
            fit_gpytorch_model(mll)
        gp_train_time = time.time() - start_time
        Logger.log(
            f"Regression surrogate GP trained in {round(gp_train_time,3)} sec",
            "INFO",
        )
        return model

    def _ask(self) -> List[ParameterVector]:
        """query the planner for a batch of new parameter points to measure"""
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

            use_p_feas_only = False
            # check to see if we are using the naive approaches
            if "naive-" in self.feas_strategy:
                infeas_ix = torch.where(self.train_y_scaled_cla == 1.0)[0]
                feas_ix = torch.where(self.train_y_scaled_cla == 0.0)[0]
                # checking if we have at least one objective function measurement
                #  and at least one infeasible point (i.e. at least one point to replace)
                if np.logical_and(
                    self.train_y_scaled_reg.size(0) >= 1,
                    infeas_ix.shape[0] >= 1,
                ):
                    if self.feas_strategy == "naive-replace":
                        # NOTE: check to see if we have a trained regression surrogate model
                        # if not, wait for the following iteration to make replacements
                        if hasattr(self, "reg_model"):
                            # if we have a trained regression model, go ahead and make replacement
                            new_train_y_scaled_reg = deepcopy(
                                self.train_y_scaled_cla
                            ).double()

                            input = self.train_x_scaled_cla[infeas_ix].double()

                            posterior = self.reg_model.posterior(X=input)
                            pred_mu = posterior.mean.detach()

                            new_train_y_scaled_reg[
                                infeas_ix
                            ] = pred_mu.squeeze(-1)
                            new_train_y_scaled_reg[
                                feas_ix
                            ] = self.train_y_scaled_reg.squeeze(-1)

                            self.train_x_scaled_reg = deepcopy(
                                self.train_x_scaled_cla
                            ).double()
                            self.train_y_scaled_reg = (
                                new_train_y_scaled_reg.view(
                                    self.train_y_scaled_cla.size(0), 1
                                ).double()
                            )

                        else:
                            use_p_feas_only = True

                    elif self.feas_strategy == "naive-0":
                        new_train_y_scaled_reg = deepcopy(
                            self.train_y_scaled_cla
                        ).double()

                        worst_obj = torch.amax(
                            self.train_y_scaled_reg[
                                ~self.train_y_scaled_reg.isnan()
                            ]
                        )

                        to_replace = torch.ones(infeas_ix.size()) * worst_obj

                        new_train_y_scaled_reg[infeas_ix] = to_replace.double()
                        new_train_y_scaled_reg[
                            feas_ix
                        ] = self.train_y_scaled_reg.squeeze()

                        self.train_x_scaled_reg = (
                            self.train_x_scaled_cla.double()
                        )
                        self.train_y_scaled_reg = new_train_y_scaled_reg.view(
                            self.train_y_scaled_cla.size(0), 1
                        )

                    else:
                        raise NotImplementedError
                else:
                    # if we are not able to use the naive strategies, propose randomly
                    # do nothing at all and use the feasibilty surrogate as the acquisition
                    use_p_feas_only = True

            # builds and fits the regression surrogate model
            self.reg_model = self.build_train_regression_gp(
                self.train_x_scaled_reg, self.train_y_scaled_reg
            )

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
            #f_best_argmin = torch.argmin(self.train_y_scaled_reg)
            # TODO: using UCB for MEDUSA acqf for now so we dont have to worry about
            # the incumbent point --> how do extend to EI in the future??
            #f_best_scaled = self.train_y_scaled_reg[f_best_argmin][0].float()

            # compute the ratio of infeasible to total points
            infeas_ratio = (
                torch.sum(self.train_y_scaled_cla)
                / self.train_x_scaled_cla.size(0)
            ).item()
            # get the approximate max and min of the acquisition function without the feasibility contribution
            # NOTE: we are not getting the acqf max min in this case - should we do it in the future??
            # probably will need this for unknown constraints
            #acqf_min_max = self.get_aqcf_min_max(self.reg_model, f_best_scaled)

            # generate all general params representations with empty features for functional params
            X_sns_empty, _ = self.generate_X_sns()
            # get functional dims mask
            functional_dims = np.logical_not(self.params_obj.exp_general_mask)


            # instantiate MEDUSA acquisition function
            self.acqf = MedusaAcquisition(
                    reg_model=self.reg_model,
                    params_obj=self.params_obj,
                    X_sns_empty=X_sns_empty,
                    functional_dims=functional_dims,
                    # ... 
                )

            if not self.use_random_acqf:
                Logger.log('Proceeding with MEDUSA acquisition function optimization', 'WARNING')
                # medusa always uses genetic general acqusition optimizer
                acquisition_optimizer = GeneticGeneralOptimizer(
                    params_obj=self.params_obj,
                    acquisition_type=self.acquisition_type,
                    acqf=self.acqf,
                    known_constraints=self.known_constraints,
                    batch_size=self.batch_size,
                    feas_strategy=self.feas_strategy,
                    fca_constraint=self.fca_constraint,
                    params=self._params,
                    timings_dict=self.timings_dict,
                    max_Ng=self.max_Ng,
                    func_param_space=self.func_param_space,
                    mode='acqf',
                )

                return_params = acquisition_optimizer.optimize()

            else:
                Logger.log('Proceeding with RANDOM acquisition function!', 'WARNING')
                # generate a random sample to measure next
                self.random_acqf._tell(iteration=len(self._values))
                return_params = self.random_acqf.ask()
                print('RANDOM RETURN PARAMS : ', return_params)


        return return_params
    

    def generate_X_sns(self):
        # generate Cartesian product space of the general parameter options
        param_options = []
        for ix in self.params_obj.general_dims:
            param_options.append(self.param_space[ix].options)

        cart_product = list(itertools.product(*param_options))
        cart_product = [list(elem) for elem in cart_product]

        X_sns_empty = torch.empty(
            size=(len(cart_product), self.params_obj.expanded_dims)
        ).double()
        general_expanded = []
        general_raw = []
        for elem in cart_product:
            # convert to ohe and add to currently available options
            ohe, raw = [], []
            for val, obj in zip(elem, self.param_space):
                if obj.type == "categorical":
                    ohe.append(
                        cat_param_to_feat(
                            obj, val, self.params_obj.has_descriptors
                        )
                    )
                    raw.append(val)
                else:
                    ohe.append([val])
            general_expanded.append(np.concatenate(ohe))
            general_raw.append(raw)

        general_expanded = torch.tensor(np.array(general_expanded))

        X_sns_empty[:, self.params_obj.exp_general_mask] = general_expanded
        # forward normalize
        X_sns_empty = forward_normalize(
            X_sns_empty,
            self.params_obj._mins_x,
            self.params_obj._maxs_x,
        )
        # TODO: careful of the batch size, will need to change this
        X_sns_empty = torch.unsqueeze(X_sns_empty, 1)

        return X_sns_empty, general_raw


    def optimize_proposals(self):
        """ Use genetic algorithm optimizer to optimize X_func and G proposals

        This function should return a predicted lead candiddate (X_func and G) for
        each of Ng=1,...,num_general_options

        """
        poss_Ngs = np.arange(
            len(self.param_space[self.general_parameters[0]].options)
        )+1

        best_proposals = {}

        for Ng in poss_Ngs:
            proposal_optimizer = GeneticGeneralOptimizer(
                params_obj=self.params_obj,
                acquisition_type=self.acquisition_type,
                acqf=self.acqf,
                known_constraints=self.known_constraints,
                batch_size=self.batch_size,
                feas_strategy=self.feas_strategy,
                fca_constraint=self.fca_constraint,
                params=self._params,
                timings_dict=self.timings_dict,
                max_Ng=self.max_Ng,
                func_param_space=self.func_param_space,
                fix_Ng=Ng,
                mode='proposal',
            )

            # TODO: need to somehow fix Ng here???
            X_func, G = proposal_optimizer.optimize()

            # print('='*50)
            # print('')
            # print(Ng)
            # print(X_func)
            # print(G)
            # print('')

            best_proposals[Ng] = {'X_func': X_func, 'G': G}


        return best_proposals