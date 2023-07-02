#!/usr/bin/env python

import os
import pickle
import sys
import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gpytorch
import numpy as np
import torch
from botorch.acquisition import (
    ExpectedImprovement,
    UpperConfidenceBound,
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qUpperConfidenceBound,
)
from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP, SingleTaskGP



from gpytorch.mlls import ExactMarginalLogLikelihood

# new stuff -----------

from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP 
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import PosteriorMean
from botorch.optim.optimize import optimize_acqf
from botorch.optim.optimize import optimize_acqf_mixed
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction

from botorch.acquisition.utils import project_to_target_fidelity

#----------------------

from olympus import ParameterVector
from olympus.campaigns import ParameterSpace

from atlas import Logger
from atlas.acquisition_functions.acqfs import (
    FeasibilityAwareEI,
    FeasibilityAwareGeneral,
    FeasibilityAwareLCB,
    FeasibilityAwareQEI,
    FeasibilityAwareUCB,
    FeasibilityAwareVarainceBased,
    LowerConfidenceBound,
    VarianceBased,
    create_available_options,
)
from atlas.acquisition_optimizers import (
    GeneticOptimizer,
    GradientOptimizer,
    PymooGAOptimizer
)
from atlas.base.base import BasePlanner
from atlas.gps.gps import (
    CategoricalSingleTaskGP,
    ClassificationGPMatern,
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



class MultiFidelityPlanner(BasePlanner):
    """
    """

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
        acquisition_type: str = "ei",  # qei, ei, ucb, variance, general
        acquisition_optimizer_kind: str = "gradient",  # gradient, genetic, pymoo
        vgp_iters: int = 2000,
        vgp_lr: float = 0.1,
        max_jitter: float = 1e-1,
        cla_threshold: float = 0.5,
        known_constraints: Optional[List[Callable]] = None,
        compositional_params: Optional[List[int]] = None,
        permutation_params: Optional[List[int]] = None,
        batch_constrained_params: Optional[List[int]] = None,
        general_parameters: Optional[List[int]] = None,
        is_moo: bool = False,
        value_space: Optional[ParameterSpace] = None,
        scalarizer_kind: Optional[str] = "Hypervolume",
        moo_params: Dict[str, Union[str, float, int, bool, List]] = {},
        goals: Optional[List[str]] = None,
        golem_config: Optional[Dict[str, Any]] = None,
        # new stuff ----------------
        fidelity_params: int = None, 
        fidelities: List[float] = None,
        fixed_cost: Optional[float] = None,
        **kwargs: Any,
    ):
        local_args = {
            key: val for key, val in locals().items() if key != "self"
        }
        super().__init__(**local_args)

        # check if we have any fidelity param dims specified
        if not self.fidelity_params:
            Logger.log('You must specify at least one fidelity dimension to use this planner', 'FATAL')

        # verify the fidelities
        if not self.fidelities:
            Logger.log('You must specify the fidelities use this planner', 'FATAL')
        elif not self.fidelities[-1]==1.0:
            Logger.log('Conventionally the target (final) fidelity is set to 1.0', 'FATAL')
        else:
            self.fidelities = torch.Tensor(self.fidelities).double()

        # target fidelity must always be 1.0
        self.target_fidelities = {self.fidelity_params: 1.0}

        # set cost model and utility
        if not self.fixed_cost:
            Logger.log('Fixed cost value not specified, resorting to defualt of 5.0', 'WARNING')
            self.fixed_cost = 5.
        self.cost_model = AffineFidelityCostModel(fidelity_weights=self.target_fidelities, fixed_cost=self.fixed_cost)
        self.cost_aware_utility = InverseCostWeightedUtility(cost_model=self.cost_model)

        # set current ask fidelity (default to target fidelity)
        self.current_ask_fidelity = 1.


    def _project(X: torch.Tensor):
        project_to_target_fidelity(X=X, target_fidelities=target_fidelities)
        

    def set_current_ask_fidelity(fidelity: float) -> None:
        setattr(self, 'current_ask_fidelity', fidelity)
 
    def build_train_regression_gp(
        self, train_x: torch.Tensor, train_y: torch.Tensor,
    ) -> Any:
        """ Build the regression model and likelihood
        """
        # TODO: only using continuous  parameters now and always using discrete 
        # fidelities

        # create model
        model = SingleTaskMultiFidelityGP(
            train_x, train_y, data_fidelity=self.fidelity_params
        ) 
        # create likelihood
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # fit the multi-fidelity GP
        start_time = time.time()
        with gpytorch.settings.cholesky_jitter(self.max_jitter):
            fit_gpytorch_mll(mll)
        gp_train_time = time.time() - start_time
        Logger.log(
            f"Regression surrogate GP trained in {round(gp_train_time,3)} sec",
            "INFO",
        )
        return model



    def _ask(self) -> List[ParameterVector]:

        """query the planner for a batch of new parameter points to measure"""
        # if we have all nan values, just continue with initial design
        if np.logical_or(
            len(self._values) < self.num_init_design,
            np.all(np.isnan(self._values)),
        ):
            return_params = self.initial_design()

        else:
            (
                self.train_x_scaled_cla,
                self.train_y_scaled_cla,
                self.train_x_scaled_reg,
                self.train_y_scaled_reg,
            ) = self.build_train_data()
            
            # TODO: handle unknown constraints

            # build and fit regression surrogate model
            self.reg_model = self.build_train_regression_gp(
                self.train_x_scaled_reg.double(), self.train_y_scaled_reg.double(),
            )

            # TODO: handle unknown constraints

            use_reg_only = True
            self.cla_model, self.cla_likelihood = None, None
            self.cla_surr_min_, self.cla_surr_max_ = None, None

            # build acquisition function
            curr_val_acqf = FixedFeatureAcquisitionFunction(
                acq_function=PosteriorMean(self.reg_model),
                d=len(self.param_space),
                columns=[self.fidelity_params],
                values=[1], # TODO: is this right for all cases??
            )
            print(self.params_obj.bounds)

            # optimize the fixed feature acquisition function
            _, current_value = optimize_acqf(
                acq_function=curr_val_acqf,
                bounds=self.params_obj.bounds[:, :-1], # this will only work if last dim in fidelity
                q=1, # batch_size always 1 here
                num_restarts=10,
                raw_samples=1024,
                options={"batch_limit": 10, "maxiter": 200},
            )

            self.mfkg_acqf = qMultiFidelityKnowledgeGradient(
                model=self.reg_model,
                num_fantasies=20, # change this to 128 for production
                current_value=current_value,
                cost_aware_utilty=self.cost_aware_utility,
                project=self._project,
            )

            # optimize the knowledge gradient
            fixed_features_list = [{self.fidelity_params:fidelity} for fidelity in self.fidelities]
            # res, _ = optimize_acqf_mixed(
            #     acq_function=self.mfkg_acqf,
            #     bounds=self.params_obj.bounds,
            #     fixed_features_list=fixed_features_list,
            #     q=self.batch_size,
            #     num_restarts=5,
            #     raw_samples=128,
            #     options={'batch_limit':5, 'max_iter': 200},
            # )

            # try pymoo acsqf optimization
            acquisition_optimizer = PymooGAOptimizer(
                self.params_obj,
                    self.acquisition_type,
                    self.mfkg_acqf,#self.acqf,
                    self.known_constraints,
                    self.batch_size,
                    self.feas_strategy,
                    None,# self.fca_constraint
                    self._params,
                    {},#self.timings_dict,
                    use_reg_only=use_reg_only,
            )

            return_params = acquisition_optimizer.optimize()
            


            # get the cost value
            #cost = self.cost_model(res).sum()
            
            #return_params = res.detach()


            print(return_params)
            #print(return_params.shape)

            quit()
            # convert to list of Olympus parameter vectors - include the fidelity param(s)



        return return_params


    


