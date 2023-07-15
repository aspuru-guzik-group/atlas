#!/usr/bin/env python

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gpytorch
import numpy as np
import torch
from botorch.fit import fit_gpytorch_mll


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

from atlas.acquisition_optimizers import (
    GeneticOptimizer,
    GradientOptimizer,
    PymooGAOptimizer
)
from atlas.base.base import BasePlanner

from atlas.utils.planner_utils import reverse_normalize



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
        fixed_cost: Optional[float] = 5.0,
        **kwargs: Any,
    ):
        local_args = {
            key: val for key, val in locals().items() if key != "self"
        }
        super().__init__(**local_args)
        self.tkwargs = {
            "dtype": torch.double,
            "device": "cpu", #torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }

        # check if we have any fidelity param dims specified
        if not self.fidelity_params:
            Logger.log('You must specify at least one fidelity dimension to use this planner', 'FATAL')

        # verify the fidelities
        if not self.fidelities:
            Logger.log('You must specify the fidelities use this planner', 'FATAL')
        elif not self.fidelities[-1]==1.0:
            Logger.log('Conventionally the target (final) fidelity is set to 1.0', 'FATAL')
        else:
            self.fidelities = torch.Tensor(self.fidelities).to(**self.tkwargs)

        # target fidelity must always be 1.0
        self.target_fidelities = {self.fidelity_params: 1.0}

        # set cost model and utility
        if not self.fixed_cost:
            Logger.log('Fixed cost value not specified, resorting to defualt of 5.0', 'WARNING')
            self.fixed_cost = 5.
        self.cost_model = AffineFidelityCostModel(fidelity_weights=self.target_fidelities, fixed_cost=self.fixed_cost)
        self.cost_aware_utility = InverseCostWeightedUtility(cost_model=self.cost_model)

        # set current ask fidelity (default to None)
        self.current_ask_fidelity = None



    def _project(self, X: torch.Tensor):
        return project_to_target_fidelity(X=X, target_fidelities=self.target_fidelities)
        

    def set_ask_fidelity(self, fidelity: float) -> None:
        # quickly validate the intended fidelity level
        if not fidelity in self.fidelities:
            Logger.log(
                f'Fidelity level {fidelity} not in the available options : {self.fidelities}', 
                'FATAL',
            )
        Logger.log(f'Setting ask fidelity level to {fidelity}', 'WARNING')
        setattr(self, 'current_ask_fidelity', fidelity)

    def reset_ask_fidelity(self) -> None:
        Logger.log(f'Resetting ask fidelity level', 'WARNING')
        setattr(self, 'current_ask_fidelity', None)

    def get_fixed_params(self) -> Dict[int, float]:
        if not self.current_ask_fidelity:
            return {}
        else:
            return [{self.fidelity_params: self.current_ask_fidelity}]
        
 
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
    
    def _get_mfkg_acqf(self) -> qMultiFidelityKnowledgeGradient:
        # build acquisition function
        curr_val_acqf = FixedFeatureAcquisitionFunction(
            acq_function=PosteriorMean(self.reg_model),
            d=len(self.param_space),
            columns=[self.fidelity_params],
            values=[1], # TODO: is this right for all cases??
        )

        # optimize the fixed feature acquisition function
        _, current_value = optimize_acqf(
            acq_function=curr_val_acqf,
            bounds=self.params_obj.bounds[:, np.logical_not(self.params_obj.fidelity_params_mask)].to(**self.tkwargs),
            q=1, # batch_size always 1 here
            num_restarts=5,
            raw_samples=100,
            options={"batch_limit": 10, "maxiter": 200},
        )

        mfkg_acqf = qMultiFidelityKnowledgeGradient(
            model=self.reg_model,
            num_fantasies=128, # change this to 128 for production
            current_value=current_value,
            cost_aware_utilty=self.cost_aware_utility,
            project=self._project,
        )
        return mfkg_acqf.to(self.tkwargs['device'])


    def _ask(self) -> List[ParameterVector]:
        """query the planner for a batch of new parameter points to measure"""

        # if we have all nan values, just continue with initial design
        if np.logical_or(
            len(self._values) < self.num_init_design,
            np.all(np.isnan(self._values)),
        ):
            return_params = self.initial_design()

        else:
            # convert bounds min max stuff for multi-fidelity problem
            self.params_obj.set_multi_fidelity_param_attrs(self.fidelity_params)

            (
                self.train_x_scaled_cla,
                self.train_y_scaled_cla,
                self.train_x_scaled_reg,
                self.train_y_scaled_reg,
            ) = self.build_train_data(return_scaled_input=True)

            # TODO: handle unknown constraints

            # build and fit regression surrogate model
            self.reg_model = self.build_train_regression_gp(
                self.train_x_scaled_reg.to(**self.tkwargs), self.train_y_scaled_reg.to(**self.tkwargs),
            )

            # TODO: handle unknown constraints

            use_reg_only = True
            self.cla_model, self.cla_likelihood = None, None
            self.cla_surr_min_, self.cla_surr_max_ = None, None

            # get mfkg acqusition function
            mfkg_acqf = self._get_mfkg_acqf()
            
            # get the fixed parameters according to current ask fidelity level
            fixed_params = self.get_fixed_params()

            if self.acquisition_optimizer_kind == 'gradient':
                # optimize the knowledge gradient
                fixed_features_list = [{self.fidelity_params:fidelity} for fidelity in self.fidelities]
                res, _ = optimize_acqf_mixed(
                    acq_function=mfkg_acqf,
                    bounds=self.params_obj.bounds.to(**self.tkwargs),
                    fixed_features_list=fixed_features_list,
                    q=self.batch_size,
                    num_restarts=5,
                    raw_samples=100,
                    options={'batch_limit':5, 'max_iter': 200},
                )
                res_unscaled_np = reverse_normalize(
                res.detach().numpy(), 
                self.params_obj._mins_x, 
                self.params_obj._maxs_x,
                )
                # convert to parameter vector
                return_params = []
                for res in res_unscaled_np:
                    return_params.append(
                    ParameterVector().from_dict({
                            p.name: r for p, r in zip(self.param_space, res)
                        }) 
                    )

            elif self.acquisition_optimizer_kind == 'pymoo':
                # TODO: fix genetic and pymoo optimizer for this problem
                # try pymoo acqf optimization
                acquisition_optimizer = PymooGAOptimizer(
                    self.params_obj,
                        self.acquisition_type,
                        mfkg_acqf,#self.acqf,
                        self.known_constraints,
                        self.batch_size,
                        self.feas_strategy,
                        None,# self.fca_constraint
                        self._params,
                        {},#self.timings_dict,
                        use_reg_only=use_reg_only,
                        fixed_params=fixed_params,
                        num_fantasies=128,
                )

                return_params = acquisition_optimizer.optimize()

            else:
                msg = 'MultiFidelityPlanner is only compatible with "gradient" and "pymoo" acquisition optimizers'
                Logger.log(msg, 'FATAL')


            # get the cost value
            #cost = self.cost_model(res).sum()


            print(return_params)



        return return_params


    


