#!/usr/bin/env python

import time
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from copy import deepcopy

import gpytorch
import numpy as np
import torch
from botorch.acquisition import PosteriorMean
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.knowledge_gradient import (
    qMultiFidelityKnowledgeGradient,
)
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.fit import fit_gpytorch_mll
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.gp_regression_fidelity import SingleTaskMultiFidelityGP
from botorch.optim.optimize import optimize_acqf, optimize_acqf_mixed
from gpytorch.mlls import ExactMarginalLogLikelihood
from olympus import ParameterVector
from olympus.campaigns import ParameterSpace

from atlas import Logger, tkwargs
from atlas.acquisition_optimizers import (
    GeneticOptimizer,
    GradientOptimizer,
    PymooGAOptimizer,
)
from atlas.acquisition_functions.acqfs import get_acqf_instance
from atlas.acquisition_functions.acqf_utils import create_available_options
from atlas.base.base import BasePlanner
from atlas.utils.planner_utils import reverse_normalize, infer_problem_type



class MultiFidelityPlanner(BasePlanner):
    """ Multi-fideltiy experiment planner using the trace-aware knowledge gradient 
    acquisition function
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
        tkwargs = {
            "dtype": torch.double,
            "device": "cpu",  # torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        }

        # check if we have any fidelity param dims specified
        if not self.fidelity_params:
            Logger.log(
                "You must specify at least one fidelity dimension to use this planner",
                "FATAL",
            )


        # verify the fidelities
        if not self.fidelities:
            Logger.log(
                "You must specify the fidelities use this planner", "FATAL"
            )
        elif not self.fidelities[-1] == 1.0:
            Logger.log(
                "Conventionally the target (final) fidelity is set to 1.0",
                "FATAL",
            )
        else:
            self.fidelities = torch.Tensor(self.fidelities).to(**tkwargs)

        # target fidelity must always be 1.0
        self.target_fidelities = {self.fidelity_params: 1.0}

        # set cost model and utility
        if not self.fixed_cost:
            Logger.log(
                "Fixed cost value not specified, resorting to defualt of 5.0",
                "WARNING",
            )
            self.fixed_cost = 5.0
        self.cost_model = AffineFidelityCostModel(
            fidelity_weights=self.target_fidelities, fixed_cost=self.fixed_cost
        )
        self.cost_aware_utility = InverseCostWeightedUtility(
            cost_model=self.cost_model
        )

        # set current ask fidelity (default to None)
        self.current_ask_fidelity = None


    

        Logger.log_chapter(title='Initial design phase')

    def _project(self, X: torch.Tensor):
        return project_to_target_fidelity(
            X=X, target_fidelities=self.target_fidelities
        )

    def set_ask_fidelity(self, fidelity: float) -> None:
        # quickly validate the intended fidelity level
        if not fidelity in self.fidelities:
            Logger.log(
                f"Fidelity level {fidelity} not in the available options : {self.fidelities}",
                "FATAL",
            )
        Logger.log(f"Setting ask fidelity level to {fidelity}", "WARNING")
        setattr(self, "current_ask_fidelity", fidelity)

    def reset_ask_fidelity(self) -> None:
        Logger.log(f"Resetting ask fidelity level", "WARNING")
        setattr(self, "current_ask_fidelity", None)

    def get_fixed_params(self) -> Dict[int, float]:
        if not self.current_ask_fidelity:
            return {}
        else:
            return [{self.fidelity_params: self.current_ask_fidelity}]

    def build_train_regression_gp(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
    ) -> Any:
        """Build the regression model and likelihood"""

        Logger.log_chapter(title='Training regression surrogate model')
        
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

    def _optimize_curr_val_mfkg_acqf(self, curr_val_acqf: qMultiFidelityKnowledgeGradient):
        
        if self.func_problem_type == 'fully_continuous':
            # optimize the fixed feature acquisition function with gradients
            _, current_value = optimize_acqf(
                acq_function=curr_val_acqf,
                bounds=self.params_obj.bounds[
                    :, np.logical_not(self.params_obj.fidelity_params_mask)
                ].to(**tkwargs),
                q=1,  # batch_size always 1 here
                num_restarts=5,
                raw_samples=100,
                options={"batch_limit": 10, "maxiter": 200},
            )
        elif self.func_problem_type in ['fully_discrete', 'fully_categorical']:
            # optimize cartesian product
            choices_feat, choices_cat = create_available_options(
                self.param_space,
                [],#self._params, # TODO: figure out is this is correct
                fca_constraint_callable=None,
                known_constraint_callables=self.known_constraints,
                normalize=self.has_descriptors,
                has_descriptors=self.has_descriptors,
                mins_x=self.params_obj._mins_x,
                maxs_x=self.params_obj._maxs_x,
            )
            # print(choices_feat)
            # print(choices_cat)
            # print(self.params_obj.fidelity_params_mask)
            #func_choices_feat = choices_feat[:,np.logical_not(self.params_obj.fidelity_params_mask)]
            
            func_choices_feat = choices_feat[:,1:]
            # print(func_choices_feat)
            # print(self.has_descriptors)

            func_choices_feat = func_choices_feat.view(func_choices_feat.shape[0],1,func_choices_feat.shape[-1])
            # print(func_choices_feat.shape)
            # full pass on acquisition function
            acqf_vals = curr_val_acqf(func_choices_feat).detach()
            current_value = torch.amax(acqf_vals)


        else:
            raise NotImplementedError

        
        # print(current_value)
        # print(current_value.shape)

        return current_value
    

    def recommend_target_fidelity(self, batch_size:int=1) -> List[ParameterVector]:
        """ make batch of recommendations on the target fidelity level with 
        the surrogate posterior mean as the acquisition function (greedy)

        """
        # greedy_acqf = FixedFeatureAcquisitionFunction(
        #     acq_function=PosteriorMean(self.reg_model),
        #     d=self.params_obj.expanded_dims,
        #     columns=[self.fidelity_params],
        #     values=[1],
        # )
        # setattr(greedy_acqf, 'feas_strategy', self.feas_strategy)
        # greedy_acqf.to(tkwargs["device"])

        acqf_args = dict(
            acquisition_optimizer_kind=self.acquisition_optimizer_kind,
            params_obj=self.params_obj,
            problem_type=self.problem_type,
            feas_strategy=self.feas_strategy,
            feas_param=self.feas_param,
            infeas_ratio=0., # TODO: implement
            use_reg_only=True,
            f_best_scaled=None, # TODO: implement
            batch_size=batch_size,
            use_min_filter=False, # TODO: implement 
        )

        greedy_acqf = get_acqf_instance(
            acquisition_type='greedy',
            reg_model=self.reg_model,
            cla_model=None, # TODO: implement
            cla_likelihood=None, # TODO: implement
            acqf_args=acqf_args,
        )

        acquisition_optimizer = PymooGAOptimizer(
            self.params_obj,
            self.acquisition_type,
            greedy_acqf,
            self.known_constraints,
            batch_size,
            self.feas_strategy,
            None,  # self.fca_constraint
            self._params,
            {},  # self.timings_dict,
            use_reg_only=False,
            fixed_params=[{self.fidelity_params: 1.0}], # always target fidelity
            num_fantasies=0,
            pop_size=800,
            num_gen=1000,
        )

        return_params = acquisition_optimizer.optimize()


        return return_params
        


    def _get_mfkg_acqf(self) -> qMultiFidelityKnowledgeGradient:
        # build acquisition function
        curr_val_acqf = FixedFeatureAcquisitionFunction(
            acq_function=PosteriorMean(self.reg_model),
            d=self.params_obj.expanded_dims, # remove fidelity param?? #len(self.param_space),
            columns=[self.fidelity_params],
            values=[1],  # TODO: is this right for all cases??
        )

        current_value = self._optimize_curr_val_mfkg_acqf(curr_val_acqf)

        mfkg_acqf = qMultiFidelityKnowledgeGradient(
            model=self.reg_model,
            num_fantasies=128,  # change this to 128 for production
            current_value=current_value,
            cost_aware_utility=self.cost_aware_utility,
            project=self._project,
        )
        # TODO: bad hack fix this
        setattr(mfkg_acqf, 'feas_strategy', self.feas_strategy)
        return mfkg_acqf.to(tkwargs["device"])
    

    def handle_init_design_ask_fidelity(self, return_params: List[ParameterVector]):
        new_return_params = []
        for param_vec in return_params:
            new_param_vec = deepcopy(param_vec)
            new_param_vec[self.fidelity_params_name] = self.current_ask_fidelity
            new_return_params.append(new_param_vec)
        return new_return_params



    def _ask(self) -> List[ParameterVector]:
        """query the planner for a batch of new parameter points to measure"""

        # infer the problem type minus the fidelity parameter
        # TODO: move this somewhere else...
        func_params = [
            param for ix, param in enumerate(self.param_space) if ix!=self.fidelity_params
        ]
        self.func_param_space = ParameterSpace()
        for param in func_params:
            self.func_param_space.add(param)
        self.func_problem_type = infer_problem_type(self.func_param_space)
        # print('func problem type : ', self.func_problem_type)
        self.fidelity_params_name = self.param_space.param_names[self.fidelity_params]
        


        # if we have all nan values, just continue with initial design
        if np.logical_or(
            len(self._values) < self.num_init_design,
            np.all(np.isnan(self._values)),
        ):
            return_params = self.initial_design()
            if self.current_ask_fidelity is not None:
                return_params = self.handle_init_design_ask_fidelity(return_params=return_params)
            
        
        else:
            # convert bounds min max stuff for multi-fidelity problem
            self.params_obj.set_multi_fidelity_param_attrs(
                self.fidelity_params
            )

            (
                self.train_x_scaled_cla,
                self.train_y_scaled_cla,
                self.train_x_scaled_reg,
                self.train_y_scaled_reg,
            ) = self.build_train_data(return_scaled_input=True)

            # TODO: handle unknown constraints

            # build and fit regression surrogate model
            self.reg_model = self.build_train_regression_gp(
                self.train_x_scaled_reg.to(**tkwargs),
                self.train_y_scaled_reg.to(**tkwargs),
            )

            # TODO: handle unknown constraints

            use_reg_only = True
            self.cla_model, self.cla_likelihood = None, None
            self.cla_surr_min_, self.cla_surr_max_ = None, None

            # get mfkg acqusition function
            mfkg_acqf = self._get_mfkg_acqf()

            # get the fixed parameters according to current ask fidelity level
            fixed_params = self.get_fixed_params()


            if self.acquisition_optimizer_kind == "gradient":
                # optimize the knowledge gradient
                fixed_features_list = [
                    {self.fidelity_params: fidelity}
                    for fidelity in self.fidelities
                ]
                res, _ = optimize_acqf_mixed(
                    acq_function=mfkg_acqf,
                    bounds=self.params_obj.bounds.to(**tkwargs),
                    fixed_features_list=fixed_features_list,
                    q=self.batch_size,
                    num_restarts=5,
                    raw_samples=100,
                    options={"batch_limit": 5, "max_iter": 200},
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
                        ParameterVector().from_dict(
                            {p.name: r for p, r in zip(self.param_space, res)}
                        )
                    )

            elif self.acquisition_optimizer_kind == "pymoo":
                # TODO: fix genetic and pymoo optimizer for this problem
                # try pymoo acqf optimization
                acquisition_optimizer = PymooGAOptimizer(
                    self.params_obj,
                    self.acquisition_type,
                    mfkg_acqf,  # self.acqf,
                    self.known_constraints,
                    self.batch_size,
                    self.feas_strategy,
                    None,  # self.fca_constraint
                    self._params,
                    {},  # self.timings_dict,
                    use_reg_only=use_reg_only,
                    fixed_params=fixed_params,
                    num_fantasies=128,
                    pop_size=800,
                    num_gen=1000,
                )

                return_params = acquisition_optimizer.optimize()

            else:
                msg = 'MultiFidelityPlanner is only compatible with "gradient" and "pymoo" acquisition optimizers'
                Logger.log(msg, "FATAL")

            # get the cost value
            # cost = self.cost_model(res).sum()

        return return_params
