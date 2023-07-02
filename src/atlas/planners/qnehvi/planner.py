#!/usr/bin/env python

import os
import pickle
import sys
import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gpytorch
import numpy as np
import olympus
import torch
from botorch.acquisition import (
    ExpectedImprovement,
    UpperConfidenceBound,
    qExpectedImprovement,
)
from botorch.acquisition.multi_objective.monte_carlo import (
	qNoisyExpectedHypervolumeImprovement,
)

from botorch import fit_gpytorch_mll
from botorch.fit import fit_gpytorch_mll
from botorch.models import MixedSingleTaskGP, SingleTaskGP
from botorch.models.kernels.categorical import CategoricalKernel
from botorch.models.model_list_gp_regression import ModelListGP
from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
from botorch.optim import (
    optimize_acqf,
    optimize_acqf_discrete,
    optimize_acqf_mixed,
)

from botorch.sampling.normal import SobolQMCNormalSampler

from gpytorch.mlls import ExactMarginalLogLikelihood
from olympus import ParameterVector
from olympus.campaigns import ParameterSpace
from olympus.planners import AbstractPlanner, CustomPlanner, Planner
from olympus.scalarizers import Scalarizer
from olympus.utils.misc import get_hypervolume

from atlas import Logger
from atlas.optimizers.acqfs import (
    FeasibilityAwareEI,
    FeasibilityAwareqNEHVI,
    create_available_options,
)
from atlas.optimizers.acquisition_optimizers import (
    GradientOptimizer, GeneticOptimizer
)
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


class qNEHVIPlanner(BasePlanner):   
    """ Wrapper for Bayesian optimization of multiple noisy
    objectives with expected hypervolume improvement (qNEHVI). 
    This planner reduces to using the expected hypervolume 
    improvement criterion when there is only a single objective. 
    
    Args: 

    """
    def __init__(
        self,
        goal: str,
        feas_strategy: Optional[str] = "naive-0",
        feas_param: Optional[float] = 0.2,
        use_min_filter: bool = True,
        batch_size: int = 1,
        batched_strategy: str = 'sequential', # sequential or greedy
        random_seed: Optional[int] = None,
        use_descriptors: bool = False,
        num_init_design: int = 5,
        init_design_strategy: str = "random",
        acquisition_optimizer_kind: str = "gradient",  # gradient, genetic
        vgp_iters: int = 2000,
        vgp_lr: float = 0.1,
        max_jitter: float = 1e-1,
        cla_threshold: float = 0.5,
        known_constraints: Optional[List[Callable]] = None,
        general_parameters: Optional[List[int]] = None,
        is_moo: bool = False,
        value_space: Optional[ParameterSpace] = None,
        goals: Optional[List[str]] = None,
        golem_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        local_args = {
            key: val for key, val in locals().items() if key != "self"
        }
        super().__init__(**local_args)

        if is_moo and goal=='maximize':
            Logger.log('Goal must be set to minimize for multiobjective problem', 'FATAL')


    def build_train_data(self) -> Tuple[torch.Tensor, torch.tensor]:
        """ build the training dataset at each iteration. 
        Overrides the method of the same name in BasePlanner. Here, 
        we do not scalarize the targets, if moo, rather we return 
        a 2d tensor of target values.
        """
        if self.is_moo:
            # parameters should be the same for each objective
            # nans should be in the same locations for each objective
            feas_ix = np.where(~np.isnan(self._values[:, 0]))[0]
            # generate the classification dataset
            params_cla = self._params.copy()
            values_cla = np.where(
                ~np.isnan(self._values[:, 0]), 0.0, self._values[:, 0]
            )
            train_y_cla = np.where(np.isnan(values_cla), 1.0, values_cla)
            # generate the regression dataset
            params_reg = self._params[feas_ix].reshape(-1, 1)
            train_y_reg = self._values[
                feas_ix, :
            ]  # (num_feas_observations, num_objectives)

        else:
            feas_ix = np.where(~np.isnan(self._values))[0]
            # generate the classification dataset
            params_cla = self._params.copy()
            values_cla = np.where(~np.isnan(self._values), 0.0, self._values)
            train_y_cla = np.where(np.isnan(values_cla), 1.0, values_cla)

            # generate the regression dataset
            params_reg = self._params[feas_ix].reshape(-1, 1)
            train_y_reg = self._values[feas_ix].reshape(-1, 1)

        train_x_cla, train_x_reg = [], []

        # adapt the data from olympus form to torch tensors
        for ix in range(self._values.shape[0]):
            sample_x = []
            for param_ix, (space_true, element) in enumerate(
                zip(self.param_space, params_cla[ix])
            ):
                if self.param_space[param_ix].type == "categorical":
                    feat = cat_param_to_feat(
                        space_true,
                        element,
                        has_descriptors=self.has_descriptors,
                    )
                    sample_x.extend(feat)
                else:
                    sample_x.append(float(element))
            train_x_cla.append(sample_x)
            if ix in feas_ix:
                train_x_reg.append(sample_x)

        train_x_cla, train_x_reg = np.array(train_x_cla), np.array(train_x_reg)

        # if we are using Golem, fit Golem to current regression training data,
        # and replace data with its predictions
        if self.golem is not None:
            self.golem.fit(X=train_x_reg, y=train_y_reg.flatten())
            train_y_reg = self.golem.predict(
                X=train_x_reg,
                distributions=self.golem_dists,
            ).reshape(-1, 1)

        # scale the training data - normalize inputs and standardize outputs
        self._means_y, self._stds_y = np.mean(train_y_reg, axis=0), np.std(
            train_y_reg, axis=0
        )
        self._stds_y = np.where(self._stds_y == 0.0, 1.0, self._stds_y)

        if (
            self.problem_type == "fully_categorical"
            and not self.has_descriptors
        ):
            # we dont scale the parameters if we have a fully one-hot-encoded representation
            pass
        else:
            # scale the parameters
            train_x_cla = forward_normalize(
                train_x_cla, self.params_obj._mins_x, self.params_obj._maxs_x
            )
            train_x_reg = forward_normalize(
                train_x_reg, self.params_obj._mins_x, self.params_obj._maxs_x
            )

        # always forward transform the objectives for the regression problem
        train_y_reg = forward_standardize(
            train_y_reg, self._means_y, self._stds_y
        )

        # convert to torch tensors and return
        return (
            torch.tensor(train_x_cla).float(),
            torch.tensor(train_y_cla).squeeze().float(),
            torch.tensor(train_x_reg).double(),
            torch.tensor(train_y_reg).double(),
        )

    def build_train_regression_gp(self, train_x: torch.Tensor, train_y: torch.Tensor) -> gpytorch.models.ExactGP:
        """ Build the regression GP model list and likelihood sum """
        # infer the model based on the parameter types
        if self.problem_type in [
            "fully_continuous",
            "fully_discrete",
            "mixed_disc_cont",
        ]:
            model_obj = SingleTaskGP
        elif self.problem_type == "fully_categorical":
            if self.has_descriptors:
                # we have some descriptors, use the Matern kernel
                model_obj = SingleTaskGP
            else:
                # if we have no descriptors, use a Categorical kernel
                # based on the HammingDistance
                model_obj = CategoricalSingleTaskGP
        elif "mixed_cat_" in self.problem_type:
            if self.has_descriptors:
                # we have some descriptors, use the Matern kernel
                model_obj = SingleTaskGP
            else:
                cat_dims = get_cat_dims(self.param_space)
                model_obj = MixedSingleTaskGP#(train_x, train_y, cat_dims=cat_dims)
        else:
            raise NotImplementedError
        
        models = []
        for obj_ix in range(train_y.shape[-1]):
            if "mixed_cat_" in self.problem_type and not self.has_descriptors:
                models.append(model_obj(train_x, train_y[:,obj_ix].unsqueeze(-1), cat_dims=cat_dims))
            else:
                models.append(model_obj(train_x, train_y[:,obj_ix].unsqueeze(-1)))

        model = ModelListGP(*models)
        mll = SumMarginalLogLikelihood(model.likelihood, model)
        # fit the gp
        start_time = time.time()
        with gpytorch.settings.cholesky_jitter(self.max_jitter):
            fit_gpytorch_mll(mll)
        gp_train_time = time.time() - start_time
        Logger.log(
            f"Regression surrogate GP trained in {round(gp_train_time,3)} sec",
            "INFO",
        )
        
        return model
    

    def _tell(self, observations: olympus.campaigns.observations.Observations):
        """unpack the current observations from Olympus
        Args:
            observations (obj): Olympus campaign observations object
        """
        self._params = observations.get_params(
            as_array=True
        )  # string encodings of categorical params
        self._values = observations.get_values(as_array=True)

        # flip signs depending on objectives
        if len(self._values) > 0:
            coeffs = np.array([1. if goal=='min' else -1. for goal in self.goals])
            self._values *= coeffs

        # make values 2d if they are not already
        if len(np.array(self._values).shape) == 1:
            self._values = np.array(self._values).reshape(-1, 1)

        # generate Parameters object
        self.params_obj = Parameters(
            olympus_param_space=self.param_space,
            observations=observations,
            has_descriptors=self.has_descriptors,
            general_parameters=self.general_parameters,
        )

    def _ask(self) -> List[ParameterVector]:
        """ query the planner for a batch of new parameter points to measure
        """
        # if we have all nan values, just keep randomly sampling

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

                        # TODO: here, the worst objective actually must be computed using 
                        # hypervolume -> assume all minimization objectives at this point
                        # use reference point as worst value for each objecitve observed
                        ref_point = self.get_ref_point() #  current scaled refercence point

                        # get the hypervolume for all observed points
                        hypervols = []
                        for train_y in self.train_y_scaled_reg:
        
                            hypervols.append(
                                    get_hypervolume(
                                        train_y.unsqueeze(0).detach().numpy(), 
                                        ref_point.detach().numpy(),
                                    )
                            )



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

            if not "naive-" in self.feas_strategy and torch.sum(self.train_y_scaled_cla).item() != 0.:
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
                self.cla_surr_min_, self.cla_surr_max_ = self.get_cla_surr_min_max(num_samples=5000)
                self.fca_cutoff = (self.cla_surr_max_-self.cla_surr_min_)*self.feas_param + self.cla_surr_min_

            else:
                use_reg_only = True
                self.cla_model, self.cla_likelihood = None, None
                self.cla_surr_min_, self.cla_surr_max_ = None, None

            # compute the ratio of infeasible to total points
            infeas_ratio = (
                torch.sum(self.train_y_scaled_cla)
                / self.train_x_scaled_cla.size(0)
            ).item()

            # reference point
            ref_point = self.get_ref_point()
            # get the approximate max and min of the acquisition function without the feasibility contribution
            acqf_min_max = self.get_acqf_min_max(self.reg_model, ref_point)

            
            # instantiate acquisition function
            self.acqf = FeasibilityAwareqNEHVI(
                self.reg_model,
                self.cla_model,
                self.cla_likelihood,
                self.param_space,
                self.feas_strategy,
                self.feas_param,
                infeas_ratio,
                acqf_min_max,
                ref_point=ref_point,
                sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
                X_baseline=self.train_x_scaled_reg,
                prune_baseline=False, 
                use_min_filter=self.use_min_filter,
                use_reg_only=use_reg_only,
            )


            if self.acquisition_optimizer_kind == 'gradient':
                acquisition_optimizer = GradientOptimizer(
                    self.params_obj,
                    self.acquisition_type,
                    self.acqf,
                    self.known_constraints,
                    self.batch_size,
                    self.feas_strategy,
                    self.fca_constraint,
                    self._params,
                    self.batched_strategy,
                    self.timings_dict,
                    use_reg_only=use_reg_only,
                )
            elif self.acquisition_optimizer_kind == 'genetic':
                acquisition_optimizer = GeneticOptimizer(
                    self.params_obj,
                    self.acquisition_type,
                    self.acqf,
                    self.known_constraints,
                    self.batch_size,
                    self.feas_strategy,
                    self.fca_constraint,
                    self._params,
                    self.timings_dict,
                    use_reg_only=use_reg_only,
                )

            return_params = acquisition_optimizer.optimize()

        return return_params
    

    def get_acqf_min_max(self, reg_model, ref_point: List, num_samples: int = 3000) -> Tuple[int, int]:

        acqf = qNoisyExpectedHypervolumeImprovement(
            model=reg_model, 
            ref_point=ref_point,
            X_baseline=self.train_x_scaled_reg,  
            prune_baseline=False, 
            sampler=SobolQMCNormalSampler(sample_shape=torch.Size([128])),
        )

        samples, _ = propose_randomly(
            num_samples,
            self.param_space,
            self.has_descriptors,
        )

        if (
            self.problem_type == "fully_categorical"
            and not self.has_descriptors
        ):
            # we dont scale the parameters if we have a fully one-hot-encoded representation
            pass
        else:
            # scale the parameters
            samples = forward_normalize(
                samples, self.params_obj._mins_x, self.params_obj._maxs_x
            )

        acqf_vals = acqf(
            torch.tensor(samples)
            .view(samples.shape[0], 1, samples.shape[-1])
            .double()
        )


        min_ = torch.amin(acqf_vals).item()
        max_ = torch.amax(acqf_vals).item()

        if np.abs(max_ - min_) < 1e-6:
            max_ = 1.0
            min_ = 0.0

        return min_, max_
    

    def get_ref_point(self) -> List:
        """get the worst measured points for each objective dimension (scaled)"""
        mins_ = torch.amin(self.train_y_scaled_reg, axis=0).tolist()
        maxs_ = torch.amax(self.train_y_scaled_reg, axis=0).tolist()
        ref_point = torch.tensor(
            [maxs_[ix] if self.goals[ix]=='min' else mins_[ix] for ix in range(len(self.goals))]
        )
        return ref_point
    
    

if __name__ == '__main__':

    from olympus.surfaces import Surface
    from olympus.campaigns import Campaign
    from olympus.objects import ParameterContinuous, ParameterCategorical, ParameterVector

    from olympus.utils.misc import get_pareto, get_pareto_set
    import matplotlib.pyplot as plt

    TYPE_ =  'single_obj_continuous' #'categorical' #'continuous'

    if TYPE_ == 'continuous':

        plot = True

        moo_surface = Surface(kind='MultFonseca', value_dim=2)

        planner = qNEHVIPlanner(
            goal='minimize', 
            feas_strategy="fca",
            feas_param=0.2,
            init_design_strategy='random',
            batch_size=1,
            use_descriptors=False,
            is_moo=True,
            value_space=moo_surface.value_space,
            goals=['min', 'min'],
        )

        planner.set_param_space(moo_surface.param_space)

        campaign = Campaign()
        campaign.set_param_space(moo_surface.param_space)
        campaign.set_value_space(moo_surface.value_space)

        BUDGET = 30

        if plot:
            fig, ax = plt.subplots()
            plt.ion()

        while len(campaign.observations.get_values()) < BUDGET:

            samples = planner.recommend(campaign.observations)

            for sample in samples:
                sample_arr = sample.to_array()
                measurement = moo_surface.run(sample_arr, return_paramvector=True)
                campaign.add_observation(sample_arr, measurement)

                print('SAMPLE : ', sample)
                print('MEASUREMENT : ', measurement)
                print('')


            if plot:
                ax.clear()

                params = campaign.observations.get_params()
                objs   = campaign.observations.get_values()

                pareto_front, pareto_set = get_pareto_set(params, objs)

                pareto_front_sorted = sorted(
                    [[pareto_front[i,0], pareto_front[i,1]] for i in range(len(pareto_front))], reverse=False,
                )
                pareto_front_sorted = np.array(pareto_front_sorted)

                ax.scatter(
                    objs[:,0],
                    objs[:,1],
                    s=20,
                    alpha=0.8,
                )
                ax.scatter(
                    pareto_front_sorted[:,0],
                    pareto_front_sorted[:,1],
                    s=40,
                )
                ax.plot(
                    pareto_front_sorted[:,0],
                    pareto_front_sorted[:,1],
                    lw=2,
                    ls='-'
                )

                ax.set_ylim(0.,1.)
                ax.set_xlim(0.,1.)

                plt.tight_layout()
                plt.pause(1.5)


    elif TYPE_ == 'categorical':

        plot = True

        surf1 = Surface(kind='CatDejong', num_opts=21)
        surf2 = Surface(kind='CatMichalewicz', num_opts=21)

        value_space = ParameterSpace()
        value_space.add(ParameterContinuous(name='obj0'))
        value_space.add(ParameterContinuous(name='obj1'))

        planner = qNEHVIPlanner(
            goal='minimize', 
            feas_strategy="fca",
            feas_param=0.2,
            init_design_strategy='random',
            batch_size=1,
            use_descriptors=True,
            is_moo=True,
            value_space=value_space,
            goals=['min', 'min'],
        )
        planner.set_param_space(surf1.param_space)

        campaign = Campaign()
        campaign.set_param_space(surf1.param_space)
        campaign.set_value_space(value_space)

        BUDGET = 30

        if plot:
            fig, ax = plt.subplots()
            plt.ion()

        while len(campaign.observations.get_values()) < BUDGET:

            samples = planner.recommend(campaign.observations)

            for sample in samples:
                sample_arr = sample.to_array()
                measurement1 = surf1.run(sample_arr, return_paramvector=True)[0]['value_0']
                measurement2 = surf2.run(sample_arr, return_paramvector=True)[0]['value_0']
                campaign.add_observation(
                    sample_arr, 
                    ParameterVector().from_dict({'obj0':measurement1,'obj1':measurement2})
                )

                print('SAMPLE : ', sample)
                print('MEASUREMENT1 : ', measurement1)
                print('MEASUREMENT2 : ', measurement2)
                print('')

                if plot:
                    ax.clear()

                    params = campaign.observations.get_params()
                    objs   = campaign.observations.get_values()

                    pareto_front, pareto_set = get_pareto_set(params, objs)

                    pareto_front_sorted = sorted(
                        [[pareto_front[i,0], pareto_front[i,1]] for i in range(len(pareto_front))], reverse=False,
                    )
                    pareto_front_sorted = np.array(pareto_front_sorted)

                    ax.scatter(
                        objs[:,0],
                        objs[:,1],
                        s=20,
                        alpha=0.8,
                    )
                    ax.scatter(
                        pareto_front_sorted[:,0],
                        pareto_front_sorted[:,1],
                        s=40,
                    )
                    ax.plot(
                        pareto_front_sorted[:,0],
                        pareto_front_sorted[:,1],
                        lw=2,
                        ls='-'
                    )

                    # ax.set_ylim(0.,1.)
                    # ax.set_xlim(0.,1.)

                    plt.tight_layout()
                    plt.pause(1.5)


    elif TYPE_ == 'single_obj_continuous':
       
        surface = Surface(kind='Dejong')

        planner = qNEHVIPlanner(
            goal='minimize', 
            feas_strategy="fca",
            feas_param=0.2,
            init_design_strategy='random',
            batch_size=1,
            use_descriptors=False,
            is_moo=False,
            goals=['min'],
        )

        planner.set_param_space(surface.param_space)

        campaign = Campaign()
        campaign.set_param_space(surface.param_space)

        BUDGET = 30

        while len(campaign.observations.get_values()) < BUDGET:

            samples = planner.recommend(campaign.observations)

            for sample in samples:
                sample_arr = sample.to_array()
                measurement = surface.run(sample_arr, return_paramvector=True)
                campaign.add_observation(sample_arr, measurement)



    else:
        raise ValueError









