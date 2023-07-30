#!/usr/bin/env python

import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction
from pymoo.config import Config
from pymoo.core.mixed import MixedVariableGA
from pymoo.core.population import Population
from pymoo.core.problem import Problem
from pymoo.core.variable import Choice, Integer, Real
from pymoo.optimize import minimize

Config.show_compile_hint = False

from olympus.objects import ParameterVector

from atlas import Logger, tkwargs
from atlas.acquisition_optimizers.base_optimizer import AcquisitionOptimizer
from atlas.params.params import Parameters
from atlas.sample_selector.sample_selector import batch_local_penalization_selector
from atlas.utils.planner_utils import infer_problem_type, propose_randomly


class PymooProblemWrapper(Problem):
    """Wraps pymoo problem object with abstract method _evaluate which"""

    def __init__(
        self,
        params_obj: Parameters,
        pymoo_space,
        bounds,
        acqf: AcquisitionFunction,
        batch_size: int,
        known_constraints: Union[Callable, List[Callable]],
        fca_constraint: Callable, 
        fixed_param: Dict[int, float],
        num_fantasies: int = 0,
        **kwargs,
    ):
        if not known_constraints.is_empty:
            self.num_constr = 1
        else:
            self.num_constr = 0
        super().__init__(
            vars=pymoo_space,
            n_vars=len(params_obj.param_space),
            n_obj=1,
            n_constr=self.num_constr,
            xl=bounds[0],
            xu=bounds[1],
            **kwargs,
        )

        self.params_obj = params_obj
        self.pymoo_space = pymoo_space
        self.param_space = self.params_obj.param_space
        self.acqf = acqf
        self.batch_size = batch_size
        self.known_constraints = known_constraints
        self.fca_constraint = fca_constraint
        self.fixed_param = fixed_param
        self.num_fantasies = num_fantasies

        # TODO: this only supports one fixed param now, should be fine...
        if self.fixed_param != {}:
            self.fixed_param_ix = next(iter(self.fixed_param.keys()))
            self.fixed_param_val = next(iter(self.fixed_param.values()))
            self.fixed_param_name = self.param_space[self.fixed_param_ix].name
            self.fixed_param_type = self.param_space[self.fixed_param_ix].type

            if self.param_space[self.fixed_param_ix] in [
                "discrete",
                "categorical",
            ]:
                self.fixed_param_option_ix = self.param_space[
                    self.fixed_param_ix
                ].options.index(self.fixed_param_val)
            else:
                self.fixed_param_option_ix = None
        else:
            self.fixed_param_name = uuid.uuid4().hex

    def _pymoo_to_olympus(
        self,
        samples,
        forward_transform=False,
        return_param_vec=False,
        return_expanded=False,
    ):
        """convert pymoo parameters to Olympus parameters
        (with optional forward transform)
        samples expects a list of dictionaries
        """
        if not isinstance(samples[0], dict):
            samples = self._X_to_list_dicts(samples)

        olymp_samples = []
        for sample in samples:
            olymp_sample = {}
            for elem, param in zip(sample, self.param_space):
                if param.type == "discrete":
                    # map back to olympus with integer index
                    if param.name == self.fixed_param_name:
                        val_ = self.fixed_param_val
                    else:
                        val_ = float(param.options[sample[elem]])
                    olymp_sample[param.name] = val_

                elif param.type == "continuous":
                    if param.name == self.fixed_param_name:
                        val_ = self.fixed_param_val
                    else:
                        val_ = float(sample[elem])
                    olymp_sample[param.name] = val_
                elif param.type == "categorical":
                    if param.name == self.fixed_param_name:
                        val_ = str(self.fixed_param_val)
                    else:
                        olymp_sample[param.name] = sample[elem]

            olymp_samples.append(olymp_sample)

        if return_param_vec:
            param_vecs = []
            for sample in olymp_samples:
                param_vecs.append(
                    ParameterVector().from_dict(sample, self.param_space)
                )
            if return_expanded:
                return self.params_obj.param_vectors_to_expanded(
                    param_vectors=param_vecs,
                    is_scaled=False,
                    return_scaled=forward_transform,
                )
            return param_vecs
        else:
            return olymp_samples

    def _known_constraints_wrapper(self, X):
        """wrapper for known constraints, converts atlas boolean
        output to <= 1. for feasible and > 1. for infeasible for pymoo

        params is a list of dictionaries with the parameter samples
        """
        g = []
        for X_ in X:
            X_arr = list(X_.values())
            kc_vals_bool = [kc(X_arr) for kc in self.known_constraints]
            if all(kc_vals_bool):
                g.append(-2)  # feasible
            else:  # infeasible
                g.append(2)

        return np.array(g)

    def _wrapped_fc_constraint(self, X):
        """wrapped fca constraint
        from pytorch >= 0. is feasible and < 0. is infeasible
        """
        X_expanded = self.params_obj.param_vectors_to_expanded(
            [ParameterVector().from_dict(X_, self.param_space) for X_ in X],
            is_scaled=False,
            return_scaled=True,
        )
        X_torch = torch.tensor(X_expanded, **tkwargs).view(X_expanded.shape[0], 1, X_expanded.shape[1])

        vals_np = self.fca_constraint(X=X_torch).detach().numpy().squeeze()
        g = []
        for val_np in vals_np: 
            if val_np >= 0:
                g.append(-2) # feasible
            else:
                g.append(2)

        return np.array(g)

    def _X_to_list_dicts(self, X):
        X_list_dicts = []
        for X_ in X:
            X_list_dicts.append(
                dict(
                    zip(
                        [param.name for param in self.param_space],
                        X_,
                    )
                )
            )

        return X_list_dicts

    def _replace_fixed_param(self, X, to_replace):
        X_fixed_param = []
        for X_ in X:
            X_[self.fixed_param_name] = to_replace
            X_fixed_param.append(X_)
        return X_fixed_param

    def _evaluate(self, X, out, *args, **kwargs):
        """Abstract objective and constraint evaluation method for pymoo
        Problem instance
        """

        # convert everything to list of dicts to stay consistent
        if not isinstance(X[0], dict):
            X = self._X_to_list_dicts(X)

        if self.fixed_param != {}:
            if self.fixed_param_type == "discrete":
                X = self._replace_fixed_param(X, self.fixed_param_option_ix)
            else:
                X = self._replace_fixed_param(X, self.fixed_param_val)

        # ----------------
        # acqf evaluation
        # ----------------
        # convert from pymoo to olympus
        X_olymp = self._pymoo_to_olympus(
            samples=X,
            forward_transform=True,
            return_param_vec=True,
            return_expanded=True,
        )

        # inflate to batch size for acqf eval
        X_torch = torch.tensor(
            X_olymp, **{"dtype": torch.double, "device": "cpu"}
        ).view(
            X_olymp.shape[0],
            1,
            X_olymp.shape[1],
        )
        # always use batch_size=1 for pymoo optimizer
        # num fantasies is only for multi-fidelity optimization
        X_torch = torch.tile(X_torch, dims=(1, 1 + self.num_fantasies, 1))

        with torch.no_grad():
            f = -self.acqf(X_torch)  # always minimization in pymoo

        out["F"] = f.detach().numpy()

        # -----------------------------
        # constraints evaluation
        # ----------------------------
        if self.acqf.feas_strategy == 'fca':
            # TODO: also take care of FCA constraint here ...
            g_unknown = self._wrapped_fc_constraint(X)
        else:
            g_unknown = np.ones(X.shape[0]) * -2. # all feasible

        if not self.known_constraints.is_empty:
            g_known = self._known_constraints_wrapper(X)
        else:
            g_known = np.ones(X.shape[0]) * -2.

        g = np.maximum(g_known, g_unknown)

        if self.num_constr > 0:
            out["G"] = g



def gen_initial_population(
    space, pop, param_space, has_descriptors
) -> Population:
    """custom initialization of the population"""
    _, samples_raw = propose_randomly(
        num_proposals=pop,
        param_space=param_space,
        has_descriptors=has_descriptors,
    )
    pop_list_dicts = []
    for sample in samples_raw:
        pop_dict = {}
        for elem, param in zip(sample, param_space):
            if param.type == "discrete":
                pop_dict[param.name] = int(param.options.index(elem))
            elif param.type == "continuous":
                pop_dict[param.name] = float(elem)
            else:
                pop_dict[param.name] = str(elem)

            pop_list_dicts.append(pop_dict)

    return Population.new(X=pop_list_dicts)


class PymooGAOptimizer(AcquisitionOptimizer):
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
        timings_dict: Dict,
        use_reg_only: bool = False,
        # pymoo config
        pop_size: int = 200,
        repair: bool = False,
        verbose: bool = False,
        save_history: bool = False,
        num_gen: int = 5000,
        eliminate_duplicates: bool = True,
        fixed_params: Optional[List[Dict[int, float]]] = [],
        num_fantasies: int = 0,
        acqf_args=None,
        **kwargs: Any,
    ):
        """
        Genetic algorithm acquisition optimizer from pymoo
        """
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
        self.batch_size = batch_size
        self.feas_strategy = feas_strategy
        self.fca_constraint = fca_constraint
        self.known_constraints = known_constraints
        self.use_reg_only = use_reg_only
        self.has_descriptors = self.params_obj.has_descriptors
        self._params = params  # already measured params

        self.pop_size = pop_size
        self.repair = repair
        self.verbose = verbose
        self.save_history = save_history
        self.num_gen = num_gen
        self.eliminate_duplicates = eliminate_duplicates

        self.fixed_params = fixed_params
        self.num_fantasies = num_fantasies

        self.kind = "pymoo"

        # check that the batch_size is samller than pop_size
        if not self.batch_size < self.pop_size:
            Logger.log(
                "You must use a larger pop_size for pymoo optimizer than the batch_size",
                "FATAL",
            )

        with torch.no_grad():
            # set pymoo parameter space
            self.pymoo_space, self.xl, self.xu = self._set_pymoo_param_space()

    def _set_pymoo_param_space(self):
        """convert Olympus parameter space to pymoo"""
        pymoo_space = {}
        xl, xu = [], []
        for param in self.param_space:
            if param.type == "continuous":
                pymoo_space[param.name] = Real(bounds=(param.low, param.high))
                xl.append(param.low)
                xu.append(param.high)
            elif param.type == "discrete":
                # TODO: need to map the discrete params to an integer
                pymoo_space[param.name] = Integer(
                    bounds=(0, len(param.options) - 1)
                )
                xl.append(param.low)
                xu.append(param.high)
            elif param.type == "categorical":
                pymoo_space[param.name] = Choice(options=param.options)
                xl.append(0)
                xu.append(len(param.options) - 1)
            else:
                raise ValueError

        return pymoo_space, np.array(xl), np.array(xu)

    def _batch_sample_selector(self, final_pop):
        """select batch of samples from the pymoo minimize results"""
        olymp_samples_arr = self.pymoo_problem._pymoo_to_olympus(
            [ind.X for ind in final_pop],
            forward_transform=False,
            return_param_vec=True,
        )
        olymp_samples_pvec = self.pymoo_problem._pymoo_to_olympus(
            [ind.X for ind in final_pop],
            forward_transform=False,
            return_param_vec=True,
        )
        batch_samples = []
        # final pop ordered by increasing (less optimal) acqf value
        for sample_arr, sample_pvec in zip(
            olymp_samples_arr, olymp_samples_pvec
        ):
            # check to see if the set of perviously measured params
            # contains this sample
            if not any((self._params[:] == sample_arr).all(1)):
                batch_samples.append(sample_pvec)
            else:
                # avoid duplicated sample
                pass
            if len(batch_samples) == self.batch_size:
                break

        return batch_samples

    def _optimize(self) -> List[ParameterVector]:
        """Perform (constrained) acqf optimization with pymoo minimize"""
        Logger.log(
            f"Optimizing acquisition function with pymoo GA for {len(self.fixed_params)} fixed parameters...",
            "INFO",
        )
        # perform acqf optimization
        all_res = []
        if len(self.fixed_params) > 0:
            for fixed_param_ix, fixed_param in enumerate(self.fixed_params):
                # set pymoo problem
                self.pymoo_problem = PymooProblemWrapper(
                    params_obj=self.params_obj,
                    pymoo_space=self.pymoo_space,
                    bounds=(self.xl, self.xu),
                    acqf=self.acqf,
                    batch_size=self.batch_size,
                    known_constraints=self.known_constraints,
                    fca_constraint=self.fca_constraint,
                    fixed_param=fixed_param,
                    num_fantasies=self.num_fantasies,
                )

                # instantiate algorithm
                self.algorithm = MixedVariableGA(
                    pop_size=self.pop_size,
                    # sampling=gen_initial_population,
                    # eliminate_duplicates=self.eliminate_duplicates,
                )

                res = minimize(
                    self.pymoo_problem,
                    self.algorithm,
                    termination=("n_evals", self.num_gen),
                    verbose=self.verbose,
                    save_history=self.save_history,
                    copy_algorithm=False,
                )
                all_res.append(res)

        else:
            # set pymoo problem
            self.pymoo_problem = PymooProblemWrapper(
                params_obj=self.params_obj,
                pymoo_space=self.pymoo_space,
                bounds=(self.xl, self.xu),
                acqf=self.acqf,
                batch_size=self.batch_size,
                known_constraints=self.known_constraints,
                fca_constraint=self.fca_constraint,
                fixed_param={},
                num_fantasies=self.num_fantasies,
            )

            # instantiate algorithm
            self.algorithm = MixedVariableGA(
                pop_size=self.pop_size,
                # sampling=gen_initial_population,
                # eliminate_duplicates=self.eliminate_duplicates,
                mu=1000.,
            )

            res = minimize(
                self.pymoo_problem,
                self.algorithm,
                termination=("n_evals", self.num_gen),
                verbose=self.verbose,
                save_history=self.save_history,
                copy_algorithm=False,
            )
            all_res.append(res)

            total_exec_time = sum([res.exec_time for res in all_res])
            Logger.log(f"Completed in {round(total_exec_time, 3)} sec", "INFO")

        # select batch of samples
        # NOTE: this will only work for a single fixed parameter now but that
        # should be fine.... might need to change later
        return_params = self._batch_sample_selector(all_res[0].pop)
        # return_params = batch_local_penalization_selector(
        #     pymoo_results=all_res,
        #     pymoo_problem=self.pymoo_problem,
        #     batch_size=self.batch_size,
        #     dist_param=0.5,
        # )




        return return_params
