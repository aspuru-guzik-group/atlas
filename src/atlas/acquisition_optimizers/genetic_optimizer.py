#!/usr/bin/env python

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from botorch.acquisition import AcquisitionFunction
from deap import base, creator, tools
from olympus import ParameterVector
from olympus.campaigns import ParameterSpace
from rich.progress import track

from atlas import Logger
from atlas.acquisition_functions.acqf_utils import (
    create_available_options,
    get_batch_initial_conditions,
)
from atlas.acquisition_optimizers.base_optimizer import AcquisitionOptimizer
from atlas.params.params import Parameters
from atlas.utils.planner_utils import (
    cat_param_to_feat,
    forward_normalize,
    forward_standardize,
    get_cat_dims,
    get_fixed_features_list,
    infer_problem_type,
    param_vector_to_dict,
    propose_randomly,
    reverse_normalize,
    reverse_standardize,
)


class GeneticOptimizer(AcquisitionOptimizer):
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
        acqf_args=None,
        **kwargs: Any,
    ):
        """
        constraints : list or None
            List of callables that are constraints functions. Each function takes a parameter dict, e.g.
            {'x0':0.1, 'x1':10, 'x2':'A'} and returns a bool indicating
            whether it is in the feasible region or not.
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
        self._params = params
        self._mins_x = self.params_obj._mins_x
        self._maxs_x = self.params_obj._maxs_x

        self.kind = "genetic"

        # range of opt domain dimensions
        self.param_ranges = self._get_param_ranges()

    def _wrapped_fca_constraint(self, params):
        # >= 0 is a feasible point --> True
        # < 0 is an infeasible point --> False
        # transform dictionary rep of x to expanded format
        expanded = self.params_obj.param_vectors_to_expanded(
            [ParameterVector().from_dict(params, self.param_space)],
            is_scaled=True,
            return_scaled=False,  # should already be scaled
        )
        val = (
            self.fca_constraint(
                torch.tensor(expanded).view(
                    expanded.shape[0], 1, expanded.shape[1]
                )
            )
            .detach()
            .numpy()[0][0]
        )

        if val >= 0:
            return True
        else:
            return False

    def _get_param_ranges(self):
        param_ranges = []
        counter = 0
        for param in self.param_space:
            if param.type == "continuous":
                param_ranges.append(
                    self.bounds[1, counter] - self.bounds[0, counter]
                )
                counter += 1
            elif param.type == "discrete":
                param_ranges.append(len(param.options))
                counter += 1
            elif param.type == "categorical":
                param_ranges.append(len(param.options))
                if self.has_descriptors:
                    counter += len(param.descriptors[0])
                else:
                    counter += len(param.options)
        return np.array(param_ranges)

    def indexify(self):
        samples = []
        counter = 0
        for cond, cond_raw in zip(
            self.batch_initial_conditions, self.raw_conditions
        ):
            sample = []
            counter = 0
            for elem, p in zip(cond_raw, self.param_space):
                if p.type == "continuous":
                    sample.append(float(cond[counter]))
                    counter += 1
                elif p.type == "discrete":
                    sample.append(float(p.options.index(float(elem))))
                    counter += 1
                elif p.type == "categorical":
                    sample.append(float(p.options.index(elem)))
                    if self.has_descriptors:
                        counter += len(p.descriptors[0])
                    else:
                        counter += len(p.options)
            samples.append(sample)
        return np.array(samples)

    # def indexify_x(self, x):
    #     """ same function as above but indexifies a given set of
    #     inputs  x
    #     """
    #     samples = []
    #     for x_ in x:
    #         sample = []
    #         counter = 0
    #         for elem, p in zip(x_, self.param_space):

    #     return None

    def deindexify(self, x):
        samples = []
        for x_ in x:
            sample = []
            counter = 0
            for elem, p in zip(x_, self.param_space):
                if p.type == "continuous":
                    sample.append(float(elem))
                    counter += 1
                elif p.type == "discrete":
                    sample.append(float(p.options[int(elem)]))
                    counter += 1
                elif p.type == "categorical":
                    sample.extend(
                        cat_param_to_feat(
                            p,
                            p.options[int(elem)],
                            self.has_descriptors,
                        )
                    )
            samples.append(sample)
        return np.array(samples)

    def acquisition(self, x: np.ndarray) -> Tuple:
        x = self.deindexify(x.reshape((1, x.shape[0])))
        # x = torch.tensor(
        #     x.reshape((1, self.batch_size, x.shape[1]))
        # )
        # inflate to batch_size for acqf
        x = torch.tile(torch.tensor(x), dims=(1, self.batch_size, 1))
        # return the negative of the acqf - this is conventionally minimized by
        # deap, but we want to maximize acqf
        return (-self.acqf(x).detach().numpy()[0],)

    def _optimize(
        self, max_iter: int = 10, show_progress: bool = True
    ) -> List[ParameterVector]:
        """
        Returns list of parameter vectors with the optimized recommendations

        show_progress : bool
            whether to display the optimization progress. Default is False.
        """
        (
            self.nonlinear_inequality_constraints,
            self.batch_initial_conditions,
            self.raw_conditions,
        ) = self.gen_initial_conditions()

        self.batch_initial_conditions = (
            self.batch_initial_conditions.squeeze().numpy()
        )  # scaled

        if type(self.nonlinear_inequality_constraints) == type(None):
            self.nonlinear_inequality_constraints = []

        # define which single-step optimization function to use, based on whether or not
        # we have known constraints
        if self.nonlinear_inequality_constraints != []:
            Logger.log(
                "GA acquisition optimizer using constrained evolution", "INFO"
            )
            self._one_step_evolution = self._constrained_evolution
        else:
            Logger.log(
                "GA acquisition optimizer using unconstrained evolution",
                "INFO",
            )
            self._one_step_evolution = self._evolution

        # indexify the discrete and categorical options
        samples = self.indexify()  # scaled

        # crossover and mutation probabilites
        CXPB = 0.5
        MUTPB = 0.4

        # setup GA with DEAP
        creator.create(
            "FitnessMin", base.Fitness, weights=[-1.0]
        )  # we minimize negative of the acquisition
        creator.create("Individual", list, fitness=creator.FitnessMin)

        # ------------
        # make toolbox
        # ------------
        toolbox = base.Toolbox()
        toolbox.register("population", param_vectors_to_deap_population)
        toolbox.register("evaluate", self.acquisition)
        # use custom mutations for continuous, discrete, and categorical variables
        toolbox.register("mutate", self._custom_mutation, indpb=0.2)
        toolbox.register("select", tools.selTournament, tournsize=3)

        # mating type depends on how many genes we have
        if np.shape(samples)[1] == 1:
            toolbox.register("mate", cxDummy)  # i.e. no crossover
        elif np.shape(samples)[1] == 2:
            toolbox.register(
                "mate", tools.cxUniform, indpb=0.5
            )  # uniform crossover
        else:
            toolbox.register("mate", tools.cxTwoPoint)  # two-point crossover

        # Initialise population
        population = toolbox.population(samples)

        # Evaluate pop fitnesses
        fitnesses = list(map(toolbox.evaluate, np.array(population)))
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        # create hall of fame
        num_elites = int(
            round(0.05 * len(population), 0)
        )  # 5% of elite individuals
        halloffame = tools.HallOfFame(
            num_elites
        )  # hall of fame with top individuals
        halloffame.update(population)

        # register some statistics and create logbook
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        logbook = tools.Logbook()
        logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])
        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(population), **record)

        # ------------------------------
        # Begin the generational process
        # ------------------------------
        if show_progress is True:
            # run loop with progress bar
            iterable = track(
                range(1, max_iter + 1),
                total=max_iter,
                description="Optimizing proposals...",
                transient=False,
            )
        else:
            # run loop without progress bar
            iterable = range(1, max_iter + 1)

        for gen in iterable:
            offspring = self._one_step_evolution(
                population=population,
                toolbox=toolbox,
                halloffame=halloffame,
                cxpb=CXPB,
                mutpb=MUTPB,
            )

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, np.array(invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # add the best back to population
            offspring.extend(halloffame.items)

            # Update the hall of fame with the generated individuals
            halloffame.update(offspring)

            # Replace the current population by the offspring
            population[:] = offspring

            # Append the current generation statistics to the logbook
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)

            # convergence criterion, if the population has very similar fitness, stop
            # we quit if the population is found in the hypercube with edge 10% of the optimization domain
            if self._converged(population, slack=0.1) is True:
                break

        # DEAP cleanup
        del creator.FitnessMin
        del creator.Individual

        # select best recommendations and return them as param vectors
        acqf_vals = [self.acquisition(x)[0] for x in np.array(population)]

        batch_pop, batch_acqf_vals = self.batch_sample_selector(
            acqf_vals,
            population,
            exp_scale_factor=0.01,
        )

        # TODO: this bit is pretty hacky...
        batch_pop_deindex = self.deindexify(batch_pop)
        batch_pop_deindex = reverse_normalize(
            batch_pop_deindex,
            self.params_obj._mins_x,
            self.params_obj._maxs_x,
        )

        batch = []
        for best_index, best_deindex in zip(batch_pop, batch_pop_deindex):
            sample = []
            counter = 0
            for elem, p in zip(best_index, self.param_space):
                if p.type == "continuous":
                    sample.append(
                        _project_bounds(
                            best_deindex[counter],
                            p.low,
                            p.high,
                        )
                    )
                    counter += 1
                elif p.type == "discrete":
                    sample.append(
                        _project_bounds(
                            p.options[int(elem)],
                            p.low,
                            p.high,
                        )
                    )
                    counter += 1
                elif p.type == "categorical":
                    # sample.append(elem)
                    sample.append(p.options[int(elem)])
                    if self.has_descriptors:
                        counter += len(p.descriptors[0])
                    else:
                        counter += len(p.options)
            batch.append(sample)

        # batch_dicts = [param_vector_to_dict(sample, self.param_space) for sample in np.array(batch)]
        batch_dicts = []
        for sample in batch:
            batch_dicts.append(
                {p.name: elem for elem, p in zip(sample, self.param_space)}
            )

        return_params = [
            ParameterVector().from_dict(dict_, self.param_space)
            for dict_ in batch_dicts
        ]
        return return_params

    def batch_sample_selector(self, acqf_vals, population, exp_scale_factor):
        """select batch of samples from GA population"""
        sort_idx = np.argsort(acqf_vals)
        sort_acqf_vals = np.array(acqf_vals)[sort_idx]
        sort_pop = np.array(population)[sort_idx]

        if self.batch_size == 1:
            # return best sample
            # TODO: check inclusion in previous observations
            batch_pop = sort_pop[0]
            batch_acqf_vals = sort_acqf_vals[0]

        else:
            # return batch of samples

            # set probs
            decay = np.arange(len(acqf_vals))[::-1]
            probs = np.exp(decay) / (np.sum(np.exp(decay)))

            batch_pop = np.empty((self.batch_size, self.params_obj.num_params))
            batch_acqf_vals = []

            # probabilistic sample
            sample_idxs = np.random.choice(
                np.arange(len(acqf_vals)),
                size=(len(acqf_vals)),
                replace=False,
                p=probs,
            )

            print(sample_idxs)
            num_added = 0
            for sample_idx in sample_idxs:
                # avoid duplicate samples
                if not any(np.equal(batch_pop, sort_pop[sample_idx]).all(1)):
                    # print(sort_pop[sample_idx])
                    # print(sort_pop.shape)
                    # print(batch_pop.shape)
                    # quit()
                    batch_pop[num_added, :] = sort_pop[sample_idx]
                    batch_acqf_vals.append(sort_acqf_vals[sample_idx])
                    num_added += 1
                    if num_added == self.batch_size:
                        break
                else:
                    pass

            print(batch_pop)
            print(batch_acqf_vals)

        return batch_pop, batch_acqf_vals

    def _converged(self, population, slack=0.1):
        """If all individuals within specified subvolume, the population is not very diverse"""
        pop_ranges = np.max(population, axis=0) - np.min(
            population, axis=0
        )  # range of values in population
        normalized_ranges = pop_ranges / self.param_ranges  # normalised ranges
        bool_array = normalized_ranges < slack
        return all(bool_array)

    @staticmethod
    def _evolution(population, toolbox, halloffame, cxpb=0.5, mutpb=0.3):
        # size of hall of fame
        hof_size = len(halloffame.items) if halloffame.items else 0

        # Select the next generation individuals (allow for elitism)
        offspring = toolbox.select(population, len(population) - hof_size)

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < cxpb:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random() < mutpb:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        return offspring

    def _constrained_evolution(
        self, population, toolbox, halloffame, cxpb=0.5, mutpb=0.3
    ):
        # size of hall of fame
        hof_size = len(halloffame.items) if halloffame.items else 0

        # Select the next generation individuals (allow for elitism)
        offspring = toolbox.select(population, len(population) - hof_size)

        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if np.random.random() < cxpb:
                parent1 = list(
                    map(toolbox.clone, child1)
                )  # both are parents to both children, but we select one here
                parent2 = list(map(toolbox.clone, child2))
                # mate
                toolbox.mate(child1, child2)
                # apply constraints
                self._apply_feasibility_constraint(child1, parent1)
                self._apply_feasibility_constraint(child2, parent2)
                # clear fitness values
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if np.random.random() < mutpb:
                parent = list(map(toolbox.clone, mutant))
                # mutate
                toolbox.mutate(mutant)
                # apply constraints
                self._apply_feasibility_constraint(mutant, parent)
                # clear fitness values
                del mutant.fitness.values
        return offspring

    def _evaluate_feasibility(self, sample):
        # evaluate whether the optimized sample violates the known constraints
        # TODO: dont pass the parameter space here?? These should be scaled so they
        # might register a 'parameter out of bounds warning message' ...
        param = param_vector_to_dict(
            sample=sample, param_space=self.param_space
        )
        param_list = list(param.values())
        # feasible = [constr(param) for constr in self.known_constraints]
        feasible = [
            constr(param_list)
            for constr in self.nonlinear_inequality_constraints
        ]

        return all(feasible)

    @staticmethod
    def _update_individual(ind, value_vector):
        for i, v in enumerate(value_vector):
            ind[i] = v

    def _apply_feasibility_constraint(self, child, parent):
        child_vector = np.array(
            child, dtype=object
        )  # object needed to allow strings of different lengths
        feasible = self._evaluate_feasibility(child_vector)
        # if feasible, stop, no need to project the mutant
        if feasible is True:
            return

        # If not feasible, we try project parent or child onto feasibility boundary following these rules:
        # - for continuous parameters, we do stick breaking that is like a continuous version of a binary tree search
        #   until the norm of the vector connecting parent and child is less than a chosen threshold.
        # - for discrete parameters, we do the same until the "stick" is as short as possible, i.e. the next step
        #   makes it infeasible
        # - for categorical variables, we first reset them to the parent, then after having changed continuous
        #   and discrete, we reset the child. If feasible, we keep the child's categories, if still infeasible,
        #   we keep the parent's categories.

        parent_vector = np.array(
            parent, dtype=object
        )  # object needed to allow strings of different lengths
        new_vector = child_vector

        child_continuous = child_vector[self.params_obj.cont_mask]
        child_discrete = child_vector[self.params_obj.disc_mask]
        child_categorical = child_vector[self.params_obj.cat_mask]

        parent_continuous = parent_vector[self.params_obj.cont_mask]
        parent_discrete = parent_vector[self.params_obj.disc_mask]
        parent_categorical = parent_vector[self.params_obj.cat_mask]

        # ---------------------------------------
        # (1) assign parent's categories to child
        # ---------------------------------------
        if any(self.params_obj.cat_mask) is True:
            new_vector[self.params_obj.cat_mask] = parent_categorical
            # If this fixes is, update child and return
            # This is equivalent to assigning the category to the child, and then going to step 2. Because child
            # and parent are both feasible, the procedure will converge to parent == child and will return parent
            if self._evaluate_feasibility(new_vector) is True:
                self._update_individual(child, new_vector)
                return

        # -----------------------------------------------------------------------
        # (2) follow stick breaking/tree search procedure for continuous/discrete
        # -----------------------------------------------------------------------
        if (
            any(self.params_obj.cont_mask)
            or any(self.params_obj.disc_mask) is True
        ):
            # data needed to normalize continuous values\
            # TODO: do we actually need to do this??
            lowers = self.bounds[0][self.params_obj.exp_cont_mask].numpy()
            uppers = self.bounds[1][self.params_obj.exp_cont_mask].numpy()
            inv_range = 1.0 / (uppers - lowers)
            counter = 0
            while True:
                # update continuous
                new_continuous = np.mean(
                    np.array([parent_continuous, child_continuous]), axis=0
                )
                # update discrete, note that it can happen that child_discrete reverts to parent_discrete
                # add noise so that we can converge to the parent if needed
                noisy_mean = np.abs(
                    np.mean([parent_discrete, child_discrete], axis=0)
                    + np.random.uniform(
                        low=-0.1, high=0.1, size=len(parent_discrete)
                    )
                )

                new_discrete = np.round(noisy_mean.astype(np.double), 0)

                new_vector[self.params_obj.cont_mask] = new_continuous
                new_vector[self.params_obj.disc_mask] = new_discrete

                # if child is now feasible, parent becomes new_vector (we expect parent to always be feasible)
                if self._evaluate_feasibility(new_vector) is True:
                    parent_continuous = new_vector[self.params_obj.cont_mask]
                    parent_discrete = new_vector[self.params_obj.disc_mask]
                # if child still infeasible, child becomes new_vector (we expect parent to be the feasible one
                else:
                    child_continuous = new_vector[self.params_obj.cont_mask]
                    child_discrete = new_vector[self.params_obj.disc_mask]

                # convergence criterion is that length of stick is less than 1% in all continuous dimensions
                # for discrete variables, parent and child should be same
                if (
                    np.sum(parent_discrete - child_discrete) < 0.1
                ):  # check all differences are zero
                    parent_continuous_norm = (
                        parent_continuous - lowers
                    ) * inv_range
                    child_continuous_norm = (
                        child_continuous - lowers
                    ) * inv_range
                    # check all differences are within 1% of range
                    if all(
                        np.abs(parent_continuous_norm - child_continuous_norm)
                        < 0.01
                    ):
                        break

                counter += 1
                if (
                    counter > 150
                ):  # convergence above should be reached in 128 iterations max
                    Logger.log(
                        "constrained evolution procedure ran into trouble - using more iterations than "
                        "theoretically expected",
                        "ERROR",
                    )

        # last parent values are the feasible ones
        new_vector[self.params_obj.cont_mask] = parent_continuous
        new_vector[self.params_obj.disc_mask] = parent_discrete

        # ---------------------------------------------------------
        # (3) Try reset child's categories, otherwise keep parent's
        # ---------------------------------------------------------
        if any(self.params_obj.cat_mask) is True:
            new_vector[self.params_obj.cat_mask] = child_categorical
            if self._evaluate_feasibility(new_vector) is True:
                self._update_individual(child, new_vector)
                return
            else:
                # This HAS to be feasible, otherwise there is a bug
                new_vector[self.params_obj.cat_mask] = parent_categorical
                self._update_individual(child, new_vector)
                return
        else:
            self._update_individual(child, new_vector)
            return

    def _custom_mutation(
        self, individual, indpb=0.3, continuous_scale=0.1, discrete_scale=0.1
    ):
        """Custom mutation that can handled continuous, discrete, and categorical variables.
        Parameters
        ----------
        individual :
        indpb : float
            Independent probability for each attribute to be mutated.
        continuous_scale : float
            Scale for normally-distributed perturbation of continuous values.
        discrete_scale : float
            Scale for normally-distributed perturbation of discrete values.
        """

        assert len(individual) == len(self.param_space)

        bounds_ix = 0

        for i, param in enumerate(self.param_space):
            param_type = param["type"]

            if param_type == "continuous":
                if np.random.random() < indpb:
                    # Gaussian perturbation with scale being 0.1 of domain range
                    bound_low = self.bounds[0, bounds_ix]
                    bound_high = self.bounds[1, bounds_ix]
                    scale = (bound_high - bound_low) * continuous_scale
                    individual[i] += np.random.normal(loc=0.0, scale=scale)
                    individual[i] = _project_bounds(
                        individual[i], bound_low, bound_high
                    )
                bounds_ix += 1
            elif param_type == "discrete":
                if np.random.random() < indpb:
                    # add/substract an integer by rounding Gaussian perturbation
                    # scale is 0.1 of domain range
                    bound_low = 0
                    bound_high = len(param.options) - 1
                    # if we have very few discrete variables, just move +/- 1
                    if bound_high - bound_low < 10:
                        delta = np.random.choice([-1, 1])
                        individual[i] += delta
                    else:
                        scale = (bound_high - bound_low) * discrete_scale
                        delta = np.random.normal(loc=0.0, scale=scale)
                        individual[i] += np.round(delta, decimals=0)
                    individual[i] = _project_bounds(
                        individual[i], bound_low, bound_high
                    )
                bounds_ix += 1

            elif param_type == "categorical":
                if np.random.random() < indpb:
                    # resample a random category
                    num_options = float(
                        self.param_ranges[i]
                    )  # float so that np.arange returns doubles
                    individual[i] = np.random.choice(
                        list(np.arange(num_options))
                    )
                if not self.params_obj.has_descriptors:
                    bounds_ix += len(self.param_space[i].options)
                else:
                    bounds_ix += len(self.param_space[i].descriptors[0])
            else:
                raise ValueError()

        return (individual,)


def cxDummy(ind1, ind2):
    """Dummy crossover that does nothing. This is used when we have a single gene in the chromosomes, such that
    crossover would not change the population.
    """
    return ind1, ind2


def _project_bounds(x, x_low, x_high):
    if x < x_low:
        return x_low
    elif x > x_high:
        return x_high
    else:
        return x


def param_vectors_to_deap_population(param_vectors):
    population = []
    for param_vector in param_vectors:
        ind = creator.Individual(param_vector)
        population.append(ind)
    return population
