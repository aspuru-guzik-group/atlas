#!/usr/bin/env python

import numpy as np
import pytest
from olympus.campaigns import Campaign, ParameterSpace
from olympus.datasets import Dataset
from olympus.emulators import Emulator
from olympus.objects import (
    ParameterCategorical,
    ParameterContinuous,
    ParameterDiscrete,
    ParameterVector,
)
from olympus.scalarizers import Scalarizer
from olympus.surfaces import Surface
from problem_generator import (
    HybridSurface,
    KnownConstraintsGenerator,
    ProblemGenerator,
)

from atlas.planners.gp.planner import GPPlanner

CONT = {
    "init_design_strategy": [
        "random",
        # "sobol",
        # "lhs",
    ],  # init design strategies
    "batch_size": [1],  # batch size
    "use_descriptors": [False],  # use descriptors
    "acquisition_type": ["ucb"],
    "acquisition_optimizer": ["pymoo"],  # ['pymoo', 'genetic'],
}

DISC = {
    "init_design_strategy": ["random"],
    "batch_size": [1],
    "use_descriptors": [False],
    "acquisition_type": ["ucb"],
    "acquisition_optimizer": ["pymoo"],  # ['pymoo', 'genetic'],
}

CAT = {
    "init_design_strategy": ["random"],
    "batch_size": [1],
    "use_descriptors": [False, True],
    "acquisition_type": ["ucb"],
    "acquisition_optimizer": ["pymoo"],  # ['pymoo', 'genetic'],
}

MIXED_CAT_CONT = {
    "init_design_strategy": ["random"],
    "batch_size": [1],
    "use_descriptors": [False, True],
    "acquisition_type": ["ucb"],
    "acquisition_optimizer": ["pymoo"],  # ['pymoo', 'genetic'],
}

MIXED_DISC_CONT = {
    "init_design_strategy": ["random"],
    "batch_size": [1],
    "use_descriptors": [False],
    "acquisition_type": ["ucb"],
    "acquisition_optimizer": ["pymoo"],  # ['pymoo', 'genetic'],
}


MIXED_CAT_DISC = {
    "init_design_strategy": ["random"],
    "batch_size": [1],
    "use_descriptors": [False, True],
    "acquisition_type": ["ucb"],
    "acquisition_optimizer": ["pymoo"],  # ['pymoo', 'genetic'],
}

MIXED_CAT_DISC_CONT = {
    "init_design_strategy": ["random"],
    "batch_size": [1],
    "use_descriptors": [False, True],
    "acquisition_type": ["ucb"],
    "acquisition_optimizer": ["pymoo"],  # ['pymoo', 'genetic'],
}

BATCHED = {
    "problem_type": [
        "cont",
        "disc",
        "cat",
        "mixed_cat_cont",
        "mixed_disc_cont",
        "mixed_cat_disc",
        "mixed_cat_disc_cont",
    ],
    "init_design_strategy": ["random"],
    "batch_size": [2, 4],
    "acquisition_optimizer": ["pymoo"],  # ['pymoo', 'genetic'],
}


@pytest.mark.parametrize("problem_type", BATCHED["problem_type"])
@pytest.mark.parametrize(
    "init_design_strategy", BATCHED["init_design_strategy"]
)
@pytest.mark.parametrize("batch_size", BATCHED["batch_size"])
@pytest.mark.parametrize(
    "acquisition_optimizer", BATCHED["acquisition_optimizer"]
)
def test_batched(
    problem_type, init_design_strategy, batch_size, acquisition_optimizer
):
    run_batched(
        problem_type, init_design_strategy, batch_size, acquisition_optimizer
    )


@pytest.mark.parametrize("init_design_strategy", CONT["init_design_strategy"])
@pytest.mark.parametrize("batch_size", CONT["batch_size"])
@pytest.mark.parametrize("use_descriptors", CONT["use_descriptors"])
@pytest.mark.parametrize("acquisition_type", CONT["acquisition_type"])
@pytest.mark.parametrize(
    "acquisition_optimizer", CONT["acquisition_optimizer"]
)
def test_cont(
    init_design_strategy,
    batch_size,
    use_descriptors,
    acquisition_type,
    acquisition_optimizer,
):
    run_continuous(
        init_design_strategy,
        batch_size,
        use_descriptors,
        acquisition_type,
        acquisition_optimizer,
    )


@pytest.mark.parametrize("init_design_strategy", DISC["init_design_strategy"])
@pytest.mark.parametrize("batch_size", DISC["batch_size"])
@pytest.mark.parametrize("use_descriptors", DISC["use_descriptors"])
@pytest.mark.parametrize("acquisition_type", DISC["acquisition_type"])
@pytest.mark.parametrize(
    "acquisition_optimizer", DISC["acquisition_optimizer"]
)
def test_disc(
    init_design_strategy,
    batch_size,
    use_descriptors,
    acquisition_type,
    acquisition_optimizer,
):
    run_discrete(
        init_design_strategy,
        batch_size,
        use_descriptors,
        acquisition_type,
        acquisition_optimizer,
    )


@pytest.mark.parametrize("init_design_strategy", CAT["init_design_strategy"])
@pytest.mark.parametrize("batch_size", CAT["batch_size"])
@pytest.mark.parametrize("use_descriptors", CAT["use_descriptors"])
@pytest.mark.parametrize("acquisition_type", CAT["acquisition_type"])
@pytest.mark.parametrize("acquisition_optimizer", CAT["acquisition_optimizer"])
def test_cat(
    init_design_strategy,
    batch_size,
    use_descriptors,
    acquisition_type,
    acquisition_optimizer,
):
    run_categorical(
        init_design_strategy,
        batch_size,
        use_descriptors,
        acquisition_type,
        acquisition_optimizer,
    )


@pytest.mark.parametrize(
    "init_design_strategy", MIXED_CAT_CONT["init_design_strategy"]
)
@pytest.mark.parametrize("batch_size", MIXED_CAT_CONT["batch_size"])
@pytest.mark.parametrize("use_descriptors", MIXED_CAT_CONT["use_descriptors"])
@pytest.mark.parametrize(
    "acquisition_type", MIXED_CAT_CONT["acquisition_type"]
)
@pytest.mark.parametrize(
    "acquisition_optimizer", MIXED_CAT_CONT["acquisition_optimizer"]
)
def test_mixed_cat_cont(
    init_design_strategy,
    batch_size,
    use_descriptors,
    acquisition_type,
    acquisition_optimizer,
):
    run_mixed_cat_cont(
        init_design_strategy,
        batch_size,
        use_descriptors,
        acquisition_type,
        acquisition_optimizer,
    )


@pytest.mark.parametrize(
    "init_design_strategy", MIXED_DISC_CONT["init_design_strategy"]
)
@pytest.mark.parametrize("batch_size", MIXED_DISC_CONT["batch_size"])
@pytest.mark.parametrize("use_descriptors", MIXED_DISC_CONT["use_descriptors"])
@pytest.mark.parametrize(
    "acquisition_type", MIXED_DISC_CONT["acquisition_type"]
)
@pytest.mark.parametrize(
    "acquisition_optimizer", MIXED_DISC_CONT["acquisition_optimizer"]
)
def test_mixed_disc_cont(
    init_design_strategy,
    batch_size,
    use_descriptors,
    acquisition_type,
    acquisition_optimizer,
):
    run_mixed_disc_cont(
        init_design_strategy,
        batch_size,
        use_descriptors,
        acquisition_type,
        acquisition_optimizer,
    )


@pytest.mark.parametrize(
    "init_design_strategy", MIXED_CAT_DISC["init_design_strategy"]
)
@pytest.mark.parametrize("batch_size", MIXED_CAT_DISC["batch_size"])
@pytest.mark.parametrize("use_descriptors", MIXED_CAT_DISC["use_descriptors"])
@pytest.mark.parametrize(
    "acquisition_type", MIXED_CAT_DISC["acquisition_type"]
)
@pytest.mark.parametrize(
    "acquisition_optimizer", MIXED_CAT_DISC["acquisition_optimizer"]
)
def test_mixed_cat_disc(
    init_design_strategy,
    batch_size,
    use_descriptors,
    acquisition_type,
    acquisition_optimizer,
):
    run_mixed_cat_disc(
        init_design_strategy,
        batch_size,
        use_descriptors,
        acquisition_type,
        acquisition_optimizer,
    )


@pytest.mark.parametrize(
    "init_design_strategy", MIXED_CAT_DISC_CONT["init_design_strategy"]
)
@pytest.mark.parametrize("batch_size", MIXED_CAT_DISC_CONT["batch_size"])
@pytest.mark.parametrize(
    "use_descriptors", MIXED_CAT_DISC_CONT["use_descriptors"]
)
@pytest.mark.parametrize(
    "acquisition_type", MIXED_CAT_DISC_CONT["acquisition_type"]
)
@pytest.mark.parametrize(
    "acquisition_optimizer", MIXED_CAT_DISC_CONT["acquisition_optimizer"]
)
def test_mixed_cat_disc_cont(
    init_design_strategy,
    batch_size,
    use_descriptors,
    acquisition_type,
    acquisition_optimizer,
):
    run_mixed_cat_disc_cont(
        init_design_strategy,
        batch_size,
        use_descriptors,
        acquisition_type,
        acquisition_optimizer,
    )


def run_batched(
    problem_type, init_design_strategy, batch_size, acquisition_optimizer
):
    if problem_type == "cont":
        run_continuous(
            init_design_strategy,
            batch_size,
            False,
            "ucb",
            acquisition_optimizer,
            num_init_design=4,
        )
    elif problem_type == "disc":
        run_discrete(
            init_design_strategy,
            batch_size,
            False,
            "ucb",
            acquisition_optimizer,
            num_init_design=4,
        )
    elif problem_type == "cat":
        run_categorical(
            init_design_strategy,
            batch_size,
            False,
            "ucb",
            acquisition_optimizer,
            num_init_design=4,
        )
        run_categorical(
            init_design_strategy,
            batch_size,
            True,
            "ucb",
            acquisition_optimizer,
            num_init_design=4,
        )
    elif problem_type == "mixed_cat_cont":
        run_mixed_cat_cont(
            init_design_strategy,
            batch_size,
            False,
            "ucb",
            acquisition_optimizer,
            num_init_design=4,
        )
        run_mixed_cat_cont(
            init_design_strategy,
            batch_size,
            True,
            "ucb",
            acquisition_optimizer,
            num_init_design=4,
        )
    elif problem_type == "mixed_disc_cont":
        run_mixed_disc_cont(
            init_design_strategy,
            batch_size,
            False,
            "ucb",
            acquisition_optimizer,
            num_init_design=4,
        )
    elif problem_type == "mixed_cat_disc":
        run_mixed_cat_disc(
            init_design_strategy,
            batch_size,
            False,
            "ucb",
            acquisition_optimizer,
            num_init_design=4,
        )
        run_mixed_cat_disc(
            init_design_strategy,
            batch_size,
            True,
            "ucb",
            acquisition_optimizer,
            num_init_design=4,
        )
    elif problem_type == "mixed_cat_disc_cont":
        run_mixed_cat_disc_cont(
            init_design_strategy,
            batch_size,
            False,
            "ucb",
            acquisition_optimizer,
            num_init_design=4,
        )
        run_mixed_cat_disc_cont(
            init_design_strategy,
            batch_size,
            True,
            "ucb",
            acquisition_optimizer,
            num_init_design=4,
        )
    else:
        pass


def run_continuous(
    init_design_strategy,
    batch_size,
    use_descriptors,
    acquisition_type,
    acquisition_optimizer,
    num_init_design=5,
):
    problem_gen = ProblemGenerator(problem_type="continuous")
    surface_callable, param_space = problem_gen.generate_instance()
    known_constraints = KnownConstraintsGenerator().get_constraint(
        "continuous"
    )

    planner = GPPlanner(
        goal="minimize",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        use_descriptors=use_descriptors,
        acquisition_type=acquisition_type,
        acquisition_optimizer_kind=acquisition_optimizer,
        known_constraints=[known_constraints],
    )

    planner.set_param_space(param_space)

    campaign = Campaign()
    campaign.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 4

    while len(campaign.observations.get_values()) < BUDGET:
        samples = planner.recommend(campaign.observations)
        for sample in samples:
            # sample_arr = sample.to_array()
            measurement = surface_callable.run(sample)
            campaign.add_observation(sample, measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET

    # check that all the measured values pass the known constraint
    meas_params = campaign.observations.get_params()

    kcs = [known_constraints(param) for param in meas_params]
    print(kcs)
    assert all(kcs)


def run_discrete(
    init_design_strategy,
    batch_size,
    use_descriptors,
    acquisition_type,
    acquisition_optimizer,
    num_init_design=5,
):
    problem_gen = ProblemGenerator(problem_type="discrete")
    surface_callable, param_space = problem_gen.generate_instance()
    known_constraints = KnownConstraintsGenerator().get_constraint("discrete")

    param_test = np.array([0.1, 0.9])
    print(known_constraints(param_test))

    planner = GPPlanner(
        goal="minimize",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        use_descriptors=use_descriptors,
        acquisition_type=acquisition_type,
        acquisition_optimizer_kind=acquisition_optimizer,
        known_constraints=[known_constraints],
    )

    planner.set_param_space(param_space)

    campaign = Campaign()
    campaign.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 4

    while len(campaign.observations.get_values()) < BUDGET:
        samples = planner.recommend(campaign.observations)
        for sample in samples:
            # sample_arr = sample.to_array()
            measurement = surface_callable.run(sample)
            campaign.add_observation(sample, measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET

    # check that all the measured values pass the known constraint
    meas_params = campaign.observations.get_params()
    kcs = [known_constraints(param) for param in meas_params]
    print(kcs, meas_params)
    assert all(kcs)


def run_categorical(
    init_design_strategy,
    batch_size,
    use_descriptors,
    acquisition_type,
    acquisition_optimizer,
    num_init_design=5,
):
    problem_gen = ProblemGenerator(problem_type="categorical")
    surface_callable, param_space = problem_gen.generate_instance()
    known_constraints = KnownConstraintsGenerator().get_constraint(
        "categorical"
    )

    campaign = Campaign()
    campaign.set_param_space(surface_callable.param_space)

    planner = GPPlanner(
        goal="minimize",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        use_descriptors=use_descriptors,
        acquisition_type=acquisition_type,
        acquisition_optimizer_kind=acquisition_optimizer,
        known_constraints=[known_constraints],
    )
    planner.set_param_space(surface_callable.param_space)

    BUDGET = num_init_design + batch_size * 4

    while len(campaign.observations.get_values()) < BUDGET:
        samples = planner.recommend(campaign.observations)
        for sample in samples:
            sample_arr = sample.to_array()
            measurement = np.array(surface_callable.run(sample_arr))
            campaign.add_observation(sample_arr, measurement[0])

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET

    # check that all the measured values pass the known constraint
    meas_params = campaign.observations.get_params()
    kcs = [known_constraints(param) for param in meas_params]
    assert all(kcs)


def run_mixed_disc_cont(
    init_design_strategy,
    batch_size,
    use_descriptors,
    acquisition_type,
    acquisition_optimizer,
    num_init_design=5,
):
    problem_gen = ProblemGenerator(problem_type="mixed_disc_cont")
    surface_callable, param_space = problem_gen.generate_instance()
    known_constraints = KnownConstraintsGenerator().get_constraint("disc_cont")

    planner = GPPlanner(
        goal="minimize",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        use_descriptors=use_descriptors,
        acquisition_type=acquisition_type,
        acquisition_optimizer_kind=acquisition_optimizer,
        known_constraints=[known_constraints],
    )

    planner.set_param_space(param_space)

    campaign = Campaign()
    campaign.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 4

    while len(campaign.observations.get_values()) < BUDGET:
        samples = planner.recommend(campaign.observations)

        for sample in samples:
            measurement = surface_callable.run(sample)
            campaign.add_observation(sample, measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET

    # check that all the measured values pass the known constraint
    meas_params = campaign.observations.get_params()
    kcs = [known_constraints(param) for param in meas_params]
    assert all(kcs)


def run_mixed_cat_disc(
    init_design_strategy,
    batch_size,
    use_descriptors,
    acquisition_type,
    acquisition_optimizer,
    num_init_design=5,
):
    problem_gen = ProblemGenerator(problem_type="mixed_cat_disc")
    surface_callable, param_space = problem_gen.generate_instance()
    known_constraints = KnownConstraintsGenerator().get_constraint("cat_disc")

    planner = GPPlanner(
        goal="minimize",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        use_descriptors=use_descriptors,
        acquisition_type=acquisition_type,
        acquisition_optimizer_kind=acquisition_optimizer,
        known_constraints=[known_constraints],
    )

    planner.set_param_space(param_space)

    campaign = Campaign()
    campaign.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 4

    while len(campaign.observations.get_values()) < BUDGET:
        samples = planner.recommend(campaign.observations)

        for sample in samples:
            measurement = surface_callable.run(sample)
            campaign.add_observation(sample, measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET

    # check that all the measured values pass the known constraint
    meas_params = campaign.observations.get_params()
    kcs = [known_constraints(param) for param in meas_params]
    assert all(kcs)


def run_mixed_cat_cont(
    init_design_strategy,
    batch_size,
    use_descriptors,
    acquisition_type,
    acquisition_optimizer,
    num_init_design=5,
):
    problem_gen = ProblemGenerator(problem_type="mixed_cat_cont")
    surface_callable, param_space = problem_gen.generate_instance()
    known_constraints = KnownConstraintsGenerator().get_constraint("cat_cont")

    planner = GPPlanner(
        goal="minimize",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        use_descriptors=use_descriptors,
        acquisition_type=acquisition_type,
        acquisition_optimizer_kind=acquisition_optimizer,
        known_constraints=[known_constraints],
    )

    planner.set_param_space(param_space)

    campaign = Campaign()
    campaign.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 4

    while len(campaign.observations.get_values()) < BUDGET:
        samples = planner.recommend(campaign.observations)

        for sample in samples:
            measurement = surface_callable.run(sample)
            campaign.add_observation(sample, measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET

    # check that all the measured values pass the known constraint
    meas_params = campaign.observations.get_params()
    kcs = [known_constraints(param) for param in meas_params]
    assert all(kcs)


def run_mixed_cat_disc_cont(
    init_design_strategy,
    batch_size,
    use_descriptors,
    acquisition_type,
    acquisition_optimizer,
    num_init_design=5,
):
    problem_gen = ProblemGenerator(problem_type="mixed_cat_disc_cont")
    surface_callable, param_space = problem_gen.generate_instance()
    known_constraints = KnownConstraintsGenerator().get_constraint(
        "cat_disc_cont"
    )

    planner = GPPlanner(
        goal="minimize",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        use_descriptors=use_descriptors,
        acquisition_type=acquisition_type,
        acquisition_optimizer_kind=acquisition_optimizer,
        known_constraints=[known_constraints],
    )

    planner.set_param_space(param_space)

    campaign = Campaign()
    campaign.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 5

    while len(campaign.observations.get_values()) < BUDGET:
        samples = planner.recommend(campaign.observations)
        for sample in samples:
            measurement = surface_callable.run(sample)
            campaign.add_observation(sample, measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET

    # check that all the measured values pass the known constraint
    meas_params = campaign.observations.get_params()
    kcs = [known_constraints(param) for param in meas_params]
    assert all(kcs)


def run_compositional_constraint_cont(
    init_design_strategy, batch_size, num_init_design=5
):
    param_space = ParameterSpace()
    param_0 = ParameterContinuous(name="param_0", low=0.0, high=1.0)
    param_1 = ParameterContinuous(name="param_1", low=0.0, high=1.0)
    param_2 = ParameterContinuous(name="param_2", low=0.0, high=1.0)
    param_space.add(param_0)
    param_space.add(param_1)
    param_space.add(param_2)

    planner = GPPlanner(
        goal="minimize",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        acquisition_type="ei",
        acquisition_optimizer_kind="gradient",
        # all params must sum to 1. (simplex/compositional constraint)
        compositional_params=[0, 1, 2],
    )

    planner.set_param_space(param_space)

    campaign = Campaign()
    campaign.set_param_space(param_space)

    BUDGET = 12

    while len(campaign.observations.get_values()) < BUDGET:
        samples = planner.recommend(campaign.observations)
        for sample in samples:
            sample_arr = sample.to_array()
            measurement = np.random.uniform()
            campaign.add_observation(sample_arr, measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET

    # check that all the parameters sum to 1.
    meas_params = campaign.observations.get_params()
    sums = [np.sum(param) for param in meas_params]
    assert all([sum_ == 1.0 for sum_ in sums])


def run_permutation_constraint_mixed_cat_disc(
    init_design_strategy, batch_size, num_init_design=5
):
    param_space = ParameterSpace()
    param_0 = ParameterDiscrete(
        name="param_0",
        options=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    param_1 = ParameterCategorical(
        name="param_1",
        options=["x0", "x1", "x2"],
        descriptors=[None, None, None],
    )
    param_2 = ParameterCategorical(
        name="param_2",
        options=["x0", "x1", "x2"],
        descriptors=[None, None, None],
    )
    param_3 = ParameterCategorical(
        name="param_3",
        options=["x0", "x1", "x2"],
        descriptors=[None, None, None],
    )
    param_space.add(param_0)
    param_space.add(param_1)
    param_space.add(param_2)
    param_space.add(param_3)

    planner = GPPlanner(
        goal="minimize",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        use_descriptors=False,
        batch_size=batch_size,
        acquisition_type="ei",
        acquisition_optimizer_kind="gradient",
        # permutation constraint on 3 categorical parameters
        permutation_params=[1, 2, 3],
    )

    planner.set_param_space(param_space)

    campaign = Campaign()
    campaign.set_param_space(param_space)

    BUDGET = 12

    while len(campaign.observations.get_values()) < BUDGET:
        samples = planner.recommend(campaign.observations)
        for sample in samples:
            sample_arr = sample.to_array()
            measurement = np.random.uniform()
            campaign.add_observation(sample_arr, measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET

    # validate permutation constraint
    meas_params = campaign.observations.get_params()
    vals = []
    for param in meas_params:
        val = []
        for idx in planner.known_constraints.permutation_params:
            if param_space[idx].type == "categorical":
                val.append(param_space[idx].options.index(param[idx]))
            else:
                val.append(float(param[idx]))
        vals.append(val)

    assert all([val == sorted(val) for val in vals])


def run_pending_experiment_constraint_cat(
    init_design_strategy, batch_size, use_descriptors, num_init_design=5
):
    surface_kind = "CatDejong"
    surface = Surface(kind=surface_kind, param_dim=2, num_opts=5)

    campaign = Campaign()
    campaign.set_param_space(surface.param_space)

    planner = GPPlanner(
        goal="minimize",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        use_descriptors=use_descriptors,
        acquisition_optimizer_kind="gradient",
        known_constraints=[known_constraints_cat],
    )
    planner.set_param_space(surface.param_space)

    BUDGET = num_init_design + batch_size * 4

    while len(campaign.observations.get_values()) < BUDGET:
        if len(campaign.observations.get_values()) >= num_init_design:
            # TODO: only use the pending experiments after the initial design has completed
            pending_experiments = [
                {
                    f"param_0": f"x{np.random.randint(5)}",
                    "param_1": f"x{np.random.randint(5)}",
                },
                {
                    f"param_0": f"x{np.random.randint(5)}",
                    "param_1": f"x{np.random.randint(5)}",
                },
                {
                    f"param_0": f"x{np.random.randint(5)}",
                    "param_1": f"x{np.random.randint(5)}",
                },
                {
                    f"param_0": f"x{np.random.randint(5)}",
                    "param_1": f"x{np.random.randint(5)}",
                },
                {
                    f"param_0": f"x{np.random.randint(5)}",
                    "param_1": f"x{np.random.randint(5)}",
                },
            ]
            pending_experiments = [
                ParameterVector().from_dict(pending_exp, surface.param_space)
                for pending_exp in pending_experiments
            ]

            #  set pending experiment constraint
            planner.set_pending_experiments(
                pending_experiments=pending_experiments
            )

            print(
                "has pending experiment constriant : ",
                planner.known_constraints.has_pending_experiment_constraint,
            )
            print(planner.known_constraints.known_constraints[-1])

        samples = planner.recommend(campaign.observations)
        for sample in samples:
            if len(campaign.observations.get_values()) >= num_init_design:
                # assert that the proposed sample in not in the list of pending experiments
                for pending_exp in pending_experiments:
                    sample_arr = sample.to_array()
                    pending_exp_arr = pending_exp.to_array().astype(str)
                    assert not (
                        pending_exp_arr == sample_arr.astype(str)
                    ).all()

            sample_arr = sample.to_array()
            measurement = np.array(surface.run(sample_arr))
            # print(sample, measurement)
            campaign.add_observation(sample_arr, measurement[0])

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET


def run_batch_constrained_disc(
    init_design_strategy, batch_size, use_descriptors, num_init_design=5
):
    assert batch_size > 1
    num_init_design = batch_size

    def surface(x):
        return np.sin(8 * x[0]) - 2 * np.cos(6 * x[1]) + np.exp(-2.0 * x[2])

    param_space = ParameterSpace()
    param_0 = ParameterDiscrete(
        name="param_0",
        options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
    )
    param_1 = ParameterDiscrete(
        name="param_1",
        options=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    param_2 = ParameterDiscrete(
        name="param_2",
        options=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    param_space.add(param_0)
    param_space.add(param_1)
    param_space.add(param_2)

    planner = GPPlanner(
        goal="minimize",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        acquisition_type="qei",
        acquisition_optimizer_kind="gradient",
        known_constraints=[known_constraints_fully_disc],
        batch_constrained_params=[0],  # first param same across batch
    )

    planner.set_param_space(param_space)

    campaign = Campaign()
    campaign.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 4

    while len(campaign.observations.get_values()) < BUDGET:
        samples = planner.recommend(campaign.observations)
        for sample in samples:
            sample_arr = sample.to_array()
            measurement = surface(sample_arr)
            campaign.add_observation(sample_arr, measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET

    # check if batch constrained param is the same across all batches

    # validate batch constraint
    meas_params = campaign.observations.get_params()
    meas_params = meas_params.reshape(
        (int(meas_params.shape[0] / batch_size), batch_size, len(param_space))
    )
    for batch in meas_params:
        assert (batch == batch[0, :])[
            0, :
        ].all()  # first param is batch constrained


def run_pymoo(
    init_design_strategy, batch_size, use_descriptors, num_init_design=5
):
    def surface(x):
        return (
            np.sin(8 * float(x[0]))
            - 2 * np.cos(6 * float(x[1]))
            + np.exp(-2.0 * float(x[2]))
        )

    param_space = ParameterSpace()
    param_0 = ParameterContinuous(name="param_0", low=0.0, high=1.0)
    param_1 = ParameterContinuous(name="param_1", low=0.0, high=1.0)
    # param_1 = ParameterDiscrete(name="param_1", options=[0.0, 0.5, 1.0])
    param_2 = ParameterContinuous(name="param_2", low=0.0, high=1.0)
    # param_2 = ParameterCategorical(name="param_2", options=['x1', 'x2', 'x3'])
    param_space.add(param_0)
    param_space.add(param_1)
    param_space.add(param_2)

    if batch_size > 1:
        acquisition_type = "qei"
    else:
        acquisition_type = "ei"

    planner = GPPlanner(
        goal="minimize",
        feas_strategy="fwa",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        acquisition_type=acquisition_type,
        acquisition_optimizer_kind="pymoo",
        known_constraints=[known_constraints_cont],
    )

    planner.set_param_space(param_space)

    campaign = Campaign()
    campaign.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 4

    while len(campaign.observations.get_values()) < BUDGET:
        samples = planner.recommend(campaign.observations)
        for sample in samples:
            sample_arr = sample.to_array()
            measurement = surface(sample_arr)
            campaign.add_observation(sample_arr, measurement)

            print("-" * 75)
            print("SAMPLE : ", sample)
            print("MEASUREMENT : ", measurement)
            print("-" * 75)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET

    # check that all the measured values pass the known constraint
    meas_params = campaign.observations.get_params()

    kcs = [known_constraint(param) for param in meas_params]

    assert all(kcs)


if __name__ == "__main__":
    # run_continuous('random', 1, False, 'ucb', 'pymoo')
    # run_discrete('random', 1, False, 'ucb', 'pymoo')
    # run_categorical('random', 1, False, 'ucb', 'pymoo')
    # run_categorical('random', 1, True, 'ucb', 'pymoo')

    # run_mixed_cat_cont('random', 1, False, 'ucb', 'pymoo')
    # run_mixed_cat_cont('random', 1, True, 'ucb', 'pymoo')

    # run_mixed_disc_cont('random', 1, False, 'ucb', 'pymoo')

    # run_mixed_cat_disc('random', 1, False, 'ucb', 'pymoo')
    # run_mixed_cat_disc('random', 1, True, 'ucb', 'pymoo')

    run_mixed_cat_disc_cont("random", 1, False, "ucb", "pymoo")
    run_mixed_cat_disc_cont("random", 1, True, "ucb", "pymoo")

    # WORKING
    # run_discrete('random', 1, False, 'gradient')
    # run_categorical('random', 1, False, 'pymoo')
    # run_mixed_cat_disc('random', 1, False, 'pymoo')
    # run_mixed_cat_cont('random', 1, False, 'pymoo')
    # run_mixed_cat_disc_cont('random', 1, False, 'pymoo')
    # run_compositional_constraint_cont('random', 1, num_init_design=5)
    # run_permutation_constraint_mixed_cat_disc('random', 1, num_init_design=5)
    # run_mixed_disc_cont('random', 1, False, 'pymoo')

    # NOT WORKING

    # run_mixed_cat_disc('random', 3, False, 'gradient')
    # run_discrete('random', 2, False, 'gradient')
    # run_mixed_cat_disc_large_space('random', 5, False, 'pymoo')
    # run_pending_experiment_constraint_cat('random', 1, False, 'gradient')
    # run_batch_constrained_disc('random', 3, False, 3)
    # run_pymoo('random', 2, False, 5)
