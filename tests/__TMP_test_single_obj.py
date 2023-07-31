#!/usr/bin/env python

import numpy as np
import pytest
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import (
    ParameterCategorical,
    ParameterContinuous,
    ParameterDiscrete,
)
from olympus.surfaces import Surface
from problem_generator import ProblemGenerator

from atlas.planners.gp.planner import GPPlanner

CONT = {
    "init_design_strategy": [
        "random",
        "sobol",
        "lhs",
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
def test_init_design_cont(
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
def test_init_design_disc(
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
def test_init_design_cat(
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
def test_init_design_mixed_cat_cont(
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
def test_init_design_mixed_disc_cont(
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
def test_init_design_mixed_cat_disc(
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
def test_init_design_mixed_cat_disc_cont(
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

    planner = GPPlanner(
        goal="minimize",
        feas_strategy="naive-0",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        acquisition_type=acquisition_type,
        acquisition_optimizer_kind=acquisition_optimizer,
    )

    planner.set_param_space(param_space)

    campaign = Campaign()
    campaign.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 4

    while len(campaign.observations.get_values()) < BUDGET:
        samples = planner.recommend(campaign.observations)
        for sample in samples:
            sample_arr = sample.to_array()
            measurement = surface_callable.run(sample_arr)
            campaign.add_observation(sample_arr, measurement)

            print("SAMPLE : ", sample)
            print("MEASUREMENT : ", measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET


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

    planner = GPPlanner(
        goal="minimize",
        feas_strategy="naive-0",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        acquisition_type=acquisition_type,
        acquisition_optimizer_kind=acquisition_optimizer,
    )

    planner.set_param_space(param_space)

    campaign = Campaign()
    campaign.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 4

    while len(campaign.observations.get_values()) < BUDGET:
        samples = planner.recommend(campaign.observations)
        for sample in samples:
            sample_arr = sample.to_array()
            measurement = surface_callable.run(sample_arr)
            campaign.add_observation(sample_arr, measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET


def run_categorical(
    init_design_strategy,
    batch_size,
    use_descriptors,
    acquisition_type,
    acquisition_optimizer,
    num_init_design=5,
):
    problem_gen = ProblemGenerator(
        use_descriptors=use_descriptors, problem_type="categorical"
    )
    surface_callable, param_space = problem_gen.generate_instance()

    campaign = Campaign()
    campaign.set_param_space(param_space)

    planner = GPPlanner(
        goal="minimize",
        feas_strategy="naive-0",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        use_descriptors=use_descriptors,
        acquisition_type=acquisition_type,
        acquisition_optimizer_kind=acquisition_optimizer,
    )
    planner.set_param_space(surface_callable.param_space)

    BUDGET = num_init_design + batch_size * 4

    while len(campaign.observations.get_values()) < BUDGET:
        samples = planner.recommend(campaign.observations)
        for sample in samples:
            sample_arr = sample.to_array()
            measurement = np.array(surface_callable.run(sample_arr))
            campaign.add_observation(sample_arr, measurement[0])

            print("SAMPLE : ", sample)
            print("MEASUREMENT : ", measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET


def run_mixed_cat_cont(
    init_design_strategy,
    batch_size,
    use_descriptors,
    acquisition_type,
    acquisition_optimizer,
    num_init_design=5,
):
    problem_gen = ProblemGenerator(
        use_descriptors=use_descriptors, problem_type="mixed_cat_cont"
    )
    surface_callable, param_space = problem_gen.generate_instance()

    campaign = Campaign()
    campaign.set_param_space(param_space)

    planner = GPPlanner(
        goal="maximize",
        feas_strategy="naive-0",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        use_descriptors=use_descriptors,
        acquisition_type=acquisition_type,
        acquisition_optimizer_kind=acquisition_optimizer,
    )
    planner.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 4

    while len(campaign.observations.get_values()) < BUDGET:
        samples = planner.recommend(campaign.observations)
        for sample in samples:
            measurement = surface_callable.run(sample)
            # print(f'ITER : {iter}\tSAMPLES : {sample}\t MEASUREMENT : {measurement}')
            campaign.add_observation(sample, measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET


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

    campaign = Campaign()
    campaign.set_param_space(param_space)

    planner = GPPlanner(
        goal="minimize",
        feas_strategy="naive-0",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        use_descriptors=use_descriptors,
        acquisition_type=acquisition_type,
        acquisition_optimizer_kind=acquisition_optimizer,
    )
    planner.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 4

    while len(campaign.observations.get_values()) < BUDGET:
        samples = planner.recommend(campaign.observations)
        for sample in samples:
            measurement = surface_callable.run(sample)
            campaign.add_observation(sample, measurement)

            print("SAMPLE : ", sample)
            print("MEASUREMENT : ", measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET


def run_mixed_cat_disc(
    init_design_strategy,
    batch_size,
    use_descriptors,
    acquisition_type,
    acquisition_optimizer,
    num_init_design=5,
):
    problem_gen = ProblemGenerator(
        use_descriptors=use_descriptors, problem_type="mixed_cat_disc"
    )
    surface_callable, param_space = problem_gen.generate_instance()

    campaign = Campaign()
    campaign.set_param_space(param_space)

    planner = GPPlanner(
        goal="minimize",
        feas_strategy="naive-0",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        use_descriptors=use_descriptors,
        acquisition_type=acquisition_type,
        acquisition_optimizer_kind=acquisition_optimizer,
    )
    planner.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 4

    while len(campaign.observations.get_values()) < BUDGET:
        samples = planner.recommend(campaign.observations)
        for sample in samples:
            measurement = surface_callable.run(sample)
            campaign.add_observation(sample, measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET


def run_mixed_cat_disc_cont(
    init_design_strategy,
    batch_size,
    use_descriptors,
    acquisition_type,
    acquisition_optimizer,
    num_init_design=5,
):
    problem_gen = ProblemGenerator(
        use_descriptors=use_descriptors, problem_type="mixed_cat_disc"
    )
    surface_callable, param_space = problem_gen.generate_instance()

    campaign = Campaign()
    campaign.set_param_space(param_space)

    planner = GPPlanner(
        goal="minimize",
        feas_strategy="naive-0",
        init_design_strategy=init_design_strategy,
        num_init_design=num_init_design,
        batch_size=batch_size,
        use_descriptors=use_descriptors,
        acquisition_type=acquisition_type,
        acquisition_optimizer_kind=acquisition_optimizer,
    )
    planner.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 4

    while len(campaign.observations.get_values()) < BUDGET:
        samples = planner.recommend(campaign.observations)
        for sample in samples:
            measurement = surface_callable.run(sample)
            campaign.add_observation(sample, measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET


if __name__ == "__main__":
    # pass

    # run_discrete(
    # 	init_design_strategy='random',
    # 	batch_size=2,
    # 	use_descriptors=False,
    # 	acquisition_type='ucb',
    # 	acquisition_optimizer='gradient',
    # 	num_init_design=6,
    # )

    # run_continuous(
    #     init_design_strategy="random",
    #     batch_size=1,
    #     use_descriptors=False,
    #     acquisition_type="ucb",
    #     acquisition_optimizer="pymoo",
    #     num_init_design=4,
    # )

    # run_categorical(
    # 	init_design_strategy='random',
    # 	batch_size=2,
    # 	use_descriptors=False,
    # 	acquisition_type='ucb',
    # 	acquisition_optimizer='gradient',
    # 	num_init_design=6,
    # )

    # run_mixed_cat_cont(
    # 	init_design_strategy='random',
    # 	batch_size=1,
    # 	use_descriptors=True,
    # 	acquisition_type='ei',
    # 	acquisition_optimizer='pymoo',
    # 	num_init_design=5,
    # 	)

    # run_mixed_disc_cont(
    # 	init_design_strategy='random',
    # 	batch_size=2,
    # 	use_descriptors=False,
    # 	acquisition_type='ucb',
    # 	acquisition_optimizer='gradient',
    # 	num_init_design=6,
    # 	)

    # run_mixed_cat_disc_cont(
    # 	init_design_strategy='random',
    # 	batch_size=1,
    # 	use_descriptors=True,
    # 	acquisition_type='ei',
    # 	acquisition_optimizer='pymoo',
    # 	num_init_design=5,
    # )

    # run_mixed_cat_disc(
    # 	init_design_strategy='random',
    # 	batch_size=1,
    # 	use_descriptors=True,
    # 	acquisition_type='ei',
    # 	acquisition_optimizer='pymoo',
    # 	num_init_design=5,
    # )

    run_batched(
        problem_type='cont', 
        init_design_strategy='random', 
        batch_size=2, 
        acquisition_optimizer='pymoo',
    )   
