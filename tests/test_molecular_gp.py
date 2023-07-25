#!/usr/bin/env python

import numpy as np
import pandas as pd
import pytest
import torch
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import ParameterCategorical

from atlas.gps.gps import TanimotoGP
from atlas.planners.gp.planner import BoTorchPlanner

tkwargs = {"dtype": torch.double, "device": "cpu"}


def run_delaney():
    # delaney = pd.read_csv('data/delaney.csv')
    # y = delaney['measured log(solubility:mol/L)'].values.reshape(-1, 1)
    features = np.load("data/delaney_features_mfp.npz", mmap_mode="r")
    X = torch.tensor(features["values"], **tkwargs)  # (1116, 2048)
    y = torch.rand(size=(X.shape[0], 1), **tkwargs)

    model = TanimotoGP(X, y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)

    posterior = model.posterior(X=X)
    mean = posterior.mean.squeeze(-2).squeeze(-1)
    sigma = posterior.variance.clamp_min(1e-12).sqrt().view(mean.shape)

    assert len(mean.shape) == 1
    assert len(sigma.shape) == 1
    assert mean.shape[0] == y.shape[0]
    assert sigma.shape[0] == y.shape[0]


def run_single_molecular_param(num_init_design, batch_size):
    def obj(params):
        return np.random.uniform()

    # get some of the delaney smiles
    # NOTE: the smiles and descriptors dont match up here - just a test
    delaney = pd.read_csv("data/delaney.csv")
    smiles = list(delaney["SMILES"])[
        :200
    ]  # categorical param molecular options

    features = np.load("data/delaney_features_mfp.npz", mmap_mode="r")
    descriptors = features["values"][:200, :]  # (1116, 2048)
    descriptors = list(
        [list(desc) for desc in descriptors]
    )  # convert to list of lists

    param_space = ParameterSpace()
    param_space.add(
        ParameterCategorical(
            name="molecule",
            options=smiles,
            descriptors=descriptors,
        )
    )

    campaign = Campaign()
    campaign.set_param_space(param_space)

    planner = BoTorchPlanner(
        goal="minimize",
        init_design_strategy="random",
        num_init_design=num_init_design,
        batch_size=batch_size,
        use_descriptors=True,
        acquisition_type="ucb",
        acquisition_optimizer_kind="gradient",
        molecular_params=[0],  # list of ints rep molecular dims
    )
    planner.set_param_space(param_space)

    BUDGET = num_init_design + batch_size * 4

    while len(campaign.observations.get_values()) < BUDGET:
        samples = planner.recommend(campaign.observations)
        for sample in samples:
            measurement = obj(sample)
            campaign.add_observation(sample, measurement)

            print("SAMPLE : ", sample)
            print("MEASUREMENT : ", measurement)

    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET

    assert isinstance(planner.reg_model, TanimotoGP)


if __name__ == "__main__":
    # run_delaney()
    run_single_molecular_param(
        num_init_design=5,
        batch_size=1,
    )
