#!/usr/bin/env python

import os
import pickle
import sys

import gpytorch
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from atlas import Logger, tkwargs
from atlas.utils.network_utils import get_args, parse_params

torch.set_default_dtype(torch.double)


class Feature(nn.Module):
    """
    Feature extractor
    """

    def __init__(self, x_dim, h_dim, z_dim):
        super(Feature, self).__init__()
        self.layer1 = nn.Linear(x_dim, h_dim)
        self.layer2 = nn.Linear(h_dim, h_dim)
        self.layer3 = nn.Linear(h_dim, z_dim)
        # self.layer4 = nn.Linear(h_dim, z_dim)

    def forward(self, x):
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer3(out))
        # out = F.relu(self.layer4(out))
        return out


# TODO: this is redundant definition I think
class ExactGPModel(gpytorch.models.ExactGP):
    """Exact GP model"""

    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.MaternKernel(nu=2.5)
        )
        # self.covar_module = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=40)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class DKT:
    DEFAULT_HYPERPARAMS = {
        "model": {
            "device": "cpu",
            "params_scaling": "same",
            "values_scaling": "same",
            "epochs": 40000,
            "pred_int": 1000,
            "learning_rate_net": 1e-3,
            "learning_rate_gp": 1e-3,
            "batch_size": 100,
            "h_dim": 48,
            "z_dim": 40,
        }
    }

    def __init__(
        self,
        x_dim,
        y_dim,
        from_disk=False,
        model_path="./tmp_model/",
        hyperparams={},
    ):
        self.x_dim = x_dim
        self.y_dim = (y_dim,)
        self.from_disk = from_disk
        self.model_path = model_path

        # parse hyperparams
        self.hp = {}
        for key, def_dict in self.DEFAULT_HYPERPARAMS.items():
            if key in hyperparams.keys():
                true = parse_params(hyperparams[key], def_dict)
            else:
                true = def_dict
            self.hp[key] = true

        # TODO: check for video card and deal with dataset stats

        # build the modules
        self.net = Feature(
            self.x_dim,
            self.hp["model"]["h_dim"],
            self.hp["model"]["z_dim"],
        )

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.dummy_params = torch.zeros(
            (self.hp["model"]["batch_size"], self.hp["model"]["z_dim"])
        )
        # NOTE: this assumes that y_dim=1, will be the case here
        self.dummy_values = torch.zeros([self.hp["model"]["batch_size"]])

        self.gp = ExactGPModel(
            self.dummy_params, self.dummy_values, self.likelihood
        )

        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.gp
        )

        self.optimizer = torch.optim.Adam(
            [
                {
                    "params": self.gp.parameters(),
                    "lr": self.hp["model"]["learning_rate_gp"],
                },
                {
                    "params": self.net.parameters(),
                    "lr": self.hp["model"]["learning_rate_net"],
                },
            ]
        )

        if self.from_disk:
            self.restore_model()

    def forward(
        self,
        context_x,
        context_y,
        target_x,
        training=False,
        scaled=False,
        return_unscaled=False,
    ):
        if len(target_x.shape) == 3:
            # remove middle dimension
            target_x = target_x[:, 0, :]
        # check the shape of the context_y data, needs to be 1 deimensional
        if len(context_y.shape) == 2:
            context_y = torch.squeeze(context_y)

        # prime the deep kernel on the context set
        context_z = self.net(context_x).detach()
        self.gp.train()
        self.gp.set_train_data(
            inputs=context_z, targets=context_y, strict=False
        )
        self.gp.eval()

        # with torch.no_grad():
        # evaluate on the target set
        target_z = self.net(target_x)
        pred = self.gp(target_z)
        likelihood = self.likelihood(pred)
        mu, sigma = likelihood.mean, torch.sqrt(likelihood.variance)
        # lower, upper = likelihood.confidence_region()

        return mu, sigma, likelihood

    def generator(
        self,
        tasks,
        set="train",
    ):
        """select a task and produce samples of context and target points"""
        if set in ["train", "valid"]:
            task = np.random.choice(tasks)

            params, values = task["params"], task["values"]

            np.random.shuffle(params)
            np.random.shuffle(values)

            num_target = np.random.randint(
                low=int(0.7 * params.shape[0]),
                high=int(1.0 * params.shape[0]),
                size=None,
            )

            num_context = np.random.randint(
                low=2, high=int(0.7 * num_target), size=None
            )

            context_x = params[:num_context, :]
            context_y = values[:num_context]
            target_x = params[:num_target, :]
            target_y = values[:num_target]

            num_repeats_context = (
                self.hp["model"]["batch_size"] // num_context
            ) + 1
            num_repeats_target = (
                self.hp["model"]["batch_size"] // num_target
            ) + 1

            context_x = torch.tile(context_x, (num_repeats_context, 1))[
                : self.hp["model"]["batch_size"], :
            ]
            context_y = torch.tile(context_y, (num_repeats_context, 1))[
                : self.hp["model"]["batch_size"], :
            ].flatten()

            target_x = torch.tile(target_x, (num_repeats_target, 1))[
                : self.hp["model"]["batch_size"], :
            ]
            target_y = torch.tile(target_y, (num_repeats_target,))[
                : self.hp["model"]["batch_size"], :
            ].flatten()

        return context_x, context_y, target_x, target_y

    def train(
        self,
        train_tasks,
        valid_tasks=None,
    ):
        """train the dkt model"""

        # check to see if the input data is numpy arrays,
        # if so, convert to torch.Tensor
        _train_tasks, _valid_tasks = [], []
        for train_task in train_tasks:
            tmp_train_task = {}
            if isinstance(train_task["params"], np.ndarray):
                tmp_train_task["params"] = torch.from_numpy(
                    train_task["params"]
                )
            elif isinstance(train_task["params"], torch.Tensor):
                tmp_train_task["params"] = train_task["params"]
            if isinstance(train_task["values"], np.ndarray):
                tmp_train_task["values"] = torch.from_numpy(
                    train_task["values"]
                )
            elif isinstance(train_task["values"], torch.Tensor):
                tmp_train_task["values"] = train_task["values"]
            _train_tasks.append(tmp_train_task)
        if not isinstance(valid_tasks, type(None)):
            for valid_task in valid_tasks:
                tmp_valid_task = {}
                if isinstance(valid_task["params"], np.ndarray):
                    tmp_valid_task["params"] = torch.from_numpy(
                        valid_task["params"]
                    )
                elif isinstance(valid_task["params"], torch.Tensor):
                    tmp_valid_task["params"] = valid_task["params"]
                if isinstance(valid_task["values"], np.ndarray):
                    tmp_valid_task["values"] = torch.from_numpy(
                        valid_task["values"]
                    )
                elif isinstance(valid_task["values"], torch.Tensor):
                    tmp_valid_task["values"] = valid_task["values"]
                _valid_tasks.append(tmp_valid_task)

        train_tasks = _train_tasks
        valid_tasks = _valid_tasks

        self.likelihood.train()
        self.gp.train()
        self.net.train()

        criterion = nn.MSELoss()

        for epoch in range(self.hp["model"]["epochs"]):
            self.optimizer.zero_grad()

            # training
            # train_context_x_batch, train_context_y_batch, _, __ = self.generator(train_tasks, 'train')

            # generate the training example

            task = np.random.choice(train_tasks)
            params, values = task["params"], task["values"]

            indices = np.arange(params.shape[0])
            np.random.shuffle(indices)

            n_context = np.random.randint(2, int(0.8 * params.shape[0]))

            context_x = params[:n_context, :]
            context_y = values[:n_context, :].flatten()

            num_reps = (self.hp["model"]["batch_size"] // n_context) + 1

            train_context_x_batch = torch.tile(context_x, (num_reps, 1))[
                : self.hp["model"]["batch_size"], :
            ]
            train_context_y_batch = torch.tile(context_y, (num_reps,))[
                : self.hp["model"]["batch_size"]
            ]

            z = self.net(train_context_x_batch)
            self.gp.set_train_data(inputs=z, targets=train_context_y_batch)
            preds = self.gp(z)

            loss = -self.mll(preds, self.gp.train_targets)
            loss.backward()
            self.optimizer.step()
            mse = criterion(preds.mean, train_context_y_batch)

            if epoch % self.hp["model"]["pred_int"] == 0:
                loss = np.around(loss.detach().numpy(), 3)
                mse = np.around(mse.detach().numpy(), 3)
                Logger.log(
                    f"[EPOCH {epoch}] - Train loss: {round(loss,2)} Train mse: {round(mse,2)}",
                    "INFO",
                )

        # set the likelihood and net to evaluation mode
        self.likelihood.eval()
        self.net.eval()
        # save the model here
        # TODO: implement early stopping
        self._save_model()

    def _save_model(self):
        os.makedirs(self.model_path, exist_ok=True)
        torch.save(
            {
                "gp_state_dict": self.gp.state_dict(),
                "likelihood_state_dict": self.likelihood.state_dict(),
                "net_state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            os.path.join(self.model_path, "model.pkl"),
        )

    def restore_model(self):
        checkpoint = torch.load(os.path.join(self.model_path, "model.pkl"))
        if isinstance(checkpoint, dict):
            self.gp.load_state_dict(checkpoint["gp_state_dict"])
            self.likelihood.load_state_dict(
                checkpoint["likelihood_state_dict"]
            )
            self.net.load_state_dict(checkpoint["net_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            self.log("Restore checkpoint has unexpected type! (not dict)")


# DEBUG:
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import seaborn as sns

    train_tasks = []
    valid_tasks = []

    for i in range(10):
        params = np.linspace(-1, 1, 100).reshape(-1, 1)
        values = np.sin(4 * params)
        train_tasks.append({"params": params, "values": values})

    for i in range(2):
        params = np.linspace(-1, 1, 100).reshape(-1, 1)
        values = np.sin(4 * params + 0.2)
        valid_tasks.append({"params": params, "values": values})

    model = DKT(x_dim=1, y_dim=1, hyperparams={"model": {"epochs": 10000}})

    # train the model
    model.train(train_tasks, valid_tasks)

    # load the model
    model.restore_model()

    # make a prediction on a new function
    for n_context in [1, 2, 3, 5, 10]:  # [1, 2, 3, 4, 5]:
        n_target = 100

        context_indices = np.arange(100)
        np.random.shuffle(context_indices)
        context_indices = context_indices[:n_context]

        context_x = torch.from_numpy(
            valid_tasks[0]["params"][context_indices, :]
        )
        context_y = torch.from_numpy(
            valid_tasks[0]["values"][context_indices, :]
        )

        target_x = torch.from_numpy(valid_tasks[0]["params"])
        target_y = torch.from_numpy(valid_tasks[0]["values"])

        mu, sigma, likelihood = model.forward(
            context_x, context_y, target_x, None
        )

        lower, upper = likelihood.confidence_region()

        print(mu.shape, lower.shape, upper.shape)

        # print(mu, sigma)

        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots()

        ax.plot(target_x, target_y, label="true surface")

        ax.plot(context_x, context_y, ls="", marker="o", markersize=10)

        ax.plot(
            target_x, mu.detach().numpy(), label="pred mean", color="red", lw=2
        )

        ax.fill_between(
            target_x.flatten(),
            lower.detach().numpy().flatten(),
            upper.detach().numpy().flatten(),
            alpha=0.1,
            color="red",
        )

        plt.tight_layout()
        plt.show()
