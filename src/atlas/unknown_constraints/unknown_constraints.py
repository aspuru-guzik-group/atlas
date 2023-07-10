#!/usr/bin/env python


from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from copy import deepcopy

import numpy as np
import torch
import gpytorch

class UnknownConstraints:

    def __init__(
        self, 
        params_obj, 
        feas_strategy,
        feas_param,

    ):
        self.params_obj = params_obj
        self.feas_strategy = feas_strategy
        self.feas_param = feas_param



    def handle_naive_feas_strategies(
        self,
        train_x_scaled_reg: torch.Tensor, 
        train_y_scaled_reg: torch.Tensor,
        train_x_scaled_cla: torch.Tensor,
        train_y_scaled_cla: torch.Tensor,
        reg_model: Optional[gpytorch.models.ExactGP] = None,
    ) -> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,bool]:

        use_p_feas_only = False

        if "naive-" in self.feas_strategy:
            infeas_ix = torch.where(train_y_scaled_cla == 1.0)[0]
            feas_ix = torch.where(train_y_scaled_cla == 0.0)[0]
            # checking if we have at least one objective function measurement
            #  and at least one infeasible point (i.e. at least one point to replace)
            if np.logical_and(
                train_y_scaled_reg.size(0) >= 1,
                infeas_ix.shape[0] >= 1,
            ):
                if self.feas_strategy == "naive-replace":
                    # NOTE: check to see if we have a trained regression surrogate model
                    # if not, wait for the following iteration to make replacements
                    if reg_model:
                        # if we have a trained regression model, go ahead and make replacement
                        new_train_y_scaled_reg = deepcopy(
                            train_y_scaled_cla
                        ).double()

                        input_ = train_x_scaled_cla[infeas_ix].double()

                        posterior = reg_model.posterior(X=input_)
                        pred_mu = posterior.mean.detach()

                        new_train_y_scaled_reg[
                            infeas_ix
                        ] = pred_mu.squeeze(-1)
                        new_train_y_scaled_reg[
                            feas_ix
                        ] = train_y_scaled_reg.squeeze(-1)

                        train_x_scaled_reg = deepcopy(
                            train_x_scaled_cla
                        ).double()
                        train_y_scaled_reg = (
                            new_train_y_scaled_reg.view(
                                self.train_y_scaled_cla.size(0), 1
                            ).double()
                        )

                    else:
                        use_p_feas_only = True

                elif self.feas_strategy == "naive-0":
                    new_train_y_scaled_reg = deepcopy(
                        train_y_scaled_cla
                    ).double()

                    worst_obj = torch.amax(
                        train_y_scaled_reg[~train_y_scaled_reg.isnan()]
                    )

                    to_replace = torch.ones(infeas_ix.size()) * worst_obj

                    new_train_y_scaled_reg[infeas_ix] = to_replace.double()
                    new_train_y_scaled_reg[
                        feas_ix
                    ] = train_y_scaled_reg.squeeze()

                    train_x_scaled_reg = (
                        train_x_scaled_cla.double()
                    )
                    train_y_scaled_reg = new_train_y_scaled_reg.view(
                        train_y_scaled_cla.size(0), 1
                    )

                else:
                    raise NotImplementedError
            else:
                # if we are not able to use the naive strategies, propose randomly
                # do nothing at all and use the feasibilty surrogate as the acquisition
                use_p_feas_only = True

        return ( 
            train_x_scaled_reg,
            train_y_scaled_reg,
            train_x_scaled_cla,
            train_y_scaled_cla,
            use_p_feas_only
        )


