#!/usr/bin/env python

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from botorch.acquisition import AcquisitionFunction

from olympus.objects import ABCMeta, Config, Object, abstract_attribute



class AdamOptimizer:
    """ Adam optimizer as reported in https://arxiv.org/abs/1412.6980
    """
    
    # ATT_TKWARGS = {
    #         'dtype': torch.double,
    #         'device': 'cpu',#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     } 


    def __init__(
        self, 
        acqf:Optional[AcquisitionFunction] = None,#AcquisitionFunction,
        select_params:List[bool] = None,
        eta: torch.Tensor = torch.Tensor([0.01]), 
        beta_1: torch.Tensor = torch.Tensor([0.9]), 
        beta_2: torch.Tensor = torch.Tensor([0.999]), 
        epsilon: torch.Tensor = torch.Tensor([1e-8]), 
        decay: bool = False,
        *args, 
        **kwargs,
    ) -> None: 

        self.acqf = acqf
        self.select_params = select_params
        self.eta = eta 
        self.beta_1 = beta_1 
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.decay = decay
        self.iterations = 0

        self.tkwargs = {
            'dtype': torch.double,
            'device': 'cpu',#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        } 

        if select_params is not None:
            self._init_params(select_params)
        else:
            self.select_bool = None
            self.select_idx = None
            self.num_dims = None
            self.ms = None
            self.vs = None

        # step used to estimate gradients numerically
        self.dx = 1e-6

        
    def _init_params(self, select_params: List[int]) -> None:
        self.select_bool = torch.tensor(select_params)
        self.num_dims = len(self.select_bool)

        self.select_idx = torch.arange(self.num_dims)[self.select_bool]
        self.ms = torch.zeros(self.num_dims)  # moment vector (length is size of input vector, i.e. opt domain)
        self.vs = torch.zeros(self.num_dims)  # exponentially weighted infinity norm


    def reset(self) -> None:
        self.iterations = 0
        self.ms = torch.zeros((self.num_dims,))
        self.vs = torch.zeros((self.num_dims,))


    def set_acqf(self, acqf, select_params=None):
        """
        """
        self.acqf = acqf
        self.reset()
        if select_params is not None:
            self._init_params(select_params)


    def grad(self, sample):
        """ 
        """
        gradients = torch.zeros((len(sample),), **self.tkwargs)
        perturb = torch.zeros((len(sample),), **self.tkwargs)

        sample_to_acqf = sample.view(1, sample.shape[0])

        for i in self.select_idx:
            perturb[i] += self.dx
            gradient = (self.acqf(sample_to_acqf + perturb) - self.acqf(sample_to_acqf - perturb)) / (2. * self.dx)
            gradients[i] = gradient
            perturb[i] -= self.dx

        return gradients


    def compute_update(self, sample: torch.Tensor) -> torch.Tensor:
        grads = self.grad(sample)

        self.iterations += 1

        if self.decay is True:
            eta = self.eta / torch.sqrt(self.iterations)
        else:
            eta = self.eta

        
        eta_next = eta * (torch.sqrt(1. - torch.pow(self.beta_2, self.iterations)) /
                        (1. - torch.pow(self.beta_1, self.iterations)))
        ms_next = (self.beta_1 * self.ms) + (1. - self.beta_1) * grads
        
        vs_next = (self.beta_2 * self.vs) + (1. - self.beta_2) * torch.square(grads)

        sample_next = sample - eta_next * ms_next / (torch.sqrt(vs_next) + self.epsilon)

        # update params
        self.ms = ms_next
        self.vs = vs_next

        return sample_next


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import seaborn

    def acqf(x):
        print(type(x))
        return torch.pow(x-1,2)

    adam = AdamOptimizer(acqf=acqf, select_params=[True])
    print(adam.tkwargs)
    #adam.set_acqf(acqf, select_params=[True])

    domain = torch.linspace(-1, 3, 200)
    values = acqf(domain)

    print(domain.shape, values.shape)

    start = torch.zeros((1,)) - 0.8

    plt.ion()

    for _ in range(10**3):

        plt.clf()
        plt.plot(domain.detach().numpy(), values.detach().numpy())
        plt.plot(start.detach().numpy(), acqf(start).detach().numpy(), marker='o', color='k')

        start = adam.compute_update(start)

        plt.pause(0.05)
