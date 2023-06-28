#!/usr/bin/env python
import pytest
import numpy as np
from olympus.surfaces import Surface
from olympus.campaigns import Campaign
from olympus.campaigns import Campaign, ParameterSpace
from olympus.objects import (
    ParameterCategorical,
    ParameterContinuous,
    ParameterDiscrete,
)

class ProblemGenerator():


     def __init__(self, problem_type: str):
         self.problem_type = problem_type
         self.accepted_problem_types = ['continuous', 'discrete', 'categorical']


     def check_problem_type(self):
         return self.problem_type in self.accepted_problem_types
        

     def generate_instance(self):
         """
         return surface_callable and campaign 

         - vary the number of parameters, nnumber of obejctives etc.
         - vary the problem itself
         - vary the .... 

         """
         if self.problem_type == 'continuous':
             surface_names = ['Dejong', 'Schwefel', 'Branin']
             surface_choice = str(np.random.choice(surface_names, size=None))
             print(surface_choice)
             surface_callable = Surface(kind=surface_choice)
             # return some surface_callable 

             print(surface_callable.param_space)

             return surface_callable, param_space

         elif self.problem_type == 'discrete':
         	surface_names = ['Dejong', 'Schwefel', 'Branin']
         	surface_choice = str(np.random.choice(surface_names, size=None))
         	surface_callable = Surface(kind=surface_choice)
         	print(surface_callable.param_space)
             # return something else

         elif self.problem_type == 'categorical':
            surface_names = ['CatDejong', 'CatAckley', 'CatMichalewicz']
            surface_choice = str(np.random.choice(surface_names, size=None))
            surface_callable = Surface(kind=surface_choice)
            print(surface_callable.param_space)
             # return something else



if __name__ == '__main__':

	problem_gen = ProblemGenerator(problem_type='continuous')

	surface_callable, param_space = problem_gen.generate_instance()








        
        