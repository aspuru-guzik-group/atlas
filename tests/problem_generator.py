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


     def __init__(self, problem_type: str, use_descriptors=False):
         self.problem_type = problem_type
         self.accepted_problem_types = ['continuous', 'discrete', 'categorical']
         self.use_descriptors = use_descriptors

     def check_problem_type(self):
         return self.problem_type in self.accepted_problem_types

     @property
     def allowed_cont_surfaces(self):
         return ['Dejong', 'Schwefel', 'Branin',  'AckleyPath', 'Denali']


     @property
     def allowed_cat_surfaces(self):
         return ['CatDejong', 'CatAckley', 'CatMichalewicz']

     def add_descriptors(self, cat_surface):
        if self.use_descriptors:
            for param in cat_surface.param_space:
                desc = [[float(i), float(i)] for i in range(len(param.options))]
                param.descriptors = desc
        else:
            pass
        return cat_surface
        
     def generate_instance(self):

         if self.problem_type == 'continuous':

             surface_choice = str(np.random.choice(self.allowed_cont_surfaces, size=None))
             print(surface_choice)
             surface_callable = Surface(kind=surface_choice)

             print(surface_callable.param_space)

             return surface_callable, surface_callable.param_space

         elif self.problem_type == 'discrete':
             surface_choice = str(np.random.choice(self.allowed_cont_surfaces, size=None))
             print(surface_choice)
             surface_callable = Surface(kind=surface_choice)

             print(surface_callable.param_space)

             discrete_param_space = ParameterSpace()
             param_0 = ParameterDiscrete(
                 name="param_0",
                options=[0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
             )
             param_1 = ParameterDiscrete(
                 name="param_1",
                 options=[0.0, 0.25, 0.5, 0.75, 1.0],
             )
             discrete_param_space.add(param_0)
             discrete_param_space.add(param_1)

             return surface_callable, discrete_param_space


         elif self.problem_type == 'categorical':
            surface_names = ['CatDejong', 'CatAckley', 'CatMichalewicz']
            surface_choice = str(np.random.choice(self.allowed_cat_surfaces, size=None))
            surface_callable = Surface(kind=surface_choice, num_opts=21)
            surface_callable = self.add_descriptors(surface_callable)
            return surface_callable, surface_callable.param_space



         elif self.problem_type == 'mixed_cat_cont':
            surface_choice_cat = str(np.random.choice(self.allowed_cat_surfaces, size=None))
            surface_choice_cont = str(np.random.choice(self.allowed_cont_surfaces, size=None))
            surface_callable_cat = Surface(kind=surface_choice_cat, num_opts=21)
            surface_callable_cont  = Surface(kind=surface_choice_cont)
            surface_callable_cat = self.add_descriptors(surface_callable_cat)
            hybrid_surface_callable = HybridSurface(surfaces=[surface_callable_cat, surface_callable_cont])

            return hybrid_surface_callable, hybrid_surface_callable.param_space


         elif self.problem_type == 'mixed_disc_cont':
          surface_choice_disc = str(np.random.choice(self.allowed_cont_surfaces, size=None))
          surface_choice_cont = str(np.random.choice(self.allowed_cont_surfaces, size=None))
          surface_callable_disc = Surface(kind=surface_choice_disc)
          surface_callable_cont  = Surface(kind=surface_choice_cont)

          hybrid_surface_callable = HybridSurface(surfaces=[surface_callable_disc, surface_callable_cont])

          return hybrid_surface_callable, hybrid_surface_callable.param_space


         elif self.problem_type == 'mixed_cat_disc':
          surface_choice_cat = str(np.random.choice(self.allowed_cat_surfaces, size=None))
          surface_choice_disc = str(np.random.choice(self.allowed_cont_surfaces, size=None))
          surface_callable_cat = Surface(kind=surface_choice_cat, num_opts=21)
          surface_callable_disc  = Surface(kind=surface_choice_disc)
          surface_callable_cat = self.add_descriptors(surface_callable_cat)
          hybrid_surface_callable = HybridSurface(surfaces=[surface_callable_cat, surface_callable_disc])

          return hybrid_surface_callable, hybrid_surface_callable.param_space

         elif self.problem_type == 'mixed_cat_disc_cont':
          surface_choice_cat = str(np.random.choice(self.allowed_cat_surfaces, size=None))
          surface_choice_disc = str(np.random.choice(self.allowed_cont_surfaces, size=None))
          surface_choice_cont = str(np.random.choice(self.allowed_cont_surfaces, size=None))
          surface_callable_cat = Surface(kind=surface_choice_cat, num_opts=21)
          surface_callable_disc  = Surface(kind=surface_choice_disc)
          surface_callable_cont  = Surface(kind=surface_choice_cont)
          surface_callable_cat = self.add_descriptors(surface_callable_cat)
          
          hybrid_surface_callable = HybridSurface(surfaces=[surface_callable_cat, surface_callable_disc, surface_callable_cont])

          return hybrid_surface_callable, hybrid_surface_callable.param_space


class HybridSurface:
    def __init__(self, surfaces):
        self.surfaces = surfaces 
        self.param_space = ParameterSpace()
        counter = 0 
        for surface in self.surfaces:
            for param in surface.param_space:
                param.name = f'param_{counter}'
                self.param_space.add(param)

                counter+=1

    def run(self, params):
    
        params_arr = params.to_array()
        counter = 0
        objs = []
        for surface in self.surfaces:
            end_counter = counter+len(surface.param_space)
            relevant_params_arr  = params_arr[counter:end_counter]
            if surface.param_space[0].type == 'categorical':
                pass
            else:
                relevant_params_arr = relevant_params_arr.astype(float)
            obj = surface.run(relevant_params_arr)
            objs.append(obj)

            counter = end_counter

        return np.sum(objs)


if __name__ == '__main__':

    problem_gen = ProblemGenerator(problem_type='continuous')
    problem_gen = ProblemGenerator(problem_type='discrete')
    problem_gen = ProblemGenerator(problem_type='categorical')
    surface_callable, param_space = problem_gen.generate_instance()








        
        