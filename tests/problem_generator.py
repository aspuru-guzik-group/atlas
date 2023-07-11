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
    ParameterVector,
)



class ProblemGenerator():

     def __init__(self, problem_type: str, use_descriptors=False, is_moo=False):
         self.problem_type = problem_type
         self.accepted_problem_types = ['continuous', 'discrete', 'categorical', 'contrained_continuous', 'contrained_discrete', 'contrained_cat']
         self.use_descriptors = use_descriptors
         self.is_moo = is_moo


     def check_problem_type(self):
         return self.problem_type in self.accepted_problem_types

     @property
     def allowed_cont_surfaces(self):
         return ['Dejong', 'Schwefel', 'Branin',  'AckleyPath', 'Denali']
     
     @property
     def allowed_cont_moo_surfaces(self):
         return ['MultFonseca', 'MultZdt1', 'MultZdt2', 'MultZdt3']


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
            
             if self.is_moo:
                 # multi-objective
                 surface_choice = str(np.random.choice(self.allowed_cont_moo_surfaces, size=None))
                 surface_callable = Surface(kind=surface_choice, value_dim=2)
             else:
                 # single objective
                 surface_choice = str(np.random.choice(self.allowed_cont_surfaces, size=None))
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
            hybrid_surface_callable = HybridSurface(
                surfaces=[surface_callable_cat, surface_callable_cont],
                is_moo=self.is_moo,
            )

            return hybrid_surface_callable, hybrid_surface_callable.param_space


         elif self.problem_type == 'mixed_disc_cont':
          surface_choice_disc = str(np.random.choice(self.allowed_cont_surfaces, size=None))
          surface_choice_cont = str(np.random.choice(self.allowed_cont_surfaces, size=None))
          surface_callable_disc = Surface(kind=surface_choice_disc)
          surface_callable_cont  = Surface(kind=surface_choice_cont)

          hybrid_surface_callable = HybridSurface(
            surfaces=[surface_callable_disc, surface_callable_cont],
            is_moo=self.is_moo,
          )

          return hybrid_surface_callable, hybrid_surface_callable.param_space


         elif self.problem_type == 'mixed_cat_disc':
          surface_choice_cat = str(np.random.choice(self.allowed_cat_surfaces, size=None))
          surface_choice_disc = str(np.random.choice(self.allowed_cont_surfaces, size=None))
          surface_callable_cat = Surface(kind=surface_choice_cat, num_opts=21)
          surface_callable_disc  = Surface(kind=surface_choice_disc)
          surface_callable_cat = self.add_descriptors(surface_callable_cat)
          hybrid_surface_callable = HybridSurface(
            surfaces=[surface_callable_cat, surface_callable_disc],
            is_moo=self.is_moo,
          )

          return hybrid_surface_callable, hybrid_surface_callable.param_space

         elif self.problem_type == 'mixed_cat_disc_cont':
          surface_choice_cat = str(np.random.choice(self.allowed_cat_surfaces, size=None))
          surface_choice_disc = str(np.random.choice(self.allowed_cont_surfaces, size=None))
          surface_choice_cont = str(np.random.choice(self.allowed_cont_surfaces, size=None))
          surface_callable_cat = Surface(kind=surface_choice_cat, num_opts=21)
          surface_callable_disc  = Surface(kind=surface_choice_disc)
          surface_callable_cont  = Surface(kind=surface_choice_cont)
          surface_callable_cat = self.add_descriptors(surface_callable_cat)
          
          hybrid_surface_callable = HybridSurface(
            surfaces=[surface_callable_cat, surface_callable_disc, surface_callable_cont],
            is_moo=self.is_moo,
          )

          return hybrid_surface_callable, hybrid_surface_callable.param_space
         


class KnownConstraintsGenerator():


    def __init__(self):
        pass


    def get_constraint(self, problem_type:str):
        return getattr(self, f'known_constraint_{problem_type}')
    

    @staticmethod
    def known_constraint_continuous(params):
        x0 = params[0]
        x1 = params[1]
        y = (x0-0.5)**2 + (x1-0.5)**2
        if np.abs(x0-x1)<0.1:
            return False
        
        if 0.05 < y < 0.15:
            return False
        
        else:
            return True
  

    @staticmethod
    def known_constraint_discrete(params):  
        x0 = params[0]
        x1 = params[1]
        y = (x0-0.5)**2 + (x1-0.5)**2
        if np.abs(x0-x1)<0.1:
            return False
        
        if 0.05 < y < 0.15:
            return False
        
        else:
            return True


    @staticmethod
    def known_constraint_categorical(params):
       # if params[0] == 'x13' and params[1] =='x2': --> as "params[0]" is x and "params[1]" is y, imagine a grid.
       # if params[0] == 'x13' this blocks out the entirity of the x axis of x = 13
       # if params[0] == 'x13' and params[1] in ['x2', 'x15'] --> this is how to do straight line
       # total of 441 blocks --> need 30-50% infesable
        x0 = params[0]
        x1 = params[1]
        
        conditions = (
            (x0 == ['x2', 'x6', 'x9', 'x13', 'x17']),
            (x1 == ['x3', 'x6', 'x10', 'x15', 'x19']),
            (x0 == 'x5' and x1 in ['x7']),
            (x0 == 'x4' and x1 in ['x12', 'x13', 'x17']),
            (x0 == 'x11' and x1 in ['x8', 'x12', 'x13', 'x17']),
            (x0 == 'x15' and x1 in ['x12', 'x13', 'x17']),
            (x0 == 'x19' and x1 in ['x8', 'x12', 'x13', 'x21']),
            (x0 == 'x20' and x1 in ['x8', 'x13', 'x17', 'x21']),
            (x1 == 'x2' and x0 in ['x3', 'x4']),
            (x1 == 'x1' and x0 in ['x18', 'x19'])
        )
        #print(params, conditions)
        return not any(conditions)
    

    @staticmethod
    def known_constraint_disc_cont(params):
        x0 = float(params[0]) #continuous
        x1 = float(params[1]) #continuous
        x2 = float(params[2]) #discrete
        x3 = float(params[3]) #discrete

        if np.abs(x0-x1)<0.1:
            return False
        
        if 0.15 < x0 < 0.5:
            return False
        
        if np.abs(x2-x3)<0.1:
            return False
        
        if 0.15 < x2 < 0.5:
            return False
        
        return True


    @staticmethod
    def known_constraint_cat_disc(params):
        x0 = params[0] # categorical
        x1 = params[1] # categorical
        x2 = float(params[2]) # discrete
        x3 = float(params[3]) # discrete

        np.random.seed(100700)
        arr = np.random.randint(21, size=(208,2))
        for x in arr:
            if [x0, x1] == [f'x{x[0]}', f'x{1}']:
                return False

        if np.abs(x2-x3)<0.1:
            return False
        
        if 0.5 < x2 < 0.15:
            return False
        
        return True


    @staticmethod
    def known_constraint_cat_cont(params):
        x0 = params[0] # categorical
        x1 = params[1] # categorical
        x2 = float(params[2]) # continuous
        x3 = float(params[3]) # continuous

        np.random.seed(100700)
        arr = np.random.randint(21, size=(208,2))
        for x in arr:
            if [x0, x1] == [f'x{x[0]}', f'x{1}']:
                return False
        
        if np.abs(x2-x3)<0.1:
            return False
        
        if 0.15 < x2 < 0.5:
            return False
        
        return True


    @staticmethod
    def known_constraint_cat_disc_cont(params):
        x0 = params[0] # categorical
        x1 = params[1] # categorical
        x2 = float(params[2]) # discrete
        x3 = float(params[3]) # discrete
        x4 = float(params[4]) # continuous
        x5 = float(params[5]) # continuous
       
        np.random.seed(100700)
        arr = np.random.randint(21, size=(208,2))
        for x in arr:
            if [x0, x1] == [f'x{x[0]}', f'x{1}']:
                return False
        
        if np.abs(x2-x3)<0.1:
            return False
        
        if 0.15 < x2 < 0.5:
            return False
        
        if np.abs(x4-x5)<0.1:
            return False
        
        if 0.15 < x4 < 0.5:
            return False
        
        return True
    


class HybridSurface:
    def __init__(self, surfaces, is_moo=False):
        self.surfaces = surfaces 
        self.is_moo = is_moo
        self.param_space = ParameterSpace()
        counter = 0 
        for surface in self.surfaces:
            for param in surface.param_space:
                param.name = f'param_{counter}'
                self.param_space.add(param)

                counter+=1

    def run(self, params):
    
        if isinstance(params, ParameterVector):
            params = params.to_array()
        elif isinstance(params, np.ndarray):
            pass
        else:
            raise TypeError

        counter = 0
        objs = []
        for surface in self.surfaces:
            end_counter = counter+len(surface.param_space)
            relevant_params_arr  = params[counter:end_counter]
            if surface.param_space[0].type == 'categorical':
                pass
            else:
                relevant_params_arr = relevant_params_arr.astype(float)
            obj = surface.run(relevant_params_arr)
            objs.append(obj)

            counter = end_counter

        if self.is_moo:
            return np.array(objs)[:2]
        else:
            return np.sum(objs)



if __name__ == '__main__':

    problem_gen = ProblemGenerator(problem_type='continuous')
    problem_gen = ProblemGenerator(problem_type='discrete')
    problem_gen = ProblemGenerator(problem_type='categorical')
    surface_callable, param_space = problem_gen.generate_instance()








        
        