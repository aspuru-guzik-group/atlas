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

from atlas.planners.gp.planner import BoTorchPlanner

from problem_generator import ProblemGenerator


CAT = {
    'batch_size': [1, 5], 
    'use_descriptors': [False, True]
}

CAT_MOO = {
    'batch_size': [1],
    'use_descriptors': [False, True]
}

DISC = {
    'batch_size': [1],
}

CAT_MULT = {

}


def surface_cat_general_cont_func(x, s, surfaces):
    return surfaces[s].run(x)

def surface_cat_general_cat_func(x, s):
    if s == '0':
        return float(x[0]) + float(x[1])
    elif s == '1':
        return float(x[0]) + 2*float(x[1])
    elif s == '2':
        return -2*float(x[0]) - 0.5*float(x[1])
    
# def surface_cat_general_mixed_cat_cont_func(x, s):
#     if s == '0':
#         return np.random.uniform()

#     elif s == '1':
#         return np.random.uniform()

#     elif s == '2':
#         return np.random.uniform()
    

# def surface_cat_moo(x, s):
#     if s == '0':
#         return  np.array([
#                     np.sin(x[0])+ 12*np.cos(x[1]) - 0.1*x[2],
#                     3*np.sin(x[0])+ 0.01*np.cos(x[1]) + 1.*x[2]**2,
#                 ])
#     elif s == '1':
#         return  np.array([
#                     3*np.sin(x[0])+ 0.01*np.cos(x[1]) + 1.*x[2]**2,
#                     5*np.cos(x[0])+ 0.01*np.cos(x[1]) + 2.*x[2]**3,
#                 ])
#     elif s == '2':
#         return np.array([
#                     5*np.cos(x[0])+ 0.01*np.cos(x[1]) + 2.*x[2]**3,
#                     np.sin(x[0])+ 12*np.cos(x[1]) - 0.1*x[2],
#                 ])
    
# def surface_disc(x, s):
#     if s == 0.:
#         return  np.sin(x[0])+ 12*np.cos(x[1]) - 0.1*x[2]
#     elif s == 1.:
#         return 3*np.sin(x[0])+ 0.01*np.cos(x[1]) + 1.*x[2]**2
#     elif s == 2.:
#         return 5*np.cos(x[0])+ 0.01*np.cos(x[1]) + 2.*x[2]**3

# def surface_mult_cat(x, s):

#     return None
    



def run_cat_general_cont_func(batch_size, use_descriptors, acquisition_optimizer):
    """ single categorical general parameter
    """
    param_space = ParameterSpace()

    # general parameter
    param_space.add(
        ParameterCategorical(
            name='s',
            options=[str(i) for i in range(3)],
            descriptors=[[float(i),float(i)] for i in range(3)],   
        )
    )
    # functional parameters
    param_space.add(ParameterContinuous(name='param_0',low=0.,high=1.))
    param_space.add(ParameterContinuous(name='param_1',low=0.,high=1.))

    surfaces = {}
    problem_gen = ProblemGenerator(problem_type='continuous')
    for general_param_option in param_space[0].options:
        surface_callable, _ = problem_gen.generate_instance()
        surfaces[general_param_option] = surface_callable
        

    campaign = Campaign()
    campaign.set_param_space(param_space)

    planner = BoTorchPlanner(
        goal='minimize',
        init_design_strategy='random',
        num_init_design=5,
        batch_size=batch_size,
        use_descriptors=use_descriptors,
        acquisition_type='general',
        acquisition_optimizer_kind=acquisition_optimizer,
        general_parameters=[0],
        
    )
    planner.set_param_space(param_space)

    BUDGET = 5 + batch_size * 6
    true_measurements = []

    while len(campaign.observations.get_values()) < BUDGET:

        samples = planner.recommend(campaign.observations)
        for sample in samples:
            measurement = surface_cat_general_cont_func(
                [float(sample.param_0), float(sample.param_1)],
                sample.s,
                surfaces,
            )

            all_measurements = []
            for s in param_space[0].options:
                all_measurements.append(
                    surface_cat_general_cont_func(
                        [float(sample.param_0), float(sample.param_1)],
                        s,
                        surfaces,
                    )
                )
            true_measurements.append(np.mean(all_measurements))

            campaign.add_observation(sample, measurement)

    
    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET


def run_cat_general_cat_func(batch_size, use_descriptors, acquisition_optimizer):
    """ single categorical general parameter
    """
    param_space = ParameterSpace()

    # general parameter
    param_space.add(
        ParameterCategorical(
            name='s',
            options=[str(i) for i in range(3)],
            descriptors=[[float(i),float(i)] for i in range(3)],   
        )
    )
    # functional parameters
    param_space.add(
        ParameterCategorical(
            name='x_1',
            options=[str(i) for i in range(21)],
            descriptors=[[float(i),float(i)] for i in range(21)],   
        )
    )
    param_space.add(
        ParameterCategorical(
            name='x_2',
            options=[str(i) for i in range(21)],
            descriptors=[[float(i),float(i)] for i in range(21)],   
        )
    )

    campaign = Campaign()
    campaign.set_param_space(param_space)

    planner = BoTorchPlanner(
        goal='minimize',
        init_design_strategy='random',
        num_init_design=5,
        batch_size=batch_size,
        use_descriptors=use_descriptors,
        acquisition_type='general',
        acquisition_optimizer_kind=acquisition_optimizer,
        general_parameters=[0],
        
    )
    planner.set_param_space(param_space)

    BUDGET = 5 + batch_size * 6
    true_measurements = []

    while len(campaign.observations.get_values()) < BUDGET:

        samples = planner.recommend(campaign.observations)
        for sample in samples:
            measurement = surface_cat_general_cat_func([sample.x_1, sample.x_2], sample.s)

            print('SAMPLE : ', sample)
            print('MEASUREMENT : ', measurement)

            all_measurements = []
            for s in param_space[0].options:
                all_measurements.append(
                    surface_cat_general_cat_func([sample.x_1, sample.x_2], s)
                )
            true_measurements.append(np.mean(all_measurements))

            campaign.add_observation(sample, measurement)

    
    assert len(campaign.observations.get_params()) == BUDGET
    assert len(campaign.observations.get_values()) == BUDGET


# def run_cat_general_mixed_cat_cont_func(batch_size, use_descriptors, acquisition_optimizer):
#     """ single categorical general parameter
#     """
#     param_space = ParameterSpace()

#     # general parameter
#     param_space.add(
#         ParameterCategorical(
#             name='s',
#             options=[str(i) for i in range(3)],
#             descriptors=[[float(i),float(i)] for i in range(3)],   
#         )
#     )
#     # functional parameters
#     param_space.add(ParameterContinuous(name='x_1',low=0.,high=1.))
#     param_space.add(ParameterContinuous(name='x_2',low=0.,high=1.))
#     param_space.add(
#         ParameterCategorical(
#             name='x_3',
#             options=[str(i) for i in range(21)],
#             descriptors=[[float(i),float(i)] for i in range(21)],   
#         )
#     )

#     campaign = Campaign()
#     campaign.set_param_space(param_space)

#     planner = BoTorchPlanner(
#         goal='minimize',
#         init_design_strategy='random',
#         num_init_design=5,
#         batch_size=batch_size,
#         use_descriptors=use_descriptors,
#         acquisition_type='general',
#         acquisition_optimizer_kind=acquisition_optimizer,
#         general_parameters=[0],
        
#     )
#     planner.set_param_space(param_space)

#     BUDGET = 5 + batch_size * 6
#     true_measurements = []

#     while len(campaign.observations.get_values()) < BUDGET:

#         samples = planner.recommend(campaign.observations)
#         for sample in samples:
#             measurement = surface_cat_general_mixed_cat_cont_func([sample.x_1, sample.x_2, sample.x_3], sample.s)

#             print('SAMPLE : ', sample)
#             print('MEASUREMENT : ', measurement)

#             all_measurements = []
#             for s in param_space[0].options:
#                 all_measurements.append(
#                     surface_cat_general_mixed_cat_cont_func([sample.x_1, sample.x_2, sample.x_3], s)
#                 )
#             true_measurements.append(np.mean(all_measurements))

#             campaign.add_observation(sample, measurement)

    
#     assert len(campaign.observations.get_params()) == BUDGET
#     assert len(campaign.observations.get_values()) == BUDGET


# def run_general_disc(batch_size):
#     """ single discrete general parameter
#     """

#     param_space = ParameterSpace()

#     # general parameter
#     param_space.add(
#         ParameterDiscrete(
#             name='s',options=[float(i) for i in range(3)], 
#         )
#     )
#     # functional parameters
#     param_space.add(ParameterContinuous(name='x_1',low=0.,high=1.))
#     param_space.add(ParameterContinuous(name='x_2',low=0.,high=1.))
#     param_space.add(ParameterContinuous(name='x_3',low=0.,high=1.))


#     campaign = Campaign()
#     campaign.set_param_space(param_space)

#     planner = BoTorchPlanner(
#         goal='minimize',
#         init_design_strategy='random',
#         num_init_design=5,
#         batch_size=batch_size,
#         acquisition_optimizer_kind='genetic',
#         acquisition_type='general',
#         general_parmeters=[0],
        
#     )
#     planner.set_param_space(param_space)

#     BUDGET = 5 + batch_size * 4
#     true_measurements = []

#     while len(campaign.observations.get_values()) < BUDGET:

#         samples = planner.recommend(campaign.observations)
#         for sample in samples:
#             measurement = surface_disc(
#                 [float(sample.x_1), float(sample.x_2), float(sample.x_3)],
#                 float(sample.s),
#             )

#             all_measurements = []
#             for s in param_space[0].options:
#                 all_measurements.append(
#                     surface_disc(
#                         [float(sample.x_1), float(sample.x_2), float(sample.x_3)],
#                         float(s),
#                     )
#                 )
#             true_measurements.append(np.mean(all_measurements))

#             campaign.add_observation(sample, measurement)

    
#     assert len(campaign.observations.get_params()) == BUDGET
#     assert len(campaign.observations.get_values()) == BUDGET


# def run_general_cat_moo(batch_size, use_descriptors):
#     """ multiple categorical general parameters
#     """

#     param_space = ParameterSpace()

#     # general parameter
#     param_space.add(
#         ParameterCategorical(
#             name='s',
#             options=[str(i) for i in range(3)],
#             descriptors=[[i,i] for i in range(3)],   
#         )
#     )
#     # functional parameters
#     param_space.add(ParameterContinuous(name='x_1',low=0.,high=1.))
#     param_space.add(ParameterContinuous(name='x_2',low=0.,high=1.))
#     param_space.add(ParameterContinuous(name='x_3',low=0.,high=1.))
        
#     campaign = Campaign()
#     campaign.set_param_space(param_space)
    
#     value_space = ParameterSpace()
#     value_space.add(ParameterContinuous(name='obj1'))
#     value_space.add(ParameterContinuous(name='obj2'))

#     planner = BoTorchPlanner(
#         goal='minimize',
#         init_design_strategy='random',
#         num_init_design=5,
#         batch_size=1,
#         use_descriptors=use_descriptors,
#         acquisition_optimizer_kind='genetic',
#         acquisition_type='general',
#         general_parmeters=[0],
#         is_moo=True,
#         scalarizer_kind='Hypervolume', 
#         value_space=value_space,
#         goals=['min', 'min'],   
#     )
#     planner.set_param_space(param_space)

#     BUDGET = 5 + batch_size * 4

#     for iter_ in range(BUDGET):

#         samples = planner.recommend(campaign.observations)
#         for sample in samples:
#             measurement = surface_cat_moo(
#                 [float(sample.x_1), float(sample.x_2), float(sample.x_3)],
#                 sample.s,
#             )

#             # all_measurements = []
#             # for s in param_space[0].options:
#             #     all_measurements.append(
#             #         surface_cat_moo(
#             #             [float(sample.x_1), float(sample.x_2), float(sample.x_3)],
#             #             s,
#             #         )
#             #     )
            
#         campaign.add_observation(samples, measurement)
    

#     assert len(campaign.observations.get_params()) == BUDGET
#     assert len(campaign.observations.get_values()) == BUDGET



if __name__ == '__main__':

    run_cat_general_cont_func(batch_size=1, use_descriptors=False, acquisition_optimizer='gradient') 
    #run_cat_general_cat_func(batch_size=1, use_descriptors=False, acquisition_optimizer='gradient')  
    #run_cat_general_mixed_cat_cont_func(batch_size=1, use_descriptors=False, acquisition_optimizer='gradient')  
   

