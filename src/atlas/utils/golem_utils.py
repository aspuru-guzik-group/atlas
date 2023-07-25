#!/usr/bin/env python

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from golem import *
from olympus.campaigns import ParameterSpace

from atlas import Logger

supported_distributions = [
    "BoundedUniform",
    "FoldedNormal",
    "FrozenCategorical",
    "Normal",
    "Uniform",
    "TruncatedUniform",
    "FrozenNormal",
    "Gamma",
    "FrozenPoisson",
    "TruncatedNormal",
    "Poisson",
    "Categorical",
    "FrozenDiscreteLaplace",
    "FrozenGamma",
    "DiscreteLaplace",
    "FrozenUniform",
]


def get_golem_dists(
    golem_config: Dict[str, Any],
    param_space: ParameterSpace,
) -> Union[List[BaseDist], None]:
    golem_params = list(golem_config.keys())
    if len(golem_params) > len(param_space):
        msg = f"Too many parameters listed for golem (listed {len(golem_params)}, expected {len(param_space)}) "
        Logger.log(msg, "FATAL")

    distributions = []
    for param in param_space:
        if param.name in golem_params:
            if isinstance(golem_config[param.name], dict):
                dist_type = golem_config[param.name]["dist_type"]
                try:
                    dist_params = golem_config[param.name]["dist_params"]
                    if dist_params is None:
                        dist_params = {}
                except:
                    dist_params = {}

                distributions.append(
                    get_dist_from_type(dist_type, dist_params)
                )

            elif isinstance(golem_config[param.name], BaseDist):
                distributions.append(golem_config[param.name])
            else:
                msg = f"Golem config of type {type(golem_config[param.name])} for parameter {param.name} not understood."
                Logger.log(msg, "FATAL")

        else:
            msg = f"No distribution requested for parameter {param.name}. Resorting to Delta distribution..."
            Logger.log(msg, "WARNING")
            distributions.append(Delta())

    # special case where all the distributions are Delta, return None and
    # do not use Golem
    if all([isinstance(dist, Delta) for dist in distributions]):
        msg = "All parameters have Delta distributions. Will not use Golem for optimization."
        Logger.log(msg, "WARNING")
        distributions = None

    return distributions


def get_dist_from_type(dist_type: str, dist_params: Dict):
    module = import_module(".".join(("golem", dist_type)))
    return module(**dist_params)


def import_module(name):
    components = name.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == "__main__":
    from olympus.campaigns import Campaign, ParameterSpace
    from olympus.objects import ParameterContinuous, ParameterVector

    param_space = ParameterSpace()
    param_space.add(ParameterContinuous(name="param0"))
    param_space.add(ParameterContinuous(name="param1"))

    golem_config_1 = {
        "param0": {"dist_type": "Normal", "dist_params": {"std": 0.2}},
        "param1": {"dist_type": "Delta", "dist_params": None},
    }

    distributions = get_golem_dists(golem_config_1, param_space)
    print("case 1 : ", distributions)

    golem_config_2 = {
        "param0": {"dist_type": "Normal", "dist_params": {"std": 0.2}},
        # 'param1': {'dist_type': 'Delta', 'dist_params': None}
    }
    distributions = get_golem_dists(golem_config_2, param_space)
    print("case 2 : ", distributions)

    # check for:
    #   param being omitted completely --> replace with Delta in distribution
    #   dist_type added, but dist_params not added --> this is OK for Delta
    #   if all dist_types are Delta, do not use Golem at all, return None
    #   check for bad params? Or let Golem handle this? it should have a built in check
    #   check if the wrong parameter name is entered

    """
    distributions = [Normal(0.8), Normal(0.8)]
    beta = 0
    forest_type = 'dt'  # replaced by script
    ntrees = '50'         # replaced by script
    try:
        ntrees = int(ntrees)
    except:
        pass

    # parameter space
    domain = [
          {'name': 'x0', 'type': 'continuous', 'domain': (func.xlims[0], func.xlims[1])},
          {'name': 'x1', 'type': 'continuous', 'domain': (func.ylims[0], func.ylims[1])}
      ]

    # instantiate golem
    golem = Golem(
        forest_type=forest_type,
        ntrees=ntrees,
        goal='min',
        random_state=random_seed,
        verbose=True
)

    """
