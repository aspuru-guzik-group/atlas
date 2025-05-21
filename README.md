# Atlas

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

[![aspuru-guzik-group](https://circleci.com/gh/aspuru-guzik-group/atlas.svg?style=shield)](https://app.circleci.com/pipelines/github/aspuru-guzik-group/atlas)
[![codecov](https://codecov.io/gh/aspuru-guzik-group/atlas/branch/main/graph/badge.svg?token=1Z8FA25WVO)](https://codecov.io/gh/aspuru-guzik-group/atlas)
![Docs](https://readthedocs.org/projects/matter-atlas/badge/?version=latest&style=flat-default)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


`atlas` is a Python package for Bayesian optimization in the experimental science. At its core, the package provides high-performing, easy-to-use Bayesian optimization based
on Gaussian processes (with help from the GPyTorch and BoTorch libraries). `atlas` attempts to cater directly to the needs of researchers in the experimental sciences,
and provides additional optimization capabilities to tackle problems typically encountered in such disciplines. These capabilities include optimization of categorical, discrete, and mixed parameter
spaces, multi-objective optimization, noisy optimization (noise on input parameters and objective measurements), constrained optimization (known and unknown constraints on the parameter space), multi-fidelity
optimization, meta-learning optimization, data-driven search space expansion/contraction, and more!

`atlas` is intended serve as the brain or experiment planner for self-driving laboratories.

You can find more details about `atlas` in our [documentation](https://matter-atlas.readthedocs.io/en/latest/). 

You can see the peer-reviewed publication in our [publication](https://pubs.rsc.org/en/content/articlelanding/2025/dd/d4dd00115j) at RSC Digital Discovery.


`atlas` is proudly developed in 🇨🇦 at the University of Toronto and the Vector Institute for Artificial Intelligence.


![alt text](https://github.com/aspuru-guzik-group/atlas/blob/main/static/atlas_logo.png)

## Installation

Currently, `atlas` does not work with Python>=3.11, and we recommend using 3.9 or 3.10. This will be fixed in later updates. You can install `atlas` from source in edit mode by executing the following commands

```bash
git clone git@github.com:aspuru-guzik-group/atlas.git
cd atlas
pip install -e .
pip install -r requirements.txt
```

If you require `olympus` resources, you can install it from source by doing
```bash
git clone -b olympus-atlas --single-branch git@github.com:aspuru-guzik-group/olympus.git
cd olympus 
pip install -e .
```

## Usage

This section gives minimal code examples showcasing some of the primary features of `atlas`.
You can also familiarize yourself with the package by checking out the following Google colab
notebook.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/aspuru-guzik-group/atlas/blob/main/atlas_get_started.ipynb)


Here we provide a minimal code example in which the `GPPlanner` from `atlas` is used to minimize the Branin-Hoo surface $f : \mathcal{X} \in \mathbb{R}^2 \mapsto \mathbb{R}$. "Ask-tell" experimentation proceeds iteratively by generating parameters to be measured using the planner's `recommend()` method, and informing the `olympus` `Campaign` instance about the corresponding measurement using its `add_observation()` method. We opt to use a flexible "ask-tell" interface in order to remain SDL application-agnostic. Measurement steps usually involve calls to specialized robotic laboratory equipment or computational simulation packages, which can be fully customized by the user.

```python
from olympus import Surface, Campaign
from atlas.planners.gp.planner import GPPlanner

surface = Surface(kind='Branin') # instantiate 2d Branin-Hoo objective function

campaign = Campaign() # define Olympus campaign object 
campaign.set_param_space(surface.param_space)

planner = GPPlanner(goal='minimize', num_init_design=5) # instantiate Atlas planner 
planner.set_param_space(surface.param_space)

while len(campaign.observations.get_values()) < 30:
    samples = planner.recommend(campaign.observations) # ask planner for batch of parameters 
    for sample in samples:
        measurement = surface.run(sample) # measure Branin-Hoo function
        campaign.add_observation(sample, measurement) # tell planner about most recent observation
```

![alt text](https://github.com/aspuru-guzik-group/atlas/blob/main/static/2d_branin_minimal_code.png)


## License

Distributed under the [MIT](https://choosealicense.com/licenses/mit/)
 license. See `LICENSE` for more information.

## Contact

Academic collaborations and extensions/improvements to the code by the community
are encouraged. Please reach out to [Riley](riley.hickman@mail.utoronto.ca) or [Gary](gary.tom@mail.mcgill.ca) by email if you have questions.

## Citation

`atlas` is an academic research software. If you use `atlas` in a scientific publication, please cite the following article.

```
@misc{hickman_atlas_2025,
	title={Atlas: a brain for self-driving laboratories},
        author={Hickman, Riley J and Sim, Malcolm and Pablo-Garc{\'\i}a, Sergio and Tom, Gary and Woolhouse, Ivan and Hao, Han and Bao, Zeqing and Bannigan, Pauric and Allen, Christine and Aldeghi, Matteo and Aspuru-Guzik, Al{\'a}n}},
        journal={Digital Discovery},
        volume={4},
        number={4},
        pages={1006--1029},
        year={2025},
        publisher={Royal Society of Chemistry}
}
```

`olympus` works hand-in-hand with `atlas`. We would be grateful for citations of the following publications as well. 

```
@article{hase_olympus_2021,
      author = {H{\"a}se, Florian and Aldeghi, Matteo and Hickman, Riley J. and Roch, Lo{\"\i}c M. and Christensen, Melodie and Liles, Elena and Hein, Jason E. and Aspuru-Guzik, Al{\'a}n},
      doi = {10.1088/2632-2153/abedc8},
      issn = {2632-2153},
      journal = {Machine Learning: Science and Technology},
      month = jul,
      number = {3},
      pages = {035021},
      title = {Olympus: a benchmarking framework for noisy optimization and experiment planning},
      volume = {2},
      year = {2021}
}

@misc{hickman_olympus_2023,
	author = {Hickman, Riley and Parakh, Priyansh and Cheng, Austin and Ai, Qianxiang and Schrier, Joshua and Aldeghi, Matteo and Aspuru-Guzik, Al{\'a}n},
	doi = {10.26434/chemrxiv-2023-74w8d},
	language = {en},
	month = may,
	publisher = {ChemRxiv},
	shorttitle = {Olympus, enhanced},
	title = {Olympus, enhanced: benchmarking mixed-parameter and multi-objective optimization in chemistry and materials science},
	urldate = {2023-06-21},
	year = {2023},
}
```
