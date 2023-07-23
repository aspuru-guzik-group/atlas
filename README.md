# Atlas


[![rileyhickman](https://circleci.com/gh/rileyhickman/atlas.svg?style=svg&circle-token=96039a8d33f9fade7e4c1a5420312b0711b16cde)](https://app.circleci.com/pipelines/github/rileyhickman/atlas)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


`atlas` is a Python package for Bayesian optimization in the experimental science. At its core, the package provides high-performing, easy-to-use Bayesian optimization based
on Gaussian processes (with help from the GPyTorch and BoTorch libraries). `atlas` attempts to cater directly to the needs of researchers in the experimental sciences,
and provides additional optimization capabilities to tackle problems typically encountered in such disciplines. These capabilities include optimization of categorical, discrete, and mixed parameter
spaces, multi-objective optimization, noisy optimization (noise on input parameters and objective measurements), constrained optimization (known and unknown constraints on the parameter space), multi-fidelity
optimization, meta-learning optimization, data-driven search space expansion/contraction, and more!

`atlas` is intended serve as the brain or experiment planner for self-driving laboratories.


`atlas` is proudly developed in 🇨🇦 at the University of Toronto and the Vector Institute for Artificial Intelligence.


![alt text](https://github.com/rileyhickman/atlas/blob/main/static/atlas_logo.png)

## Installation

Install `atlas` from source by executing the following commands

```bash
git clone git@github.com:rileyhickman/atlas.git
cd atlas
pip install -e .
```

To use the Google doc feature, you must install the Google client library

```
pip install --upgrade google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

and `gspread`

```
pip install gspread
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

![alt text](https://github.com/rileyhickman/atlas/blob/main/static/2d_branin_minimal_code.png)


## License

Distributed under the [MIT](https://choosealicense.com/licenses/mit/)
 license. See `LICENSE` for more information.

## Contact

Academic collaborations and extensions/improvements to the code by the community
are encouraged. Please reach out to [Riley](riley.hickman@mail.utoronto.ca) by email if you have questions.

## Citation

`atlas` is an academic research software. If you use `atlas` in a scientific publication, please cite the following article.

```
@misc{hickman_atlas_2023,

}
```
