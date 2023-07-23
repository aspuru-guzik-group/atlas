# atlas


[![rileyhickman](https://circleci.com/gh/rileyhickman/atlas.svg?style=svg&circle-token=96039a8d33f9fade7e4c1a5420312b0711b16cde)](https://app.circleci.com/pipelines/github/rileyhickman/atlas)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


`atlas` is a Python package for Bayesian optimization in the experimental science. At its core, the package provides high-performing, easy-to-use Bayesian optimization based
on Gaussian processes (with help from the GPyTorch and BoTorch libraries). `atlas` attempts to cater directly to the needs of researchers in the experimental sciences,
and provides additional optimization capabilities to tackle problems typically encountered in such disciplines. These capabilities include optimization of categorical, discrete, and mixed parameter
spaces, multi-objective optimization, noisy optimization (noise on input parameters and objective measurements), constrained optimization (known and unknown constraints on the parameter space), multi-fidelity
optimization, meta-learning optimization, data-driven search space expansion/contraction, and more!

`atlas` is intended serve as the brain or experiment planner for self-driving laboratories.


`atlas` is proudly developed in :ca: at the University of Toronto and the Vector Institute for Artificial Intelligence.


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

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rileyhickman/atlas/blob/main/atlas_get_started.ipynb)



### Proof-of-concept optimization




### Optimization of mixed-parameter spaces


### Optimization with a-priori known constraints


### Optimization with a-priori unknown constraints


### Multi-objective optimization


### Robust optimization with Golem


### Optimization for a generalizable set of parameters


Often, researchers may like to find parameters that are _generalizable_.
For example, one might want to find a single set of chemical reaction conditions which give good yield across several different substrates. [cite MADNESS Science paper]

Consider an optimization problem with $d$ continuous reaction parameters, $\mathcal{X} \in \mathbb{R}^d$
(functional parameters), and a set of $n$ substrates $\mathcal{S} = { s_i }_{i=1}^n$ (non-functional
parameters). The goal of such an optimization is to maximize the objective function $f(\mathbf{x})$, which is
the average response across all molecules,

$$ f_{\mathcal{C}} = \frac{1}{n} \sum_{i=1}^n f(\mathbb{x}, s_i)  . $$

For a minimization problem, the best performing parameters are

$$  \mathbf{x}^* = argmin_{\mathbf{x}\in \mathcal{X}, s_i \in \mathcal{C}} f_{\mathcal{C}}  .$$

`atlas` employs an approach which removes the need to measure $f_{\mathcal{C}}$ at each iteration. Consider a toy problem,
where $n=3$, and the following piecewise function is used for $f_{\mathcal{C}}$, and is to be minimized.

$$ f(\mathbf{x}, s) = \sin(x_1) + 12\cos(x_2) - 0.1x_3   \text{  if}  s = s_1$$

$$ f(\mathbf{x}, s) = 3\sin(x_1) + 0.01\cos(x_2) + x_3^2  \text{  if }  s = s_2$$

$$ f(\mathbf{x}, s) = 5\cos(x_1) + 0.01\cos(x_2) + 2x_3^3  \text{  if } s = s_3$$


The variable $s$ is a categorical parameter with 3 options. $f_{\mathcal{C}}$ has a minimum value of approximately
3.830719 at $\mathbf{x}^* = (0.0, 1.0, 0.0404)$. Given the appropriate `olympus` parameter space, one can instantiate
a planner as follows.

```python

param_space = ParameterSpace()

# add general parameter, one-hot-encoded
param_space.add(
    ParameterCategorical(
        name='s',
        options=[str(i) for i in range(3)],
        descriptors=[None for i in range(3)],       
    )
)
param_space.add(
    ParameterContinuous(name='x_1',low=0.,high=1.)
)
param_space.add(
    ParameterContinuous(name='x_2',low=0.,high=1.)
)
param_space.add(
    ParameterContinuous(name='x_3',low=0.,high=1.)
)

planner  = BoTorchPlanner(
    goal='minimize',
    batch_size=1,
    num_init_design=5,
    general_parameters=[0] # indices of general parameters
)

planner.set_param_space(param_space)

```

The `general_parameters` argument to the constructor takes a list of integers, which
represent the parameter space indices which are intended to be treated as _general_ or _non functional_
parameters. The figure below shows the performance of `atlas` compared to random sampling on this toy
problem (10 repeats).

![alt text](https://github.com/rileyhickman/atlas/blob/main/static/synthetic_general_conditions_gradient.png)


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
