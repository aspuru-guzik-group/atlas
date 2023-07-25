#!/usr/bin/env python

import GPy
import matplotlib.pyplot as plt
import numpy as np
import olympus
import seaborn as sns
import sobol_seq
import torch
from olympus.surfaces import Surface
from olympus.surfaces.surface_cat_ackley import CatAckley
from olympus.surfaces.surface_cat_camel import CatCamel
from olympus.surfaces.surface_cat_dejong import CatDejong
from olympus.surfaces.surface_cat_michalewicz import CatMichalewicz
from scipy.stats import norm, pearsonr, spearmanr

torch.set_default_dtype(torch.double)

OLYMP_SURFACES = olympus.surfaces.get_surfaces_list()

ALL_SYNTHETIC = ["trig", "gp", "olympus", "bra", "gprice", "hm3"]


def list_all_synthetic_data():
    print(ALL_SYNTHETIC)


def trig_factory(
    amplitude_range=[0.8, 1.2],
    shift_range=[-0.1, 0.1],
    scale_range=[[7, 9], [7, 9]],
    num_samples=100,
    as_numpy=False,
):
    """generate train tasks using sine and cosine functions
    in 1d on the domain [-pi, pi]
    """
    train_tasks = []
    domain = torch.linspace(0, 1, 100)
    for i in range(num_samples):
        # select random region
        scale_range_ix = np.random.choice([0, 1])
        sr = scale_range[scale_range_ix]
        amplitude = (amplitude_range[0] - amplitude_range[1]) * torch.rand(
            1
        ) + amplitude_range[1]
        shift = (shift_range[0] - shift_range[1]) * torch.rand(
            1
        ) + shift_range[1]
        scale = (sr[0] - sr[1]) * torch.rand(1) + sr[1]
        values = amplitude * (torch.sin(scale * (domain + shift)))
        if as_numpy:
            train_tasks.append(
                {
                    "params": domain.detach().numpy().reshape(-1, 1),
                    "values": values.detach().numpy().reshape(-1, 1),
                }
            )
        else:
            train_tasks.append(
                {
                    "params": domain.reshape(-1, 1),
                    "values": values.reshape(-1, 1),
                }
            )
    return train_tasks


def gp_factory(
    param_dim=1,
    kernel="rbf",
    noise_var_range=[0.01, 1.0],
    length_scale_range=[0.05, 0.1],
    num_samples=100,
    resolution=1000,
    plot=False,
):
    """generate num_samples GP prior samples using specified kernel, length scale,
    noise variance and domain
    These are always generated on the unit hypercube, i.e. [0, 1]^{param_dim}
    """
    feat = sobol_seq.i4_sobol_generate(param_dim, resolution)

    tasks = []
    kernel_fn = gp_kernel(kind=kernel)
    k = kernel_fn(
        input_dim=param_dim,
        lengthscale=np.random.uniform(
            length_scale_range[0], length_scale_range[1]
        ),
        variance=np.random.uniform(noise_var_range[0], noise_var_range[1]),
    )
    mu = np.zeros((feat.shape[0]))
    cov = k.K(feat, feat)
    # generate the samples from the prior
    Z = np.random.multivariate_normal(mu, cov, num_samples)
    for z in Z:
        # print(Z.shape)
        # print(z.shape)
        tasks.append(
            {
                "params": torch.from_numpy(feat),
                # "values": torch.from_numpy(z.reshape(-1, param_dim)),
                "values": torch.from_numpy(z.reshape(-1, 1)),
            }
        )

    if plot:
        fig = plt.figure(figsize=(8, 4))
        for i in range(num_samples):
            plt.plot(tasks[i]["params"], tasks[i]["values"])

        plt.tight_layout()
        plt.show()

    return tasks


def gp_kernel(kind="rbf"):
    if kind == "rbf":
        kernel_fn = GPy.kern.RBF
    elif kind == "matern32":
        kernel_fn = GPy.kern.Matern32
    else:
        raise NotImplementedError

    return kernel_fn


# def olymp_factory(
#     param_dim,  # dimension of the parameter space
#     surface_kind,       # name of the olympus surface
#     num_train, # number of training point for the fitted model
#     num_samples,  # the number of meta datasets to return for each case
#     corr_metric='spearman',  # pearson or spearman
#     corr_red='mean', # mean, max, min
#     num_sobol=1000,  # number of initial sobol points
#     fit_models=['gp_rbf', 'gp_matern32'],
#     seed=100700,
# ):
#     ''' Generate perturbed Olympus surfaces where the oracle is the
#     ground truth surface
#     '''
#     # set random seed
#     np.random.seed(seed)

#     # register olympus surface
#     surf = Surface(kind=surface_kind, param_dim=param_dim)

#     # generate initial point with Sobol sequence
#     domain = sobol_seq.i4_sobol_generate(param_dim, num_sobol)
#     y_oracle = np.array(surf.run(domain))

#     # sample a training set from the oracle points
#     indices = np.arange(y_oracle.shape[0])
#     np.random.shuffle(indices)

#     indices = indices[:num_train]

#     train_params = domain[indices, :]
#     train_values = domain[indices, :]

#     if num_samples % 2 == 0:
#         num_rbf, num_mat32 = num_samples//2, num_samples//2
#     elif num_samples %2 == 1:
#         num_rbf, num_mat32 = num_samples//2, num_samples//2 + 1

#     # train a GP with RBF
#     kernel=GPy.kern.RBF(input_dim=train_params.shape[0], variance=3.0, lengthscale=0.1)
#     model = GPy.models.GPRegression(train_params, train_values)
#     model.optimize_restarts(num_restarts=5)

#     post_rbf_samples = model.posterior_samples_f(
#             domain, full_cov=True, size=num_rbf,
#         ).reshape((
#             num_rbf, domain.shape[0], param_dim
#         ))                                           # [num_samples, domain.shape[0], param_dim]

#     # all_post_samples = np.concatenate((
#     #     post_rbf_samples, post_mat32_samples
#     # ))
#     all_post_samples = post_rbf_samples

#     tasks = []
#     corrs = []
#     for ix, sample in enumerate(all_post_samples):
#         # compute the surface value
#         z = np.array(surf.run(sample)).reshape(-1, 1)
#         if corr_metric == 'pearson':
#             p = pearsonr(y_oracle.ravel(), z.ravel())
#             corrs.append(p)
#         elif corr_metric == 'spearman':
#             s = spearmanr(y_oracle.ravel(), z.ravel())
#             corrs.append(s)

#         tasks.append({'params': sample, 'values': z})

#     corr_stats = {
#         'mean': np.mean(corrs),
#         'std': np.std(corrs),
#         'min': np.amin(corrs),
#         'max': np.amax(corrs),
#     }

#     return tasks, corr_stats


def olymp_factory_cat(
    param_dim,
    surface_kind,
    num_opts,
    noise_level,
    descriptors=False,  # whether or not to use descriptors
):
    """Generate "noisy" Olympus categorical surfaces"""
    # define original surface
    if surface_kind == "CatDejong":
        surface = CatDejong(param_dim=param_dim, num_opts=num_opts)
    elif surface_kind == "CatAckley":
        surface = CatAckley(param_dim=param_dim, num_opts=num_opts)
    elif surface_kind == "CatMichalewicz":
        surface = CatMichalewicz(param_dim=param_dim, num_opts=num_opts)
    else:
        raise NotImplementedError

    domain = [f"x{element}" for element in np.arange(num_opts)]
    # meshgrid for plotting
    X, Y = np.meshgrid(domain, domain)
    # make 2d for params
    params = [list(element) for element in np.dstack([X, Y]).reshape(-1, 2)]
    Z = np.zeros((num_opts, num_opts))
    values = []
    for y_index, y in enumerate(domain):
        for x_index, x in enumerate(domain):
            scal_val = surface.run(np.array([x, y]))[0][0]

            Z[x_index, y_index] = scal_val
            values.append(scal_val)

    # add some noise, if needed
    max_ = np.amax(values)
    min_ = np.amin(values)

    values_noise = []
    Z_noise = np.zeros((num_opts, num_opts))

    for y_index, y in enumerate(domain):
        for x_index, x in enumerate(domain):
            scal_val = surface.run(np.array([x, y]))[0][0]

            upper_ = max_ - scal_val
            lower_ = min_ - scal_val

            #             sample = truncnorm.rvs(lower_, upper_, loc=0., scale=noise_level, size=None)
            is_sat = False
            while not is_sat:
                sample = norm.rvs(loc=0, scale=noise_level, size=None)
                if lower_ <= sample <= upper_:
                    is_sat = True
                else:
                    pass

            Z_noise[x_index, y_index] = scal_val + sample
            values_noise.append(scal_val + sample)

    values = np.array(values_noise).reshape(-1, 1)
    Z = Z_noise

    if descriptors:
        # use descriptors - the number of the
        # params_ shape = (num_opts**param_dim, param_dim)
        params_ = np.zeros((num_opts**param_dim, param_dim))
        for param_ix, param in enumerate(params):
            descs = []
            for element in param:
                desc = int(element[1:])
                descs.append(desc)
            params_[param_ix] = np.array(descs)
    else:
        # params shape = (num_opts**param_dim, num_opts*param_dim)
        params_ = np.zeros((num_opts**param_dim, num_opts * param_dim))
        for param_ix, param in enumerate(params):
            one_hots = []
            for element in param:
                one_hot = np.zeros(num_opts)
                ix = int(element[1:])
                one_hot[ix] = 1.0
                one_hots.extend(one_hot)
            params_[param_ix] = np.array(one_hots)

    # return domain and Z for plotting purposes
    return params_, values, domain, Z


# BRANIN =======================================================================
def bra(x):
    # the Branin function (2D)
    # https://www.sfu.ca/~ssurjano/branin.html
    x1 = x[:, 0]
    x2 = x[:, 1]
    # scale x
    x1 = x1 * 15.0
    x1 = x1 - 5.0
    x2 = x2 * 15.0
    # parameters
    a = 1
    b = 5.1 / (4 * np.pi**2)
    c = 5 / np.pi
    r = 6
    s = 10
    t = 1 / (8 * np.pi)
    bra = (
        a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
    )
    # normalize
    mean = 54.44
    std = 51.44
    bra = 1 / std * (bra - mean)
    # maximize
    # bra = -bra
    return bra.reshape(x.shape[0], 1)


def bra_var(x, t, s):
    x_new = x.copy()
    # apply translation
    # bound the translations s.t. upper left max lies in domain
    t_range = np.array([[-0.12, 0.87], [-0.81, 0.18]])
    t = np.clip(t, t_range[:, 0], t_range[:, 1])
    x_new = x_new - t
    return s * bra(x_new)


def bra_max_min():
    max_pos = np.array([[-np.pi, 12.275]])
    max_pos[0, 0] += 5.0
    max_pos[0, 0] /= 15.0
    max_pos[0, 1] /= 15.0
    max = bra(max_pos)
    min_pos = np.array([[0.0, 0.0]])
    min = bra(min_pos)
    return max_pos, max, min_pos, min


def bra_max_min_var(t, s):
    max_pos, max, min_pos, min = bra_max_min()
    # apply translation
    # clip the translations s.t. upper left max lies in domain
    t_range = np.array([[-0.12, 0.87], [-0.81, 0.18]])
    t = np.clip(t, t_range[:, 0], t_range[:, 1])
    max_pos = max_pos + t
    min_pos = min_pos + t
    return max_pos, s * max, min_pos, s * min


# GOLDSTEIN-PRICE ==============================================================
def gprice(x):
    # the goldstein price function (2D)
    # https://www.sfu.ca/~ssurjano/goldpr.html
    x1 = x[:, 0]
    x2 = x[:, 1]
    # scale x
    x1 = x1 * 4.0
    x1 = x1 - 2.0
    x2 = x2 * 4.0
    x2 = x2 - 2.0
    gprice = (
        1
        + (x1 + x2 + 1) ** 2
        * (19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2)
    ) * (
        30
        + (2 * x1 - 3 * x2) ** 2
        * (18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2)
    )
    # lognormalize
    mean = 8.693
    std = 2.427
    gprice = 1 / std * (np.log(gprice) - mean)
    # maximize
    # gprice = -gprice
    return gprice.reshape(x.shape[0], 1)


def gprice_max_min():
    max_pos = np.array([[0.0, -1.0]])
    max_pos[0, 0] += 2.0
    max_pos[0, 0] /= 4.0
    max_pos[0, 1] += 2.0
    max_pos[0, 1] /= 4.0
    max = gprice(max_pos)
    min_pos = np.array([[0.066, 1.0]])
    min = gprice(min_pos)
    return max_pos, max, min_pos, min


def gprice_var(x, t, s):
    x_new = x.copy()
    # apply translation
    # clip the translations s.t. upper left max lies in domain
    t_range = np.array([[-0.5, 0.5], [-0.25, 0.75]])
    t = np.clip(t, t_range[:, 0], t_range[:, 1])
    x_new = x_new - t
    return s * gprice(x_new)


def gprice_max_min_var(t, s):
    # do the transformation in opposite order as in hm3_var!
    max_pos, max, min_pos, min = gprice_max_min()
    # apply translation
    t_range = np.array([[-0.5, 0.5], [-0.25, 0.75]])
    t = np.clip(t, t_range[:, 0], t_range[:, 1])
    max_pos = max_pos + t
    min_pos = min_pos + t
    return max_pos, s * max, min_pos, s * min


# HARTMAN-3 =========================================================
def hm3(x):
    # the hartmann3 function (3D)
    # https://www.sfu.ca/~ssurjano/hart3.html
    # parameters
    alpha = np.array([1.0, 1.2, 3.0, 3.2])
    A = np.array([[3.0, 10, 30], [0.1, 10, 35], [3.0, 10, 30], [0.1, 10, 35]])
    P = 1e-4 * np.array(
        [
            [3689, 1170, 2673],
            [4699, 4387, 7470],
            [1091, 8732, 5547],
            [381, 5743, 8828],
        ]
    )
    x = x.reshape(x.shape[0], 1, -1)
    B = x - P
    B = B**2
    exponent = A * B
    exponent = np.einsum("ijk->ij", exponent)
    C = np.exp(-exponent)
    hm3 = -np.einsum("i, ki", alpha, C)
    # normalize
    mean = -0.93
    std = 0.95
    hm3 = 1 / std * (hm3 - mean)
    # maximize
    # hm3 = -hm3
    return hm3.reshape(x.shape[0], 1)


def hm3_max_min():
    max_pos = np.array([[0.114614, 0.555649, 0.852547]])
    max = hm3(max_pos)
    min_pos = np.array([[1.0, 1.0, 0.0]])
    min = hm3(min_pos)
    return max_pos, max, min_pos, min


def hm3_var(x, t, s):
    x_new = x.copy()
    # apply translation
    # clip the translations s.t. upper left max lies in domain
    t_range = np.array([[-0.11, 0.88], [-0.55, 0.44], [-0.85, 0.14]])
    t = np.clip(t, t_range[:, 0], t_range[:, 1])
    x_new = x_new - t
    return s * hm3(x_new)


def hm3_max_min_var(t, s):
    # do the transformation in opposite order as in hm3_var!
    max_pos, max, min_pos, min = hm3_max_min()
    # apply translation
    t_range = np.array([[-0.11, 0.88], [-0.55, 0.44], [-0.85, 0.14]])
    t = np.clip(t, t_range[:, 0], t_range[:, 1])
    max_pos = max_pos + t
    min_pos = min_pos + t
    return max_pos, s * max, min_pos, s * min


def metaBO_factory(
    num_samples,
    kind,
    t_range=[-0.1, 0.1],
    s_range=[0.9, 1.1],
    num_sobol=200,
):
    """sample perturbed funcs in metaBO paper"""
    if kind in ["bra", "gprice"]:
        param_dim = 2
    elif kind == ["hm3"]:
        param_dim = 3
    domain = sobol_seq.i4_sobol_generate(param_dim, num_sobol)
    tasks = []
    for sample_ix in range(num_samples):
        t = np.random.uniform(t_range[0], t_range[1])
        s = np.random.uniform(s_range[0], s_range[1])
        if kind == "bra":
            z = bra_var(domain, t, s)
        elif kind == "gprice":
            z = gprice_var(domain, t, s)
        elif kind == "hm3":
            z = hm3(domain, t, s)

        tasks.append({"params": domain, "values": z.reshape(-1, 1)})

    return tasks


def olymp_cat_source_task_gen(
    num_train_tasks, num_valid_tasks, num_opts, use_descriptors=False
):
    train_tasks = []
    for task in range(num_train_tasks):
        surf_name = str(
            np.random.choice(
                ["CatDejong", "CatMichalewicz", "CatAckley"], size=None
            )
        )
        params_, values, _, __ = olymp_factory_cat(
            param_dim=2,
            surface_kind=surf_name,
            num_opts=num_opts,
            noise_level=0.2,
            descriptors=use_descriptors,  # whether or not to use descriptors
        )
        train_tasks.append({"params": params_, "values": values})

    valid_tasks = []
    for task in range(num_valid_tasks):
        surf_name = str(
            np.random.choice(
                ["CatDejong", "CatMichalewicz", "CatAckley"], size=None
            )
        )
        params_, values, _, __ = olymp_factory_cat(
            param_dim=2,
            surface_kind=surf_name,
            num_opts=num_opts,
            noise_level=0.2,
            descriptors=use_descriptors,  # whether or not to use descriptors
        )
        valid_tasks.append({"params": params_, "values": values})

    return train_tasks, valid_tasks


def mixed_source_code(problem_type):
    all_train_tasks = []
    all_valid_tasks = []
    split_ = problem_type.split("_")

    if "cat" in split_:
        train_tasks_cat, valid_tasks_cat = olymp_cat_source_task_gen(
            num_train_tasks=10,
            num_valid_tasks=5,
            num_opts=5,
            use_descriptors=False,
        )

        all_train_tasks.append(train_tasks_cat)
        all_valid_tasks.append(valid_tasks_cat)

    if "disc" in split_:
        train_tasks_disc = gp_factory(
            param_dim=2,
            kernel="rbf",
            noise_var_range=[0.01, 1.0],
            length_scale_range=[0.05, 0.1],
            num_samples=10,
            resolution=25,
            plot=False,
        )
        valid_tasks_disc = train_tasks_disc[:5]

        all_train_tasks.append(train_tasks_disc)
        all_valid_tasks.append(valid_tasks_disc)

    if "cont" in split_:
        train_tasks_cont = gp_factory(
            param_dim=2,
            kernel="rbf",
            noise_var_range=[0.01, 1.0],
            length_scale_range=[0.05, 0.1],
            num_samples=10,
            resolution=25,
            plot=False,
        )
        valid_tasks_cont = train_tasks_cont[:5]

        all_train_tasks.append(train_tasks_cont)
        all_valid_tasks.append(valid_tasks_cont)

    train_tasks = []
    for tasks in zip(*all_train_tasks):
        params = torch.cat(
            [torch.tensor(task["params"]) for task in tasks], dim=1
        )
        values = torch.sum(
            torch.cat([torch.tensor(task["values"]) for task in tasks], dim=1),
            dim=1,
        )
        train_tasks.append(
            {"params": params, "values": values.view(values.shape[0], 1)}
        )

    valid_tasks = []
    for tasks in zip(*all_valid_tasks):
        params = torch.cat(
            [torch.tensor(task["params"]) for task in tasks], dim=1
        )
        values = torch.sum(
            torch.cat([torch.tensor(task["values"]) for task in tasks], dim=1),
            dim=1,
        )
        valid_tasks.append(
            {"params": params, "values": values.view(values.shape[0], 1)}
        )

    return train_tasks, valid_tasks


if __name__ == "__main__":
    # tasks = gp_factory(param_dim=2, kernel='rbf', plot=False)
    # print(len(tasks))
    # print(OLYMP_SURFACES)
    # tasks, corr_stats = olymp_factory(
    #     param_dim=2, surface_kind="Everest", num_train=500, num_samples=20
    # )

    # print(len(tasks))
    # print(tasks[0]["params"].shape)
    # print(tasks[0]["values"].shape)
    # print(corr_stats)

    # params_, values, domain, Z = olymp_factory_cat(
    #     param_dim=2,
    #     surface_kind='CatDejong',
    #     num_opts=21,
    #     noise_level=0.1,
    #     descriptors=False,  # whether or not to use descriptors
    # )

    # print(params_.shape)
    # print(values.shape)

    train_tasks_cat, valid_tasks_cat = olymp_cat_source_task_gen(
        num_train_tasks=20,
        num_valid_tasks=5,
        use_descriptors=False,
    )

    # print(len(train_tasks))
    # print(len(valid_tasks))

    train_tasks_cont = gp_factory(
        param_dim=2,
        kernel="rbf",
        noise_var_range=[0.01, 1.0],
        length_scale_range=[0.05, 0.1],
        num_samples=20,
        resolution=441,
        plot=False,
    )
    valid_tasks_cont = gp_factory(
        param_dim=2,
        kernel="rbf",
        noise_var_range=[0.01, 1.0],
        length_scale_range=[0.05, 0.1],
        num_samples=5,
        resolution=441,
        plot=False,
    )

    train_tasks = []

    for task_cat, task_cont in zip(train_tasks_cat, train_tasks_cont):
        params = torch.cat(
            (torch.tensor(task_cat["params"]), task_cont["params"]), dim=1
        )
        values = torch.sum(
            torch.cat(
                (torch.tensor(task_cat["values"]), task_cont["values"]), dim=1
            ),
            dim=1,
        )

        train_tasks.append(
            {"params": params, "values": values.view(values.shape[0], 1)}
        )

    valid_tasks = []

    for task_cat, task_cont in zip(valid_tasks_cat, valid_tasks_cont):
        params = torch.cat(
            (torch.tensor(task_cat["params"]), task_cont["params"]), dim=1
        )
        values = torch.sum(
            torch.cat(
                (torch.tensor(task_cat["values"]), task_cont["values"]), dim=1
            ),
            dim=1,
        )

        valid_tasks.append(
            {"params": params, "values": values.view(values.shape[0], 1)}
        )

    print(valid_tasks[0]["params"].shape)
    print(valid_tasks[0]["values"].shape)

    train_tasks_cont = trig_factory(
        param_dim=2,
        kernel="rbf",
        noise_var_range=[0.01, 1.0],
        length_scale_range=[0.05, 0.1],
        num_samples=20,
        resolution=441,
        plot=False,
    )
    valid_tasks_cont = trig_factory(
        param_dim=2,
        kernel="rbf",
        noise_var_range=[0.01, 1.0],
        length_scale_range=[0.05, 0.1],
        num_samples=5,
        resolution=441,
        plot=False,
    )

    train_tasks = []

    for task_cont in zip(train_tasks_cont):
        params = torch.cat(
            (torch.tensor(task_cat["params"]), task_cont["params"]), dim=1
        )
        values = torch.sum(
            torch.cat(
                (torch.tensor(task_cat["values"]), task_cont["values"]), dim=1
            )
        )

        train_tasks.append(
            {"params": params, "values": values.view(values.shape[0], 1)}
        )

    valid_tasks = []

    for task_cont in zip(valid_tasks_cont):
        params = torch.cat(torch.tensor(task_cont["params"]), dim=1)
        values = torch.sum(
            torch.cat((torch.tensor(task_cont["values"])), dim=1)
        )

        valid_tasks.append(
            {"params": params, "values": values.view(values.shape[0], 1)}
        )

    print(valid_tasks[0]["params"].shape)
    print(valid_tasks[0]["values"].shape)
