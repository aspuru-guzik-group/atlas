#!/usr/bin/env python

import time
import timeit
from itertools import permutations, product
from math import exp

import numpy as np
from olympus.surfaces import CatCamel, CatDejong, CatMichalewicz, CatSlope
from rich.progress import track
from scipy.special import binom


def rbf_network_py(X, beta, theta):

    N = X.shape[0]
    D = X.shape[1]
    Y = np.zeros(N)

    for i in range(N):
        for j in range(N):
            r = 0
            for d in range(D):
                r += (X[j, d] - X[i, d]) ** 2
            r = r**0.5
            Y[i] += beta[j] * exp(-((r * theta) ** 2))

    return Y


def stirling_sum_py(Ns):
    """..."""
    stirling = lambda n, k: int(
        1.0
        / math.factorial(k)
        * np.sum([(-1.0) ** i * binom(k, i) * (k - i) ** n for i in range(k)])
    )
    return np.sum([stirling(Ns, k) for k in range(Ns + 1)])


def partition_py(S):
    """..."""
    if len(S) == 1:
        yield [S]
        return

    first = S[0]
    for smaller in partition_py(S[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n] + [[first] + subset] + smaller[n + 1 :]
        yield [[first]] + smaller


def gen_partitions_py(S):
    """
    generate all possible partitions of Ns-element set S

    Args:
        S (list): list of non-functional parameters S
    """
    return [p for _, p in enumerate(partition_py(S), 1)]


def gen_permutations_py(X_funcs, Ng):
    """generate all possible functional parameter permutations
    given number of non-functional parameter subsets Ng

    Args:
        X_funcs (np.ndarray): numpy array with all functional
            possile functional parameters
        Ng (int): number of non-functional parameter subsets

    Returns
        (np.ndarray): array of parameter permutations of
            shape (# perms, Ng, # params)
    """

    return np.array(list(permutations(X_funcs, Ng)))


def measure_objective_py(xgs, G, surf_map):
    """..."""

    f_x = 0.0
    for g_ix, Sg in enumerate(G):
        f_xg = 0.0
        for si in Sg:
            f_xg += measure_single_obj_py(xgs[g_ix], si, surf_map)
        f_x += f_xg / len(Sg)

    return f_x


def record_merits_py(S, surf_map, X_func_truncate=20):

    # list of dictionaries to store G, X_func, f_x
    f_xs = []

    start_time = time.time()

    # generate all the partitions of non-functional parameters
    Gs = gen_partitions_py(S)
    print("total non-functional partitions : ", len(Gs))

    # generate all the possible values of functional parametres
    # assume all surfs have same # options, params
    param_opts = [f"x{i}" for i in range(surf_map[0].num_opts)]
    params_opts = [param_opts for _ in range(surf_map[0].param_dim)]
    cart_product = list(product(*params_opts))
    X_funcs = np.array([list(elem) for elem in cart_product])

    if isinstance(X_func_truncate, int):
        X_funcs = X_funcs[:X_func_truncate, :]
    print("cardnality of functional params : ", X_funcs.shape[0])

    # Gs = Gs[::-1]
    np.random.seed(100700)
    np.random.shuffle(Gs)
    print(Gs)
    for G_ix, G in enumerate(Gs):
        Ng = len(G)
        # generate permutations of functional params
        X_func_perms = gen_permutations_py(X_funcs, Ng)
        print(G)
        print(X_func_perms)

        if G_ix % 1 == 0:
            print(
                f"[INFO] Evaluating partition {G_ix+1}/{len(Gs)+1} with len {Ng} ({len(X_func_perms)} perms)"
            )
        for X_func in track(
            X_func_perms, description="Evaluating permutations..."
        ):
            # measure objective
            print(X_func)
            quit()
            f_x = measure_objective_py(X_func, G, surf_map)
            # store values
            f_xs.append(
                {
                    "G": G,
                    "X_func": X_func,
                    "f_x": f_x,
                }
            )
    total_time = round(time.time() - start_time, 2)
    print(f"[INFO] Done in {total_time} s")

    return f_xs


def measure_single_obj_py(X_func, si, surf_map):
    return surf_map[si].run(X_func)[0][0]


if __name__ == "__main__":

    from test_cython import *

    S = [0, 1, 2]
    surf_map = {
        0: CatCamel(param_dim=2, num_opts=2),
        1: CatDejong(param_dim=2, num_opts=2),
        2: CatMichalewicz(param_dim=2, num_opts=2),
        3: CatSlope(param_dim=2, num_opts=2),
    }

    param_opts = [f"x{i}" for i in range(surf_map[0].num_opts)]
    params_opts = [param_opts for _ in range(surf_map[0].param_dim)]
    cart_product = list(product(*params_opts))
    X_funcs = np.array([list(elem) for elem in cart_product])

    # make all surface measurements
    surf_meas = {}
    for s in S:
        surf = surf_map[s]
        res = {}
        for X_func in X_funcs:
            meas = surf.run(X_func)[0][0]
            res[f"{X_func[0]}_{X_func[1]}"] = meas
        surf_meas[s] = res

    # print(surf_meas.keys())
    # meas = surf_meas[0]['x0_x0']
    # print(meas)
    # quit()

    parts = gen_partitions_py(S)
    print(parts)

    # meas = measure_single_obj(np.array(['x0', 'x1']), 0, surf_map)
    # print(meas)

    # cython
    # start_time = time.time()
    # f_xs = record_merits(S, surf_meas, X_func_truncate=None)
    # cython_time_ = time.time() - start_time
    # print('cython time (s) : ', round(cython_time_,5))

    # # python
    start_time = time.time()
    f_xs = record_merits_py(S, surf_map, X_func_truncate=None)
    python_time_ = time.time() - start_time
    print("python time (s) : ", round(python_time_, 5))

    # print('speedup : ', round(python_time_/cython_time_, 2))

    # print(len(f_xs))
