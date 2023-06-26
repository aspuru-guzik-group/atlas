from math import exp, factorial

import numpy as np

cimport numpy as np

from scipy.special import binom

from libc.stdio cimport printf

import time
from itertools import permutations, product

import pandas as pd


def rbf_network(double[:, :] X,  double[:] beta, double theta):

    cdef int N = X.shape[0]
    cdef int D = X.shape[1]
    cdef double[:] Y = np.zeros(N)
    cdef int i, j, d
    cdef double r = 0

    for i in range(N):
        for j in range(N):
            r = 0
            for d in range(D):
                r += (X[j, d] - X[i, d]) ** 2
            r = r**0.5
            Y[i] += beta[j] * exp(-(r * theta)**2)

    return Y


def stirling_sum(int Ns):
    """ ...
    """
    cdef int n, k
    stirling = lambda n,k: int(1./factorial(k) * np.sum([(-1.)**i * binom(k,i)*(k-i)**n for i in range(k)]))
    return np.sum([stirling(Ns, k) for k in range(Ns+1)])

def partition(list S):
    if len(S) == 1:
        yield [S]
        return 
    cdef int first = S[0]
    for smaller in partition(S[1:]):
        for n, subset in enumerate(smaller):
            yield smaller[:n]+[[first] + subset]+smaller[n+1:]
        yield [[first]]+smaller 


def gen_partitions(list S):
    return [p for _, p in enumerate(partition(S))]


def measure_objective(np.ndarray xgs, list G, surf_meas):

    cdef double f_x = 0.
    cdef double f_xg
    cdef int g_ix, s_ix
    cdef list Sg

    for g_ix in range(len(G)):
        f_xg = 0.
        Sg = G[g_ix]
        for s_ix in range(len(Sg)):
            f_xg += measure_single_obj(xgs[g_ix], Sg[s_ix], surf_meas)
        f_x += f_xg / len(Sg)

    return f_x


def measure_single_obj(np.ndarray X_func, int si, surf_meas):
    cdef str param0 = str(X_func[0])
    cdef str param1 = str(X_func[1])
    return surf_meas[si]['_'.join((param0, param1))]

def gen_permutations(np.ndarray X_funcs, int Ng):
    return np.array(list(permutations(X_funcs, Ng)))


def record_merits(list S, surf_meas, X_func_truncate):
    
    cdef list f_xs = [] 
    cdef list param_opts, params_opts, cart_product, G
    cdef np.ndarray X_funcs, X_func_perms
    cdef int Ng, G_ix, X_func_ix

    cdef list Gs = gen_partitions(S)
    
    param_opts = [f'x{i}' for i in range(21)] 
    params_opts = [param_opts for _ in range(2)]
    cart_product = list(product(*params_opts))
    X_funcs = np.array([list(elem) for elem in cart_product])
    
    if isinstance(X_func_truncate,int):
        X_funcs = X_funcs[:X_func_truncate, :]
    
    for G_ix in range(len(Gs)): 
        Ng = len(Gs[G_ix])
        X_func_perms = gen_permutations(X_funcs, Ng)
        
        for X_func_ix in range(len(X_func_perms)):
            f_x = measure_objective(X_func_perms[X_func_ix], Gs[G_ix], surf_meas)

    
    return f_xs


