# -*- coding: utf-8 -*-
"""
Created on Mon 11 Jan 2016
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Pairwise performance measures
"""

import numpy as np
from sklearn.metrics import roc_auc_score as auc
import numba

# PERFORMANCE MEASURES
# --------------------

rmse = lambda Y, P : np.mean((Y - P)**2)**0.5

# mean squared error
mse = lambda Y, P : np.mean( (Y - P)**2 )

# micro AUC
micro_auc = lambda Y, P : auc(np.ravel(Y) > 0, np.ravel(P))

# instance AUC
def instance_auc(Y, P):
    n, m = Y.shape
    return np.mean([auc(Y[i] > 0, P[i]) for i in range(n) if Y[i].var()])

# macro AUC
def macro_auc(Y, P):
    n, m = Y.shape
    return np.mean([auc(Y[:,i] > 0, P[:,i]) for i in range(m) if Y[:,i].var()])

# C-index
@numba.jit
def c_index(y, p):
    """
    C-index for vectors
    """
    compared = 0.0
    ordered = 0.0
    n = len(y)
    for i in range(n):
        for j in range(n):
            if y[i] > y[j]:
                compared += 1
                if p[i] > p[j]:
                    ordered += 1
                elif p[i] == p[j]:
                    ordered += 0.5
    return ordered / compared

# C-index for matrices
def matrix_c_index(Y, P, axis=0):
    nrows, ncols = Y.shape
    if axis==0:
        return np.mean([c_index(Y[:,i], P[:,i]) for i in range(ncols)
                    if np.var(Y[:,i]) > 1e-8])
    elif axis==1:
        return np.mean([c_index(Y[[i]], P[[i]]) for i in range(nrows)
                    if np.var(Y[[i]]) > 1e-8])
    else:
        raise KeyError
