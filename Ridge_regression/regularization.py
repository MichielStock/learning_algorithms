"""
Created on Tue 13 Oct 2015
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Module for studying the effect of the regulatrization parameters for
pairwise problems
"""

from Kronecker_regularized_least_squares import KroneckerRegularizedLeastSquares
import numpy as np
from sklearn.metrics import auc

# some functions for measuring performance on labels
# --------------------------------------------------

# mean squared error
mse = lambda Y, P : np.mean( (Y -P)**2 )

# macro AUC
macro_auc = lambda Y, P : auc(Y.ravel(), P.ravel())

# instance AUC
def instance_auc(Y, P):
    n, m = Y.shape
    return np.mean([auc(Y[i], P[i]) for i in range(n) if Y[i].var()])

# micro AUC
def micro_auc(Y, P):
    n, m = Y.shape
    return np.mean([auc(Y[:,i], P[:,i]) for i in range(m) if Y[:,i].var()])
