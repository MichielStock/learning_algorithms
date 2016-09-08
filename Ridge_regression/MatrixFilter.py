# -*- coding: utf-8 -*-
"""
Created on Mon 15 Aug 2016
Last update: Mon 9 May 2016

@author: Michiel Stock
michielfmstock@gmail.com

Simple matrix filter
"""

import numpy as np
from sklearn.metrics import roc_auc_score

mse = lambda Y, F : np.mean( (Y - F)**2 )
auc = lambda Y, F : roc_auc_score(Y.flatten()>0, F.flatten())

def filter_matrix(Y, alphas=(0.25, 0.25, 0.25, 0.25)):
    """
    Filters the matrix
    """
    a1, a2, a3, a4 = alphas
    return a1 * Y + a2 * Y.mean(0, keepdims=True) +\
            a3 * Y.mean(1, keepdims=True) + a4 * Y.mean()

def impute_matrix(Y, F, leverage):
    """
    Computes leave-one-out imputation of a matrix using the filter
    """
    return (F - Y * leverage) / (1 - leverage)

class MatrixFilter:
    """
    Simple class for filtering a matrix
    """
    def __init__(self, Y, alphas=(0.25, 0.25, 0.25, 0.25)):
        self._Y = Y
        a1, a2, a3, a4 = alphas
        n, m = Y.shape
        self.n, self.m = n, m
        self._alphas = alphas
        self._F = filter_matrix(Y, alphas)
        self._leverage = a1 + a2 / n + a3 / m + a4 / n / m
        self._Yimp = impute_matrix(Y, self._F, self._leverage)

    def iterative_impuation(self, mask, n_iterations=100,
                            performance_measure=None):
        """
        Impute missing values of a matrix.

        Inputs:
            - mask : boolean mask of size of Y indicating the values to be
                    imputated
            - n_iterations : number of iterations
            - performance_measure (optional) : caculate performance on the
                        imputed values

        Output:
            Yimp : the imputed matrix
        """
        Yimp = np.zeros_like(self._Y)
        Yimp[mask==False] = self._Y[mask==False]
        for i in range(n_iterations):
            Yimp[mask] = filter_matrix(Yimp, self._alphas)[mask]
        if performance_measure is None:
            return Yimp
        else:
            return Yimp, performance_measure(self._Y[mask], Yimp[mask])

    def set_alphas(self, alphas):
        """
        Sets the alphas
        """
        n, m = self.n, self.m
        self._alphas = alphas
        self._F = filter_matrix(self._Y, alphas)
        a1, a2, a3, a4 = alphas
        self._leverage = a1 + a2 / n + a3 / m + a4 / n / m
        self._Yimp = impute_matrix(self._Y, self._F, self._leverage)

    def get_performance(self, performance_measure=mse):
        """
        Gets the performance of the imputed vs. real values
        default = mse
        """
        return performance_measure(self._Y, self._Yimp)
