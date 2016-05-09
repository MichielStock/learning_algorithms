# -*- coding: utf-8 -*-
"""
Created on Mon 11 Jan 2016
Last update: Mon 9 May 2016

@author: Michiel Stock
michielfmstock@gmail.com

Kronecker Kernel Ridge regression, with the shortcuts
"""

import numpy as np
from PairwiseModel import *


def kronecker_ridge(Y, U, Sigma, V, S, reg):
    """
    Small method to generate parameters for Kroncker kernel ridge regression
    given de eigenvalue decomposition of the kernels
    """
    L = (np.dot(Sigma.reshape((-1, 1)), S.reshape((1, -1))) + reg)**-1
    L *= (U.T).dot(Y).dot(V)
    A = U.dot(L).dot(V.T)
    return A


def kernel_split(K, indice):
    """
    Removes an indice (row and column) of a kernel matrix and returns both
    """
    Ksubset = np.delete(np.delete(K, indice, axis=1), indice, axis=0)
    ktest = np.delete(K[:, indice], indice, axis=0)
    return Ksubset, ktest


def loocv_setA(Y, U, Sigma, V, S, regularization, Yhoo=None):
    """
    Leave-one-pair out for Kronecker kernel ridge regression setting A
    """
    E = np.dot(Sigma.reshape((-1, 1)), S.reshape((1, -1)))
    E /= (E + regularization)  # filtered eigenvalues of hat matrix
    Yhat = (U).dot(U.T.dot(Y).dot(V) * E).dot(V.T)
    L = np.dot((U)**2, E).dot(V.T**2)  # leverages, structured as a matrix
    return (Yhat - Y * L) / (1 - L)


def loocv_setB(Y, U, Sigma, V, S, regularization, Yhoo):
    """
    Leave-one-pair out for Kronecker kernel ridge regression setting B
    """
    G = (V * S).dot(V.T)
    K = (U * Sigma).dot(U.T)
    for indice in range(Y.shape[0]):  # iterate over rows
        Ksubset, ktest = kernel_split(K, indice)
        Sigma_new, U_new = np.linalg.eigh(Ksubset)
        A = kronecker_ridge(np.delete(Y, indice, axis=0), U_new, Sigma_new, V,
                                                    S, regularization)
        Yhoo[indice, :] = ktest.dot(A).dot(G)
    return Yhoo


def loocv_setC(Y, U, Sigma, V, S, regularization, Yhoo):
    """
    Leave-one-pair out for Kronecker kernel ridge regression setting C
    """
    Yhoo[:] = loocv_setB(Y.T, V, S, U, Sigma, regularization, Yhoo.T).T
    return Yhoo


def loocv_setD(Y, U, Sigma, V, S, regularization, Yhoo):
    """
    Leave-one-pair out for Kronecker kernel ridge regression setting D
    """
    G = (V * S).dot(V.T)
    K = (U * Sigma).dot(U.T)
    for i in range(Y.shape[0]):  # iterate over rows
        Ksubset, ktest = kernel_split(K, i)
        Sigma_new, U_new = np.linalg.eigh(Ksubset)
        for j in range(Y.shape[1]):  # iterate over columns
            Gsubset, gtest = kernel_split(G, j)
            S_new, V_new = np.linalg.eigh(Gsubset)
            Y_new = np.delete(np.delete(Y, i, axis=0), j, axis=1)
            A = kronecker_ridge(Y_new, U_new, Sigma_new, V_new, S_new,
                                    regularization)
            Yhoo[i, j] = ktest.dot(A).dot(gtest.T)
    return Yhoo


class KroneckerKernelRidgeRegression(PairwiseModel):
    """
    Kronecker kernel ridge regression, with the corresponding shortcuts
    """
    def __init__(self, Y, K, G):
        """
        Initialize the model from the two kernel matrices as feature
        descriptions
        """
        Sigma, U = np.linalg.eigh(K)
        S, V = np.linalg.eigh(G)
        self._Y = Y
        self._U = U
        self._V = V
        self._Sigma = Sigma
        self._S = S
        self.nrows, self.ncols = Y.shape

    def train_model(self, regularization=1):
        """
        Trains an Kronecker kernel ridge regression model
        """
        # make filtered values
        L = (np.dot(self._Sigma.reshape((-1, 1)), self._S.reshape((1, -1))) +
                                                        regularization)**-1
        self._filtered_vals = L  # save the filtered values
        # make parameters
        self._A = self._parameters_from_filtered_vals(L)
        self.regularization = regularization

    def lo_setting_A(self, regularization=1):
        """
        Imputation for setting A
        """
        return loocv_setA(self._Y, self._U, self._Sigma, self._V, self._S,
                                                          regularization)

    def lo_setting_B(self, regularization=1):
        """
        Imputation for setting B
        Uses a for-loop so is slow
        """
        return loocv_setB(self._Y, self._U, self._Sigma, self._V, self._S,
                                      regularization, np.zeros_like(self._Y))

    def lo_setting_C(self, regularization=1):
        """
        Imputation for setting C
        Uses a for-loop so is slow
        """
        return loocv_setC(self._Y, self._U, self._Sigma, self._V, self._S,
                                      regularization, np.zeros_like(self._Y))

    def lo_setting_D(self, regularization=1):
        """
        Imputation for setting D
        Uses two for-loops so is VERY slow
        """
        return loocv_setD(self._Y, self._U, self._Sigma, self._V, self._S,
                                      regularization, np.zeros_like(self._Y))

    def loocv_grid_search(self, grid, setting='A', performance=rmse):
        """
        Explores the performance for a grid of the regularization parameter
        """
        n_steps = len(grid)
        # initialize matrices
        performance_grid = np.zeros(n_steps)
        Yhoo = np.zeros_like(self._Y)
        # choose setting
        if setting == 'A':
            loocv_function = loocv_setA
        elif setting == 'B':
            loocv_function = loocv_setB
        elif setting == 'C':
            loocv_function = loocv_setC
        elif setting == 'D':
            loocv_function = loocv_setD
        for i, reg in enumerate(grid):
            # calculate holdout values
            Yhoo[:] = loocv_function(self._Y, self._U, self._Sigma,
                                            self._V, self._S, reg, Yhoo)
            performance_grid[i] = performance(self._Y, Yhoo)
        return performance_grid

    def tune_loocv(self, grid, setting='A', performance=rmse):
        """
        Tunes a model for a certain setting by grid search.
        Gives the model with the LOWEST value for performance metric
        """
        # initialize matrices
        best_perf = np.inf
        best_reg = 0
        Yhoo = np.zeros_like(self._Y)
        # choose setting
        if setting == 'A':
            loocv_function = loocv_setA
        elif setting == 'B':
            loocv_function = loocv_setB
        elif setting == 'C':
            loocv_function = loocv_setC
        elif setting == 'D':
            loocv_function = pyloocv_setD
        for i, reg in enumerate(grid):
            # calculate holdout values
            Yhoo[:] = loocv_function(self._Y, self._U, self._Sigma,
                                        self._V, self._S, reg, Yhoo)
            perf = performance(self._Y, Yhoo)
            if perf < best_perf:
                best_reg = reg
                best_perf = perf
        self.train_model(best_reg)
        print('Best regularization {} gives {}'.format(best_reg, best_perf))

if __name__ == '__main__':

    n_rows, n_cols = 100, 250
    dim_1, dim_2 = 300, 600

    Y = np.random.randn(n_rows, n_cols)
    X1 = np.random.randn(n_rows, dim_1)
    X2 = np.random.rand(n_cols, dim_2)

    K = X1.dot(X1.T)
    G = X2.dot(X2.T)

    Sigma, U = np.linalg.eigh(K)
    S, V = np.linalg.eigh(G)

    model = KroneckerKernelRidgeRegression(Y, K, G)
    model.train_model(regularization=100)

    # test prediction function
    print('These values should be the same:')
    print(np.allclose(model.predict(), model.predict(K, G)))
