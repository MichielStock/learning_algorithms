# -*- coding: utf-8 -*-
"""
Created on Sat 17 Jul 2016
Last update: Thu 2 Jul 2016

@author: Michiel Stock
michielfmstock@gmail.com

Method for conditional ranking using Kronecker ridge regression or
two-step ridge regression
"""

from PairwiseModel import rmse
import numpy as np

class ConditionalRankingKronecker():

    def __init__(self, Y, K, G, axis=0):
        m, q = Y.shape
        if axis==0:  # ranking of u
            C = np.eye(m) - np.ones((m, m)) / m
            K = C.dot(K)
            Y = C.dot(Y)
        elif axis==1:  # ranking of v
            C = np.eye(q) - np.ones((q, q)) / q
            G = C.dot(G)
            Y = Y.dot(C)
        else:
            print('Axis should be 0 or 1')
            raise KeyError
        Sigma, U = np.linalg.eig(K)
        S, V = np.linalg.eig(G)
        self._Y = Y
        self._U = U
        self._Uinv = np.linalg.inv(U)
        self._V = V
        self._Vinv = np.linalg.inv(V)
        self._Sigma = Sigma
        self._S = S
        self.nrows, self.ncols = Y.shape

    def train_model(self, regularization=1):
        """
        Construct the dual parameters from the eigenvalue decomposition
        """
        # filtered eigenvalues
        L = np.dot(self._Sigma.reshape((-1, 1)), self._S.reshape((1, -1)))
        E = self._Uinv.dot(self._Y).dot(self._Vinv.T) / (L + regularization)
        # the dual parameters
        self._A_complex = (self._U.dot(E).dot(self._V.T))
        self._A = self._A_complex.real  # only real part
        self.regularization = regularization

    def predict(self, k, g):
        """
        Makes predictions
        """
        predictions = k.dot(self._A).dot(g.T)
        return predictions.real

    def lo_setting_A(self, regularization=1):
        """
        Imputation for setting A
        """
        # eigenvalues Kronecker product kernels
        E = np.dot(self._Sigma.reshape((-1, 1)), self._S.reshape((1, -1)))
        E /= (E + regularization)  # filtered eigenvalues of hat matrix
        Yhat = self._U.dot(self._Uinv.dot(self._Y).dot(self._Vinv.T) *\
                    E).dot(self._V.T)
        leverages = (self._U * self._Uinv.T).dot(E).dot((self._V * self._Vinv.T).T)
        loo_values = (Yhat - leverages * self._Y) / (1 - leverages)
        return loo_values.real

    def tune_loocv(self, grid, performance=rmse):
        """
        Tunes a model for Setting A by grid search.
        Gives the model with the LOWEST value for performance metric
        """
        loo_predictions = np.zeros_like(self._Y)
        best_perf = 10**10
        for reg in grid:
            loo_predictions[:] = self.lo_setting_A(reg)
            perf = performance(self._Y, loo_predictions)
            if perf < best_perf:
                best_perf = perf
                best_reg = reg
        self.train_model(best_reg)
        print('Best regularization {} gives {}'.format(best_reg, best_perf))

if __name__ == '__main__':

    from PairwiseModel import c_index, matrix_c_index, rmse
    micro_c_index = lambda Y, P : c_index(Y.ravel(), P.ravel())

    X_u = np.random.randn(200, 201)
    X_v = np.random.randn(200, 201)
    # set bias
    X_u[:, -1] = 1
    X_v[:, -1] = 1

    Y = X_u[:, [0, 1]].dot(np.ones((2, 200))) + np.random.randn(200, 200)
    Y += np.random.randn(200, 1) * 2
    Y += np.random.randn(1, 200) * 3


    Ktrain = X_u[:100].dot(X_u[:100].T)
    Ktest = X_u[100:].dot(X_u[:100].T)

    Gtrain = X_v[:100].dot(X_v[:100].T)
    Gtest = X_v[100:].dot(X_v[:100].T)
    Ytrain = Y[:, :100][:100, :]
    Ytest = Y[:, 100:][100:, :]

    model0 = ConditionalRankingKronecker(Ytrain, Ktrain, Gtest, axis=0)
    model1 = ConditionalRankingKronecker(Ytrain, Ktrain, Gtest, axis=1)

    model0.train_model(1)
    model1.train_model(1)

    P0 = model0.predict(Ktest, Gtest)
    P1 = model0.predict(Ktest, Gtest)
