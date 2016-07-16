# -*- coding: utf-8 -*-
"""
Created on Mon 11 Jan 2016
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Method for conditional ranking using Kronecker ridge regression or
two-step ridge regression
"""

from KroneckerRidge import KroneckerKernelRidgeRegression
from TwoStepRidge import TwoStepRidgeRegression
import numpy as np

class ConditionalRankingKronecker(KroneckerKernelRidgeRegression):

    def __init__(self, Y, K, G, axis=0):
        m, q = Y.shape
        if axis==0:  # ranking of u
            K = K - K.sum(axis=1, keepdims=True) / m
            Y = Y - Y.sum(axis=0, keepdims=True) / m
        elif axis==1:  # ranking of v
            G = G - G.sum(axis=1, keepdims=True) / q
            Y = Y - Y.sum(axis=1, keepdims=True) / q
        else:
            print('Axis should be 0 or 1')
            raise KeyError
        Sigma, U = np.linalg.eig(K)
        S, V = np.linalg.eig(G)
        self._Y = Y
        self._U = U.real
        self._V = V.real
        self._Sigma = Sigma.real
        self._S = S.real
        self.nrows, self.ncols = Y.shape

class ConditionalRankingTwoStep(TwoStepRidgeRegression):

    def __init__(self, Y, K, G, axis=0):
        m, q = Y.shape
        if axis==0:  # ranking of u
            K = K - K.sum(axis=1, keepdims=True) / m
            Y = Y - Y.sum(axis=0, keepdims=True) / m
        elif axis==1:  # ranking of v
            G = G - G.sum(axis=1, keepdims=True) / q
            Y = Y - Y.sum(axis=1, keepdims=True) / q
        else:
            print('Axis should be 0 or 1')
            raise KeyError
        Sigma, U = np.linalg.eig(K)
        S, V = np.linalg.eig(G)
        self._Y = Y
        self._U = U.real
        self._V = V.real
        self._Sigma = Sigma.real
        self._S = S.real
        self.nrows, self.ncols = Y.shape

if __name__ == '__main__':

    from PairwiseModel import c_index, matrix_c_index
    micro_c_index = lambda Y, P : c_index(Y.ravel(), P.ravel())

    X_u = np.random.randn(200, 11)
    X_v = np.random.randn(200, 11)
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

    for axis in [0, 1]:
        model = ConditionalRankingTwoStep(Ytrain, Ktrain, Gtrain, axis)
        model.train_model((1, 1))
        model.tune_loocv(np.logspace(-5, 5, 11), 'D'
                        #, performance=lambda Y, F : -micro_c_index(Y, F)
                        )
        predictions = model.predict(Ktest, Gtest)
        print('TS : axis {}: macro c-index = {:.4f}, instance c-index'.format(
            axis,
            matrix_c_index(Ytest, predictions, axis=1),
            matrix_c_index(Ytest, predictions, axis=0)
            ))

    for axis in [0, 1]:
        model = ConditionalRankingKronecker(Ytrain, Ktrain, Gtrain, axis)
        model.train_model(0.01)
        model.tune_loocv(np.logspace(-5, 5, 11), 'A'
                        #, performance=lambda Y, F : -micro_c_index(Y, F)
                        )
        predictions = model.predict(Ktest, Gtest)
        print('KK : axis {}: macro c-index = {:.4f}, instance c-index'.format(
            axis,
            matrix_c_index(Ytest, predictions, axis=1),
            matrix_c_index(Ytest, predictions, axis=0)
            ))
