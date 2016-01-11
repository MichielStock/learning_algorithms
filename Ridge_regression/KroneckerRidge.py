# -*- coding: utf-8 -*-
"""
Created on Mon 11 Jan 2016
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Kronecker Kernel Ridge regression, with the shortcuts
"""

import numpy as np
from PairwiseModel import PairwiseModel


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

    def train_model(self, regularization=1, return_Yhat=False):
        """
        Trains an Kronecker kernel ridge regression model
        """
        # make leverages
        L = (np.dot(self._Sigma.reshape((-1, 1)), self._S.reshape((1, -1))) +
                                                        regularization)**-1
        # make parameters
        self._A = self._parameters_from_leverages(L)

if __name__ == '__main__':

    Y = np.random.randn(10, 20)
    X1 = np.random.randn(10, 10)
    X2 = np.random.rand(20, 20)

    K = X1.dot(X1.T)
    G = X2.dot(X2.T)

    model = KroneckerKernelRidgeRegression(Y, K, G)
    model.train_model(regularization=100)

    # test prediction function
    print('These values should be the same:')
    print(np.allclose(model.predict(), model.predict(K, G)))
