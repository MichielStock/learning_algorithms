# -*- coding: utf-8 -*-
"""
Created on Mon 11 Jan 2016
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

General class for pairwise model
"""

import numpy as np


class PairwiseModel:
    """
    General pairwise model to use as a base for Kronecker ridge models and
    two-step ridge regression.

    Has as input just the eigenvalues and eigenvectors and performs
    Kronecker ordinary least squares
    """
    def __init__(self, Y, U, Sigma, V, S):
        self._Y = Y
        self._U = U
        self._V = V
        self._Sigma = Sigma
        self._S = S

    def transform_fisher(self, axis=None):
        """
        Perform a Fisher transformantion on the labels for binary data
        such that the positive labels have a value of N/N_+ and the negative of
        -N/N_-. This transformation makes the Ridge regression equivalent with
        Fisher discriminant analysis. Optionally specify the axis: None
        (default) for transforming the whole matrix, 0 for the rows, and 1 for
        the colums.
        """
        # only for binary data
        assert self._Y.min() == 0 and self._Y.max() == 1
        self._Y = self._Y.astype(float)
        if axis is None:
            N = self.n_u * self.n_v * 1.0
            Npos = self._Y.sum()
            self._Y[self._Y > 0] = N / Npos
            self._Y[self._Y <= 0] = - N / (N - Npos)
        elif axis == 0:
            rowsums = self._Y.sum(1)
            for i in range(self.n_u):
                y = self._Y[i]
                y[y > 0] = self.n_v / rowsums[i]
                y[y == 0] = - self.n_v / (self.n_v - rowsums[i])
                self._Y[i] = y
        elif axis == 1:
            colsums = self._Y.sum(0)
            for i in range(self.n_v):
                y = self._Y[:, i]
                y[y > 0] = self.n_u / colsums[i]
                y[y == 0] = - self.n_u / (self.n_u - colsums[i])
                self._Y[:, i] = y

    def _parameters_from_leverages(self, L):
            """
            Given the leverages, construct the parameters
            """
            B = L * (self._U.T).dot(self._Y).dot(self._V)
            A = self._U.dot(B).dot(self._V.T)
            return A

    def train_model(self, return_Yhat=False):
        """
        Trains an Kronecker kernel ordinary least-squares model
        """
        # make leverages
        L = np.dot(self._Sigma.reshape((-1, 1)), self._S.reshape((1, -1)))**-1
        # make parameters
        self._A = self._parameters_from_leverages(L)

    def get_parameters(self):
        return self._A

    def predict(self, k=None, g=None):
        """
        Makes predictions
        """
        if k is None:
            # use training instances
            predictions = (self._U * self._Sigma).dot(U.T).dot(self._A)
        else:
            # use given instances
            predictions = k.dot(self._A)
        if g is None:
            # use training tasks
            predictions = predictions.dot(self._V * self._S).dot(self._V.T)
        else:
            predictions = predictions.dot(g.T)
        return predictions

if __name__ == '__main__':

    Y = np.random.randn(10, 20)
    X1 = np.random.randn(10, 10)
    X2 = np.random.rand(20, 20)

    K = X1.dot(X1.T)
    G = X2.dot(X2.T)

    Sigma, U = np.linalg.eigh(K)
    S, V = np.linalg.eigh(G)

    model = PairwiseModel(Y, U, Sigma, V, S)
    model.train_model()

    # test prediction function
    print('These values should be the same:')
    print(np.allclose(model.predict(), model.predict(K, G)))
