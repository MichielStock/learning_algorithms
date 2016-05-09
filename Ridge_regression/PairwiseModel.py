# -*- coding: utf-8 -*-
"""
Created on Mon 11 Jan 2016
Last update: Mon 9 May 2016

@author: Michiel Stock
michielfmstock@gmail.com

General class for pairwise model
"""

import numpy as np
from sklearn.metrics import roc_auc_score as auc

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
        self.nrows, self.ncols = Y.shape
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
            N = self.nrows * self.ncols * 1.0
            Npos = self._Y.sum()
            self._Y[self._Y > 0] = N / Npos
            self._Y[self._Y <= 0] = - N / (N - Npos)
        elif axis == 0:
            rowsums = self._Y.sum(1)
            for i in range(self.nrows):
                y = self._Y[i]
                y[y > 0] = self.ncols / rowsums[i]
                y[y == 0] = - self.ncols / (self.ncols - rowsums[i])
                self._Y[i] = y
        elif axis == 1:
            colsums = self._Y.sum(0)
            for i in range(self.ncols):
                y = self._Y[:, i]
                y[y > 0] = self.nrows / colsums[i]
                y[y == 0] = - self.nrows / (self.nrows - colsums[i])
                self._Y[:, i] = y

    def _parameters_from_filtered_vals(self, L):
        """
        Given the leverages, construct the parameters
        """
        B = L * (self._U.T).dot(self._Y).dot(self._V)
        A = self._U.dot(B).dot(self._V.T)
        return A

    def train_model(self, regularization=None):
        """
        Trains an Kronecker kernel ordinary least-squares model
        """
        # make filted values
        L = np.dot(self._Sigma.reshape((-1, 1)), self._S.reshape((1, -1)))**-1
        self._filtered_vals = L  # save the filtered values
        # make parameters
        self._A = self._parameters_from_filtered_vals(L)

    def get_parameters(self):
        return self._A

    def predict(self, k=None, g=None):
        """
        Makes predictions, if no inputs are given, this method returns the
        predicted values for the training matrix. Optionally, one can give
        a row-vector with the kernel values or matrix with the rows corresponding
        to instances to make new predictions.
        """
        if k is None:
            # use training instances
            predictions = (self._U * self._Sigma).dot(self._U.T).dot(self._A)
        else:
            # use given instances
            predictions = k.dot(self._A)
        if g is None:
            # use training tasks
            predictions = predictions.dot(self._V * self._S).dot(self._V.T)
        else:
            predictions = predictions.dot(g.T)
        return predictions

    def reestimate(self, Ynew=None):
        """
        Re-estimates the label matrix, if no new matrix is provided, the matrix
        with the training labels will be used. If a new conformable label matrix
        is given, it will be re-estimated using the given model.
        """
        if Ynew is None: Ynew = self._Y
        # the required variables
        U, Sigma, V, S = self._U, self._Sigma, self._V, self._S
        return (U * Sigma).dot(U.T.dot(Ynew).dot(V) * self._filtered_vals).dot(
                        (V * S).T
            )

    def impute_iter(self, mask, Y=None, max_iter=100, epsilon=None):
        """
        Uses iteration to impute missing vales in a label matrix

        Inputs:
            - mask : boolean matrix with the trues denoting the elements that
                    have to be imputed and the false the given/observed labels
            - Y : new label matrix (optional argument) with values that have
                    to be imputed. WARNING! WILL BE OVERWRITTEN! MAKE DEEP COPY
                    IF IMPORTANT
            - max_iter : maximum number of iteration (default is 100)
            - epsilon : tolerance (optional argument), if the mean squared
                    difference between the imputed values of two iterations is
                    smaller than epsilon, the updating will terminate

        """
        if Y is None:  # use given labels
            Y = self._Y.copy()
        Y[mask] = np.mean(Y[mask==False])
        F = np.zeros_like(Y)
        iteration = 1
        while iteration <= max_iter:
            F[:] = self.reestimate(Y)
            print(np.mean((Y[mask] - F[mask])**2))
            if epsilon is not None and np.mean((Y[mask] - F[mask])**2) < epsilon:
                print('Converged after {} iterations'.format(iteration))
                Y[mask] = F[mask]  # update missing values
                break
            Y[mask] = F[mask]  # update missing values
            iteration += 1
        return Y


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

    model = PairwiseModel(Y, U, Sigma, V, S)
    model.train_model()

    # test prediction function
    print('These values should be the same:')
    print(np.allclose(model.predict(), model.predict(K, G)))
