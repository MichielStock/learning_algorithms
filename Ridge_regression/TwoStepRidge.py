"""
Created on Wed 13 Jan 2016
Last update: Sun 21 Feb 2016

@author: Michiel Stock
michielfmstock@gmail.com

Implementations of the two-step kernel ridge regression method
"""

import numpy as np
from KroneckerRidge import KroneckerKernelRidgeRegression
from PairwiseModel import *


# CROSS VALIDATION
# ----------------

def loocv_setA(Y, H_k, H_g):
    """
    Leave-one-pair out for two-step ridge regression setting A
    """
    return ( H_k.dot(Y).dot(H_g) -  # Yhat
            np.diag(H_g) * Y * np.diag(H_k).reshape((-1, 1))) / (1 -
            np.dot(np.diag(H_k).reshape((-1, 1)),
                        np.diag(H_g).reshape((1, -1))))


def loocv_setB(Y, H_k, H_g):
    """
    Leave-one-pair out for two-step ridge regression setting B
    """
    return (H_k - np.diag(np.diag(H_k))).dot(Y).dot(H_g) / (1 -
            np.diag(H_k).reshape((-1, 1)))

def loocv_setC(Y, H_k, H_g):
    """
    Leave-one-pair out for two-step ridge regression setting C
    """
    return H_k.dot(Y).dot(H_g - np.diag(np.diag(H_g))) / (1 -
            np.diag(H_g))


#@numba.jit
def loocv_setD(Y, H_k, H_g, Y_hoo):
    """
    Leave-one-pair out for two-step ridge regression setting D
    """
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y_hoo[i, j] = H_k[i, :].dot(Y.dot(H_g[:,j ]) - Y[:, j] * H_g[j, j])
            Y_hoo[i, j] -= H_k[i, i] * (Y[[i]].dot(H_g[:, j]) - Y[i, j] *
                    H_g[j, j])
            Y_hoo[i, j] /= (1 - H_k[i, i]) * (1 - H_g[j, j])
    return Y_hoo

def regularization_map_2sridge(Y, Sigma, U, S, V, H_k, H_g, loocv_function,
                               Yhoo, grid, performance, performance_matrix):
    for i, reg_1 in enumerate(grid):
        H_k[:] = (U * Sigma / (Sigma + reg_1)).dot(U.T)
        for j, reg_2 in enumerate(grid):
            H_g[:] = (V * S / (S + reg_2)).dot(V.T)
            # calculate holdout values
            Yhoo[:] = loocv_function(Y, H_k, H_g)
            performance_matrix[i, j] = performance(Y, Yhoo)
    return performance_matrix


# MAIN CLASS
# ----------

class TwoStepRidgeRegression(KroneckerKernelRidgeRegression):
    """
    Kronecker kernel ridge regression, with the corresponding shortcuts
    """
    def train_model(self, regularization=(1, 1), return_Yhat=False):
        """
        Trains an Kronecker kernel ridge regression model
        """
        reg_1, reg_2 = regularization
        self.regularization = regularization
        self._A = (self._U / (self._Sigma + reg_1)).dot(
                    self._U.T).dot(self._Y).dot(self._V / (self._S +
                    reg_2)).dot(self._V.T)

    def lo_setting_A(self, regularization=(1, 1), H_k=None, H_g=None):
        """
        Imputation for setting A
        """
        reg_1, reg_2 = regularization
        if H_k is None:
            H_k = (self._U * self._Sigma / (self._Sigma + reg_1)).dot(
                        self._U.T)
        if H_g is None:
            H_g = (self._V * self._S / (self._S + reg_2)).dot(self._V.T)
        return loocv_setA(self._Y, H_k, H_g)

    def lo_setting_B(self, regularization=(1, 1), H_k=None, H_g=None):
        """
        Imputation for setting B
        """
        reg_1, reg_2 = regularization
        if H_k is None:
            H_k = (self._U * self._Sigma / (self._Sigma + reg_1)).dot(
                        self._U.T)
        if H_g is None:
            H_g = (self._V * self._S / (self._S + reg_2)).dot(self._V.T)
        leverages_k = np.diag(H_k).reshape((-1, 1))
        rep_leverages = np.repeat(leverages_k, self.ncols, axis=1)
        Y_loo = (H_k.dot(self._Y) - rep_leverages * self._Y).dot(H_g)
        Y_loo /= 1.0 - rep_leverages
        return Y_loo

    def lo_setting_C(self, regularization=(1, 1), H_k=None, H_g=None):
        """
        Imputation for setting C
        """
        reg_1, reg_2 = regularization
        if H_k is None:
            H_k = (self._U * self._Sigma / (self._Sigma + reg_1)).dot(
                        self._U.T)
        if H_g is None:
            H_g = (self._V * self._S / (self._S + reg_2)).dot(self._V.T)
        Y_loo = H_k.dot(self._Y.dot(H_g) - self._Y * np.diag(H_g))
        Y_loo /= 1 - np.diag(H_g)
        return Y_loo

    def lo_setting_D(self, regularization=(1, 1), H_k=None, H_g=None):
        """
        Imputation for setting D
        """
        reg_1, reg_2 = regularization
        if H_k is None:
            H_k = (self._U * self._Sigma / (self._Sigma + reg_1)).dot(
                        self._U.T)
        if H_g is None:
            H_g = (self._V * self._S / (self._S + reg_2)).dot(self._V.T)
        leverages_k = np.diag(H_k).reshape((-1, 1))
        leverages_g = np.diag(H_g).reshape((1, -1))
        Y_loo = H_k.dot(self._Y).dot(H_g)
        Y_loo -= H_k.dot(self._Y) * leverages_g
        Y_loo -= leverages_k * (self._Y.dot(H_g))
        Y_loo += leverages_k * self._Y * leverages_g
        Y_loo /= (1 - leverages_k).dot(1 - leverages_g)
        return Y_loo

    def loocv_grid_search(self, grid, setting='A', performance=rmse):
        """
        Explores the performance for a grid of the two regularization
        parameters
        """
        n_steps = len(grid)
        # initialize matrices
        performance_grid = np.zeros((n_steps, n_steps))
        Yhoo = np.zeros_like(self._Y)
        # choose setting
        H_k = np.zeros((self.nrows, self.nrows))
        H_g = np.zeros((self.ncols, self.ncols))
        for i, reg_1 in enumerate(grid):
            H_k[:] = (self._U * self._Sigma / (self._Sigma + reg_1)).dot(
                                self._U.T)
            for j, reg_2 in enumerate(grid):
                H_g[:] = (self._V * self._S / (self._S + reg_2)).dot(self._V.T)
                # calculate holdout values
                if setting == 'A':
                    Yhoo[:] = self.lo_setting_A((reg_1, reg_2), H_k, H_g)
                elif setting == 'B':
                    Yhoo[:] = self.lo_setting_B((reg_1, reg_2), H_k, H_g)
                elif setting == 'C':
                    Yhoo[:] = self.lo_setting_C((reg_1, reg_2), H_k, H_g)
                elif setting == 'D':
                    Yhoo[:] = self.lo_setting_D((reg_1, reg_2), H_k, H_g)
                performance_grid[i, j] = performance(self._Y, Yhoo)
        return performance_grid

    def tune_loocv(self, grid, setting='A', performance=rmse):
        """
        Tunes a model for a certain setting by grid search.
        Gives the model with the LOWEST value for performance metric
        """
        # initialize matrices
        best_perf = np.inf
        best_regs = (0, 0)
        Yhoo = np.zeros_like(self._Y)
        H_k = np.zeros((self.nrows, self.nrows))
        H_g = np.zeros((self.ncols, self.ncols))
        # choose setting
        for i, reg_1 in enumerate(grid):
            H_k[:] = (self._U * self._Sigma / (self._Sigma + reg_1)).dot(
                                self._U.T)
            for j, reg_2 in enumerate(grid):
                H_g[:] = (self._V * self._S / (self._S + reg_2)).dot(self._V.T)
                # calculate holdout values
                if setting == 'A':
                    Yhoo[:] = self.lo_setting_A((reg_1, reg_2), H_k, H_g)
                elif setting == 'B':
                    Yhoo[:] = self.lo_setting_B((reg_1, reg_2), H_k, H_g)
                elif setting == 'C':
                    Yhoo[:] = self.lo_setting_C((reg_1, reg_2), H_k, H_g)
                elif setting == 'D':
                    Yhoo[:] = self.lo_setting_D((reg_1, reg_2), H_k, H_g)
                performance_ij = performance(self._Y, Yhoo)
                if performance_ij < best_perf:
                    best_regs = (reg_1, reg_2)
                    best_perf = performance_ij
        self.train_model(best_regs)
        print('Best regularization {} gives {}'.format(best_regs, best_perf))

if __name__ == '__main__':

    nrow = 110
    ncol = 150

    Y = np.random.randn(nrow, ncol)
    X1 = np.random.randn(nrow, nrow)
    X2 = np.random.rand(ncol, ncol)

    K = X1.dot(X1.T)
    G = X2.dot(X2.T)

    Sigma, U = np.linalg.eigh(K)
    S, V = np.linalg.eigh(G)

    H_k = (U * Sigma / (Sigma + 10)).dot(U.T)
    H_g = (V * S / (S + 0.1)).dot(V.T)

    model = TwoStepRidgeRegression(Y, K, G)
    model.train_model(regularization=(10, 0.1))

    # test prediction function
    print('These values should be the same:')
    print(np.allclose(model.predict(), model.predict(K, G)))
