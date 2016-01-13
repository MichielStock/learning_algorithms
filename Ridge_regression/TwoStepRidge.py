"""
Created on Wed 13 Jan 2016
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Implementations of the two-step kernel ridge regression method
"""

import numpy as np
from KroneckerRidge import KroneckerKernelRidgeRegression
import numba


def loocv_setA(Y, H_k, H_g):
    """
    Leave-one-pair out for two-step ridge regression setting A
    """
    return ( H_k.dot(Y).dot(H_g) -  # Yhat
            np.diag(H_g) * Y * np.diag(H_k).reshape((-1, 1))) / (1 -
            np.dot(np.diag(H_k).reshape((-1, 1)), 
                        np.diag(H_g).reshape((1, -1))))


def loocv_setB(Y, H_k, H_g, Yhoo):
    """
    Leave-one-pair out for two-step ridge regression setting A
    """
    
    


def loocv_setC(Y, H_k, H_g, Yhoo):
    """
    Leave-one-pair out for two-step ridge regression setting A
    """
    H_g = 1


def loocv_setD(Y, H_k, H_g, Yhoo):
    """
    Leave-one-pair out for two-step ridge regression setting A
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


class TwoStepRidgeRegression(KroneckerKernelRidgeRegression):
    """
    Kronecker kernel ridge regression, with the corresponding shortcuts
    """
    def train_model(self, regularization=(1,1), return_Yhat=False):
        """
        Trains an Kronecker kernel ridge regression model
        """
        reg_1, reg_2 = regularization
        L = (np.dot(self._Sigma.reshape((-1, 1)) + reg_1,
                                self._S.reshape((1, -1)) + reg_2))**-1
        # make parameters
        self._A = self._parameters_from_leverages(L)

    def lo_setting_A(self, regularization=(1,1)):
        """
        Imputation for setting A
        """
        return loocv_setA(self._Y, self._U, self._Sigma, self._V, self._S,
                                                          regularization)

    def lo_setting_B(self, regularization=(1,1)):
        """
        Imputation for setting B
        Uses a for-loop so is slow
        """
        return loocv_setB(self._Y, self._U, self._Sigma, self._V, self._S,
                                      regularization, np.zeros_like(self._Y))

    def lo_setting_C(self, regularization=(1,1)):
        """
        Imputation for setting C
        Uses a for-loop so is slow
        """
        return loocv_setC(self._Y, self._U, self._Sigma, self._V, self._S,
                                      regularization, np.zeros_like(self._Y))

    def lo_setting_D(self, regularization=(1,1)):
        """
        Imputation for setting D
        Uses two for-loops so is VERY slow   
        """
        return loocv_setD(self._Y, self._U, self._Sigma, self._V, self._S,
                                      regularization, np.zeros_like(self._Y))
                                      
                                      
if __name__ == '__main__':

    nrow = 25
    ncol = 43

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
