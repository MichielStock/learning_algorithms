# -*- coding: utf-8 -*-
"""
Created on Mon 11 Jan 2016
Last update: Tue 12 Jan 2016

@author: Michiel Stock
michielfmstock@gmail.com

Kronecker Kernel Ridge regression, with the shortcuts
"""

import numpy as np
import numba
from PairwiseModel import PairwiseModel

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
    for indice in range(Y.shape[0]): # iterate over rows
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
        
    def lo_setting_A(self, regularization=1):
        """
        Imputation for setting A
        """
        return loocv_setA(self._Y, self._U, self._Sigma, self._V, self._S,
                                                          regularization)
 
    def lo_setting_B(self, regularization=1):
        """
        Imputation for setting B
        """
        return loocv_setB(self._Y, self._U, self._Sigma, self._V, self._S,
                                      regularization, np.zeros_like(self._Y))

    def lo_setting_C(self, regularization=1):
        """
        Imputation for setting B
        """
        return loocv_setC(self._Y, self._U, self._Sigma, self._V, self._S,
                                      regularization, np.zeros_like(self._Y))                                                        

if __name__ == '__main__':

    nrow = 110
    ncol = 55
    
    Y = np.random.randn(nrow, ncol)
    X1 = np.random.randn(nrow, nrow)
    X2 = np.random.rand(ncol, ncol)

    K = X1.dot(X1.T)
    G = X2.dot(X2.T)
    
    Sigma, U = np.linalg.eigh(K)
    S, V = np.linalg.eigh(G)

    model = KroneckerKernelRidgeRegression(Y, K, G)
    model.train_model(regularization=100)

    # test prediction function
    print('These values should be the same:')
    print(np.allclose(model.predict(), model.predict(K, G)))
