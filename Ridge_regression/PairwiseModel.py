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
        self._sigma = Sigma
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
        if axis == None:
            N = self.n_u * self.n_v * 1.0
            Npos = self._Y.sum()
            self._Y[self._Y > 0] = N / Npos
            self._Y[self._Y <= 0] = - N / (N - Npos)
        elif axis ==  0:
            rowsums = self._Y.sum(1)
            for i in range(self.n_u):
                y = self._Y[i]
                y[y > 0] =  self.n_v / rowsums[i]
                y[y == 0] = - self.n_v / (self.n_v - rowsums[i])
                self._Y[i] = y
        elif axis ==  1:
            colsums = self._Y.sum(0)
            for i in range(self.n_v):
                y = self._Y[:,i]
                y[y > 0] =  self.n_u / colsums[i]
                y[y == 0] = - self.n_u / (self.n_u- colsums[i])
                self._Y[:,i] = y