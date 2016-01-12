# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 16:23:13 2016

@author: michielstock
"""

import unittest
from KroneckerRidge import KroneckerKernelRidgeRegression, loocv_setA
import numpy as np


def make_problem():
    """
    Makes a pairwise problem
    """
    Y = np.random.randn(10, 20)
    X1 = np.random.randn(10, 10)
    X2 = np.random.rand(20, 20)
    K = X1.dot(X1.T)
    G = X2.dot(X2.T)
    reg = np.random.randint(1, 100)
    return Y, K, G, reg


class TestKroneckerRidge(unittest.TestCase):
    """
    Test Kronecker kernel ridge regression implementation
    """
    def test_prediction(self):
        """
        Test the prediction function
        using KroneckerKernelRidgeRegression.predict() using no arguments
        should return Yhat
        """
        print 'Testing prediction hat'
        Y, K, G, reg = make_problem()
        model = KroneckerKernelRidgeRegression(Y, K, G)
        model.train_model(regularization=reg)
        self.assertTrue((np.allclose(model.predict(), model.predict(K, G))))

    def test_prediction_correct(self):
        """
        Test if prediction is correct
        """
        print 'testing prediction'
        Y, K, G, reg = make_problem()   
        N = np.prod(Y.shape)
        model = KroneckerKernelRidgeRegression(Y, K, G)
        model.train_model(regularization=reg)
        P = model.predict()  # according to model
        P = np.reshape(P, (-1, ), order='F')  # vecotrize
        Ptrue = np.kron(G, K).dot(np.linalg.inv(np.kron(G, K) + np.eye(N) *
                                reg)).dot(np.reshape(Y, -1, order='F'))
        self.assertTrue(np.allclose(P, Ptrue))
        
                
if __name__ == '__main__':
    unittest.main()
