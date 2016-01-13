# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 2015
Last update: Tue Feb 17 2015

@author: Michiel Stock
michielfmstock@gmail.com

Implementations of the kronecker RLS methods for pairs
"""

import unittest
from KroneckerRidge import KroneckerKernelRidgeRegression, loocv_setA
import numpy as np


def make_problem(nrow=40, ncol=25):
    """
    Makes a pairwise problem
    """
    Y = np.random.randn(nrow, ncol)
    X1 = np.random.randn(nrow, nrow)
    X2 = np.random.rand(ncol, ncol)
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
        Y, K, G, reg = make_problem()
        model = KroneckerKernelRidgeRegression(Y, K, G)
        model.train_model(regularization=reg)
        self.assertTrue((np.allclose(model.predict(), model.predict(K, G))))

    def test_prediction_correct(self):
        """
        Test if prediction is correct
        """
        Y, K, G, reg = make_problem(20, 11)
        N = np.prod(Y.shape)
        model = KroneckerKernelRidgeRegression(Y, K, G)
        model.train_model(regularization=reg)
        P = model.predict()  # according to model
        P = np.reshape(P, (-1, ), order='F')  # vecotrize
        Ptrue = np.kron(G, K).dot(np.linalg.inv(np.kron(G, K) + np.eye(N) *
                                reg)).dot(np.reshape(Y, -1, order='F'))
        self.assertTrue(np.allclose(P, Ptrue))

    def test_imputation_holdout(self):
        """
        Test for setting A, based on the idea that if you replace Yij with
        the predicted imputated value, this should not change the predicted
        value for Yhatij
        """
        Y, K, G, reg = make_problem(20, 40)  # do for a small problem
        model = KroneckerKernelRidgeRegression(Y, K, G)
        Yhoo = model.lo_setting_A(reg)
        tests = np.zeros(Y.shape, dtype=bool)
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                Yhelp = Y.copy()
                Yhelp[i, j] = Yhoo[i, j]  # replacing this should not matter
                model._Y = Yhelp
                model.train_model(reg)  # train again, with new Y
                pred_ij = model.predict()[i, j]
                tests[i, j] = np.allclose(pred_ij, Yhoo[i, j])  # should close!
                # uncomment to see in detail
                # print(Yhoo[i,j], pred_ij, Y[i,j])
        correct = np.all(tests)
        self.assertTrue(correct)

    def test_settingB_holdout(self):
        Y, K, G, reg = make_problem(20, 40)  # do for a small problem
        model = KroneckerKernelRidgeRegression(Y, K, G)
        Yhoo = model.lo_setting_B(reg)
        tests = np.zeros(Y.shape[0], dtype=bool)
        for i in range(Y.shape[0]):
            Yhelp = Y.copy()
            Yhelp[i, :] = Yhoo[i, :]  # replacing this should not matter
            model._Y = Yhelp
            model.train_model(reg)  # train again, with new Y
            pred_i = model.predict()[i, :]
            tests[i] = np.allclose(pred_i, Yhoo[i, :])  # should close!
            # uncomment to see in detail
            print(Yhoo[i, :], pred_i, Y[i, :])
        correct = np.all(tests)
        self.assertTrue(correct)

        
if __name__ == '__main__':
    unittest.main()
