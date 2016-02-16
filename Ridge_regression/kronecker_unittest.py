# -*- coding: utf-8 -*-
"""
Created on Tue Feb 17 2015
Last update: Tue Feb 17 2015

@author: Michiel Stock
michielfmstock@gmail.com

Implementations of the kronecker RLS methods for pairs
"""

import unittest
from KroneckerRidge import KroneckerKernelRidgeRegression
from TwoStepRidge import TwoStepRidgeRegression
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
            # print(Yhoo[i, :], pred_i, Y[i, :])
        correct = np.all(tests)
        self.assertTrue(correct)

class TestTwoStepRidge(unittest.TestCase):
    """
    Test Kronecker kernel ridge regression implementation
    """
    def test_prediction(self):
        """
        Test the prediction function
        using KroneckerKernelRidgeRegression.predict() using no arguments
        should return Yhat
        """
        Y, K, G, reg_1 = make_problem()
        reg_2 = np.random.rand() * 10
        model = TwoStepRidgeRegression(Y, K, G)
        model.train_model(regularization=(reg_1, reg_2))
        self.assertTrue((np.allclose(model.predict(), model.predict(K, G))))

    def test_prediction_correct(self):
        """
        Test if prediction is correct
        """
        Y, K, G, reg_1 = make_problem()
        reg_2 = np.random.rand() * 10
        N = Y.shape
        model = TwoStepRidgeRegression(Y, K, G)
        model.train_model(regularization=(reg_1, reg_2))
        P = model.predict()  # according to model
        Ptrue = K.dot(np.linalg.inv(K + reg_1 * np.eye(N[0]))).dot(Y).dot(
                        G.dot(np.linalg.inv(G + reg_2 * np.eye(N[1]))))
        self.assertTrue(np.allclose(P, Ptrue))

    def test_imputation_holdout(self):
        """
        Test for setting A, based on the idea that if you replace Yij with
        the predicted imputated value, this should not change the predicted
        value for Yhatij
        """
        Y, K, G, reg_1 = make_problem(10, 40)
        reg_2 = np.random.rand() * 10        
        model = TwoStepRidgeRegression(Y, K, G)
        Yhoo = model.lo_setting_A((reg_1, reg_2))
        tests = np.zeros(Y.shape, dtype=bool)
        for i in range(Y.shape[0]):
            for j in range(Y.shape[1]):
                Yhelp = Y.copy()
                Yhelp[i, j] = Yhoo[i, j]  # replacing this should not matter
                model._Y = Yhelp
                model.train_model((reg_1, reg_2))  # train again, with new Y
                pred_ij = model.predict()[i, j]
                tests[i, j] = np.allclose(pred_ij, Yhoo[i, j])  # should close!
                # uncomment to see in detail
                # print(Yhoo[i,j], pred_ij, Y[i,j])
        correct = np.all(tests)
        self.assertTrue(correct)
        
    """
    def test_settingB_holdout(self):
        Y, K, G, reg_1 = make_problem(10, 40)
        reg_2 = np.random.rand() * 10
        model = TwoStepRidgeRegression(Y, K, G)
        Yhoo = model.lo_setting_B((reg_1, reg_2))
        tests = np.zeros(Y.shape[0], dtype=bool)
        for i in range(Y.shape[0]):
            Yhelp = Y.copy()
            Yhelp[i, :] = Yhoo[i, :]  # replacing this should not matter
            model._Y = Yhelp
            model.train_model((reg_1, reg_2))  # train again, with new Y
            pred_i = model.predict()[i, :]
            tests[i] = np.allclose(pred_i, Yhoo[i, :])  # should close!
            # uncomment to see in detail
            # print(Yhoo[i, :], pred_i, Y[i, :])
        correct = np.all(tests)
        self.assertTrue(correct)
    """
    
    def test_one_line_imputation(self):
        Y, K, G, reg_1 = make_problem(10, 40)
        reg_2 = np.random.rand() * 10
        model = TwoStepRidgeRegression(Y, K, G)
        model2 = TwoStepRidgeRegression(Y[1:], K[1:,:][:,1:], G)
        model2.train_model(regularization=(reg_1, reg_2))
        Hoopreds = model2.predict(k = np.delete(K[[0],:], 0, axis=1))
        Hoocalc = model.lo_setting_B((reg_1, reg_2))[[0]]
        # print(Hoopreds, Hoocalc)
        self.assertTrue(np.allclose(Hoopreds, Hoocalc))
        
    def test_one_col_imputation(self):
        Y, K, G, reg_1 = make_problem(25, 40)
        reg_2 = np.random.rand() * 10
        model = TwoStepRidgeRegression(Y, K, G)
        model2 = TwoStepRidgeRegression(Y[:,1:], K, G[1:,:][:,1:])
        model2.train_model(regularization=(reg_1, reg_2))
        Hoopreds = model2.predict(g = np.delete(G[[0],:], 0, axis=1))
        Hoocalc = model.lo_setting_C((reg_1, reg_2))[:, 0]
        # print(Hoopreds, Hoocalc)
        self.assertTrue(np.allclose(Hoopreds.ravel(), Hoocalc))
        
    def test_D(self):
        Y, K, G, reg_1 = make_problem(20, 50)
        reg_2 = np.random.rand() * 10
        model = TwoStepRidgeRegression(Y, K, G)
        model2 = TwoStepRidgeRegression(Y[1:][:,1:], K[1:,:][:,1:],
                                        G[1:,:][:,1:])
        model2.train_model(regularization=(reg_1, reg_2))
        Hoopreds = model2.predict(k = np.delete(K[[0],:], 0, axis=1),
                                  g = np.delete(G[[0],:], 0, axis=1))[0, 0]
        Hoocalc = model.lo_setting_D((reg_1, reg_2))[0, 0]
        # print(Hoopreds, Hoocalc)
        self.assertTrue(np.allclose(Hoopreds.ravel(), Hoocalc))
        
        
if __name__ == '__main__':
    unittest.main()
