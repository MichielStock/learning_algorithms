# -*- coding: utf-8 -*-

"""
Created on Sun 21 Feb 2016
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Test of Karaman dataset
"""

from KroneckerRidge import KroneckerKernelRidgeRegression
from TwoStepRidge import TwoStepRidgeRegression
import numpy as np
import random as rd
import matplotlib.pyplot as plt



Y = np.genfromtxt('Y.txt').T
Y = np.log(Y)

indices = range(Y.shape[0])
rd.shuffle(indices)
train_indices = indices[:25]
test_indices = indices[400:]

G = np.genfromtxt('68study3_3D_SFS.txt')
K = np.genfromtxt('targets_WS_normalized.txt')

lambda_grid = np.logspace(-10, 10, 21)

# models
two_step = TwoStepRidgeRegression(Y[train_indices],
                    K[train_indices,:][:, train_indices], G)
ridge = TwoStepRidgeRegression(Y[train_indices],
                    K[train_indices,:][:, train_indices], np.eye(Y.shape[1]))
kronecker = KroneckerKernelRidgeRegression(Y[train_indices],
                    K[train_indices,:][:, train_indices], G)

two_step.tune_loocv(lambda_grid, 'A')
ridge.tune_loocv(lambda_grid, 'A')
kronecker.tune_loocv(lambda_grid, 'A')

print matrix_c_index(Y[test_indices], two_step.predict(K[test_indices,:][:, train_indices]))
print matrix_c_index(Y[test_indices], ridge.predict(K[test_indices,:][:, train_indices]))
print matrix_c_index(Y[test_indices], kronecker.predict(K[test_indices,:][:, train_indices]))