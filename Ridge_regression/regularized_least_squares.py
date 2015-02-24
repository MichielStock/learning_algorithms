"""
Created on Wed Nov 2 2014
Last update: Mon Jan 7 2015

@author: Michiel Stock
michielfmstock@gmail.com

Implementations of the RLS methods for one object
"""

import numpy as np

def svd_long_matrix(matrix, epsilon=1e-5):
    """
    Performs a singular value decomposition of a matrix with more
    rows than columns
    returns U, Esq, VE
    """
    XTX = np.dot(matrix.T, matrix)
    Esq, V = np.linalg.eigh(XTX)
    V = V[:, Esq>epsilon]
    Esq = Esq[Esq>epsilon]
    VE = V*(Esq**0.5)
    del XTX  # remove to free space
    U = X.dot(VE)/Esq
    return U, Esq, VE


class RegularizedLeastSquaresGeneral:
    """
    Implementation of the regularized least squares,
    using only the eigendecomposition
    Input: a vector or matrix Y containing the labels, a matrix X with
    the features
    """
    def __init__(self, Y, X, Q = None, R = None):
        """
        Initialization of the instances, works with decomposition of X
        Optionally, give Q (or its decomposition R such that Q = R^TR,
        a positive semi-definite matrix to weight the observations
        """
        self._Y = Y
        if Q:
            E, R = np.linalg.eigh(Q)
            R = R[:, E>0]*(E[E>0]**0.5)
        if R:
            print 'WARNING: Overwriting features and labels for reweighting!'
            self._R = R
            Y = R.dot(Y)
            X = R.dot(X)
        self._N, self._P = X.shape
        # perform decomposition of X
        # eigvects are the left eigenvectors
        # eigvals are the eigenvalues of XTX
        # projection matrix is such that
        #       X = eigvects * projection_matrix
        self._eigvects, self._eigvals, self._projection_matrix = svd_long_matrix(X)


    def spectral_filter(self, regularisation, return_values = False):
        """
        Filters the eigenvalues using the following filter function:
            f(sigma_i) = 1/(sigma_i + l_i)
        with sigma_i the ith eigenvalue and l_i the elements from the regularisation
        (can be fixed)
        """
        self._filtered_values = 1.0/(self._eigvals + regularisation)
        if return_values:
            return self._filtered_values

    def train_model(self, l = 1.0, Q = None, R = None):
        """
        Calculate the weight vectors for a given regularization term
        Can solve a more general function, with weights for
        regularization:
            J(w) = (y - Xw)^TQ(y - Xw) + w^TLw
        with L = diag(l) and Q = R^TR (R is orthogonal)
        (none-diagonal weightings not supported)
        If a Q or R is given (to reweight the instances, the problem is solved
        accordingly (Q is positive definite)
        """
        assert np.all(l) > 0  # ensure numerical stability
        # apply the spectral filter on the eigenvalues
        self.spectral_filter(l)
        self._regularization = l  # last lambda(matrix) used for training
        # estimate training error
        Yhat = (self._eigvects*(self._filtered_values*self._eigvals)).dot(
                np.dot(self._eigvects.T, self._Y))
        self._mse_train = np.linalg.norm(Yhat-self._Y, ord = 2)**2  # Frobenius

    def generate_W(self):
        '''
        Generates the matrix W, used for predictions
        '''
        self._W = self._projection_matrix.dot(np.diag(self._filtered_values)).dot(
                np.dot(self._eigvects.T, self._Y))

    def get_parameters(self):
        """
        Returns the estimated parameter vector/matrix
        """
        if not hasattr(self, '_W'):
            self.generate_W()
        return self._W

    def set_labels(self, Y):
        """
        Change the labels for the learner
        """
        self._Y = Y

    def predict(self, X_new):
        """
        Make new prediction, X_new can be features or a kernel matrix of
        appropriate size
        """
        if not hasattr(self, '_W'):
            self.generate_W()
        return X_new.dot(self._W)

    def predict_HOO(self, val_inds, l = None):
        """
        Makes a prediction for the instances indexed by val_inds, by using a
        model trained by all the remaining instances
        Optinally give a new lamda parameter
        returns: estimated labels for tuning instances, parameters estimated
        on the train set
        """
        if l:
            self.spectral_filter(l)
        else:
            l = self._regularization
        number_inds = len(val_inds)
        mask_train_indices = np.ones(self._N, dtype=bool)
        mask_train_indices[val_inds] = False  # mask for training indices
        eigenvectors_HO = self._eigvects[val_inds]
        P = eigenvectors_HO*self._filtered_values*self._eigvals
        submatrix = np.linalg.inv(
                P.dot(eigenvectors_HO.T) - np.eye(number_inds) )
        predictions_HOO = (P + eigenvectors_HO.dot(P.T).dot(submatrix).dot(P)).dot(np.dot(self._eigvects[
                mask_train_indices].T, self._Y[mask_train_indices]))
        return predictions_HOO

    def predict_LOOCV(self, l = None, predictions = True, MSE = False):
        """
        Makes a prediction for each instance by means of leave-one-out
        cross validation
        returns the estimated labels for each instance (if predictions is True)
        and/or estimated mean squared error (if MSE is True)
        """
        if l:
            self.train_model(l)
        else:
            l = self._regularization
        leverages = ((self._eigvects*(self._eigvals*self._filtered_values
                )*self._eigvects)).sum(1)
        self._leverages = leverages
        Yhat = (self._eigvects*(self._filtered_values*self._eigvals)).dot(
                np.dot(self._eigvects.T, self._Y))
        if predictions:
            LOOCVpreds = self._Y - (self._Y - Yhat)/(
                    1-np.reshape(leverages, (-1,1)))
        if MSE:
            MSE = np.mean( ((self._Y - Yhat)/(
                    1-np.reshape(leverages, (-1,1))))**2)
        if predictions and MSE:
            return LOOCVpreds, MSE
        if predictions and not MSE:
            return LOOCVpreds
        if not predictions and MSE:
            return MSE
        else:
            print 'You have to specify to calculate something!'


    def LOOCV_model_selection(self, l_grid):
        """
        Does model model selection based on LOOCV estimated MSE for all the
        possible regularization parameters from l_grid
        returns best lambda and best MSE and trains the model according to
        the best found lambda
        """
        best_lambda = 1
        best_MSE = 1e100
        for l in l_grid:
            MSE_l = self.predict_LOOCV(l, predictions = False,
                MSE = True)
            if MSE_l < best_MSE:
                best_lambda = l
                best_MSE = MSE_l
        print 'Best lambda = %s (MSE = %s)' %(best_lambda, best_MSE)
        self.train_model(best_lambda)
        return best_lambda, best_MSE

    def __repr__(self):
        print 'General RLS model'
        print 'Dimensionality of output is %s' %self._Y.shape[1]
        if hasattr(self, '_filtered_values'):
            print 'MSE train is %s' %self._mse_train
            print 'MSE LOOCV is %s' %self.predict_LOOCV(predictions = False,
                                                        MSE = True)
            print 'Regularization is %s' %np.mean(self._regularization)
        return ''

class RegularizedLeastSquares(RegularizedLeastSquaresGeneral):
    """
    Implementation of the standard regularized least squares,
    also known as ridge regression, only in primal form
    Input: a vector or matrix Y containing the labels, a matrix X with
    the features
    """


class KernelRegularizedLeastSquares(RegularizedLeastSquaresGeneral):
    """
    Modification of the standard regularized least squares to cope
    with kernels as feature descriptions.
    This module works by simple calculating a decomposition of the kernel
    matrix using this in primal form
    Input: a vector or matrix Y (N x K) containing the labels, a matrix K
    (N x N) with the kernel values
    """
    def __init__(self, Y, K):
        """
        Initialization of the instances
        automatically performs a decomposition on the kernel matrix
        """
        self._Y = Y
        self._N = Y.shape[0]
        assert self._N == K.shape[0] and self._N == K.shape[0]
        # perform decomposition of X
        eigvals, eigvects = np.linalg.eigh(K)
        # remove eigenvalues smaller than epsilon
        epsilon = 1e-10
        self._eigvals = eigvals[eigvals>epsilon]
        self._eigvects = eigvects[:,eigvals>epsilon]

    def generate_W(self):
        '''
        Generates the matrix W, used for predictions
        In kernel terms usually denoted as A:
        Y = Knew W
        '''
        self._W = self._eigvects.dot(np.diag(self._filtered_values)).dot(
                np.dot(self._eigvects.T, self._Y))




if __name__ == "__main__":
    import random as rd

    test_linear = True
    test_kernel = True

    n = 1000  # number of instances
    p = 10  # number of features
    k = 2  # number of tasks

    X = np.random.randn(n, p)
    w = np.random.rand(p, k)*5
    y = np.dot(X, w) + np.random.randn(n, k)/2

    if test_linear:
        RLS = RegularizedLeastSquares(y, X)
        RLS.train_model(0.1)
        for i in range(p):
            print w[i], RLS.get_parameters()[i]

        HO_set = rd.sample(range(n), n/10)
        Yhat = RLS.predict_HOO(HO_set, 0.01)
        print Yhat.shape

        for i in range(n/10):
            print Yhat[i], y[HO_set[i]]

        CVpreds = RLS.predict_LOOCV()
        for i in range(n):
            print CVpreds[i], y[i]

        print RLS.LOOCV_model_selection([10**i for i in range(-5, 5)])
        YhatF = RLS.predict(X)

    if test_kernel:
        K = X.dot(X.T)
        KRLS = KernelRegularizedLeastSquares(y, K)
        KRLS.train_model(1)
        YhatK = KRLS.predict(K)
        print KRLS.get_parameters()
        HO_set = rd.sample(range(n), n/10)
        YHOO =  KRLS.predict_HOO(HO_set, 1)
        for i in range(n/10):
            print y[HO_set[i]], YHOO[i]

        print KRLS.LOOCV_model_selection([10**i for i in range(-5, 5)])
