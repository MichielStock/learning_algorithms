"""
Created on Wed Nov 2 2014
Last update: Fri Feb 27 2015

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
    def __init__(self, Y, U, Sigma, V):
        """
        Initialization of the instances, works with decomposition of X
        Sigma contains the SQUARED eigenvalues
        """
        self._Y = Y
        self._U = U
        self._V = V
        self._Sigma = Sigma
        self._N, self._P = U.shape

    def train_model(self, l = 1.0):
        """
        Trains model and calculates mse
        """
        assert np.all(l) > 0  # ensure numerical stability
        # apply the spectral filter on the eigenvalues
        print 'This is only for theoretical purposes'
        print 'Cannot do predictions'
        Yhat = (self._U * self._Sigma*(self._Sigma + l)**-1).dot(self._U.T)\
                .dot(self._Y)
        self.mse_train = np.mean((Yhat-self._Y)**2)  # Frobenius

    def get_parameters(self):
        """
        Returns the estimated parameter vector/matrix
        """
        if not hasattr(self, '_W'):
            return self._W
        else: raise AttributeError

    def set_labels(self, Y):
        """
        Change the labels for the learner
        """
        self._Y = Y

    '''
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
        '''

    def predict_LOOCV(self, l = None, predictions = True, MSE = False):
        """
        Makes a prediction for each instance by means of leave-one-out
        cross validation
        returns the estimated labels for each instance (if predictions is True)
        and/or estimated mean squared error (if MSE is True)
        """
        Hat = self._U*(self._Sigma/(self._Sigma + l)**-1).dot(self._U.T)
        leverages = np.diag(Hat)
        upper_part = self._Y - self._U.dot(self._U.T) - Hat * l
        loo_residuals = (upper_part.T/(1 - leverages))
        if MSE:
            MSE = np.mean((loo_residuals)**2)
        if predictions and MSE:
            return self._Y + loo_residuals, MSE
        if predictions and not MSE:
            return self._Y + loo_residuals
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

    def __str__(self):
        print 'General RLS model'
        print 'Dimensionality of output is %s' %self._Y.shape[1]
        if hasattr(self, '_filtered_values'):
            print 'MSE train is %s' %self.mse_train
            print 'MSE LOOCV is %s' %self.predict_LOOCV(predictions = False,
                                                        MSE = True)
            print 'Regularization is %s' %np.mean(self._regularization)
            print 'Model norm is %s' %self.model_norm
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
    test_kernel = False

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

    # test if LOOCV works
    RLS = RegularizedLeastSquares(y, X)
    RLS.train_model(l = 5)
    hoo_ther = RLS.predict_LOOCV(l = 5)[0]
    RLS_ho = RegularizedLeastSquares(y[1:], X[1:])
    RLS_ho.train_model(l = 5)
    ho_exp = RLS_ho.predict(X[0])
