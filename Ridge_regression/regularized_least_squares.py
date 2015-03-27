"""
Created on Wed Nov 2 2014
Last update: Fri Mar 6 2015

@author: Michiel Stock
michielfmstock@gmail.com
Implementations of the RLS methods for one object
"""

import numpy as np

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
        Yhat = (self._U * self._Sigma*(self._Sigma + l)**-1).dot(self._U.T)\
                .dot(self._Y)
        self.model_norm = np.sum((self._Y.T.dot(self._U) / (self._Sigma + l))**2\
                * self._Sigma)
        self.model_norm = np.sqrt( self.model_norm )
        self.mse_train = np.mean((Yhat-self._Y)**2)  # Frobenius

    def get_parameters(self):
        """
        Returns the estimated parameter vector/matrix
        """
        if hasattr(self, '_W'):
            return self._W
        else: raise AttributeError

    def set_labels(self, Y):
        """
        Change the labels for the learner
        """
        self._Y = Y

    def predict_HOO(self, val_inds, l = 1.0, predictions = True, MSE = False):
        """
        Makes a prediction for the instances indexed by val_inds, by using a
        model trained by all the remaining instances
        Always provide a regularization parameter
        returns: estimated labels for tuning instances, parameters estimated
        on the train set
        """
        train_inds = [i for i in range(self._N) if i not in val_inds]
        reg_eigvals = (self._Sigma + 1.0)
        U_val = self._U[val_inds]
        U_train = self._U[train_inds]
        B_block = np.dot(U_val / reg_eigvals, U_train.T).T
        C_block = np.linalg.inv(np.dot(U_val / reg_eigvals, U_val.T))
        A = np.dot(U_train / reg_eigvals, U_train.T)
        K_val = np.dot(U_val * self._Sigma, U_train.T)
        Hat_HO = K_val.dot(A - B_block.dot(C_block).dot(B_block.T))
        loo_values = Hat_HO.dot(Y[train_inds])
        if MSE:
            loo_mse = np.mean( (loo_values - self._Y[val_inds])**2 )
            if predictions:
                return predictions, loo_mse
            else:
                return loo_mse
        elif predictions:
            return loo_values

    def predict_LOOCV(self, l = 1.0, predictions = True, MSE = False):
        """
        Makes a prediction for each instance by means of leave-one-out
        cross validation
        returns the estimated labels for each instance (if predictions is True)
        and/or estimated mean squared error (if MSE is True)
        """
        Hat = (self._U*(self._Sigma/(self._Sigma + l))).dot(self._U.T)
        leverages = np.diag(Hat)
        upper_part = self._Y - Hat.dot(self._Y)
        loo_residuals = (upper_part.T/(1 - leverages)).T
        if MSE:
            MSE = np.mean((loo_residuals)**2)
        if predictions and MSE:
            return self._Y - loo_residuals, MSE
        if predictions and not MSE:
            return self._Y - loo_residuals
        if not predictions and MSE:
            return MSE
        else:
            print 'You have to specify to calculate something!'

    def LOOCV_model_selection(self, l_grid, verbose=False):
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
            if verbose: print 'Lambda = %s (MSE = %s)' %(l, MSE_l)
        if verbose: print 'Best lambda of %s gives a mse of %s'\
                %(best_lambda, best_MSE)
        self.train_model(best_lambda)
        return best_lambda, best_MSE

    def get_norm(self):
        """
        Returns the norm of the trained model
        """
        return self.model_norm


class RegularizedLeastSquares(RegularizedLeastSquaresGeneral):
    """
    Implementation of the standard regularized least squares,
    also known as ridge regression, only in primal form
    Input: a vector or matrix Y containing the labels, a matrix X with
    the features
    """
    def __init__(self, Y, X):
        """
        Initialization of the instances, works with decomposition of X
        Sigma contains the SQUARED eigenvalues
        """
        U, Sigma_sqrt, VT = np.linalg.svd(X, full_matrices=False)
        self._X = X
        self._Y = Y
        self._U = U
        self._V = VT.T  # use transpose to be algebraic correct
        self._Sigma = Sigma_sqrt**2  # need square for implementation
        self._N, self._P = U.shape

    def train_model(self, l = 1.0):
        """
        Trains model and calculates mse
        """
        assert np.all(l) > 0  # ensure numerical stability
        # apply the spectral filter on the eigenvalues
        self._W = (self._V * (self._Sigma + l)**-1).dot(self._V.T).dot(np.dot(self._X.T, self._Y))
        Yhat = self._X.dot(self._W)
        self.mse_train = np.mean((Yhat-self._Y)**2)  # Frobenius
        self.model_norm = np.sum((self._Y.T.dot(self._U) / (self._Sigma + l))**2\
                * self._Sigma)
        self.model_norm = np.sqrt( self.model_norm )

    def predict(self, Xnew):
        return np.dot(Xnew, self._W)


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
        assert self._N == K.shape[0] and self._N == K.shape[1]
        # perform decomposition of X
        self._Sigma, self._U = np.linalg.eigh(K)
        self._U = self._U[:, self._Sigma > 1e-8]
        self._Sigma = self._Sigma[self._Sigma > 1e-8]

    def train_model(self, l = 1.0):
        """
        Trains model and calculates mse
        """
        assert np.all(l) > 0  # ensure numerical stability
        # apply the spectral filter on the eigenvalues
        self._A = (self._U/(self._Sigma + l)).dot(np.dot(self._U.T, self._Y))
        Yhat = (self._U*self._Sigma/(self._Sigma + l)).dot(np.dot(self._U.T, self._Y))
        self.mse_train = np.mean((Yhat-self._Y)**2)  # Frobenius
        self.model_norm = np.sum((self._Y.T.dot(self._U) / (self._Sigma + l))**2\
                * self._Sigma)
        self.model_norm = np.sqrt( self.model_norm )

    def predict(self, Knew):
        return np.dot(Knew, self._A)

    def get_parameters(self):
        """
        Returns the estimated parameter vector/matrix
        """
        if hasattr(self, '_A'):
            return self._A
        else: raise AttributeError

if __name__ == "__main__":
    import random as rd

    test_linear = True
    test_kernel = True

    n = 1000  # number of instances
    p = 10  # number of features
    k = 2  # number of tasks

    X = np.random.randn(n, p)
    w = np.random.rand(p, k)*5
    y = np.dot(X, w) + np.random.randn(n, k)

    if test_linear:
        RLS = RegularizedLeastSquares(y, X)
        RLS.train_model(0.1)
        for i in range(p):
            print w[i], RLS.get_parameters()[i]

        """
        HO_set = rd.sample(range(n), n/10)
        Yhat = RLS.predict_HOO(HO_set, 0.01)
        print Yhat.shape
        for i in range(n/10):
            print Yhat[i], y[HO_set[i]]
        """

        CVpreds = RLS.predict_LOOCV()
        for i in range(n):
            print CVpreds[i], y[i]

        print RLS.LOOCV_model_selection([10**i for i in range(-5, 5)])
        YhatF = RLS.predict(X)

    #########################################
    ##              Test LOOCV             ##
    #########################################

        RLS = RegularizedLeastSquares(y, X)
        RLS.train_model(l = 5)
        hoo_ther = RLS.predict_LOOCV(l = 0.001)[0]
        RLS_ho = RegularizedLeastSquares(y[1:], X[1:])
        RLS_ho.train_model(l = 0.001)
        ho_exp = RLS_ho.predict(X[0])

        print 'Should be the same:',np.allclose(hoo_ther, ho_exp)


    if test_kernel:
        K = X.dot(X.T)
        KRLS = KernelRegularizedLeastSquares(y, K)
        KRLS.train_model(1)
        YhatK = KRLS.predict(K)
        print KRLS.get_parameters()
        indices = range(n)
        rd.shuffle(indices)
        '''
        HO_set = indices[:n/10]
        HI_set = indices[n/10:]
        YHOO =  KRLS.predict_HOO(HO_set, 10)

        KRLS_HO = KernelRegularizedLeastSquares(y[HI_set],\
                X[HI_set].dot(X[HI_set].T))
        KRLS_HO.train_model(10)
        YHOO_exp = KRLS_HO.predict(X[HO_set].dot(X[HI_set].T))
        for i in range(n/10):
            print y[HO_set[i]], YHOO[i], YHOO_exp[i]
        '''
        print KRLS.LOOCV_model_selection([10**i for i in range(-5, 5)],\
                verbose=True)

    # test on Synthetic data
    from sklearn.datasets import make_regression
    from sklearn.metrics import mean_squared_error
    from sklearn.ensemble import RandomForestRegressor

    X, Y, coef = make_regression(n_samples=200, n_features=50, n_informative=10, n_targets=10, bias=0.0, effective_rank=20, tail_strength=0.5, noise=1, shuffle=True, coef=True)

    RLS = RegularizedLeastSquares(Y[:100], X[:100])
    RLS.LOOCV_model_selection([10**i for i in range(-5, 5)],verbose=True)
    Yhat_rls = RLS.predict(X[100:])
    mse_rls = mean_squared_error(Y[100:], Yhat_rls)

    RFR = RandomForestRegressor(n_estimators=50)
    RFR.fit(X[:100], Y[:100])
    Yhat_rls = RFR.predict(X[100:])
    mse_rfr = mean_squared_error(Y[100:], Yhat_rls)

    print 'LOOCV and HOO are the same:', np.allclose(RLS.predict_LOOCV()[0], RLS.predict_HOO([0]))
