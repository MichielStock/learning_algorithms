"""
Created on Tue Feb 17 2015
Last update: Wed Feb 25 2015

@author: Michiel Stock
michielfmstock@gmail.com

Implementations of the kronecker RLS methods for pairs
"""

import numpy as np

class KroneckerRegularizedLeastSquaresGeneral:
    """
    General class for Kronecker-based linear methods
    """
    def __init__(self, Y, U, Sigma, V, Delta):
        self._Y = Y
        self._U = U  # eigenvectors first type of objects
        self._V = V  # eigenvectors second type of objects
        self._Sigma = Sigma  # eigenvalues of first type of objects
        self._Delta = Delta  # eigenvalues of second type of objects

    def spectral_filter(self, regularisation, return_values=False,
                algorithm='2SRLS'):
        """
        Filter function on the eigenvalues, can do standard Kronecker RLS (KRLS)
        or two-step RLS (2SRLS).
        If 2SRLS is used, regularisation contains a pair of regularisation
        parameters
        """
        if algorithm == '2SRLS':
            lambda_u, lambda_v = regularisation
            self._filtered_values = 1/np.dot((self._Sigma.reshape((-1, 1))\
                + lambda_u), (self._Delta.reshape((-1, 1)) + lambda_v).T)
        elif algorithm == 'KRLS':
            self._filtered_values = 1/(np.dot(self._Sigma.reshape((-1, 1)),\
                self._Delta.reshape((-1, 1)).T) + regularisation)
        else:
            raise KeyError
        if return_values:
            return self._filtered_values

    def train_model(self, regularisation, algorithm='2SRLS', return_Yhat=False):
        self.spectral_filter(regularisation, return_values=False,
                algorithm=algorithm)
        self._W =  self._U.dot(self._filtered_values * self._U.T.dot(
                self._Y.dot(self._V))).dot(self._V.T)

    def get_parameters(self):
        """
        Returns the estimated parameter vector/matrix
        """
        return self._W

    def predict(self, U_new, V_new):
        """
        Make new prediction for U_new and V_new
        """
        return U_new.dot(self._W.dot(V_new.T))

    def predict_LOOCV_rows_2SRLS(self, (reg_1, reg_2), mse=False):
        """
        Predict Y holdout for new rows using 2SRLS
        set mse to True to only get mse estimated by LOOCV
        """
        filtered_values = self.spectral_filter((reg_1, reg_2), return_values=True,
                algorithm='2SRLS')
        Yhat = ((self._U.dot(np.diag(self._Sigma)).dot(filtered_values*(self._U.T\
                .dot(self._Y).dot(self._V))))*self._Delta).dot(self._V.T)
        hat_u_diags = np.sum(self._U**2/(self._Sigma + reg_1)*self._Sigma)
        residual_HOO = ((Y-Yhat).T/(1-hat_u_diags)).T
        if mse:
            return np.sum(residual_HOO**2)
        else:
            return self._Y - residual_HOO

    def LOOCV_model_selection_2SRLS(self, reg_1_grid, reg_2_grid):
        self.best_performance_LOOCV = 1e10
        for reg_1 in reg_1_grid:
            for reg_2 in reg_2_grid:
                performance = self.predict_LOOCV_rows_2SRLS((reg_1, reg_2), mse=True)
                if performance < self.best_performance_LOOCV:
                    self.best_performance_LOOCV = performance
                    self.best_regularisation = (reg_1, reg_2)
                print reg_1, reg_2, performance
        self.train_model(self.best_regularisation, algorithm='2SRLS')

class KroneckerRegularizedLeastSquares(KroneckerRegularizedLeastSquaresGeneral):

    def __init__(self, Y, K_u, K_v, loss='squared_error'):
        self._Y = Y
        n_u, n_v = Y.shape
        if loss == 'squared_error':
            Sigma, U = np.linalg.eigh(K_u)
            Delta, V = np.linalg.eigh(K_v)
        elif loss == 'instance_conditional':
            C = np.eye(n_u) - np.ones((n_u, n_u))/n_u
            Sigma, U = np.linalg.eigh(C.dot(K_u))
            Delta, V = np.linalg.eigh(K_v)
            self._Y = np.dot(C, Y)
        elif loss == 'micro_conditional':
            C = np.eye(n_v) - np.ones((n_v, n_v))/n_v
            Sigma, U = np.linalg.eigh(K_u)
            Delta, V = np.linalg.eigh(np.dot(C, K_v))
            self._Y = np.dot(Y, C)
        self._U = U[:,Sigma>1e-12]  # eigenvectors first type of objects
        self._V = V[:,Delta>1e-12]  # eigenvectors second type of objects
        self._Sigma = Sigma[Sigma>1e-12]  # eigenvalues of first type of objects
        self._Delta = Delta[Delta>1e-12]  # eigenvalues of second type of objects

if __name__ == "__main__":
    import random as rd

    # number of objects
    n_u = 2600
    n_v = 4100

    # dimension of objects
    p_u = 200
    p_v = 100

    noise = 100

    X_u = np.random.randn(n_u, p_u)
    K_u = np.dot(X_u, X_u.T)

    X_v = np.random.randn(n_v, p_v)
    K_v = np.dot(X_v, X_v.T)

    W = np.random.randn(p_u, p_v)

    Y = X_u.dot(W.dot(X_v.T)) + np.random.randn(n_u, n_v)*noise
    #Y = X_u.dot(np.random.randn(p_u, n_v)) + np.random.randn(n_u, n_v)*noise

    KRLS = KroneckerRegularizedLeastSquares(Y, K_u, K_v)
    KRLS.train_model((0.1, 0.1))
    #KRLS.train_model(0.1, algorithm='KRLS')
    Yhat = KRLS.predict(K_u, K_v)

    print np.mean((Y-Yhat)**2)

    print KRLS.predict_LOOCV_rows_2SRLS((.1,.1), mse = True)

    KRLS.LOOCV_model_selection_2SRLS([10**i for i in range(-10, 10)], [10**i for i in range(-10, 10)])

    print KRLS.best_performance_LOOCV
    print KRLS.best_regularisation


    """
    from sklearn.metrics import roc_auc_score


    Y = Y > 0

    def micro_auc(Y, Yhat):
        n_u, n_v = Y.shape
        return np.mean([roc_auc_score(Y[:,i], Yhat[:,i]) for i in range(n_v)])

    def instance_auc(Y, Yhat):
        n_u, n_v = Y.shape
        return np.mean([roc_auc_score(Y[i], Yhat[i]) for i in range(n_u)])

    def macro_auc(Y, Yhat):
        return roc_auc_score(Y.reshape(-1), Yhat.reshape(-1))

    KRLS = KroneckerRegularizedLeastSquares(Y, K_u, K_v)
    KRLS.train_model((10, 10))
    Yhat = KRLS.predict(K_u, K_v)

    print
    print 'Regular MSE'
    print 'instance: %.5f' %instance_auc(Y, Yhat)
    print 'micro: %.5f' %micro_auc(Y, Yhat)
    print 'macro: %.5f' %macro_auc(Y, Yhat)
    print '='*50

    KRLS = KroneckerRegularizedLeastSquares(Y, K_u, K_v, loss='instance_conditional')
    KRLS.train_model((10, 10))
    Yhat = KRLS.predict(K_u, K_v)

    print
    print 'instance'
    print 'instance: %.5f' %instance_auc(Y, Yhat)
    print 'micro: %.5f' %micro_auc(Y, Yhat)
    print 'macro: %.5f' %macro_auc(Y, Yhat)
    print '='*50

    KRLS = KroneckerRegularizedLeastSquares(Y, K_u, K_v, loss='micro_conditional')
    KRLS.train_model((10, 10))
    Yhat = KRLS.predict(K_u, K_v)

    print
    print 'micro'
    print 'instance: %.5f' %instance_auc(Y, Yhat)
    print 'micro: %.5f' %micro_auc(Y, Yhat)
    print 'macro: %.5f' %macro_auc(Y, Yhat)
    print '='*50
    """
