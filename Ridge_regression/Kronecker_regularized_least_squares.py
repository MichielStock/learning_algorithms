"""
Created on Tue Feb 17 2015
Last update: Mon Mar 2 2015

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
            self._filtered_values = 1/np.dot((self._Sigma.reshape((-1, 1), order='F')\
                + lambda_u), (self._Delta.reshape((-1, 1), order='F') + lambda_v).T)
        elif algorithm == 'KRLS':
            self._filtered_values = 1/(np.dot(self._Sigma.reshape((-1, 1), order='F'),\
                self._Delta.reshape((-1, 1), order='F').T) + regularisation)
        else:
            raise KeyError
        if return_values:
            return self._filtered_values

    def train_model(self, regularisation, algorithm='2SRLS', return_Yhat=False):
        self.spectral_filter(regularisation, return_values=False,
                algorithm=algorithm)
        projected_labels_filtered = self._filtered_values * self._U.T.dot(\
                self._Y.dot(self._V))
        self._W =  self._U.dot(projected_labels_filtered).dot(self._V.T)
        self.model_norm = np.sum(np.dot(self._Sigma.reshape((-1, 1)),\
                self._Delta.reshape((-1, 1)).T)*projected_labels_filtered**2)
        self.model_norm = self.model_norm**0.5

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

    def predict_LOOCV_rows_2SRLS(self, (reg_1, reg_2), preds=True, mse=False):
        """
        Uses LOOCV for the rows to estimate the mse,
        preds: return predictions
        mse: return mse estimated for LOOCV
        """
        hat_matrix_u = (self._U*self._Sigma/(self._Sigma + reg_1)).dot(self._U.T)
        hat_matrix_v = (self._V*self._Delta/(self._Delta + reg_2)).dot(self._V.T)
        leverages = np.diag(hat_matrix_u)
        Yhat = np.dot(hat_matrix_u, self._Y).dot(hat_matrix_v)
        residual_HOO = (((self._Y-Yhat).T/(1-leverages))).T
        mse_loocv = np.mean(residual_HOO**2)
        if mse and not preds:
            return mse_loocv
        elif not mse and preds:
            return self._Y - residual_HOO
        elif mse and preds:
            return self._Y - residual_HOO, mse_loocv

    def predict_LOOCV_colums_2SRLS(self, (reg_1, reg_2), preds=True, mse=False):
        """
        Uses LOOCV for the colums to estimate the mse,
        preds: return predictions
        mse: return mse estimated for LOOCV
        """
        hat_matrix_u = (self._U*self._Sigma/(self._Sigma + reg_1)).dot(self._U.T)
        hat_matrix_v = (self._V*self._Delta/(self._Delta + reg_2)).dot(self._V.T)
        leverages = np.diag(hat_matrix_v)
        Yhat = np.dot(hat_matrix_u, self._Y).dot(hat_matrix_v)
        residual_HOO = (((self._Y-Yhat)/(1-leverages)))
        mse_loocv = np.mean(residual_HOO**2)
        if mse and not preds:
            return mse_loocv
        elif not mse and preds:
            return self._Y - residual_HOO
        elif mse and preds:
            return self._Y - residual_HOO, mse_loocv

    def predict_LOOCV_both_2SRLS(self, (reg_1, reg_2), preds=True, mse=False):
        """
        Uses LOOCV for the new objects to estimate the mse,
        preds: return predictions
        mse: return mse estimated for LOOCV
        """
        hat_matrix_u = (self._U*self._Sigma/(self._Sigma + reg_1)).dot(self._U.T)
        hat_matrix_v = (self._V*self._Delta/(self._Delta + reg_2)).dot(self._V.T)
        leverages_u = np.diag(hat_matrix_u)
        leverages_v = np.diag(hat_matrix_v)
        Yhat = np.dot(hat_matrix_u, self._Y).dot(hat_matrix_v)
        residual_HOO = (((self._Y-Yhat)/(1-leverages_v)))
        residual_HOO = (residual_HOO.T/(1-leverages_u)).T
        mse_loocv = np.mean(residual_HOO**2)
        if mse and not preds:
            return mse_loocv
        elif not mse and preds:
            return self._Y - residual_HOO
        elif mse and preds:
            return self._Y - residual_HOO, mse_loocv

    def predict_LOOCV_rows_KRLS(self, reg, mse=False):
        """
        Uses a semi-efficient way to predict Y holdout for new rows using KRLS
        set mse to True to only get mse estimated by LOOCV
        """
        K_u = (self._U*self._Sigma).dot(self._U.T)  # recover kernel matrix for u
        K_v = (self._V*self._Delta).dot(self._V.T)  # recover kernel matrix for v
        n_instances = len(K_u)
        Y_loocv = np.zeros(self._Y.shape)
        for row_indice in range(n_instances):
            #  perform eigendecomposition of matrix with one instance removed
            Sigma_min_i, U_min_i = np.linalg.eigh(\
                    np.delete(np.delete(K_u, row_indice, 0), row_indice, 1))
            temp_model = KroneckerRegularizedLeastSquaresGeneral(\
                    np.delete(self._Y, row_indice, 0), U_min_i, Sigma_min_i,\
                    self._V, self._Delta)
            temp_model.train_model(reg, algorithm='KRLS')
            Y_loocv[row_indice] = temp_model.predict(\
                    np.delete(K_u[row_indice], row_indice, 0),\
                    K_v)
        if mse:
            return np.mean((Y_loocv - self._Y)**2)
        else:
            return Y_loocv

    def LOOCV_model_selection_2SRLS(self, reg_1_grid, reg_2_grid, verbose=False):
        """
        Seaches for the combination of paramters that give the best generalization
        error where both items are new
        """

        self.best_performance_LOOCV = 1e10
        for reg_1 in reg_1_grid:
            for reg_2 in reg_2_grid:
                performance = self.predict_LOOCV_both_2SRLS((reg_1, reg_2),\
                        preds=False, mse=True)
                print 'Regulariser pair (%s, %s) gives MSE of %s'\
                        %(reg_1, reg_2, performance)
                if performance < self.best_performance_LOOCV:
                    self.best_performance_LOOCV = performance
                    self.best_regularisation = (reg_1, reg_2)
        self.train_model(self.best_regularisation, algorithm='2SRLS')


    def LOOCV_model_selection_KRLS(self, reg_grid, verbose=False):
        self.best_performance_LOOCV = 1e10
        for reg in reg_grid:
            performance = self.predict_LOOCV_rows_KRLS(reg, mse=True)
            if performance < self.best_performance_LOOCV:
                self.best_performance_LOOCV = performance
                self.best_regularisation = reg
            if verbose:
                print 'Regulariser: %s gives MSE of %s' %(reg, performance)
        self.train_model(self.best_regularisation, algorithm='KRLS')

    def get_norm(self):
        """
        Returns the norm of the trained model
        """
        return self.model_norm

    def __str__(self):
        print 'Kronecker RLS model'
        print 'Dimensionality of label matrix is (%s, %s)' %self._Y.shape
        if hasattr(self, '_filtered_values'):
            # to do: add errors both side
            #print 'MSE train is %s' %self.mse_train
            print 'Model norm is %s' %self.model_norm
        return ''


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
    n_u = 100
    n_v = 200

    # dimension of objects
    p_u = 180
    p_v = 1000

    noise = 10

    X_u = np.random.randn(n_u, p_u)
    K_u = np.dot(X_u, X_u.T)

    X_v = np.random.randn(n_v, p_v)
    K_v = np.dot(X_v, X_v.T)

    W = np.random.randn(p_u, p_v)

    Y = X_u.dot(W.dot(X_v.T)) + np.random.randn(n_u, n_v)*noise
    #Y = X_u.dot(np.random.randn(p_u, n_v)) + np.random.randn(n_u, n_v)*noise






    KRLS = KroneckerRegularizedLeastSquares(Y, K_u, K_v)
    KRLS.train_model((0.1, 0.1))


    # testing LOOCV
    row_HO_ther = KRLS.predict_LOOCV_rows_2SRLS((1e-7, 1e-7))[0]
    KRLS_HO = KroneckerRegularizedLeastSquares(Y[1:], K_u[1:][:, 1:], K_v)
    KRLS_HO.train_model((1e-7, 1e-7), '2SRLS')
    row_HO_exp = KRLS_HO.predict(K_u[0, 1:], K_v)


    #KRLS.train_model(0.1, algorithm='KRLS')
    Yhat = KRLS.predict(K_u, K_v)

    print np.mean((Y-Yhat)**2)

    print 'Testing for two-step RLS'

    KRLS.LOOCV_model_selection_2SRLS([10**i for i in range(-10, 10)],\
        [10**i for i in range(-5, 5)], verbose = True)

    print 'Estimated row CV error: %s'\
            %KRLS.predict_LOOCV_rows_2SRLS(KRLS.best_regularisation, preds=False,\
                    mse=True)


    print KRLS.best_performance_LOOCV
    print KRLS.best_regularisation
    KRLS.train_model((1, 1))
    X_u_new = np.random.randn(n_u, p_u)
    Ynew = X_u_new.dot(W.dot(X_v.T)) + np.random.randn(n_u, n_v)*noise

    Yhat_new = KRLS.predict(X_u_new.dot(X_u.T), K_v)
    print np.mean((Ynew-Yhat_new)**2)

    print 'Testing for Kronecker ridge regression'

    KRLS.LOOCV_model_selection_KRLS([10**i for i in range(-2, 10)],\
            verbose = True)

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
