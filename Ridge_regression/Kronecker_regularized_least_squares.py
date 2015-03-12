"""
Created on Tue Feb 17 2015
Last update: Thu Mar 12 2015

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
        self.n_u, self.n_v = Y.shape

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
                self._Delta.reshape((1, -1), order='F')) + regularisation)
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
        if return_Yhat:
            Yhat = (self._U * self._Sigma).dot(projected_labels_filtered)\
                    .dot((self._V*self._Delta).T)
            return Yhat

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
        n_u = len(hat_matrix_u)
        hat_matrix_v = (self._V*self._Delta/(self._Delta + reg_2)).dot(self._V.T)
        n_v = len(hat_matrix_v)
        leverages_u = np.diag(hat_matrix_u)
        partial_residues = (hat_matrix_u - np.diag(leverages_u)).dot(self._Y).\
                dot(hat_matrix_v)
        loov = (partial_residues.T/(1 - leverages_u)).T
        mse_loocv = np.mean( ( self._Y - loov )**2 )
        if mse and not preds:
            return mse_loocv
        elif not mse and preds:
            return loov
        elif mse and preds:
            return loov, mse_loocv

    def predict_LOOCV_columns_2SRLS(self, (reg_1, reg_2), preds=True, mse=False):
        """
        Uses LOOCV for the columns to estimate the mse,
        preds: return predictions
        mse: return mse estimated for LOOCV
        """
        hat_matrix_u = (self._U*self._Sigma/(self._Sigma + reg_1)).dot(self._U.T)
        n_u = len(hat_matrix_u)
        hat_matrix_v = (self._V*self._Delta/(self._Delta + reg_2)).dot(self._V.T)
        n_v = len(hat_matrix_v)
        leverages_v = np.diag(hat_matrix_v)
        partial_residues = (hat_matrix_u).dot(self._Y).\
                dot(hat_matrix_v - np.diag(leverages_v))
        loov = (partial_residues/(1 - leverages_v))
        mse_loocv = np.mean( ( self._Y - loov )**2 )
        if mse and not preds:
            return mse_loocv
        elif not mse and preds:
            return loov
        elif mse and preds:
            return loov, mse_loocv

    def predict_LOOCV_both_2SRLS(self, (reg_1, reg_2), preds=True, mse=False):
        """
        Uses LOOCV for new pairs to estimate the mse,
        preds: return predictions
        mse: return mse estimated for LOOCV
        """
        hat_matrix_u = (self._U*self._Sigma/(self._Sigma + reg_1)).dot(self._U.T)
        n_u = len(hat_matrix_u)
        leverages_u = np.diag(hat_matrix_u)
        hat_matrix_v = (self._V*self._Delta/(self._Delta + reg_2)).dot(self._V.T)
        n_v = len(hat_matrix_v)
        leverages_v = np.diag(hat_matrix_v)
        partial_residues = (hat_matrix_u - np.diag(leverages_u)).dot(self._Y).\
                dot(hat_matrix_v - np.diag(leverages_v))
        loov = (partial_residues/(1 - leverages_v))
        loov = (loov.T/(1 - leverages_u)).T
        mse_loocv = np.mean( ( self._Y - loov )**2 )
        if mse and not preds:
            return mse_loocv
        elif not mse and preds:
            return loov
        elif mse and preds:
            return loov, mse_loocv

    def predict_LOOCV_rows_KRLS(self, reg, preds=True, mse=False):
        """
        Uses a semi-efficient way to predict Y holdout for new rows using KRLS
        set mse to True to only get mse estimated by LOOCV
        """
        K_u = (self._U*self._Sigma).dot(self._U.T)  # recover kernel matrix for u
        K_v = (self._V*self._Delta).dot(self._V.T)  # recover kernel matrix for v
        n_instances = len(K_u)
        loov = np.zeros(self._Y.shape)
        for row_indice in range(n_instances):
            #  perform eigendecomposition of matrix with one instance removed
            Sigma_min_i, U_min_i = np.linalg.eigh(\
                    np.delete(np.delete(K_u, row_indice, 0), row_indice, 1))
            temp_model = KroneckerRegularizedLeastSquaresGeneral(\
                    np.delete(self._Y, row_indice, 0), U_min_i, Sigma_min_i,\
                    self._V, self._Delta)
            temp_model.train_model(reg, algorithm='KRLS')
            loov[row_indice] = temp_model.predict(\
                    np.delete(K_u[row_indice], row_indice, 0),\
                    K_v)
        mse_loocv = np.mean( (self._Y - loov)**2 )
        if mse and not preds:
            return mse_loocv
        elif not mse and preds:
            return loov
        elif mse and preds:
            return loov, mse_loocv

    def predict_LOPO(self, regularisation, algorithm, preds=True, mse=False):
        """
        Predicts for one pair out for the given algorithm and regularisation
        => could be made more efficient
        """
        n_u, n_v = self._Y.shape
        Yhat = self.train_model(regularisation, algorithm, return_Yhat=True)
        mod_eigvals = self._filtered_values * \
                np.dot(self._Sigma.reshape((-1,1)),self._Delta.reshape((1,-1)))
        leverages = (self._U**2).dot(mod_eigvals).dot(self._V.T**2)
        looe = (self._Y - Yhat)/(1 - leverages)
        if mse:
            mse_loocv = np.mean( (looe)**2 )
        if preds:
            loov = self._Y - looe
        if mse and not preds:
            return mse_loocv
        elif not mse and preds:
            return loov
        elif mse and preds:
            return loov, mse_loocv

    def LOOCV_model_selection_2SRLS(self, reg_1_grid, reg_2_grid, verbose=False,\
            method='rows'):
        """
        Seaches for the combination of parameters for two-step RLS that give
        the best generalization error for given grids.
        Provided stategies for cross validation:
            - pairs: leave one pair out
            - rows: leave one row out
            - columns: leave one column out
            - both: leave one colums and row out
        """
        self.best_performance_LOOCV = 1e10
        for reg_1 in reg_1_grid:
            for reg_2 in reg_2_grid:
                if method ==  'rows':
                    performance = self.predict_LOOCV_rows_2SRLS((reg_1, reg_2),\
                            preds=False, mse=True)
                elif method ==  'columns':
                    performance = self.predict_LOOCV_columns_2SRLS((reg_1, reg_2),\
                            preds=False, mse=True)
                elif method ==  'both':
                    performance = self.predict_LOOCV_both_2SRLS((reg_1, reg_2),\
                            preds=False, mse=True)
                elif method ==  'pairs':
                    performance = self.predict_LOPO((reg_1, reg_2),\
                            preds=False, mse=True, algorithm='2SRLS')
                if verbose: print 'Regulariser pair (%s, %s) gives MSE of %s'\
                        %(reg_1, reg_2, performance)
                if performance < self.best_performance_LOOCV:
                    self.best_performance_LOOCV = performance
                    self.best_regularisation = (reg_1, reg_2)
        self.train_model(self.best_regularisation, algorithm='2SRLS')

    def LOOCV_model_selection_KRLS(self, reg_grid, verbose=False, method='rows'):
        """
        Seaches for the hyperparameter for Kronecker RLS that gives
        the best generalization error for a given grid.
        Provided stategies for cross validation:
            - pairs: leave one pair out
            - rows: leave one row out
        """
        self.best_performance_LOOCV = 1e10
        for reg in reg_grid:
            if method == 'rows':
                performance = self.predict_LOOCV_rows_KRLS(reg, mse=True, preds=False)
            elif method == 'pairs':
                performance = self.predict_LOPO(reg,\
                            preds=False, mse=True, algorithm='KRLS')
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
        self._U = U[:,Sigma>1e-15]  # eigenvectors first type of objects
        self._V = V[:,Delta>1e-15]  # eigenvectors second type of objects
        self._Sigma = Sigma[Sigma > 1e-15]  # eigenvalues of first type of objects
        self._Delta = Delta[Delta > 1e-15]  # eigenvalues of second type of objects
        self.n_u, self.n_v = Y.shape

    def train_model(self, regularisation, algorithm='2SRLS', return_Yhat=False):
        self.spectral_filter(regularisation, return_values=False,
                algorithm=algorithm)
        projected_labels_filtered = self._filtered_values * self._U.T.dot(\
                self._Y.dot(self._V))
        self._A =  self._U.dot(projected_labels_filtered).dot(self._V.T)
        self.model_norm = np.sum(np.dot(self._Sigma.reshape((-1, 1)),\
                self._Delta.reshape((-1, 1)).T)*projected_labels_filtered**2)
        self.model_norm = self.model_norm**0.5
        if return_Yhat:
            Yhat = (self._U * self._Sigma).dot(projected_labels_filtered)\
                    .dot((self._V*self._Delta).T)
            return Yhat

    def get_parameters(self):
        """
        Returns the estimated parameter vector/matrix
        """
        return self._A

    def predict(self, K_u_new, K_v_new):
        """
        Make new prediction for U_new and V_new
        """
        return K_u_new.dot(self._A.dot(K_v_new.T))

if __name__ == "__main__":
    import random as rd

    # number of objects
    n_u = 100
    n_v = 100

    # dimension of objects
    p_u = 18
    p_v = 1000

    noise = 1

    X_u = np.random.randn(n_u, p_u)
    K_u = np.dot(X_u, X_u.T)

    X_v = np.random.randn(n_v, p_v)
    K_v = np.dot(X_v, X_v.T)

    W = np.random.randn(p_u, p_v)

    Y = X_u.dot(W.dot(X_v.T)) + np.random.randn(n_u, n_v)*noise

    KRLS = KroneckerRegularizedLeastSquares(Y, K_u, K_v)
    KRLS.train_model((0.1, 0.1))

    Yhat = KRLS.predict(K_u, K_v)

    print np.mean((Y-Yhat)**2)

    print 'Testing for two-step RLS'

    KRLS.LOOCV_model_selection_2SRLS([10**i for i in range(-10, 10)],\
        [10**i for i in range(-5, 5)], verbose = True, method='both')

    print 'Estimated row CV error: %s'\
            %KRLS.best_performance_LOOCV



    print KRLS.best_regularisation
    KRLS.train_model(KRLS.best_regularisation)
    X_u_new = np.random.randn(n_u, p_u)
    Ynew = X_u_new.dot(W.dot(X_v.T)) + np.random.randn(n_u, n_v)*noise

    Yhat_new = KRLS.predict(X_u_new.dot(X_u.T), K_v)
    print np.mean((Ynew-Yhat_new)**2)

    print 'Testing for Kronecker ridge regression'

    KRLS.LOOCV_model_selection_KRLS([10**i for i in range(-2, 5)],\
            verbose = True)

    print KRLS.best_performance_LOOCV
    print KRLS.best_regularisation

    Yhat_new = KRLS.predict(X_u_new.dot(X_u.T), K_v)
    print np.mean((Ynew-Yhat_new)**2)


    #########################################
    ##              Test LOOCV             ##
    #########################################

    # number of objects
    n_u = 50
    n_v = 60

    # dimension of objects
    p_u = 187
    p_v = 100

    noise = 1

    X_u = np.random.randn(n_u, p_u)
    K_u = np.dot(X_u, X_u.T)

    X_v = np.random.randn(n_v, p_v)
    K_v = np.dot(X_v, X_v.T)

    W = np.random.randn(p_u, p_v)
    Y = X_u.dot(W.dot(X_v.T)) + np.random.randn(n_u, n_v)*noise

    r1 = 250
    r2 = 190

    # rows
    KRLS = KroneckerRegularizedLeastSquares(Y, K_u, K_v)
    KRLS.train_model((r1,r2))
    row_HO_ther = KRLS.predict_LOOCV_rows_2SRLS((r1, r2))[0, :]

    KRLS_HO = KroneckerRegularizedLeastSquares(Y[1:], K_u[1:][:,1:], K_v)
    KRLS_HO.train_model((r1, r2), '2SRLS')
    row_HO_exp = KRLS_HO.predict(K_u[0, 1:], K_v)
    print 'rows: must be the same:', np.allclose(row_HO_ther, row_HO_exp)

    # columns
    KRLS = KroneckerRegularizedLeastSquares(Y, K_u, K_v)
    KRLS.train_model((r1,r2))
    col_HO_ther = KRLS.predict_LOOCV_columns_2SRLS((r1, r2))[:, 0]

    KRLS_HO = KroneckerRegularizedLeastSquares(Y[:,1:], K_u, K_v[1:][:,1:])
    KRLS_HO.train_model((r1, r2), '2SRLS')
    col_HO_exp = KRLS_HO.predict(K_u, K_v[0, 1:])
    print 'columns: must be the same:', np.allclose(col_HO_ther, col_HO_exp)

    # both
    KRLS = KroneckerRegularizedLeastSquares(Y, K_u, K_v)
    KRLS.train_model((r1,r2))
    both_HO_ther = KRLS.predict_LOOCV_both_2SRLS((r1, r2))[0, 0]

    KRLS_HO = KroneckerRegularizedLeastSquares(Y[1:,1:], K_u[1:][:,1:], K_v[1:][:,1:])
    KRLS_HO.train_model((r1, r2), '2SRLS')
    both_HO_exp = KRLS_HO.predict(K_u[0, 1:], K_v[0, 1:])
    print 'both: must be the same:', np.allclose(both_HO_ther, both_HO_exp)

    """
    #########################################
    ##      Test conditional raning        ##
    #########################################

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
