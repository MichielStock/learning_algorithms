"""
Created on Wed Sep 07 2016
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

General model to analyse incomplete pairwise datasets
"""

import numpy as np
from Performance import auc, mse

def non_empty_fraction_mask(mask, fraction):
    frac_mask = np.zeros_like(mask)
    while not np.any(np.logical_and(mask, frac_mask)):
        frac_mask[:] = np.random.rand(*mask.shape) <= fraction
    return frac_mask

class PairwiseAnalysis:

    def __init__(self, model, mask):
        self.model = model
        self._mask_observed = mask
        self._Y = model._Y

    def imputation_study(self, fractions, n_repetitions=10,
                    performance_measure=mse):
        """
        Studies the performance when larger parts of the label matrix are
        missing.

        Inputs:
            - fractions : list with the fractions of the observed values used
            - n_repetitions : number of repetitions for each fractions
            - performance_measure : performance measure used (default is mse)

        Output:
            - performance_matrix : matrix with the performance for the fractions
                        (columns) and repetitions (rows)
        """
        performance_matrix = np.zeros((n_repetitions, len(fractions)))
        # the observed labels
        mask_observed = self._mask_observed
        # values to impute
        mask_impute = np.zeros_like(mask_observed)
        for j, fraction in enumerate(fractions):
            for i in range(n_repetitions):
                mask_subset = non_empty_fraction_mask(mask_observed, 1 - fraction)
                # mask to be imputed
                mask_impute[:] = np.logical_and(mask_subset,
                                        np.logical_not(mask_observed))
                p = self.model.impute_iter(mask_impute, Y=None,
                            max_iter=100)[mask_subset]
                y = self._Y[mask_subset]
                performance_matrix[i, j] = performance_measure(y, p)
        return performance_matrix

    def cross_validation_study(self, setting, performance_measure=mse):
        predicted_matrix = np.zeros_like(self._Y)
        n, m = Y.shape
        # the observed labels
        mask_observed = self._mask_observed
        mask_impute = np.zeros_like(mask_observed)
        if setting == 'B':  # CV on rows
            for i in range(n):
                mask_impute[:] = np.logical_not(mask_observed)
                mask_impute[i, :] = True
                predicted_matrix[i, :] = self.model.impute_iter(mask_impute,
                            Y=None, max_iter=100)[i, :]
        elif setting == 'C': # CV on columns
            for j in range(m):
                mask_impute[:] = np.logical_not(mask_observed)
                mask_impute[:, j] = True
                predicted_matrix[:, j] = self.model.impute_iter(mask_impute,
                            Y=None, max_iter=100)[:, j]
        elif setting == 'D':
            for i in range(n):  # CV on both
                for j in range(m):
                    if mask_observed[i, j]:
                        mask_impute[:] = np.logical_not(mask_observed)
                        mask_impute[i, j] = True
                        predicted_matrix[i, j] = self.model.impute_iter(mask_impute,
                                    Y=None, max_iter=100)[i, j]
        else:
            raise KeyError
        return mse(self._Y[mask_observed], predicted_matrix[mask_observed])



if __name__ == '__main__':
    Y = np.genfromtxt('Y.txt')
    Y = np.log10(Y)
    G = np.genfromtxt('targets_WS_normalized.txt')
    #G = np.log(G)
    K = np.genfromtxt('68study3_3D_SFS.txt')

    from TwoStepRidge import TwoStepRidgeRegression

    model = TwoStepRidgeRegression(Y, K, G)
    model.train_model((0.10, 0.10))

    mask = np.random.rand(*Y.shape) < 0.80  # retain 80% of the data

    pairwise_analysis_karaman = PairwiseAnalysis(model, mask)

    performance_matrix = pairwise_analysis_karaman.imputation_study(
                    [0.1, 0.5, 0.9])
