"""
Created on Tue 13 Oct 2015
Last update: Wed 14 Oct 2015

@author: Michiel Stock
michielfmstock@gmail.com

Module for studying the effect of the regulatrization parameters for
pairwise problems
"""

from Kronecker_regularized_least_squares import KroneckerRegularizedLeastSquares
import numpy as np
from sklearn.metrics import auc

# some functions for measuring performance on labels
# --------------------------------------------------

# mean squared error
mse = lambda Y, P : np.mean( (Y -P)**2 )

# macro AUC
macro_auc = lambda Y, P : auc(Y.ravel() > 0, P.ravel())

# instance AUC
def instance_auc(Y, P):
    n, m = Y.shape
    return np.mean([auc(Y[i] > 0, P[i]) for i in range(n) if Y[i].var()])

# micro AUC
def micro_auc(Y, P):
    n, m = Y.shape
    return np.mean([auc(Y[:,i] > 0, P[:,i]) for i in range(m) if Y[:,i].var()])


def regularization_settinga_kkrr(model, reg_grid, perf_measure=mse):
    """
    Computes the LOOCV performance (setting A) for an array of regularization
    values for Kronecker kernel ridge regression, give a reg_grid (vector of
    lambda values) and optionally the performance measure (default is mean
    squared error)
    """
    performance = []
    for reg in reg_grid:
        Y_loocv = model.predict_LOPO(reg, 'KRLS')
        performance.append(perf_measure(model._Y, Y_loocv))
    return performance

def regularization_map_2srls(model, reg_grid, method='pairs', perf_measure=mse):
    """
    Computes the LOO performance for a two-step RLS model, given a cross
    validation setting:
        - pairs (default)
        - rows
        - columns
        - both
    and a given performance measure.
    """
    n_steps = len(reg_grid)
    performance = np.zeros((n_steps, n_steps))
    for i in range(n_steps):
        reg_rows = reg_grid[i]
        for j in range(n_steps):
            reg_cols = reg_grid[j]
            Y_loocv = model.predict_LOOCV_general_2SRLS(reg_rows, reg_cols,
                    method)
            performance[i,j] = perf_measure(model._Y, Y_loocv)
    return performance

if __name__ == '__main__':

    # Test on predicting pixels of a picture
    # features are the location of a picture, with a radial basis kernel

    from skimage.data import coffee
    from skimage.color import rgb2gray
    import matplotlib.pyplot as plt
    from math import exp

    def make_rbk_Gram(x, sigma=1.0):
        # makes radial basis kernel matrix
        K = np.zeros((len(x), len(x)))
        for i in range(len(x)):
            for j in range(i+1):
                K[i, j] = exp( -(x[i] - x[j])**2 / sigma**2)
                K[j, i] = K[i, j]
        return K

    image = rgb2gray(coffee())
    K_u = make_rbk_Gram(range(image.shape[0]), 4)
    K_v = make_rbk_Gram(range(image.shape[1]), 4)

    model = KroneckerRegularizedLeastSquares(image, K_u, K_v)

    reg_grid = np.logspace(-5, 5, 20)
    reg_mesh1, reg_mesh2 = np.meshgrid(reg_grid, reg_grid)

    perf_setA_kkr = regularization_settinga_kkrr(model, reg_grid)
    perf_setA_2SRLS = regularization_map_2srls(model, reg_grid, method='pairs',
                perf_measure=mse)
    perf_setD_2SRLS = regularization_map_2srls(model, reg_grid, method='both',
                perf_measure=mse)

    # Show the results!
    # -----------------

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(30, 30))

    # show image

    axes[0,0].imshow(image)

    # for Kronecker RLS

    axes[0, 1].plot(reg_grid, perf_setA_kkr)
    axes[0, 1].loglog()
    axes[0, 1].set_xlabel('lambda')
    axes[0, 1].set_ylabel('mean squared error')

    # for two-step RLS
    # setting A
    for i in range(2):
        axes[1, i].loglog()
        axes[1, i].set_xlabel('reg. cols')
        axes[1, i].set_xlabel('reg. rows')
        if i == 0:
            axes[1, i].set_title('Holdout pairs')
            performance = perf_setA_2SRLS
        else:
            axes[1, i].set_title('Holdout both')
            performance = perf_setD_2SRLS
        axes[1, i].contour(reg_mesh1, reg_mesh1, performance,
                V=np.linspace(performance.min(),performance.max(),num=10), colors='k',
                 linewidth=0.1)
        axes[1, i].contourf(reg_mesh1, reg_mesh2, performance,
                 cmap='hot')
        axes[1, i].colorbar()

    #cax = fig.add_axes([0.925, 0.1, 0.02, 0.8])
    #fig.colorbar(fig, cax=cax)
    fig.show()
