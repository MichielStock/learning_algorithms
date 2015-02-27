"""
Created on Fri Jan 2 2015
Last update: Mon Jan 5 2015

@author: Michiel Stock
michielfmstock@gmail.com

Implementation of the rubust principal component analysis
Details according to Lin et al. 2013?
"""

import numpy as np

from numpy.linalg import svd, norm
from numpy import inf

def max_Frob_inf_norm(matrix, lambd):
    """
    returns max( ||Y||_2, lambd^-2 * ||Y||_inf)
    """
    return max(norm(matrix, ord=2), norm(matrix, ord=inf)/lambd)

def soft_thresholding(x, epsilon):
    """
    Returns
        x - epsilon, if x > epsilon
        x + epsilon, if x < - epsilon
        0, else
    """
    return (x > epsilon)*(x - epsilon) + (x < -epsilon)*(x + epsilon)

def RPCA_inexact(observation_matrix, lambd=1.0, max_iter=1000,
        epsilon=0.1, rho=1.2, verbose=False):
    """
    Decomposes obeservation_matrix in a low-rank and sparse matrix
        observation_matrix: real matrix (n1 x n2)
        lambd: tuning parameter (good value is (n1*n2)**-0.25
        max_iter: maximum number of iterations
        epsilon: error till convergence
        rho: inflation factor
        verbose: is equal to true gives value of target for each step
    """
    Y_k = observation_matrix/max_Frob_inf_norm(observation_matrix, lambd)
    E_k = np.zeros(Y_k.shape)
    A_k = np.zeros(Y_k.shape)
    mu_k = 0.1
    iteration = 0
    f_old = 1e10
    difference = 1e10
    while difference > epsilon and iteration < max_iter:
        f_new = np.trace(A_k.dot(A_k.T)) + lambd*norm(E_k, ord=1)
        U, S, VT = svd(observation_matrix - E_k + Y_k/mu_k, full_matrices=False)
        A_k = np.dot(U*soft_thresholding(S, 1.0/mu_k), VT)
        # just multiply instead of
        # transforming into a diagonal matrix and dot product
        E_k = soft_thresholding(observation_matrix - A_k + Y_k/mu_k, lambd/mu_k)
        Y_k += mu_k*(observation_matrix - A_k - E_k)
        mu_k *= rho
        difference = np.abs(f_old - f_new)
        f_old = f_new
        iteration += 1
        if verbose: print 'Iteration %i, f-value is %s'%(iteration, f_new)
    if iteration >= max_iter:
        # note that algorithm did not converge
        print 'Passed maximum number of iterations!'
    return A_k, E_k

def RPCA_matrix_completion(partial_observations, observed, lambd=1.0,
        max_iter=1000, epsilon=0.1, rho=1.2, verbose=False):
    # for analogy previous algorithm
    observation_matrix = partial_observations + 0
    observation_matrix[observed] = 0  # set missing values to zero
    Y_k = observation_matrix/max_Frob_inf_norm(observation_matrix, lambd)
    E_k = np.zeros(Y_k.shape)
    A_k = np.zeros(Y_k.shape)
    mu_k = 0.1
    iteration = 0
    f_old = 1e10
    difference = 1e10
    while difference > epsilon and iteration < max_iter:
        f_new = np.trace(A_k.dot(A_k.T)) + lambd*norm(E_k, ord=1)
        U, S, VT = svd(observation_matrix - E_k + Y_k/mu_k, full_matrices=False)
        A_k = np.dot(U*soft_thresholding(S, 1.0/mu_k), VT)
        # just multiply instead of
        # transforming into a diagonal matrix and dot product
        E_k = soft_thresholding(observation_matrix - A_k + Y_k/mu_k, lambd/mu_k)
        E_k[not observed] = 0
        Y_k += mu_k*(observation_matrix - A_k - E_k)
        mu_k *= rho
        difference = np.abs(f_old - f_new)
        f_old = f_new
        iteration += 1
        if verbose: print 'Iteration %i, f-value is %s'%(iteration, f_new)
    if iteration >= max_iter:
        # note that algorithm did not converge
        print 'Passed maximum number of iterations!'
    return A_k, E_k

if __name__ == "__main__":

    # first test: synthetic data

    n_1 = 1000
    n_2 = 750
    rank = 4
    f = 0.01

    # generate low-rank matrix
    D = np.random.rand(n_1, rank).dot(np.random.rand(n_2, rank).T)

    # add random sparse noise
    D += np.random.binomial(1, f, (n_1, n_2))*10

    baseline_lambda = (n_1*n_2)**-0.25
    A, E = RPCA_inexact(D, 0.01, max_iter=5000, verbose=True)
    print A
    print E


    import matplotlib.pyplot as plt

    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(8, 4))

    ax0.imshow(D, cmap='hot')
    ax0.set_title('Original')

    ax1.imshow(A, cmap='hot')
    ax1.set_title('Low rank')

    ax2.imshow(E, cmap='hot')
    ax2.set_title('Sparse')

    # remove ugly ticks...
    for ax in (ax0, ax1, ax2):
        ax.tick_params(axis='both', which='both',
        bottom='off',top='off',labelbottom='off', left='off')

    #fig.show()
    fig.savefig('Toy_example.pdf')

    # second test: real image

    def RGB_to_matrix(RGB):
        matrix = np.concatenate([RGB[:,:,i] for i in range(3)], axis=0)
        return np.log(matrix + 1.0)

    def matrix_to_RGB(matrix):
        n,m = matrix.shape
        RGB = np.zeros((n/3, m, 3))
        for i in range(3):
           RGB[:,:,i] =  matrix[i*(n/3):(i+1)*(n/3),:]
        return np.exp(RGB - 1.0)

    import matplotlib.image as mpimg
    img=mpimg.imread('bookshelf.jpg')

    D = img[:,:,0]*0.299 + 0.587*img[:,:,1] +0.114*img[:,:,2]

    original_shape = img.shape

    #D = RGB_to_matrix(img)  # turn on for complete image
    A, E = RPCA_inexact(D, 0.1, max_iter=500, verbose=True, rho=1.05)

    fig, (ax0, ax1, ax2) = plt.subplots(ncols=3, figsize=(8, 4))

    ax0.imshow(D, cmap='hot')
    ax0.set_title('Original')

    ax1.imshow(A, cmap='hot')
    ax1.set_title('Low rank')

    ax2.imshow(E, cmap='hot')
    ax2.set_title('Sparse')

    # remove ugly ticks...
    for ax in (ax0, ax1, ax2):
        ax.tick_params(axis='both', which='both',
        bottom='off',top='off',labelbottom='off', left='off')

    fig.savefig('Bookshelf_decomposition.pdf')

    """
    mpimg.imsave('bookshelf_low_rank.jpeg', matrix_to_RGB(A))
    mpimg.imsave('bookshelf_sparse.jpeg', matrix_to_RGB(E))
    """
