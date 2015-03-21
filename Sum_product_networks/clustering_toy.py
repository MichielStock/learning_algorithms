"""
Created on Fri Mar 20 2015
Last update: Fri Mar 20 2015

@author: Michiel Stock
michielfmstock@gmail.com

Toy example of clusting as a sum product network using Theano
"""

import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt

def floatX(x):
    return np.asarray(x, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.rand(shape)))

def univar_normal(x, mu=np.random.randn(), sigma=1.0):
    """
    Returns a Theano univariate normal PDF (unnormalized)
    """
    mu_shared = theano.shared(mu)
    sigma_shared = theano.shared(sigma)
    norm_pdf = 1/T.sqrt(2*np.pi*sigma**2)*T.exp(-(x - mu_shared)**2/sigma_shared**2)
    return norm_pdf, [mu_shared, sigma_shared]


# define the variables, and associated distributions

# two dimensional problem
X1 = T.scalar()
X2 = T.scalar()

# assume three clusters

parameters_pdf = []  # dump all the parameters of the pdfs
prob_dens_functions = []  #save the pdf

for clrnr in range(3):
    pdfs_group = []
    for X in [X1, X2]:
        pdf, params = univar_normal(X)
        pdfs_group.append(pdf)
        parameters_pdf += params
    prob_dens_functions.append(pdfs_group)

# make network, three weights

sum_weights = init_weights(3)

top_sum_value = T.sum(sum_weights * map(T.prod, prob_dens_functions))
top_sum_value_MAP = T.max(sum_weights * map(T.prod, prob_dens_functions))
normalization = T.sum(sum_weights)

log_MAP = T.log(top_sum_value_MAP/normalization)

gradient_weights  = T.grad(log_MAP, sum_weights)
gradient_params = T.grad(log_MAP, parameters_pdf)

train = theano.function([X1, X2], log_MAP, updates = [\
            (sum_weights, sum_weights + 1 * gradient_weights)] +
            [(p, p + 1 * g) for (p, g) in zip(parameters_pdf, gradient_params)])

# generate a random dataset

from sklearn.datasets import make_blobs
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt


X, Y = make_blobs(centers=3, cluster_std=0.7)

for iteration in range(5000):
    for x1, x2 in X:
        train(x1, x2)

map_value = theano.function((X1, X2), log_MAP)

X1_vals = np.arange(X[:,0].min(), X[:,0].max(), 0.1)
X2_vals = np.arange(X[:,1].min(), X[:,1].max(), 0.1)
MAP_loglikelihood = np.zeros((len(X1_vals), len(X2_vals)))

for i in range(len(X1_vals)):
    for j in range(len(X2_vals)):
        MAP_loglikelihood[i,j] = map_value(X1_vals[i], X2_vals[i])


fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].scatter(X[:,0], X[:,1])
ax[1].imshow(MAP_loglikelihood)

fig.show()
