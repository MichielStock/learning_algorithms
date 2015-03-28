"""
Created on Fri Mar 20 2015
Last update: Sat Mar 21 2015

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
    return theano.shared(floatX(np.ones((shape))))

def univar_normal(x, mu=np.random.randn(), sigma=1.0):
    """
    Returns a Theano univariate normal PDF (unnormalized)
    """
    mu_shared = theano.shared(mu)
    sigma_shared = theano.shared(sigma)
    norm_pdf = 1/T.sqrt(2*np.pi*sigma**2)*T.exp(-(x - mu_shared)**2/sigma_shared**2/2)
    return norm_pdf, [mu_shared, sigma_shared]


# define the variables, and associated distributions

# two dimensional problem
X1 = T.TensorType(dtype='float64', broadcastable=(False, True))('X1')
X2 = T.TensorType(dtype='float64', broadcastable=(False, True))('X2')
# assume three clusters

n_clusters = 10

mu_X1 = theano.shared(np.random.randn(1, n_clusters), broadcastable=(True,False))
sigma_X1 = theano.shared(np.ones((1, n_clusters)) * 10, broadcastable=(True,False))

mu_X2 = theano.shared(np.random.randn(1, n_clusters), broadcastable=(True,False))
sigma_X2 = theano.shared(np.ones((1, n_clusters)) * 10, broadcastable=(True,False))



# make network, three weights

sum_weights = init_weights(n_clusters)

pdf_values_X1 = T.exp(-(X1 - mu_X1)**2/sigma_X1**2)/T.sqrt(2*np.pi*sigma_X1**2)
pdf_values_X2 = T.exp(-(X2 - mu_X2)**2/sigma_X2**2)/T.sqrt(2*np.pi*sigma_X2**2)

top_sum_value = (pdf_values_X1 * pdf_values_X2 * T.abs_(sum_weights)).sum(1)
top_sum_value_MAP = (pdf_values_X1 * pdf_values_X2 * T.abs_(sum_weights)).max(1)
normalization = T.sum(T.abs_(sum_weights))

log_MAP = T.sum(T.log(top_sum_value/normalization))

params = [sum_weights, mu_X1, sigma_X1, mu_X2, sigma_X2]
learning_rate = 0.1

complexity_penalty = 0.00000001 * T.sum(map(lambda x: T.mean(T.abs_(x)), params)) # L1

gradient_params  = T.grad(log_MAP + complexity_penalty, params)


train = theano.function([X1, X2], log_MAP, updates =\
            [(p, p + learning_rate * g) for p, g in zip(params, gradient_params)])

# generate a random dataset

from sklearn.datasets import make_blobs, make_moons
from matplotlib.mlab import griddata
import matplotlib.pyplot as plt

X, Y = make_blobs(n_samples=100, centers=7, cluster_std=1)
#X, y = make_moons(n_samples=1000, noise = 10)

for iteration in range(10000):
    print train(X[:,0].reshape((-1,1)), X[:,1].reshape((-1,1)))

map_value = theano.function((X1, X2), - T.log(top_sum_value/normalization))

x1 = np.arange(X[:,0].min(), X[:,0].max(), 0.1)
x2 = np.arange(X[:,1].min(), X[:,1].max(), 0.1)
X1_vals, X2_vals = np.meshgrid(x1, x2)

MAP_loglikelihood = np.zeros(X1_vals.shape)

for j in range(X1_vals.shape[1]):
    temp = map_value(X1_vals[:,j].reshape((-1,1)), X2_vals[:,j].reshape((-1,1)))
    for i in range(X2_vals.shape[0]):
        MAP_loglikelihood[i,j] = temp[i]

CS = plt.contourf(X1_vals, X2_vals, MAP_loglikelihood)
plt.clabel(CS, inline=1, fontsize=10, cmap='hot')
plt.colorbar()
plt.scatter(X[:,0], X[:,1])
plt.show()

for par in params:
    print par.get_value()
