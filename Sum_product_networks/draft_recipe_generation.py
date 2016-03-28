import theano
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import random as rd

dataset = np.random.binomial(1, 0.2, size=(50000, 300))

def floatX(x):
    return np.asarray(x, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape)))

def generate_random_choices(n_choices, n_vars, n_elements):
    '''
    Generate n_choices choice matrice for n_vars var discribed by
    n_elements distributions
    '''
    return [[rd.randint(0, n_elements-1) for var in range(n_vars)]\
            for i in range(n_choices)]


n_distributions = 10  # for each ingredient, number of distributions
n_products = 20  # number of intial random combinations for products
n_sum_combinations = 20  # number of intial sum combinations
size_sums = 10  # consider mixtures of this size
stack_size = 5  # this number of layers
n_variables = dataset.shape[1]

x = T.TensorType(dtype='floatX', broadcastable=(True, False))('x')

p_pars_uncompr = init_weights((n_distributions, n_variables))
weights = [p_pars_uncompr]

p_pars = T.nnet.sigmoid(p_pars_uncompr)

pfm_components = p_pars**x*(1-p_pars**(1-x))



choices_prod = generate_random_choices(n_products, n_products, n_distributions)
choices_sums = generate_random_choices(n_sum_combinations, n_products, size_sums)

tree_topology = [(choices_prod, choices_sums)]

product_layer = [T.prod([pfm_components[i,p] for i,p in enumerate(ch)])\
        for ch in choices_prod]

weights_layer = init_weights((n_sum_combinations, size_sums))



sum_layer = [T.max([prod * T.abs_(weights_layer[sum_row, sum_col]) for\
        sum_col, prod in enumerate(ch)])/T.sum(T.abs_(weights_layer[sum_row, :]))\
        for sum_row, ch in enumerate(choices_sums)]
