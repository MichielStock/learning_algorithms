"""
Created on Thu Apr 14 2015
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Implementation of the modules needed for a som product network
"""

import theano
import theano.tensor as T
import numpy as np
import random as rd

def make_random_partition(n_objects, n_part):
    """
    Makes a random partition of of size n_part a range of numbers 1 .. n_objects
    """
    indices = range( n_objects )
    rd.shuffle( indices )
    return [indices[i*n_objects/n_part:(i+1)*n_objects/n_part]\
            for i in range(n_part)]

class SumProductLayer():
    """
    A layer of product sums, independent sources are combined using products
    while weighted sums are taken of different versions of the same source
    """
    def __init__(self, n_components, input_size, partition):
        self.n_components = n_components
        self.partition = partition
        self.input_size = input_size
        self.n_combinations = len( partition )
        self.parameters = theano.shared(\
                np.random.randn(n_components,\
                self.n_combinations * input_size).astype(theano.config.floatX))

    def sum_output(self, inp):
        product_layer = [ T.prod(inp[:, tes], axis=1, keepdims=True)\
                for tes in self.partition]
        input_size = self.input_size
        sum_layer = []
        for i, prod in enumerate(product_layer):
            unscaled_weights = self.parameters[:, i*input_size:(i+1)*input_size]
            scaled_weights = T.nnet.softmax(unscaled_weights)
            sum_layer.append( T.dot( scaled_weights, prod ) )
        return T.concatenate( sum_layer, axis=1 )

    def max_output(self, inp):
        product_layer = [ T.prod(inp[:, tes], axis=1, keepdims=True)\
                for tes in self.partition]
        input_size = self.input_size
        max_layer = []
        for i, prod in enumerate(product_layer):
            prod = T.addbroadcast(prod.T, 0)
            unscaled_weights = self.parameters[:, i*input_size:(i+1)*input_size]
            scaled_weights = T.nnet.softmax(unscaled_weights)
            max_layer.append( T.max(scaled_weights * prod, axis=1, keepdims=True) )
        return T.concatenate( max_layer, axis=1 )


class SumProductNetwork():

    def __init__(self, pdf, n_components_layers, partitions):
        self.parameters = [ pdf.parameters ]
        self.pdf = pdf
        self.layers = []
        n_components = n_components_layers[0]
        self.layers.append( SumProductLayer(n_components,\
                pdf.n_components, partitions[0]) )  # initial layer
        # now add the other layers
        n_inputs = n_components
        for n_components, partition in zip(n_components_layers, partitions)[1:]:
            layer = SumProductLayer(n_components, n_inputs, partition)
            self.layers.append( layer )
            self.parameters.append( layer.parameters )
            n_inputs = n_components

    def sum_output(self):
        x = self.pdf.pdf
        for layer in self.layers:
            x = layer.sum_output(x)
        return x

    def max_output(self):
        x = self.pdf.pdf
        for layer in self.layers:
            x = layer.max_output(x)
        return x


class BernoulliPDF():
    """
    Bernoulli distribution for a vector
    """
    def __init__(self, x, n_vars, n_components, marg_flags=None):
        self.n_components = n_components
        self.n_vars = n_vars
        self.parameters = theano.shared(np.random.randn(n_components, n_vars))
        prob_succes = T.nnet.sigmoid(self.parameters)
        log_odds = T.log( prob_succes / (1 - prob_succes) )
        self.log_pdf = log_odds * x + T.log( 1 - prob_succes )
        self.pdf = T.exp( self.log_pdf )
        self.type = 'pdf_node'

    def get_parameters(self):
        """
        Returns parameters of the pdf
        """
        parameters = [self.parameters]
        return parameters

def gradient_updates_momentum(cost, params, learning_rate, momentum):
    '''
    Compute updates for gradient descent with momentum

    :parameters:
        - cost : theano.tensor.var.TensorVariable
            Theano cost function to minimize
        - params : list of theano.tensor.var.TensorVariable
            Parameters to compute gradient against
        - learning_rate : float
            Gradient descent learning rate
        - momentum : float
            Momentum parameter, should be at least 0 (standard gradient descent) and less than 1

    :returns:
        updates : list
            List of updates, one for each parameter
    '''
    # Make sure momentum is a sane value
    assert momentum < 1 and momentum >= 0
    # List of update steps for each parameter
    updates = []
    # Just gradient descent on cost
    for param in params:
        # For each parameter, we'll create a param_update shared variable.
        # This variable will keep track of the parameter's update step across iterations.
        # We initialize it to 0
        param_update = theano.shared(param.get_value()*0., broadcastable=param.broadcastable)
        # Each parameter is updated by taking a step in the direction of the gradient.
        # However, we also "mix in" the previous step according to the given momentum value.
        # Note that when updating param_update, we are using its old value and also the new gradient step.
        updates.append((param, param - learning_rate*param_update))
        # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
        updates.append((param_update, momentum*param_update + (1. - momentum)*T.grad(cost, param)))
    return updates

'''
class ProductNode():
    def __init__(self, input_nodes, initial_layer_indices=False):
        if initial_layer_indices:
            # if the pdfs are proved, also give the indices of the pdfs to use
            # input nodes is a single pdf
            self.log_sum_output = T.sum(input_nodes.log_pdf[\
                    :,initial_layer_indices], axis=0)
            self.log_max_output = self.log_sum_output
            self._inputs = [input_nodes]
            self.n_components = input_nodes.n_components
        else:
            components = [node.n_components for node in input_nodes]
            assert np.var( components ) == 0
            self.n_components = len(components)
            self.log_sum_output = T.sum([node.log_sum_output for node in input_nodes]\
                    , axis=0)
            self.log_max_output = T.sum([node.log_max_output for node in input_nodes]\
                    , axis=0)
            self._inputs = input_nodes
            self.type = 'product_node'

    def get_parameters(self):
        """
        Returns parameters of this model and all lower models!
        """
        parameters = set([])
        # get the parameters of the lower models
        for prod in self._inputs:
            parameters = parameters.union(prod.get_parameters())
        return list(parameters)


class SumNode():
    def __init__(self, input_nodes, n_components):
        self.n_components = input_nodes.n_components
        self._weights = theano.shared(np.random.randn(n_components, \
                input_nodes.n_components))
        scaled_weights = T.nnet.softmax(self._weights)
        pdf_values_sum = T.exp( input_nodes.log_sum_output )
        pdf_values_max = T.exp( input_nodes.log_max_output )
        pdf_values_max = pdf_values_max.reshape((1, -1))
        sum_output = T.dot( scaled_weights, pdf_values_sum )
        max_output = T.max( scaled_weights * T.addbroadcast(pdf_values_max, 0), axis=1 )
        self.log_sum_output = T.log( sum_output )
        self.log_max_output = T.log( max_output )
        self._inputs = input_nodes

    def get_parameters(self):
        """
        Returns parameters of this model and all lower models!
        """
        parameters = set([self._weights])
        # get the parameters of the lower models
        parameters = parameters.union(self._inputs.get_parameters())
        return list(parameters)
'''

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    plt.ion()
    from collections import deque

    X = []
    for i in range(500):
        x = []
        for j in range(10):
            x += rd.choice([ [0]*10, [1]*10])
        X.append(x)

    X = np.array(X, dtype=np.int8)

    x = T.drow('x')

    indices = range(100)
    rd.shuffle(indices)

    partition = make_random_partition(100, 10)
    correct_partition = [range(i*10, (i+1)*10) for i in range(10)]
    n_components = 20
    bern_pdf = BernoulliPDF(x, 100, n_components)

    n_components_layers = [n_components, n_components, 1]
    partitions = [correct_partition, make_random_partition(10, 5), [range(5)]]

    sum_product_network = SumProductNetwork(bern_pdf, n_components_layers, partitions)

    max_funct = theano.function([x], sum_product_network.max_output())

    print max_funct(X[[0]])

    parameters = sum_product_network.parameters
    cost = -T.mean(T.log(sum_product_network.max_output()))
    cost += 0.0001 * T.sum(map(T.mean, map(T.abs_, parameters)))  # L1 regularization
    log_likelihood = T.sum(T.log(sum_product_network.sum_output()))
    updates = gradient_updates_momentum(cost, parameters, 0.01, 0.95)

    train = theano.function([x], log_likelihood, updates=updates)

    # let us train !

    fig, axes = plt.subplots(nrows=1, ncols=2)

    axes[0].imshow(X)

    ax = axes[1]
    #ax.loglog()
    ax.grid()
    ax.set_xlabel('iteration')
    ax.set_ylabel('log likelihood')

    fig.show()

    train_indices = range(500)
    for iteration in range(1000):
        log_likehood_window = []
        rd.shuffle(train_indices)
        for indice in train_indices:
            log_likehood_window.append(train(X[[indice]]))
        print iteration, np.mean(log_likehood_window)
        ax.scatter(iteration, np.mean(log_likehood_window))
