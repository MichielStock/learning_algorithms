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


class BernoulliPDF():
    """
    Bernoulli distribution for a vector
    """
    def __init__(self, x, n_vars, n_components, marg_flags=None):
        self.n_components = n_components
        self._params = theano.shared(np.random.randn(n_components, n_vars))
        prob_succes = T.nnet.sigmoid(self._params)
        log_odds = T.log( prob_succes / (1 - prob_succes) )
        self.log_pdf = log_odds * x + T.log( 1 - prob_succes )
        self.pdf = T.exp( self.log_pdf )
        self.type = 'pdf_node'

    def get_parameters(self):
        """
        Returns parameters of the pdf
        """
        parameters = [self._params]
        return parameters

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


if __name__ == "__main__":
    import random as rd
    import matplotlib.pyplot as plt

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

    tes_size = 4
    tesselation = [indices[i*tes_size : (i+1)*tes_size] for i in range(100/tes_size)]

    pdf_layer = BernoulliPDF(x, 100, 5)

    prod_layer1 = [ProductNode(pdf_layer, tes) for tes in tesselation]
    sum_layer1 = [SumNode(node, 4) for node in prod_layer1]

    prod_layer2 = [ProductNode(sum_layer1[i*5:(i+1)*5]) for i in range(5)]
    sum_layer2 = [SumNode(node, 4) for node in prod_layer2]

    final_prod_node = ProductNode(sum_layer2)
    final_sum_node = SumNode(final_prod_node, 1)

    function_map = theano.function([x], [sum_layer1[0].log_max_output])
    function_map(X[[0]])
