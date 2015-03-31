"""
Created on Fri Mar 27 2015
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Implementation of the modules needed for a som product network
"""

import theano
import theano.tensor as T
import numpy as np
import random as rd

def generate_random_choices(n_choices, n_vars, n_elements):
    '''
    Generate n_choices choice matrice for n_vars var discribed by
    n_elements distributions, useful for generating network
    topology
    '''
    return [[rd.randint(0, n_elements-1) for var in range(n_vars)]\
            for i in range(n_choices)]

vector = T.vector()
f_soft_max = theano.function([vector], T.nnet.softmax(vector))
scalar = T.dscalar()
f_sigmoid = theano.function([scalar], T.nnet.sigmoid(scalar))


def weighted_random_choice(list_to_choose_from, unstandardised_weights):
    """
    Weighted sample from a list
    """
    weights = f_soft_max(unstandardised_weights)
    random_number = rd.random()
    value = 0.0
    i = 0
    while value <= random_number:
        element = list_to_choose_from[i]
        value += weights[0,i]
        i += 1
    return element


class SumNode():
    def __init__(self, product_nodes):
        self._product_nodes = product_nodes
        self._sum_size = len(product_nodes)  # number of elements to combine
        self._sum_weights = theano.shared(np.random.randn(self._sum_size))
        self._normalized_weigths = T.nnet.softmax(self._sum_weights)
        self.sum_output = T.sum(T.mul([node.sum_output for node in product_nodes], self._normalized_weigths))
        self.max_output = T.max(T.mul([node.max_output for node in product_nodes], self._normalized_weigths))
        self.sum_output_marg = T.sum(T.mul([node.sum_output_marg for node in product_nodes], self._normalized_weigths))
        self.max_output_marg = T.max(T.mul([node.max_output_marg for node in product_nodes], self._normalized_weigths))
        #self.argmax_output = T.argmax(T.mul(product_nodes, self._normalized_weigths))

    def get_parameters(self):
        """
        Returns parameters of this model and all lower models!
        """
        parameters = set([self._sum_weights])
        # get the parameters of the lower models
        for prod in self._product_nodes:
            parameters = parameters.union(prod.get_parameters())
        return list(parameters)

    def sample(self):
        """
        Samples from the distribution
        """
        choice = weighted_random_choice(self._product_nodes, self._sum_weights.get_value())
        return choice.sample()


class ProductNode():
    def __init__(self, inputs_nodes, initial_layer=False):
        self._product_size = len(inputs_nodes)  # number of elements to combine
        if initial_layer:
            self.sum_output = T.mul(*[inp.pdf for inp in inputs_nodes])
            self.max_output = self.sum_output
            self.sum_output_marg = T.mul(*[inp.pdf_marg for inp in inputs_nodes])
            self.max_output_marg = self.sum_output_marg
        else:
            self.sum_output = T.mul(*[inp.sum_output for inp in inputs_nodes])
            self.max_output = T.mul(*[inp.max_output for inp in inputs_nodes])
            self.sum_output_marg = T.mul(*[inp.sum_output_marg for inp in inputs_nodes])
            self.max_output_marg = T.mul(*[inp.max_output_marg for inp in inputs_nodes])
        self._inputs = inputs_nodes
        self._initial_flag = initial_layer

    def get_parameters(self):
        """
        Returns parameters of this model and all lower models!
        """
        parameters = set([])
        # get the parameters of the lower models
        for prod in self._inputs:
            parameters = parameters.union(prod.get_parameters())
        return list(parameters)

    def output(self):
        '''
        Returns the output of this node
        '''
        return self.product_output

    def sample(self):
        """
        Samples from the distribution
        """
        return [inp.sample() for inp in self._inputs]


class BernoulliPDF():
    """
    Bernoulli distribution for one variable
    """
    def __init__(self, x, marginal_flag=None):
        self._params = theano.shared(np.random.randn())
        self._prob_succes = T.nnet.sigmoid(self._params)
        self.pdf = T.mul((self._prob_succes**x), (1.0 - self._prob_succes)**(1-x))
        self.pdf_marg = T.mul((self._prob_succes**x), (1.0 - self._prob_succes)**(1-x))
        if marginal_flag:
            self.pdf_marg **= marginal_flag

    def get_parameters(self):
        """
        Returns parameters of the pdf
        """
        parameters = [self._params]
        return parameters

    def sample(self):
        p = f_sigmoid(self._params.get_value())
        return np.random.binomial(1, p)


class NormalPDF():
    """
    Univariate normal distribution
    """
    def __init__(self, x, marginal_flag=None):
        self._params = theano.shared(np.random.randn(2))
        self.pdf = T.exp(-(x - self._params[0])**2/(2*self._params[1]**2))/\
                T.sqrt(2*np.pi*self._params[1]**2)
        self.pdf_marg = T.exp(-(x - self._params[0])**2/(2*self._params[1]**2))/\
                T.sqrt(2*np.pi*self._params[1]**2)
        if marginal_flag:
            self.pdf_marg **= marginal_flag


    def get_parameters(self):
        """
        Returns parameters of the pdf
        """
        parameters = [self._params]
        return parameters

    def sample(self):
        standard_random = np.random.randn()
        return (standard_random + self._params.get_value()[0])/\
                np.abs(self._params.get_value()[1])


if __name__ == "__main__":
    x1 = T.scalar('x1')
    x2 = T.scalar('x2')

    marg1 = T.scalar()
    marg2 = T.scalar()

    pdf_x1 = [BernoulliPDF(x1, marg1) for i in range(10)]
    pdf_x2 = [NormalPDF(x2) for i in range(10)]

    products_l1 = [ProductNode([pdf_x1[i], pdf_x2[i]], initial_layer=True) for i in range(10)]

    prod_f = theano.function([x1, x2], [prod.max_output for prod in products_l1])

    sum_l1 = SumNode([prod for prod in products_l1])

    max_f = theano.function([x1,x2], sum_l1.max_output)

    print max_f(0,-3.3)
    print sum_l1.sample()

    marg_f = theano.function([x1,x2, marg1], sum_l1.sum_output_marg)
    print marg_f(0, -1, 0)

    # note: use the function grad to find the map posterior
