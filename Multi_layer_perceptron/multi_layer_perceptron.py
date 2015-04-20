"""
Created on Sun Apr 19 2015
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Basic version of multi-layer artificial neural network in Theano
Based on the tutorial in:
http://nbviewer.ipython.org/github/craffel/theano-tutorial/blob/master/Theano%20Tutorial.ipynb
"""

import numpy as np
import matplotlib.pyplot as plt
import theano
# By convention, the tensor submodule is loaded as T
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

srng = RandomStreams()

def dropout(x, p=0.):
    """
    Randomly sets weights to 0 when training with probability p
    """
    if p > 0:
        retain_prob = 1 - p
        x *= srng.binomial(x.shape, p=retain_prob, dtype=theano.config.floatX)
        x /= retain_prob
    return x

class Layer(object):
    def __init__(self, W_init, b_init, activation):
        '''
        A layer of a neural network, computes s(Wx + b) where s is a nonlinearity and x is the input vector.

        :parameters:
            - W_init : np.ndarray, shape=(n_output, n_input)
                Values to initialize the weight matrix to.
            - b_init : np.ndarray, shape=(n_output,)
                Values to initialize the bias vector
            - activation : theano.tensor.elemwise.Elemwise
                Activation function for layer output
        '''
        # Retrieve the input and output dimensionality based on W's initialization
        n_output, n_input = W_init.shape
        # Make sure b is n_output in size
        assert b_init.shape == (n_output,)
        # All parameters should be shared variables.
        # They're used in this class to compute the layer output,
        # but are updated elsewhere when optimizing the network parameters.
        # Note that we are explicitly requiring that W_init has the theano.config.floatX dtype
        self.W = theano.shared(value=W_init.astype(theano.config.floatX),
                               # The name parameter is solely for printing purporses
                               name='W',
                               # Setting borrow=True allows Theano to use user memory for this object.
                               # It can make code slightly faster by avoiding a deep copy on construction.
                               # For more details, see
                               # http://deeplearning.net/software/theano/tutorial/aliasing.html
                               borrow=True)
        # We can force our bias vector b to be a column vector using numpy's reshape method.
        # When b is a column vector, we can pass a matrix-shaped input to the layer
        # and get a matrix-shaped output, thanks to broadcasting (described below)
        self.b = theano.shared(value=b_init.reshape(-1, 1).astype(theano.config.floatX),
                               name='b',
                               borrow=True,
                               # Theano allows for broadcasting, similar to numpy.
                               # However, you need to explicitly denote which axes can be broadcasted.
                               # By setting broadcastable=(False, True), we are denoting that b
                               # can be broadcast (copied) along its second dimension in order to be
                               # added to another variable.  For more information, see
                               # http://deeplearning.net/software/theano/library/tensor/basic.html
                               broadcastable=(False, True))
        self.activation = activation
        # We'll compute the gradient of the cost of the network with respect to the parameters in this list.
        self.params = [self.W, self.b]

    def output(self, x, dropout_frac=0):
        '''
        Compute this layer's output given an input

        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for layer input
            - dropout_frac : fractions of neurons to randomly remove

        :returns:
            - output : theano.tensor.var.TensorVariable
                Mixed, biased, and activated x
        '''
        # Compute linear mix
        if dropout_frac == 0:
            lin_output = T.dot(self.W, x) + self.b
        else:
            lin_output = dropout(T.dot(self.W, x) + self.b, p = dropout_frac)
        # Output is just linear mix if no activation function
        # Otherwise, apply the activation function
        return (lin_output if self.activation is None else self.activation(lin_output))


class MLP(object):
    def __init__(self, W_init, b_init, activations, dropout_fraction=0):
        '''
        Multi-layer perceptron class, computes the composition of a sequence of Layers

        :parameters:
            - W_init : list of np.ndarray, len=N
                Values to initialize the weight matrix in each layer to.
                The layer sizes will be inferred from the shape of each matrix in W_init
            - b_init : list of np.ndarray, len=N
                Values to initialize the bias vector in each layer to
            - activations : list of theano.tensor.elemwise.Elemwise, len=N
                Activation function for layer output for each layer
            - dropout_fraction : fractions of neurons to randomly remove during
                training
        '''
        # Make sure the input lists are all of the same length
        assert len(W_init) == len(b_init) == len(activations)

        # Initialize lists of layers
        self.layers = []
        self.dropout_fraction = dropout_fraction
        # Construct the layers
        for W, b, activation in zip(W_init, b_init, activations):
            self.layers.append(Layer(W, b, activation))

        # Combine parameters from all layers
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def output(self, x, train=False):
        '''
        Compute the MLP's output given an input

        :parameters:
            - x : theano.tensor.var.TensorVariable
                Theano symbolic variable for network input
            - train : flag if the dropout should be used (default is False)

        :returns:
            - output : theano.tensor.var.TensorVariable
                x passed through the MLP
        '''
        # Recursively compute output
        if not train:
            for layer in self.layers:
                x = layer.output(x, 0)
        else:
            for layer in self.layers:
                x = layer.output(x, self.dropout_fraction)
        return x


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

if __name__=='__main__':
    max_activation = lambda x : T.maximum(0, x)

    from sklearn.datasets import make_multilabel_classification

    X, Y = make_multilabel_classification(n_samples=55000,
            n_features=100, n_classes=50, n_labels=5, length=100,
            return_indicator=True)

    W_init = [0.1*np.random.randn(200, 100),
                #0.1*np.random.randn(200, 200),
                0.1*np.random.randn(200, 200),
                0.1*np.random.randn(50, 200)]

    b_init = [0.1*np.random.randn(200),
                #0.1*np.random.randn(200),
                0.1*np.random.randn(200),
                0.1*np.random.randn(50)]

    activations = [max_activation,
                    #max_activation,
                    max_activation,
                    T.nnet.sigmoid]

    multi_layer_perceptron = MLP(W_init, b_init, activations, 0.5)

    params = multi_layer_perceptron.params
    inp = T.matrix()
    target = T.matrix()
    model_output_dropout = multi_layer_perceptron.output(inp.T, True)
    model_output = multi_layer_perceptron.output(inp.T, False)

    cost = T.mean(T.nnet.binary_crossentropy(model_output_dropout.T, target))

    updates = gradient_updates_momentum(cost, params, learning_rate=0.11, momentum=0.3)

    train = theano.function([inp, target], cost, updates=updates)
    predict = theano.function([inp], model_output.T)
    error = theano.function([inp, target], cost)



    mini_batch_size = 1000
    for iteration in range(500):
        for i in range(0, 50000, mini_batch_size):
            X_tr = X[i:i+mini_batch_size]
            Y_tr = Y[i:i+mini_batch_size]
            c = train(X_tr, Y_tr)
        print iteration, ':', error(X[50000:], Y[50000:])
