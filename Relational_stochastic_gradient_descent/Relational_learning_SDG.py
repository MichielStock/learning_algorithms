"""
Created on Wed May 13 2015
Last update: -

@author: Michiel Stock
michielfmstock@gmail.com

Class for stochastic gradient descent learning of relational models

Uses mini-batches to train a low rank model for pairwise data
"""

import theano
import theano.tensor as T
import numpy as np

# LOSS FUNCTIONS
# squared loss
loss_squared = lambda y, prediction : T.mean((y - prediction)**2)
# logistic loss (0/1 classiifcation)
prob_prediction = lambda y, prediction : T.nnet.sigmoid(prediction)
loss_log = lambda y, prediction : - T.mean( y * T.log(prob_prediction) +\
        (1-y) * T.log(1 - prob_prediction))
# e-insensitive loss
epsilon = 0.1
loss_e_insens = lambda y, prediction : T.mean(T.maximum(prediction - y -\
        epsilon, T.maximum(0, y - prediction - epsilon)))
# hinge loss (-1/1 classification)
loss_hinge = lambda y, prediction : T.mean(T.maximum(-prediction*y + epsilon, 0))


class RelationalModel:
    def __init__(self, X_u, X_v, y,  Uinit, Vinit, loss=loss_squared,
            regularization=(1,1), reg_type=(lambda x:T.mean(x**2),
            lambda x:T.mean(x**2))):
        self.U = theano.shared(Uinit)
        self.V = theano.shared(Vinit)
        self.X_u = X_u
        self.X_v = X_v
        self.y = y
        self.prediction_matrix = X_u.dot(self.U).dot((X_v.dot(self.V)).T)
        self.prediction_pairs = T.diagonal( self.prediction_matrix )
        self.cost = loss(y, self.prediction_pairs)
        self.cost += regularization[0] * reg_type[0]( self.U )
        self.cost += regularization[1] * reg_type[1]( self.V )

    def make_stochastic_gradient_descent_updates(self, learning_rate=1e-5):
        dU, dV = theano.grad(self.cost, [self.U, self.V])
        updates = []
        updates.append( (self.U, self.U - learning_rate * dU) )
        updates.append( (self.V, self.V - learning_rate * dV) )
        return theano.function((self.y, self.X_u, self.X_v), self.cost,
                updates=updates)

    def make_pair_predictions(self):
        return theano.function((self.X_u, self.X_v),
                self.prediction_pairs)

    def make_matrix_predictions(self):
        return theano.function((self.X_u, self.X_v),
                self.prediction_matrix)

if __name__ == '__main__':
    labels = np.genfromtxt('Y.txt')
    labels = np.log(labels)
    n_u, n_v = labels.shape
    features_u = np.genfromtxt('68study3_3D_SFS.txt')
    features_v = np.genfromtxt('targets_WS_normalized.txt')

    rank = 20

    Uinit = np.random.randn( n_u, rank ) * 0.001
    Vinit = np.random.randn( n_v, rank ) * 0.001

    X_u = T.matrix('X_u')
    X_v = T.matrix('X_v')
    y = T.vector('y')

    model = RelationalModel( X_u, X_v, y,  Uinit, Vinit )

    train = model.make_stochastic_gradient_descent_updates()
    predict = model.make_pair_predictions()
    squared_error = theano.function( (y, X_u, X_v), T.mean((model.prediction_pairs -
                y)**2))

    n_iterations = 500
    mini_batch_size = 100

    indices = [(i, j) for i in range(n_u) for j in range(n_v)]
    from random import shuffle
    shuffle(indices)

    train_instances = indices[:25000]
    n_train_instances = len(train_instances)
    test_instances = indices[25000:]


    for iteration in range(n_iterations):
        for start in range(0, n_train_instances, mini_batch_size):
            indices = train_instances[start:start+mini_batch_size]
            y_batch = [labels[i,j] for (i,j) in indices]
            indices_i, indices_j = zip(*indices)
            indices_i = list(indices_i)
            indices_j = list(indices_j)
            cost_iteration = train(y_batch, features_u[indices_i],
                    features_v[indices_j])
        # test error
        y_batch = [labels[i,j] for (i,j) in test_instances]
        indices_i, indices_j = zip(*test_instances)
        indices_i = list(indices_i)
        indices_j = list(indices_j)
        test_error = squared_error(y_batch, features_u[indices_i],
                features_v[indices_j])
        print test_error
