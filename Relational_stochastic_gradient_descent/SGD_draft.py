import theano
import theano.tensor as T
import numpy as np

def floatX(x):
    return np.asarray(x, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape)*0.0001 ))

Y = np.genfromtxt('Y.txt')
Y = np.log(Y)
n_u, n_v = Y.shape
X_u = np.genfromtxt('68study3_3D_SFS.txt')
X_v = np.genfromtxt('targets_WS_normalized.txt')
d_u = X_u.shape[1]
d_v = X_v.shape[1]
model_rank = 40

# first define the elements of the training data

y = T.vector()
x_u = T.matrix()
x_v = T.matrix()

# the model and assiotiated elements

U = init_weights((model_rank, d_u))
V = init_weights((model_rank, d_v))

# delta for momentum
delta_U = init_weights((model_rank, d_u))
delta_V = init_weights((model_rank, d_v))

lambda_u = 10.0
lambda_v = 10.0

left_features = U.dot(x_u.T)
right_features = V.dot(x_v.T)

prediction = T.diagonal(T.dot(left_features.T, right_features))

# LOSS FUNCTIONS
# squared loss
loss_squared = T.mean((y - prediction)**2)
# logistic loss (0/1 classiifcation)
prob_prediction = T.nnet.sigmoid(prediction)
loss_log = - T.mean( y * T.log(prob_prediction) + (1-y) * T.log(1 - prob_prediction))
# e-insensitive loss
epsilon = 0.1
loss_e_insens = T.mean(T.maximum(prediction - y - epsilon, T.maximum(0, y - prediction - epsilon)))
# hinge loss (-1/1 classification)
loss_hinge = T.mean(T.maximum(-prediction*y + epsilon, 0))

# PENALTIES
cost = loss_squared + lambda_u * T.mean( U**2 ) + lambda_v * T.mean( V**2 )
gU, gV = T.grad(cost, [U, V])

learning_rate = 1e-7
momentum_factor = 0.9

train = theano.function(inputs=[y, x_u, x_v],
            outputs=cost,
            updates = ((U, U + delta_U),
                (V, V + delta_V),
                (delta_U, momentum_factor*delta_U - (1 - momentum_factor)*learning_rate*gU),
                (delta_V, momentum_factor*delta_V - (1 - momentum_factor)*learning_rate*gV)))

predict = theano.function(inputs=[x_u, x_v],
        outputs=prediction)

#get_grads = theano.function(inputs=[y, x_u, x_v], outputs=[gU, gV])

squared_error = theano.function(inputs=[y, x_u, x_v], outputs=loss_squared)

indices = [(i, j) for i in range(n_u) for j in range(n_v)]
from random import shuffle
shuffle(indices)

train_instances = indices[:25000]
n_train_instances = len(train_instances)
test_instances = indices[25000:]

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("whitegrid")

plt.ion()
plt.xlabel('training iteration')
plt.ylabel('squared error')
plt.show()

n_iterations = 500
mini_batch_size = 100
for iteration in range(n_iterations):
    # train
    for start in range(0, n_train_instances, mini_batch_size):
        indices = train_instances[start:start+mini_batch_size]
        y_batch = [Y[i,j] for (i,j) in indices]
        indices_i, indices_j = zip(*indices)
        indices_i = list(indices_i)
        indices_j = list(indices_j)
        cost_iteration = train(y_batch, X_u[indices_i], X_v[indices_j])
    # calculate test error
    y_batch = [Y[i,j] for (i,j) in test_instances]
    indices_i, indices_j = zip(*test_instances)
    indices_i = list(indices_i)
    indices_j = list(indices_j)
    test_error = squared_error(y_batch, X_u[indices_i], X_v[indices_j])
    # calculate train error (on subset)
    shuffle(train_instances)
    y_batch = [Y[i,j] for (i,j) in train_instances[:10000]]
    indices_i, indices_j = zip(*train_instances[:10000])
    indices_i = list(indices_i)
    indices_j = list(indices_j)
    train_error = squared_error(y_batch, X_u[indices_i], X_v[indices_j])
    print 'For iteration %s, test error is %.5f' %(iteration+1, test_error)
    plt.scatter(iteration+1, test_error, c='b', marker='o')
    plt.scatter(iteration+1, train_error, c='r', marker='o')
    plt.draw()
