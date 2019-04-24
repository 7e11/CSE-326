# -------------------------------------------------------------------------
'''
    Problem 5: Gradient Descent Training of Logistic Regression
'''

from problem4 import * # compute loss and grad
from problem2 import *
import numpy as np # linear algebra
import pickle
# --------------------------
def gradient_descent(X, Y, X_val, Y_val, X_test, Y_test, num_iters = 50, lr = 0.01, log=True):
    '''
    Train Logistic Regression using Gradient Descent
    X: d x m training sample vectors (sparse csc_matrix)
    Y: 1 x m labels
    X_val: validation sample vectors (sparse csc_matrix)
    Y_val: validation labels
    X_test: test sample vectors (sparse csc_matrix)
    Y_test: test labels
    num_iters: number of gradient descent iterations
    lr: learning rate
    log: True if you want to track the training process, by default True
    :return: (w, b, training_log)
    '''
    #########################################
    ## INSERT YOUR CODE HERE

    # 1. average gradients at all examples
    # 2. update the parameters simultaneously
    # 3. until convergence?

    # Initialize initial weights and biases (these don't matter b/c function is convex)
    # weights = np.zeros(X.shape)             #one weight for each feature (d features per example)
    # biases = np.zeros((1, X.shape[1]))               #one bias for each training example (m examples)

    weights = np.zeros((X.shape[0], 1))
    biases = np.zeros((1, 1))
    training_log = []

    # print(X, Y)

    # for iter in range(num_iters):
    for iter in range(1, 1000 + 1):
        Z = linear(weights, biases, X)
        A = sigmoid(Z)
        _loss = loss(A, Y)
        _dz = dZ(Z, Y)
        _dw = dw(Z, X, Y).A
        _db = db(Z, Y)

        # Adjust weights
        # weights = _dw * np.subtract(weights, lr)
        # biases = _db * np.subtract(biases, lr)
        weights = weights - lr * _dw
        biases = biases - lr * _db

        # Do validation data checking
        Z_val = linear(weights, biases, X_val)
        A_val = sigmoid(Z_val)
        _loss_val = loss(A_val, Y_val)

        # Do test loss checking ???
        Z_test = linear(weights, biases, X_test)
        A_test = sigmoid(Z_test)
        _loss_test = loss(A_test, Y_test)

        if log:
            training_log.append((_loss, _loss_val, _loss_test, np.linalg.norm(_dw) ** 2 + np.linalg.norm(_db) ** 2))

        print(iter, _loss)

    return np.matrix(weights), np.matrix(biases), training_log

    #########################################


# --------------------------
def train(**kwargs):
    #########################################
    ## INSERT YOUR CODE HERE

    # Does this just call gradient descent?

    # for a in kwargs:
    #     print(a, kwargs[a])
    #
    # print(kwargs.values())
    return gradient_descent(*kwargs.values())

    #########################################

# --------------------------
if __name__ == "__main__":
    filename = '../data/data_matrices.pkl'
    X, y = loadData(filename)
    tr_X, val_X, te_X = splitData(X)
    tr_y, val_y, te_y = splitData(y)

    kwargs = {'Training X': tr_X,
              'Training y': tr_y,
              'Validation X': val_X,
              'Validation y': val_y,
              'Test X': te_X,
              'Test y': te_y,
              'num_iters': 10,
              'lr': 0.01,
              'log': True}

    w, b, training_log = train(**kwargs)

    with open('../data/batch_outcome.pkl', 'wb') as f:
        pickle.dump((w, b, training_log), f)
