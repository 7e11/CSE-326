from problem1 import *
from problem2 import *

import numpy as np
'''
    Unit test 2:
    This file includes unit tests for problem2.py.
    You could test the correctness of your code by typing `nosetests -v test2.py` in the terminal.
'''

# -------------------------------------------------------------------------
def test_forward():
    ''' (15 points) NN:forward()'''

    X = np.asmatrix(np.array([[1,1], [-1,-1]]))
    input_dim = 2
    num_classes = 1
    # make a tiny network
    dimensions = [input_dim, 32, num_classes]

    activation_funcs = {1:Tanh, 2:Identity}
    loss_func = MSELoss
    nn = NN(dimensions, activation_funcs, loss_func, rand_seed = 42)

    output = nn.forward(X)
    assert np.allclose(output, np.array([0.22031805, 0.22031805]), atol=1e-6)

# -------------------------------------------------------------------------
def test_bacward():
    ''' (15 points) NN:bacward()'''


    X = np.asmatrix(np.array([[1,1], [-1,-1]])).T
    Y = np.asmatrix(np.array([[1], [-1]])).T
    input_dim = 2
    num_classes = 1
    # make a tiny network
    dimensions = [input_dim, 2, num_classes]

    activation_funcs = {1:Tanh, 2:Identity}
    loss_func = MSELoss
    nn = NN(dimensions, activation_funcs, loss_func, rand_seed = 42)

    output = nn.forward(X)
    dW, db = nn.backward(Y)

    assert type(dW) == dict
    assert type(dW[1]) == np.matrixlib.defmatrix.matrix
    assert np.allclose(dW[2], np.array([[-0.29580902, -1.08618959]]))
    assert type(dW[1]) == np.matrixlib.defmatrix.matrix
    assert np.allclose(dW[1], np.array([[0.18519954, 0.18519954], [0.03346838, 0.03346838]]))

    assert type(db) == dict
    assert type(db[2]) == np.matrixlib.defmatrix.matrix
    assert np.allclose(db[2], np.array([[-0.0]]))
    assert type(db[1]) == np.matrixlib.defmatrix.matrix
    assert np.allclose(db[1], np.array([[1.38777878e-17], [0.00000000e+00]]))
