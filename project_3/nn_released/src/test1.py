from problem1 import *

import numpy as np
'''
    Unit test 1:
    This file includes unit tests for problem1.py.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
'''

# -------------------------------------------------------------------------
def test_sigmoid_activation():
    ''' (5 points) Sigmoid:activate()'''

    Z = np.asmatrix(np.array([[0.1, 0.2], [-.1, -.2], [-5, 6]])).T

    assert type(Z) == np.matrixlib.defmatrix.matrix

    res = Sigmoid.activate(Z)
    assert type(res) == np.matrixlib.defmatrix.matrix
    assert res.shape == (2, 3)
    assert np.allclose(res, np.array([[0.524979, 0.47502081, 0.00669285], [0.549834, 0.450166, 0.99752738]]), atol=1e-6)

# -------------------------------------------------------------------------
def test_sigmoid_gradient():
    ''' (5 points) Sigmoid:gradient()'''

    Z = np.asmatrix(np.array([[0.1, 0.2], [-.1, -.2], [-5, 6]])).T

    assert type(Z) == np.matrixlib.defmatrix.matrix

    res = Sigmoid.gradient(Z)
    assert type(res) == np.matrixlib.defmatrix.matrix
    assert res.shape == (2, 3)
    assert np.allclose(res, np.array([[0.24937604, 0.24937604, 0.00664806], [0.24751657, 0.24751657, 0.00246651]]), atol=1e-6)

# -------------------------------------------------------------------------
def test_softmax_activation():
    ''' (5 points) Softmax:activate()'''

    Z = np.asmatrix(np.array([[0.1, 0.2], [-.1, -.2], [-5, 6]])).T

    assert type(Z) == np.matrixlib.defmatrix.matrix

    res = Softmax.activate(Z)
    assert type(res) == np.matrixlib.defmatrix.matrix
    assert res.shape == (2, 3)
    print (res)
    assert np.allclose(res, np.array([[4.75020813e-01, 5.24979187e-01, 1.67014218e-05], [5.24979187e-01, 4.75020813e-01, 9.99983299e-01]]), atol=1e-6)

# -------------------------------------------------------------------------
def test_tanh_activate():
    ''' (5 points) Tanh:activate()'''

    Z = np.asmatrix(np.array([[0.1, 0.2], [-.1, -.2], [-5, 6]])).T

    assert type(Z) == np.matrixlib.defmatrix.matrix

    res = Tanh.activate(Z)
    assert type(res) == np.matrixlib.defmatrix.matrix
    assert res.shape == (2, 3)
    print (res)
    assert np.allclose(res, np.array([[0.09966799, -0.09966799, -0.9999092], [0.19737532, -0.19737532, 0.99998771]]), atol=1e-6)

# -------------------------------------------------------------------------
def test_tanh_gradient():
    ''' (5 points) Tanh:gradient()'''

    Z = np.asmatrix(np.array([[0.1, 0.2], [-.1, -.2], [-5, 6]])).T

    assert type(Z) == np.matrixlib.defmatrix.matrix

    res = Tanh.gradient(Z)
    assert type(res) == np.matrixlib.defmatrix.matrix
    assert res.shape == (2, 3)
    print (res)
    assert np.allclose(res, np.array([[9.90066291e-01, 9.90066291e-01, 1.81583231e-04], [9.61042983e-01, 9.61042983e-01, 2.45765474e-05]]), atol=1e-6)

# -------------------------------------------------------------------------
def test_crossentropyloss_loss():
    ''' (5 points) CrossEntropyLoss:loss()'''

    Y = np.asmatrix(np.array([[1, 0], [1, 0], [0, 1]])).T
    Y_hat = np.asmatrix(np.array([[0.9, 0.1], [0.1, .9], [0, 1]])).T

    assert type(Y) == np.matrixlib.defmatrix.matrix
    assert type(Y_hat) == np.matrixlib.defmatrix.matrix

    res = CrossEntropyLoss.loss(Y, Y_hat)
    assert np.any(res < 0) == False
    assert np.allclose(res, 1.6052970724345812, atol=1e-6)

# -------------------------------------------------------------------------
def test_crossentropyloss_gradient():
    ''' (5 points) CrossEntropyLoss:gradient()'''

    Y = np.asmatrix(np.array([[1, 0], [1, 0], [0, 1]])).T
    Y_hat = np.asmatrix(np.array([[0.9, 0.1], [0.1, .9], [0, 1]])).T

    assert type(Y) == np.matrixlib.defmatrix.matrix
    assert type(Y_hat) == np.matrixlib.defmatrix.matrix

    res = CrossEntropyLoss.gradient(Y, Y_hat)
    assert np.allclose(res, np.array([[-0.1, -0.9, 0], [0.1, 0.9, 0]]))

# -------------------------------------------------------------------------
def test_mse_loss():
    ''' (5 points) MSELoss:loss()'''

    Y = np.asmatrix(np.array([[1, 0], [1, 0], [0, 1]])).T
    Y_hat = np.asmatrix(np.array([[0.9, 0.1], [0.1, .9], [0, 1]])).T

    assert type(Y) == np.matrixlib.defmatrix.matrix
    assert type(Y_hat) == np.matrixlib.defmatrix.matrix

    res = MSELoss.loss(Y, Y_hat)
    assert np.any(res < 0) == False
    assert np.allclose(res, 0.2733333333333334, atol=1e-6)

# -------------------------------------------------------------------------
def test_mse_gradient():
    ''' (5 points) MSE:gradient()'''

    Y = np.asmatrix(np.array([[1, 0], [1, 0], [0, 1]])).T
    Y_hat = np.asmatrix(np.array([[0.9, 0.1], [0.1, .9], [0, 1]])).T

    assert type(Y) == np.matrixlib.defmatrix.matrix
    assert type(Y_hat) == np.matrixlib.defmatrix.matrix

    res = MSELoss.gradient(Y, Y_hat)
    assert np.allclose(res, np.array([[-0.1, -0.9, 0], [0.1, 0.9, 0]]))
