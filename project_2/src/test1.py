from problem1 import *

import numpy as np
'''
    Unit test 1:
    This file includes unit tests for problem1.py.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
'''

# -------------------------------------------------------------------------
def test_linear_kernel():
    ''' (5 points) linear_kernel()'''

    # when both X1 and X2 contain only one feature vector
    X1 = np.asmatrix(np.array([1, 2])).T
    X2 = np.asmatrix(np.array([3, 4])).T

    assert type(X1) == np.matrixlib.defmatrix.matrix
    assert type(X2) == np.matrixlib.defmatrix.matrix

    res = linear_kernel(X1, X2)
    assert type(res) == np.matrixlib.defmatrix.matrix
    assert res.shape == (1, 1)
    assert np.allclose(res, [11], atol=1e-6)

    # when X1 and X2 contain multiple feature vectors
    X1 = np.asmatrix(np.array([[1, 2], [2, 4]])).T
    X2 = np.asmatrix(np.array([[3, 4], [1, 2], [5, 6]])).T

    assert type(X1) == np.matrixlib.defmatrix.matrix
    assert type(X2) == np.matrixlib.defmatrix.matrix

    res = linear_kernel(X1, X2)
    assert type(res) == np.matrixlib.defmatrix.matrix
    assert res.shape == (2, 3)
    assert np.allclose(res, np.array([[11, 5, 17], [22, 10, 34]]), atol=1e-6)

# -------------------------------------------------------------------------
def test_Gaussian_kernel():
    ''' (8 points) Gaussian_kernel()'''

    # when both X1 and X2 contain only one feature vector
    X1 = np.asmatrix(np.array([1, 2])).T
    X2 = np.asmatrix(np.array([3, 4])).T

    assert type(X1) == np.matrixlib.defmatrix.matrix
    assert type(X2) == np.matrixlib.defmatrix.matrix

    res = Gaussian_kernel(X1, X2, 1)
    # assert type(res) == np.matrixlib.defmatrix.matrix
    assert res.shape == (1, 1)
    assert np.allclose(res, [0.01831564], atol=1e-6)

    # when X1 and X2 contain multiple feature vectors
    X1 = np.asmatrix(np.array([[1, 2], [2, 4]])).T
    X2 = np.asmatrix(np.array([[3, 4], [1, 2], [5, 6]])).T
    # print(X2.T)

    assert type(X1) == np.matrixlib.defmatrix.matrix
    assert type(X2) == np.matrixlib.defmatrix.matrix

    res = Gaussian_kernel(X1, X2, 5)
    # assert type(res) == np.matrixlib.defmatrix.matrix
    assert res.shape == (2, 3)
    assert np.allclose(res, np.array([[0.85214379, 1., 0.52729242],[0.98019867, 0.90483742, 0.77105159]]), atol=1e-6)

# -------------------------------------------------------------------------
def test_hinge_loss():
    ''' (3 points) hinge_loss()'''
    X = np.asmatrix(np.array([[1, 2], [2, 4]])).T
    y = np.array([1, -1])
    w = np.array([1, 1])
    b = -1

    z = np.dot(w.T, X) + b

    losses = hinge_loss(z, y)
    assert type(losses) == np.matrixlib.defmatrix.matrix
    assert np.allclose(losses, np.array([0, 6]), atol=1e-6)
