from problem4 import *

import sys
import numpy as np
from scipy.sparse import csc_matrix
'''
    Unit test 4:
    This file includes unit tests for problem4.py.
    You could test the correctness of your code by typing `nosetests -v test4.py` in the terminal.
'''

# -------------------------------------------------------------------------
def test_python_version():
    ''' ---------- Problem 4 (40 points in total) ------------'''
    assert sys.version_info[0] == 3  # require python 3 (instead of python 2)

# --------------------------
def test_linear():
    ''' (7 points) linear()'''
    w = np.matrix([1, 2, 3]).T
    b = 1.5
    X = csc_matrix(np.array([[2, 1], [2, 3], [1, 3]]))


    Z = linear(w,b,X)

    # test whether or not Z is a numpy matrix
    assert type(Z) == np.matrixlib.defmatrix.matrix

    # test whether Z is of the correct shape
    assert Z.shape == (1, 2)

    # test value of Z
    assert np.allclose(Z, np.array([10.5, 17.5]), atol=1e-03)

# --------------------------
def test_sigmoid():
    ''' (7 points) sigmoid()'''
    Z = np.array([1, 0, -1])
    assert np.allclose(sigmoid(Z), np.array([0.731, 0.5, 0.2689,]), atol = 1e-03)

# -------------------------------------------------------------------------
def test_loss():
    ''' (7 points) loss()'''
    w = np.array([1, 2]).T
    b = 1.5
    X = csc_matrix(np.array([[2, 2], [1, 1]]))
    Z = linear(w, b, X)
    A = sigmoid(Z)
    Y = np.array([1, 0])

    assert np.allclose(loss(A, Y), np.mean([0.00407,5.50407]), atol=1e-03)

# -------------------------------------------------------------------------
def test_dZ():
    ''' (7 points) dZ()'''
    # Z 1 x 2 vector
    Z = np.asmatrix([5.5, 5.5])
    # Y 1 x 2 vector
    Y = np.asmatrix([1, 0])

    assert type(dZ(Z, Y)) == np.matrixlib.defmatrix.matrix

    assert dZ(Z, Y).shape == (1, 2)

    assert np.allclose(dZ(Z, Y), np.array([-0.004, 0.9959]), atol=1e-02)

# -------------------------------------------------------------------------
def test_dw():
    ''' (7 points) dw()'''
    X = np.matrix([[2, 2], [1,1]])
    Z = np.matrix([5.5, 5.5])
    Y = np.matrix([1, 0])

    assert type(dw(Z, X, Y)) == np.matrixlib.defmatrix.matrix

    assert dw(Z, X, Y).shape == (2, 1)

    assert np.allclose(dw(Z, X, Y), np.mean(np.asmatrix([[-0.008, -0.004], [1.9918, 0.9959]]).T, axis=1), atol=1e-03)

# -------------------------------------------------------------------------
def test_db():
    ''' (5 points) db()'''
    Z = np.array([5.5, 5.5])
    Y = np.array([1, 0])

    assert type(db(Z, Y)) == np.matrixlib.defmatrix.matrix

    assert db(Z, Y).shape == (1, 1)

    assert np.allclose(db(Z, Y), np.mean(np.asmatrix([[-0.004], [0.9959]]).T, axis=1), atol=1e-03)
