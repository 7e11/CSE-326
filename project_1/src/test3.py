from problem3 import *
import sys

import numpy as np
'''
    Unit test 3:
    This file includes unit tests for problem3.py.
    You could test the correctness of your code by typing `nosetests -v test3.py` in the terminal.
'''

# -------------------------------------------------------------------------
def test_python_version():
    ''' ---------- Problem 3 (30 points in total) ------------'''
    assert sys.version_info[0] == 3  # require python 3 (instead of python 2)


# -------------------------------------------------------------------------
def test_linear():
    ''' (5 points) linear()'''
    w = np.asmatrix(np.array([1, 2])).T
    x = np.asmatrix(np.array([2, 1])).T
    b = 1.5

    assert linear(w, b, x) == 5.5

# -------------------------------------------------------------------------
def test_sigmoid():
    ''' (5 points) sigmoid()'''
    z = 1
    assert np.allclose(sigmoid(z), 0.731, atol=1e-02)
    z = 0
    assert np.allclose(sigmoid(z), 0.5, atol=1e-02)
    z = -1
    assert np.allclose(sigmoid(z), 0.2689, atol=1e-02)

# -------------------------------------------------------------------------
def test_loss():
    ''' (5 points) loss()'''
    a = 0.6
    y = 1

    assert np.allclose(loss(a, y), 0.5108, atol=1e-02)
    # assert np.allclose(loss(a, y), -np.log(0.6), atol=1e-03)

    a = 0.6
    y = 0

    assert np.allclose(loss(a, y), 0.9162, atol=1e-02)
    # assert np.allclose(loss(a, y), -np.log(0.4), atol=1e-03)

# -------------------------------------------------------------------------
def test_dz():
    ''' (5 points) dz()'''
    z = 5.5
    y = 1
    assert np.allclose(dz(z, y), -0.004, atol=1e-03)

# -------------------------------------------------------------------------
def test_dw():
    ''' (5 points) dw()'''
    x = np.array([2, 1])
    z = 5.5
    y = 1
    assert np.allclose(dw(z, x, y), [-0.008, -0.004], atol=1e-03)

    y = 0
    assert np.allclose(dw(z, x, y), [1.9918, 0.9959], atol=1e-03)

# -------------------------------------------------------------------------
def test_db():
    ''' (5 points) db()'''
    z = 5.5
    y = 1
    assert np.allclose(db(z, y), -0.004, atol=1e-03)

    y = 0
    assert np.allclose(db(z, y), 0.9959, atol=1e-03)
