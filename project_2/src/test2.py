from problem1 import *
from problem2 import *

import numpy as np
'''
    Unit test 2:
    This file includes unit tests for problem2.py.
    You could test the correctness of your code by typing `nosetests -v test2.py` in the terminal.
'''

# -------------------------------------------------------------------------
def test_dual_objective_function():
    ''' (10 points) dual_objective_function()'''
    train_X = np.asmatrix(np.array([[1,1], [-1, -1]])).T
    train_y = np.asmatrix(np.array([1, -1]))
    alpha = np.asmatrix(np.array([1, 1]))

    sigma = None
    value = dual_objective_function(alpha, train_y, train_X, linear_kernel, sigma)
    assert np.allclose(value, -2, atol=1e-6)

    sigma = 1
    value = dual_objective_function(alpha, train_y, train_X, Gaussian_kernel, sigma)
    assert np.allclose(value, [1.01831564], atol=1e-6)

# -------------------------------------------------------------------------
def test_primal_objective_function():
    ''' (10 points) primal  _objective_function()'''
    train_X = np.asmatrix(np.array([[1, 1], [-1, -1]])).T
    train_y = np.asmatrix(np.array([1, -1]))
    alpha = np.asmatrix(np.array([1, 1]))
    b = -1
    C = 10

    sigma = None
    value = primal_objective_function(alpha, train_y, train_X, b, C, linear_kernel, sigma)
    assert np.allclose(value, 4, atol=1e-6)

    sigma = 1
    value = primal_objective_function(alpha, train_y, train_X, b, C, Gaussian_kernel, sigma)
    assert np.allclose(value, [11.1648407], atol=1e-6)

    # further test primal objective is no less than the dual objective, for both kernel functions
    train_X = np.asmatrix(np.array([[1, 1], [-2, -2]])).T
    train_y = np.asmatrix(np.array([1, 1]))
    alpha = np.asmatrix(np.array([1, 1]))
    b = -1
    C = 10

    sigma = None
    primal_value_linear = primal_objective_function(alpha, train_y, train_X, b, C, linear_kernel, sigma)
    assert np.allclose(primal_value_linear, 41, atol=1e-6)
    dual_value_linear = dual_objective_function(alpha, train_y, train_X, linear_kernel, sigma)
    assert np.allclose(dual_value_linear, 1, atol=1e-6)
    assert primal_value_linear - dual_value_linear >= -1e-6

    sigma = 1
    primal_value_Gaussian = primal_objective_function(alpha, train_y, train_X, b, C, Gaussian_kernel, sigma)
    assert np.allclose(primal_value_Gaussian, [20.99765521], atol=1e-6)
    dual_value_Gaussian = dual_objective_function(alpha, train_y, train_X, Gaussian_kernel, sigma)
    assert np.allclose(dual_value_Gaussian, 0.99987659, atol=1e-6)
    assert primal_value_Gaussian - dual_value_Gaussian >= -1e-6

# -------------------------------------------------------------------------
def test_decision_function():
    ''' (4 points) decision_function()'''
    train_X = np.asmatrix(np.array([[1, 1], [-1, -1]])).T
    train_y = np.asmatrix(np.array([1, -1]))
    test_X = np.asmatrix(np.array([[1, 1], [-1, -1]])).T
    alpha = np.asmatrix(np.array([1, 1]))
    b = 4

    sigma = None
    value = decision_function(alpha, train_y, train_X, b, linear_kernel, sigma, test_X)
    assert np.allclose(value, [8, 0], atol=1e-6)

    sigma = 1
    value = decision_function(alpha, train_y, train_X, b, Gaussian_kernel, sigma, test_X)
    assert np.allclose(value, [4.98168436, 3.01831564], atol=1e-6)