from problem1 import *
from problem2 import *
from problem3 import *

import numpy as np

'''
    Unit test 3:
    This file includes unit tests for problem3.py.
    You could test the correctness of your code by typing `nosetests -v test3.py` in the terminal.
'''

# -------------------------------------------------------------------------
def test_train():
    ''' (20 points) train()'''
    train_X = np.asmatrix(np.array([[1, 1], [-1, -1]])).T
    train_y = np.asmatrix(np.array([1, -1]))
    C = 10
    sigma = 1

    # Instantiate model
    model = SVMModel(train_X, train_y, C, linear_kernel, sigma)
    iter_num, duals, primals, models = train(model, max_iters=10, record_every=1)

    assert type(iter_num) == list
    assert type(duals) == list
    assert type(primals) == list
    assert type(models) == list

    # since there are randomness in the train function,
    # we won't be able to give a definite assertion here about the return values
    tol = 1e-6
    for d, p in zip(duals, primals):
        assert d < p + tol, 'dual {} is greater than primal {}'.format(d, p)

# -------------------------------------------------------------------------
def test_predict():
    ''' (5 points) predict()'''
    train_X = np.asmatrix(np.array([[1, 1], [-1, -1]])).T
    train_y = np.asmatrix(np.array([1, -1]))
    C = 10
    sigma = 1

    # Instantiate model
    model = SVMModel(train_X, train_y, C, linear_kernel, sigma)
    model.alpha[0, 0] = 1
    model.alpha[0, 1] = 1
    model.b = 1

    predicted_y = predict(model, train_X)
    assert np.allclose(predicted_y, [1, -1], atol=1e-6)