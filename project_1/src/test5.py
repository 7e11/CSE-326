from problem5 import *

import sys

import numpy as np

# -------------------------------------------------------------------------
def test_python_version():
    ''' ---------- Problem 5 (10 points in total) ------------'''
    assert sys.version_info[0] == 3  # require python 3 (instead of python 2)

# -------------------------------------------------------------------------
def test_train():
    ''' (5 points) train()'''
    filename = '../data/data_matrices.pkl'
    X, y = loadData(filename)
    # print(X, X.shape)
    # print(y, y.shape)
    # print(X.shape, y.shape)
    # print(type(X), type(y))
    tr_X, val_X, te_X = splitData(X)
    tr_y, val_y, te_y = splitData(y)

    kwargs = {'Training X': tr_X,
              'Training y': tr_y,
              'Validation X': val_X,
              'Validation y': val_y,
              'Test X': te_X,
              'Test y': te_y,
              'num_iters': 2,
              'lr': 0.01,
              'log': True}

    w, b, training_log = train(**kwargs)

    # check type of w
    assert type(w) == np.matrixlib.defmatrix.matrix

    # check shape of w
    assert w.shape == (X.shape[0], 1)

    # check type of b
    assert type(b) == np.matrixlib.defmatrix.matrix

    # check shape of b
    assert b.shape == (1, 1)

    # check training log length
    assert len(training_log) == kwargs['num_iters']

    # check each entry of the log
    for e in training_log:
        assert len(e) == 4
        for i in range(4):
            assert type(e[i]) == np.float64

