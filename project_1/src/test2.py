from problem2 import *
import sys

import numpy as np
from scipy.sparse import csc_matrix
'''
    Unit test 1:
    This file includes unit tests for problem1.py.
    You could test the correctness of your code by typing `nosetests -v test1.py` in the terminal.
'''

# -------------------------------------------------------------------------
def test_python_version():
    ''' ---------- Problem 2 (10 points in total) ------------'''
    assert sys.version_info[0] == 3  # require python 3 (instead of python 2)


# -------------------------------------------------------------------------
def test_loadData():
    ''' (5 points) loadData()'''
    sample_matrix, label_vector = loadData('../data/data_matrices.pkl')

    # test whether or not data_matrix is a numpy matrix
    assert type(sample_matrix) == csc_matrix

    # test the shape of the matrix
    assert sample_matrix.shape == (2913, 63326)

    # test whether or not label_vector is a numpy matrix
    assert type(label_vector) == np.matrixlib.defmatrix.matrix

    # test the shape of the matrix
    assert label_vector.shape == (1, 63326)


# -------------------------------------------------------------------------
def test_splitData():
    ''' (5 points) splitData()'''

    data_matrix = np.asmatrix(np.random.rand(2, 10))
    tr, val, te = splitData(data_matrix)

    # test whether tr is a numpy matrix
    assert type(tr) == np.matrixlib.defmatrix.matrix

    # test the shape of tr
    assert tr.shape == (2, 7)

    # check if tr is the first 7 columns
    assert np.allclose(data_matrix[:, 0:7], tr, atol=1e-3)

    # test whether val is a numpy matrix
    assert type(val) == np.matrixlib.defmatrix.matrix

    # test the shape of val
    assert val.shape == (2, 1)

    # check if the val is the next 2 columns
    assert np.allclose(data_matrix[:, 7:8], val, atol=1e-3)

    # test whether test is a numpy matrix
    assert type(te) == np.matrixlib.defmatrix.matrix

    # test the shape of test
    assert te.shape == (2, 2)

    # check if the test is the last column
    assert np.allclose(data_matrix[:, 8:], te, atol=1e-3)
