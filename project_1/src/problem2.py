# -------------------------------------------------------------------------
'''
    Problem 2: reading data set from a file, and then split them into training, validation and test sets.

    The package for loading data in python
'''

import pickle
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
import numpy as np # linear algebra
import math
from scipy.sparse import csr_matrix
from scipy import sparse
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


def loadData(filename):
    with open(filename, 'rb') as f:
        X, y = pickle.load(f)
    return X, y

def splitData(data_matrix, split_ratio=[0.7, 0.1, 0.2]):
    '''
	data_matrix: columns are samples
    split_ratio: the ratio of examples go into the Training, Validation, and Test sets.
    Split the whole dataset into Training, Validation, and Test sets.
    :return: return Training, Validation, and Test sets.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    # data_matrix = data_matrix.toarray()
    # sparse matrices are cancer...

    if sparse.issparse(data_matrix):
        data_matrix = data_matrix.toarray() #memory error...


    # print(type(data_matrix))
    # print(data_matrix.shape)
    length = data_matrix.shape[1]

    trainingSplit = math.floor(length * split_ratio[0])
    validationSplit = math.floor(length * split_ratio[1]) + trainingSplit

    # print(length, trainingSplit, validationSplit)
    # print(data_matrix)

    # training = data_matrix[:trainingSplit, :trainingSplit]
    # validation = data_matrix[trainingSplit:validationSplit, trainingSplit:validationSplit]
    # testing = data_matrix[validationSplit:, validationSplit:]
    # print(training, validation, testing)

    #testSplit = round(length * split_ratio[1])
    #
    # training = [data_matrix[0][:trainingSplit], data_matrix[1][:trainingSplit]]
    # validation = [data_matrix[0][trainingSplit:validationSplit], data_matrix[1][trainingSplit:validationSplit]]
    # test = [data_matrix[0][validationSplit:], data_matrix[1][validationSplit:]]
    # print(length, trainingSplit, validationSplit)

    # output = np.hsplit(data_matrix, [trainingSplit, validationSplit])
    # if data_matrix.shape[0] == 1:
    #     output = np.split(data_matrix, [trainingSplit, validationSplit], 0)
    # else:
    output = np.split(data_matrix, np.array([trainingSplit, validationSplit]), 1)

    # print(output[0].shape, output[1].shape, output[2].shape)
    # print(output)

    # print(training.shape, validation.shape, testing.shape)

    return output
    # return training, validation, testing
    #########################################
