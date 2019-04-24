# -------------------------------------------------------------------------
'''
    Problem 4: compute sigmoid(wT X +loss function, and the gradient.
    This is the vectorized version  b), the that handle multiple training examples X.
'''

import numpy as np # linear algebra
from scipy import sparse
from scipy.sparse import diags
from scipy.sparse import csc_matrix

# --------------------------
def linear(w, b, X):
    '''
    w: d x 1, Logistic parameters, ndarry
    b: 1 x 1, Logistic parameters, float
    X: d x m, m example feature vectors, sparse csc_matrix, (2,2)?
    :return: Z = wT X + b
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    if sparse.issparse(X):
        X = X.toarray()
    # print(type(w), type(b), type(X))
    prod = np.add((np.matmul(w.T, X)), b)
    # print(w.T, w.T.shape)
    # print(X, X.shape)
    # print(prod, type(prod), prod.shape)
    #print(prod[0], type(prod[0]), prod[0].shape)
    #print(prod[1], type(prod[1]), prod[1].shape)
    #print(prod[2], type(prod[2]), prod[2].shape)
    return prod
    #########################################

# --------------------------
def sigmoid(Z):
    '''
    Z: 1 x m vector. wT X + b
    :return: A = sigmoid(Z)
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    return 1/(1+np.exp(-Z))
    #########################################

# --------------------------
def loss(A, Y):
    '''
    A: 1 x m, sigmoid output on m training examples
    Y: 1 x m, labels of the m training examples
    :return: mean negative log-likelihood loss on m training examples.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    # print(A, Y)
    # yminus = np.subtract(Y, np.ones(Y.shape))
    # aminus = np.subtract(A, np.ones(A.shape))
    # print(aminus, yminus)
    # print(np.dot(Y, -np.log(A)))
    # print(np.dot(yminus, -np.log(aminus)))
    # return (np.dot(Y, -np.log(A))) + (np.dot(yminus, -np.log(aminus)))
    if type(Y) is np.matrix:
        Y = Y.A
    if type(A) is np.matrix:
        A = A.A
    # print(A.shape, Y.shape)
    # print(type(A), type(Y))
    return np.mean(Y * -np.log(A) + (1-Y) * -np.log(1-A))
    #########################################

# --------------------------
def dZ(Z, Y):
    '''
    Z: 1 x m vector. wT X + b
    Y: 1 x m, label of X
    :return: 1 x m, the gradient of the negative log-likelihood loss on all samples wrt z.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    # return ((1/(np.exp(Z)*(1+np.exp(-Z))**2))*((-Y/(1/(1+np.exp(-Z)))) + ((1-Y)/(1-(1/(1+np.exp(-Z)))))))
    # return -np.subtract(Y, 1/(1+np.exp(-Z)))
    return np.subtract(sigmoid(Z), Y)
    #########################################

# --------------------------
def dw(Z, X, Y):
    '''
    Z: 1 x m vector. wT X + b
    X: d x m, m example feature vectors. Should be in CSC scipy sparse matrix format
    Y: 1 x m, label of X
    :return: d x 1, the gradient of the negative log-likelihood loss on all samples wrt w.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    # print(Z, X, Y)
    # print(Z.shape, X.shape, Y.shape)
    # out = np.dot((-np.subtract(Y, 1/(1+np.exp(-Z)))), X.T).T

    # out = np.mean()
    # print(np.mean(np.asmatrix([[-0.008, -0.004], [1.9918, 0.9959]]).T))
    # 0.743925
    # out = np.mean(np.dot(X.T, Z))
    dz = dZ(Z, Y)
    # print(dz, dz.shape)
    # out = np.mean(np.matmul(dz, X))
    # out = (np.exp(Z) / (1 + np.exp(Z)) - Y)  #This is the same as dz (duh)
    out = (np.matmul(dz, X.T) / Z.shape[1]).T
    # print(out)
    return out
    #########################################

# --------------------------
def db(Z, Y):
    '''
    Z: 1 x m vector. wT X + b
    Y: 1 x m, label of X
    :return: 1, the gradient of the negative log-likelihood loss on (X, Y) wrt w.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    # print(np.mean(np.asmatrix([[-0.004], [0.9959]]).T))
    # 0.49595
    # out = np.matrix(np.mean(-np.subtract(Y, 1/(1 + np.exp(-Z)))))
    out = np.matrix(np.mean(dZ(Z, Y)))
    # print(out)
    # print(type(out))
    return out
    #########################################
