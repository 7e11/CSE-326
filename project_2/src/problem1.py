# -------------------------------------------------------------------------
'''
    Problem 1: Implement linear, Gaussian kernels, and hinge loss

'''

import numpy as np
from sklearn.metrics.pairwise  import euclidean_distances

# --------------------------
def linear_kernel(X1, X2):
    """
    Compute linear kernel between two set of feature vectors.
    No constant 1 is appended.
    X1: n x m1 matrix, each of the m1 column is an n-dim feature vector.
    X2: n x m2 matrix, each of the m2 column is an n-dim feature vector.
    :return: if both m1 and m2 are 1, return linear kernel on the two vectors; else return a m1 x m2 kernel matrix K,
            where K(i,j)=linear kernel of column i from X1 and column j from X2
    """

    #########################################
    ## INSERT YOUR CODE HERE
    return np.dot(X1.T, X2)
    #########################################

# --------------------------
def Gaussian_kernel(X1, X2, sigma):
    """
    Compute linear kernel between two set of feature vectors.
    No constant 1 is appended.
    For your convenience, please use euclidean_distances.

    X1: n x m1 matrix, each of the m1 column is an n-dim feature vector.
    X2: n x m2 matrix, each of the m2 column is an n-dim feature vector.
    sigma: Gaussian variance (called bandwidth)
    :return: if both m1 and m2 are 1, return Gaussian kernel on the two vectors; else return a m1 x m2 kernel matrix K,
            where K(i,j)=Gaussian kernel of column i from X1 and column j from X2

    """
    #########################################
    ## INSERT YOUR CODE HERE
    # print(X1.T, X2.T)
    return np.exp(- euclidean_distances(X1.T, X2.T)**2 / (2 * sigma ** 2))

    # X1 = X1.T
    # X2 = X2.T
    #
    # result = [0] * len(X1)
    # for i in range(len(X1)):
    #     result[i] = [0] * len(X2)
    #
    # #result = [[]]
    # for x1_index, x1 in enumerate(X1):
    #     for x2_index, x2 in enumerate(X2):
    #         #print(x1, x2)
    #         #print(np.linalg.norm(x1 - x2))
    #         result[x1_index][x2_index] = np.exp(-1 / (2 * sigma**2) * np.linalg.norm(x1 - x2)**2)
    # #print(result)
    # return np.asmatrix(result)
    # #return np.asmatrix(np.exp(-1 / (2 * sigma**2) * np.linalg.norm(X1 - X2)**2))
    #########################################
# --------------------------
def hinge_loss(z, y):
    """
    Compute the hinge loss on a set of training examples
    z: 1 x m vector, each entry is <w, x> + b (may come from a kernel function)
    y: 1 x m label vector. Each entry is -1 or 1

    z is matrix, y is ndarray

    :return: 1 x m hinge losses over the m examples
    """
    #########################################
    ## INSERT YOUR CODE HERE
    # print(z, type(z), y, type(y))
    # print(z.A1)
    # if type(z) == np.matrixlib.defmatrix.matrix:
    #     z = z.A1
    # print(z, y)
    # return np.asmatrix([max(0, 1 - y[i] * z[i]) for i in range(len(y))])
    return np.maximum(0, 1 - np.multiply(y, z))

    #########################################
