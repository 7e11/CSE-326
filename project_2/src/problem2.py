# -------------------------------------------------------------------------
'''
    Problem 2: Compute the objective function and decision function of dual SVM.

'''
from problem1 import *

import numpy as np

# -------------------------------------------------------------------------
def dual_objective_function(alpha, train_y, train_X, kernel_function, sigma):
    """
    Compute the dual objective function value.
    alpha: 1 x m learned Lagrangian multipliers (the dual variables).
    train_y: 1 x m labels (-1 or 1) of training data.
    train_X: n x m training feature matrix.
    kernel_function: as the name suggests
    sigma: need to be provided when Gaussian kernel is used.
    :return: a scalar representing the dual objective function value at alpha

    alpha is (double bagged) matrix, kernel_function is function, sigma is number
    train_y is matrix, train_x is matrix
    """
    
    #########################################
    ## INSERT YOUR CODE HERE
    # print(alpha, type(alpha), kernel_function, type(kernel_function), sigma, type(sigma))
    # print(train_y, type(train_y), train_X, type(train_X))

    # alpha = alpha.A1
    # train_y = train_y.A1

    sumresult = 0

    if kernel_function.__name__ == 'Gaussian_kernel':
        k = kernel_function(train_X, train_X, sigma)
    elif kernel_function.__name__ == 'linear_kernel':
        k = kernel_function(train_X, train_X) #2x2

    scalar = np.multiply(np.outer(alpha, alpha), np.outer(train_y, train_y))
    # print(np.sum(alpha) - 1/2 * np.sum(np.multiply(scalar, k))
    return np.sum(alpha) - 1/2 * np.sum(np.multiply(scalar, k))

    # for i in range(len(alpha)):
    #     for j in range(len(alpha)):
    #         if sigma != None:
    #             sumresult += alpha[i] * alpha[j] * train_y[i] * train_y[j] * kernel_function(train_X[:, i],
    #                                                                                          train_X[:, j],
    #                                                                                          sigma)
    #         else:
    #             sumresult += alpha[i] * alpha[j] * train_y[i] * train_y[j] * kernel_function(train_X[:, i],
    #                                                                                          train_X[:, j])
    # #print(np.sum(alpha) - 1/2 * sumresult)
    # return np.sum(alpha) - 1/2 * sumresult

    #########################################

# -------------------------------------------------------------------------
def primal_objective_function(alpha, train_y, train_X, b, C, kernel_function, sigma):
    """
    Compute the primal objective function value.
    alpha: 1 x m learned Lagrangian multipliers (the dual variables).
    train_y: 1 x m labels (-1 or 1) of training data.
    train_X: n x m training feature matrix.
    b: bias term
    C: regularization parameter of soft-SVM
    kernel_function: as the name suggests
    sigma: need to be provided when Gaussian kernel is used.
    :return: a scalar representing the primal objective function value at alpha

    alpha is (double bagged) matrix
    train_y is (double bagged) matrix
    train_x, is matrix
    kernel_function is function
    C, sigma, b are numbers
    """
    
    #########################################
    ## INSERT YOUR CODE HERE

    # print(alpha, type(alpha))
    # print(train_X, type(train_X), train_y, type(train_y))
    # print(b, type(b), C, type(C))

    # alpha = alpha.A1
    # train_y = train_y.A1

    # print(np.asmatrix(w).T)
    if kernel_function.__name__ == 'Gaussian_kernel':
        if sigma != None:
            k = kernel_function(train_X, train_X, sigma)
           # w = np.dot(np.multiply(alpha, train_y), k).T
            z = np.dot(np.multiply(alpha, train_y), k) + b
            sum_res = np.sum(hinge_loss(z, train_y))
            wNorm = np.dot(np.dot(np.multiply(alpha, train_y), k), np.multiply(alpha, train_y).T).item(0)
            return 1/2 * wNorm + C * sum_res
    elif kernel_function.__name__ == 'linear_kernel':
        w = np.dot(np.multiply(alpha, train_y), train_X.T).T
        z = kernel_function(w, train_X) + b
        sum_res = np.sum(hinge_loss(z, train_y))
        return 1/2 * np.linalg.norm(w)**2 + C * sum_res
    # print(sum_res)
    # print(1/2 * np.linalg.norm(w)**2 + C * sum_res)
    # return 1/2 * np.linalg.norm(w)**2 + C * sum_res
    #########################################
# -------------------------------------------------------------------------
def decision_function(alpha, train_y, train_X, b, kernel_function, sigma, test_X):
    """
    Predict the labels of test_X, using the current SVM.

    alpha: 1 x m learned Lagrangian multipliers (the dual variables).
    train_y: 1 x m labels (-1 or 1) of training data.
    train_X: n x m training feature matrix.
    b: scalar, the bias term in SVM w^T x + b.
    kernel_function: as the name suggests
    sigma: need to be provided when Gaussian kernel is used.
    test_X: n x m2 test feature matrix.
    :return: 1 x m2 vector w^T x + b

    Everything is a matrix.
    """
    
    #########################################
    ## INSERT YOUR CODE HERE
    # print(alpha, type(alpha))
    # print(train_y, type(train_y), train_X, type(train_X))
    # print(test_X, type(test_X))

    # First, train w, then do the calculation. (?)
    # dual_objective_function(alpha, train_y, train_X, kernel_function, sigma)

    # alpha = alpha.A1
    # train_y = train_y.A1

    if kernel_function.__name__ == 'Gaussian_kernel':
        k = kernel_function(train_X, test_X, sigma)
    elif kernel_function.__name__ == 'linear_kernel':
        k = kernel_function(train_X, test_X) #2x2

    return np.dot(np.multiply(alpha, train_y), k) + b

    # sum = b # this turns into an array???
    # if sigma == None:
    #     for i in range(len(alpha)):
    #         sum += alpha[i] * train_y[i] * kernel_function(train_X[:, i], test_X)
    # else:
    #     for i in range(len(alpha)):
    #         sum += alpha[i] * train_y[i] * kernel_function(train_X[:, i], test_X, sigma)
    # return sum

    #########################################