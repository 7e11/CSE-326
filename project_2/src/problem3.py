# -------------------------------------------------------------------------
'''
    Problem 3: SMO training algorithm

'''
from problem1 import *
from problem2 import *

import numpy as np

import copy

# -------------------------------------------------------------------------
class SVMModel():
    """
    The class containing information about the SVM model, including parameters, data, and hyperparameters
    """
    def __init__(self, train_X, train_y, C, kernel_function, sigma=1):
        # data
        self.train_X = train_X
        self.train_y = train_y
        self.n, self.m = train_X.shape

        # hyper-parameters
        self.C = C
        self.kernel_func = kernel_function
        self.sigma = sigma

        # parameters
        self.alpha = np.zeros((1, self.m))
        self.b = 0

# -------------------------------------------------------------------------

def train(model, max_iters = 10, record_every = 1, max_passes = 1, tol=1e-6):
    """
    SMO training of SVM
    model: an SVMModel
    max_iters: how many iterations of optimization
    record_every: record intermediate dual and primal objective values and models every record_every iterations
    max_passes: each iteration can have maximally max_passes without change any alpha
    tol: numerical tolerance (exact equality of two floating numbers may be impossible).
    :return: 3 lists (of dual objectives, primal objectives, and models)
    """

    iter_num = []
    duals = []
    primals = []
    models = []

    # Precompute Kernel
    if model.kernel_func.__name__ == 'linear_kernel':
        K = model.kernel_func(model.train_X, model.train_X)
    elif model.kernel_func.__name__ == 'Gaussian_kernel':
        K = model.kernel_func(model.train_X, model.train_X, model.sigma)


    for iter in range(1, max_iters):
        num_passes = 0
        while num_passes < max_passes:
            num_changes = 0
            for i in range(model.m):
                # i calculations
                y_i = model.train_y.item(i)
                E_i = np.dot(np.multiply(model.alpha, model.train_y), K[i].T) + model.b - y_i
                alpha_i = model.alpha.item(i)
                if (y_i * E_i < -tol and alpha_i < model.C) or (y_i * E_i > tol and alpha_i > 0):
                    # j calculations
                    j = np.random.randint(1, model.m)
                    while j == i:
                        j = np.random.randint(0, model.m)
                    y_j = model.train_y.item(j)
                    E_j = np.dot(np.multiply(model.alpha, model.train_y), K[j].T) + model.b - y_j
                    alpha_j = model.alpha.item(j)
                    alpha_i_old, alpha_j_old = alpha_i, alpha_j

                    # L and H calulations
                    if y_i == y_j:
                        L = max(0, alpha_j + alpha_i - model.C)
                        H = min(model.C, alpha_j + alpha_i)
                    else:
                        L = max(0, alpha_i - alpha_j)
                        H = min(model.C, model.C + alpha_i - alpha_j)
                    if L == H:
                        continue

                    # ETA Component
                    eta = K[j, j] + K[i, i] - 2 * K[j, i]
                    if eta <= 0:
                        continue
                    alpha_i = alpha_i + y_i * (E_j - E_i) / eta
                    if alpha_i < L:
                        alpha_i = L
                    elif alpha_i > H:
                        alpha_i = H
                    if abs(alpha_i - alpha_i_old) < tol:
                        continue

                    # Final Bias calculation and model updates
                    alpha_j = alpha_j - y_j * y_i * (alpha_i - alpha_i_old)
                    b_1 = model.b - E_j - y_j * (alpha_j - alpha_j_old) * K[j, j] - y_i * (alpha_i - alpha_i_old) * K[j, i]
                    b_2 = model.b - E_i - y_j * (alpha_j - alpha_j_old) * K[j, i] - y_i * (alpha_i - alpha_i_old) * K[i, i]
                    if 0 < alpha_i < model.C:
                        model.b = b_1
                    elif 0 < alpha_j < model.C:
                        model.b = b_2
                    else:
                        model.b = (b_1 + b_2) / 2

                    # Change alpha's
                    model.alpha[0][j] = alpha_j
                    model.alpha[0][i] = alpha_i

                    num_changes += 1
            if num_changes == 0:
                num_passes += 1
            else:
                num_passes = 0

        # Saving records for review
        if iter % record_every == 0:
            iter_num.append(iter)
            duals.append(dual_objective_function(model.alpha, model.train_y, model.train_X, model.kernel_func, model.sigma))
            primals.append(primal_objective_function(model.alpha, model.train_y, model.train_X, model.b, model.C, model.kernel_func, model.sigma))
            models.append(model)
    return iter_num, duals, primals, models

# -------------------------------------------------------------------------
def predict(model, test_X):
    """
    Predict the labels of test_X
    model: an SVMModel
    test_X: n x m matrix, test feature vectors
    :return: 1 x m matrix, predicted labels
    """

    #########################################
    ## INSERT YOUR CODE HERE
    return np.asmatrix([1 if item >= 0 else -1 for item in decision_function(model.alpha, model.train_y, model.train_X, model.b, model.kernel_func, model.sigma, test_X).A1])
    #########################################