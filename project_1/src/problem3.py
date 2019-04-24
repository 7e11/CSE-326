# -------------------------------------------------------------------------
'''
    Problem 3: compute sigmoid(wTx+b), the loss function, and the gradient.
    This is the single training example version.
'''

import numpy as np # linear algebra

# --------------------------
def linear(w, b, x):
    '''
    w: d x 1, Logistic parameters   numpy matrix
    b: 1 x 1, Logistic parameters   float?
    x: d x 1, an example feature vector. Must be a sparse csc_matrix
    :return: wT x + b
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    # We want to output our algorithm's prediction: z = mx+b
    # https://ml-cheatsheet.readthedocs.io/en/latest/logistic_regression.html
    # https://hackernoon.com/introduction-to-machine-learning-algorithms-logistic-regression-cbdd82d81a36
    # return np.matmul(np.transpose(w), x) + b
    return np.dot(np.transpose(w), x) + b

    # NOTE: When you have 1-D vectors, matmul and dot are the same thing.
    #########################################

# --------------------------
def sigmoid(z):
    '''
    z: scalar. wT x + b
    :return: sigmoid(z)
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    # See: https://en.wikipedia.org/wiki/Sigmoid_function
    return 1 / (1 + np.exp(-z))
    #########################################

# --------------------------
def loss(a, y):
    '''
    a: 1 x 1, sigmoid of the example (x, y)
    y: {0,1}, label of x
    :return: negative log-likelihood loss on (x, y).
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    # https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
    # Could just use an if:

    # if y == 1:
    #     return - np.log(a)
    # else:
    #     return - np.log(1 - a)

    return (y * -np.log(a)) + ((1 - y) * -np.log(1 - a))

    #########################################

# --------------------------
def dz(z, y):
    '''
    z: scalar. wT x + b
    y: {0,1}, label of x
    :return: the gradient of the negative log-likelihood loss on (x, y) wrt z.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    # https://web.stanford.edu/class/archive/cs/cs109/cs109.1178/lectureHandouts/220-logistic-regression.pdf
    return ((1/(np.exp(z)*(1+np.exp(-z))**2))*((-y/(1/(1+np.exp(-z)))) + ((1-y)/(1-(1/(1+np.exp(-z)))))))
    #########################################

# --------------------------
def dw(z, x, y):
    '''
    z: scalar. wT x + b
    x: d x 1, an example feature vector
    y: {0,1}, label of x
    :return: the gradient of the negative log-likelihood loss on (x, y) wrt w.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    return ((1/(np.exp(z)*(1+np.exp(-z))**2)) * ((-y/(1/(1+np.exp(-z))))+((1-y)/(1-(1/(1+np.exp(-z))))))) * x
    #########################################

# --------------------------
def db(z, y):
    '''
    z: scalar. wT x + b
    y: {0,1}, label of x
    :return: the gradient of the negative log-likelihood loss on (x, y) wrt w.
    '''
    #########################################
    ## INSERT YOUR CODE HERE
    return ((1/(np.exp(z)*(1+np.exp(-z))**2)) * ((-y/(1/(1+np.exp(-z))))+((1-y)/(1-(1/(1+np.exp(-z))))))) * 1
    #########################################
