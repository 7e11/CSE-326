# -------------------------------------------------------------------------
'''
    Problem 1: Implement activation and loss functions.

    Note: A static method is also a method which is bound to the class and not the object of the class.
        @staticmethod is used before the function definition to indicate that the function is static.
'''

import numpy as np
from scipy.special import logsumexp
from scipy.special import xlogy, xlog1py

# --------------------------
class Activation():
    def __init__(self):
        raise NotImplementedError

    @staticmethod
    def activate(Z):
        raise NotImplementedError

    @staticmethod
    def gradient(Z):
        raise NotImplementedError

# --------------------------
class Sigmoid(Activation):
    """
    Implement the sigmoid activation function and its gradient
    """

    @staticmethod
    def activate(Z):
        """
        Sigmoid of each element of Z.
        Z: n x m matrix
        :return: elementwise sigmoid of Z
        """
        #########################################
        ## INSERT YOUR CODE HERE
        return 1 / (1 + np.exp(-Z))
        #########################################

    @staticmethod
    def gradient(Z):
        """
        Gradient of sigmoid at Z
        Z: n x m matrix
        :return: elementwise gradient of sigmoid at Z
        """
        #########################################
        ## INSERT YOUR CODE HERE
        return np.multiply(Sigmoid.activate(Z), 1 - Sigmoid.activate(Z))
        #########################################

# --------------------------
class Softmax(Activation):
    """
    Implement the softmax activation function, for multi-class classification.
    """

    @staticmethod
    def activate(Z):
        """
        Transform each column of Z into a probability distribution through the Softmax mapping.
        Read this article

            https://blog.feedly.com/tricks-of-the-trade-logsumexp/

        to avoid numerical problem in calculating the denominator
        Z: n x m matrix
        :return: each column of Z get transformed to a probability distribution.
        """
        #########################################
        ## INSERT YOUR CODE HERE
        arr = []
        for col in Z.T:
            col -= np.max(col)  # normalizing trick
            if type(col) == np.ndarray:
                arr.append((np.exp(col) / np.sum(np.exp(col))))
            else:
                arr.append((np.exp(col) / np.sum(np.exp(col))).A1)
        return np.asmatrix(arr).T
        #########################################

    @staticmethod
    def gradient(Z):
        """
        No need to implement gradient of softmax wrt Z. The reason is that,
        cross_entropy_loss ( softmax(w^T x + b), Y) can be differentiated wrt w^T x + b = Z directly.
        This is implemented in the gradient of cross entropy.
        """
        raise NotImplementedError

# --------------------------
class Identity(Activation):
    """
        Implement the identify activation function, for regression problems.
        """

    @staticmethod
    def activate(Z):
        """
        Z: n x m matrix
        :return: just return the argument and you don't need to do anything here.
        """
        return Z

    @staticmethod
    def gradient(Z):
        """
        No need to implement gradient of softmax wrt Z. The reason is that,
        cross_entropy_loss ( softmax(w^T x + b), Y) can be differentiated wrt w^T x + b = Z directly.
        This is implemented in the gradient of cross entropy.
        """
        raise NotImplementedError


# --------------------------
class Tanh(Activation):
    """
    Implement the tanh activation function and its gradient
    """

    @staticmethod
    def activate(Z):
        """
        Z: n x m matrix
        :return: elementwise tanh of Z
        """
        #########################################
        ## INSERT YOUR CODE HERE
        return np.tanh(Z)
        #########################################

    @staticmethod
    def gradient(Z):
        """
        Gradient of tanh at Z
        :return: elementwise gradient of tanh at Z
        """
        #########################################
        ## INSERT YOUR CODE HERE
        return 1 - np.multiply(np.tanh(Z), np.tanh(Z))
        #########################################

# --------------------------
class Loss():
    @staticmethod
    def loss(Y, Y_hat):
        raise NotImplementedError

    @staticmethod
    def gradient(Y, Y_hat):
        raise NotImplementedError

# --------------------------
class CrossEntropyLoss(Loss):

    """
    Define the cross entropy loss and its gradient with respect to the input linear term (see below)
    Cross entropy loss is defined here:

        https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html#cross-entropy

    """
    @staticmethod
    def loss(Y, Y_hat):
        """
        Refer to
        https://stackoverflow.com/questions/50018625/how-to-handle-log0-when-using-cross-entropy
        to see how to avoid taking log of 0.
        Think: when and why log 0 happens? - at 100% incorrect confidence.
        Y: k x m the ground truth labels of m training examples. Y[j, i]=1 if the i-th example has label j.
        Y_hat: k x m the multi-class or multi-label predictions on the m training examples
        :return: a scalar that is the averaged cross entropy loss
        """
        #########################################
        ## INSERT YOUR CODE HERE
        res = - np.sum(xlogy(Y, Y_hat), axis=0)
        print(res)
        # print(np.mean(res))
        return np.mean(res)
        #########################################

    @staticmethod
    def gradient(Y, Y_hat):
        """
        It is the gradient of cross entropy loss with respect to Z that is used to compute Y_hat, NOT wrt Y_hat.
        Y: k x m the ground truth labels of m training examples. Y[j, i]=1 if the i-th example has label j.
        Y_hat: k x m the multi-class or multi-label predictions on the m training examples
        :return: 1 x m vector, the gradients on the m training examples
        """
        #########################################
        ## INSERT YOUR CODE HERE
        # FIXME: Is this correct?
        return Y_hat - Y
        #########################################

# --------------------------
class MSELoss(Loss):

    """
    Define the Mean Square Error loss and its gradient with respect to the input linear term (see below)
    """
    @staticmethod
    def loss(Y, Y_hat):
        """
        Y: k x m the ground truth k-values of m training examples.
        Y_hat: k x m the regression predictions on the m training examples
        :return: a scalar that is the averaged MSE loss
        """
        #########################################
        ## INSERT YOUR CODE HERE
        return np.mean(np.mean(np.square(Y_hat - Y), axis=1))
        #########################################

    @staticmethod
    def gradient(Y, Y_hat):
        """
        It is the gradient of MSE loss with respect to Y_hat.
        Y: k x m the ground truth k-values of m training examples.
        Y_hat: k x m the regression predictions on the m training examples
        :return: 1 x m vector, the gradients on the m training examples
        """
        #########################################
        ## INSERT YOUR CODE HERE
        return Y_hat - Y
        #########################################
