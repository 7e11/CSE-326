# -------------------------------------------------------------------------
'''
    Problem 2: Implement a simple feedforward neural network, the details of the architecture is given in HW6.
'''

from problem1 import *
import numpy as np
from sklearn.metrics import accuracy_score

class NN:

    #--------------------------
    def __init__(self, dimensions, activation_funcs, loss_func, rand_seed = None):
        """
        Specify a L layer feedforward network.
        Design consideration: we don't include data in this neural network class.

        dimensions: list of L+1 integers , with dimensions[i+1] and dimensions[i]
                            being the number of rows and columns for the W at layer i+1.
                            dimensions[0] is the dimension of the data.
                            dimensions[L] is the dimension of output units.
        activation_funcs: dictionary with key=layer number, value = an activation class
        loss_func: loss function at the top layer
        rand_seed: set this to a number if you want deterministic experiments.
        """

        #########################################
        ## INSERT YOUR CODE HERE
        # Code taken from piazza
        np.random.seed(rand_seed)
        self.w = {}
        self.b = {}
        self.z = {}
        self.a = {}
        self.dimensions = dimensions
        self.activation_funcs = activation_funcs
        self.loss_func = loss_func
        self.rand_seed = rand_seed
        self.num_layers = len(dimensions) - 1  # doesn't include input layer.

        for i in range(self.num_layers):
            self.w[i + 1] = np.random.randn(dimensions[i + 1], dimensions[i]) / np.sqrt(dimensions[i])
            self.b[i + 1] = np.zeros((dimensions[i + 1], 1))
        #########################################

    #--------------------------
    def forward(self, X):
        """
        Forward computation of activations at each layer.
        :param X: n x m matrix. m examples in n dimensional space.
        :return:  an n x m matrix (the activations at output layer)
        """
        #########################################
        ## INSERT YOUR CODE HERE
        self.a[0] = X
        # self.z[1] = np.matmul(W[1], X) + self.b[1]
        # self.a[1] = self.activation_funcs[1](z[1])
        for i in range(self.num_layers):
            self.z[i + 1] = np.dot(self.w[i + 1], self.a[i]) + self.b[i + 1]
            self.a[i + 1] = self.activation_funcs[i + 1].activate(self.z[i + 1])
        # print(self.a[self.num_layers])
        return self.a[self.num_layers]
        #########################################

    #--------------------------
    def backward(self, y):
        """
        Back propagation.
        Use the A and Z cached in forward
        :return: two dictionaries of gradients of W and b respectively.
                dW[i] is the gradient of the loss to W[i]
                db[i] is the gradient of the loss to b[i]
        """
        #########################################
        ## INSERT YOUR CODE HERE
        dw = {}
        db = {}
        dz = {}
        da = {}
        da[self.num_layers] = self.loss_func.gradient(y, self.a[self.num_layers])
        
        for i in range(self.num_layers, 0, -1): #num_layers 2 -> 1 (inclusive)
            try:
                dz[i] = np.multiply(da[i], self.activation_funcs[i].gradient(self.z[i]))
            except:
                dz[i] = da[i]
            dw[i] = (1 / y.size) * dz[i] * self.a[i - 1].T #where m is the number of nodes in layer?
            da[i - 1] = self.w[i].T * dz[i]
            db[i] = np.asmatrix((1 / y.size) * np.sum(dz[i].A, axis=1, keepdims=True))
        # print(dw, db)
        return dw, db
        #########################################

    #--------------------------
    def train(self, **kwargs):
        """

        :param kwargs:
        :return: the loss at the final step
        """
        X_train = kwargs['Training X']
        Y_train = kwargs['Training Y']
        iter_num = kwargs['max_iters']
        eta = kwargs['Learning rate']

        record_every = kwargs['record_every']

        losses = []
        grad_norms = []

        # iterations of gradient descent
        for it in range(iter_num):
            #########################################
            ## INSERT YOUR CODE HERE
            y_hat = self.forward(X_train)
            dw, db = self.backward(Y_train)
            norm_dw, norm_db = 0, 0
            for key in dw:
                norm_dw += np.linalg.norm(dw[key])
                self.w[key] -= eta * dw[key]
            for key in db:
                norm_db += np.linalg.norm(db[key])
                self.b[key] -= eta * db[key]
            l = self.loss_func.loss(Y_train, y_hat) # most recent loss entry
            losses.append(l)
            grad_norms.append(norm_dw + norm_db)
            #########################################

            if (it + 1) % record_every == 0:
                print('iterations = {}: loss = {}, gradient norms = {}'.format(it, l, grad_norms), end=' ')
                if 'Test X' in kwargs and 'Test Y' in kwargs:
                    prediction_accuracy = self.test(**kwargs)
                    print (', test error = {}'.format(prediction_accuracy))
                else:
                    print ()
        return l

    #--------------------------
    def test(self, **kwargs):
        """
        Test accuracy of the trained model.
        :return: classification accuracy (for classification) or
                    MSE loss (for regression)
        """
        X_test = kwargs['Test X']
        Y_test = kwargs['Test Y']

        loss_func = kwargs['Test loss function name']

        output = self.forward(X_test)

        if loss_func == '0-1 error':
            predicted_labels = np.argmax(output, axis = 0)
            true_labels = np.argmax(Y_test, axis = 0)
            return 1.0 - accuracy_score(np.array(true_labels).flatten(), np.array(predicted_labels).flatten())
            # return 1.0 - np.where(true_labels == predicted_labels)[0].shape[0] / true_labels.shape[1]
        else:
            # return the Frobenius norm of the difference between y and y_hat, divided by (2m)
            return np.linalg.norm(output - Y_test) ** 2 / (2 * Y_test.shape[1])
