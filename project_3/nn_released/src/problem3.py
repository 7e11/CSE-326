# -------------------------------------------------------------------------
'''
    Problem 3: Train and test neural networks on classification and regression tasks.
'''

import numpy as np

from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from problem1 import *
from problem2 import *

#--------------------------
def generate_binary_dataset(n_samples, distribution_name):
    """
    Generate binary classification datasets.
    n_samples: how many examples to have.
    distribution_name: shapes of the distributions.
    :return: training X (columns are examples), training Y (one-hot vectors indicating class labels),
            test X, and test Y
    """
    if distribution_name == 'blobs':
        tr_X, tr_Y = make_blobs(n_samples, centers=2, n_features=2)
        te_X, te_Y = make_blobs(n_samples, centers=2, n_features=2)
    elif distribution_name == 'circles':
        tr_X, tr_Y = make_circles(n_samples, noise=0.01, factor=0.1)
        te_X, te_Y = make_circles(n_samples, noise=0.01, factor=0.1)
    else:
        tr_X, tr_Y = make_moons(n_samples, noise=0.01)
        te_X, te_Y = make_moons(n_samples, noise=0.01)

    scaler = StandardScaler()

    tr_X_scaled = scaler.fit_transform(tr_X, tr_Y)
    tr_X_scaled = tr_X.T
    tr_X = np.asmatrix(tr_X_scaled)

    y_multi_class = np.zeros((2, n_samples))

    for i in range(2):
        y_multi_class[i, np.where(tr_Y == i)] = 1
    tr_Y = np.asmatrix(y_multi_class)

    te_X_scaled = scaler.fit_transform(te_X, te_Y)
    te_X_scaled = te_X.T
    te_X = np.asmatrix(te_X_scaled)

    y_multi_class = np.zeros((2, n_samples))

    for i in range(2):
        y_multi_class[i, np.where(te_Y == i)] = 1
    te_Y = np.asmatrix(y_multi_class)

    return tr_X, tr_Y, te_X, te_Y

#--------------------------
def generate_multi_class_dataset():
    """
    Load the MNIST handwritten digit recognition datasets for multi-class classification.
    :return: training X (columns are examples), training Y (one-hot vectors indicating class labels),
            test X, and test Y
    """
    digits = load_digits()

    X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.33, random_state=42)

    tr_X = X_train.T
    tr_Y = Y_train.T
    te_X = X_test.T
    te_Y = Y_test.T

    num_classes = len(np.unique(tr_Y))
    input_dim, n_samples = tr_X.shape

    tr_y_multi_class = np.zeros((num_classes, n_samples))
    for i in range(num_classes):
        tr_y_multi_class[i, np.where(tr_Y == i)] = 1
    tr_Y = np.asmatrix(tr_y_multi_class)

    input_dim, n_samples = te_X.shape
    te_y_multi_class = np.zeros((num_classes, n_samples))
    for i in range(num_classes):
        te_y_multi_class[i, np.where(te_Y == i)] = 1
    te_Y = np.asmatrix(te_y_multi_class)

    return tr_X, tr_Y, te_X, te_Y

#--------------------------
def generate_regression_dataset(n_samples):
    """
    Generate regression datasets.
    n_samples: how many examples to have.
    distribution_name: shapes of the distributions.
    :return: training X (columns are examples), training Y (one-hot vectors indicating class labels),
    test X, and test Y
    """
    num_classes = 1
    tr_X, tr_Y = make_regression(n_samples=n_samples, n_features=10, n_informative=2, n_targets=num_classes)
    tr_Y = np.exp((tr_Y + abs(tr_Y.min())) / 200)
    tr_Y = np.log1p(tr_Y)

    tr_X = tr_X.T
    tr_Y = np.asmatrix(tr_Y)

    te_X, te_Y = make_regression(n_samples=n_samples, n_features=10, n_informative=2, n_targets=num_classes)
    te_Y = np.exp((te_Y + abs(te_Y.min())) / 200)
    te_Y = np.log1p(te_Y)

    te_X = te_X.T
    te_Y = np.asmatrix(te_Y)

    # transforming the
    return tr_X, tr_Y, te_X, te_Y

if __name__ == '__main__':

    n_samples = 100


    num_classes = 2

    for dist in ['blobs', 'circles', 'moons']:
        print('==========={}==========='.format(dist))
        tr_X, tr_Y, te_X, te_Y = generate_binary_dataset(n_samples, dist)

        input_dim, n_samples = tr_X.shape

        kwargs = {'Training X': tr_X,
                  'Training Y': tr_Y,
                  'Test X': te_X,
                  'Test Y': te_Y,
                  'max_iters': 100,
                  'Learning rate': 0.1,
                  'record_every': 10,
                  'Test loss function name': 'CrossEntropyLoss'
                  }

        # input -> hidden -> output
        dimensions = [input_dim, 4, num_classes]

        activation_funcs = {1: Tanh, 2: Sigmoid}
        loss_func = CrossEntropyLoss
        nn = NN(dimensions, activation_funcs, loss_func)

        # also feed the test data to train() to evaluate the model during training.
        nn.train(**kwargs)

    ## test multiple class prediction
    tr_X, tr_Y, te_X, te_Y = generate_multi_class_dataset()
    print('===========MNIST===========')


    kwargs = {'Training X': tr_X,
              'Training Y': tr_Y,
              'Test X': te_X,
              'Test Y': te_Y,
              'max_iters': 1001,
              'Learning rate': 0.1,
              'record_every': 100,
                'Test loss function name' : '0-1 error'
              }

    # input -> hidden -> output
    input_dim, n_samples = tr_X.shape
    num_classes = tr_Y.shape[0]
    dimensions = [input_dim, 32, num_classes]

    activation_funcs = {1:Tanh, 2:Softmax}
    loss_func = CrossEntropyLoss
    nn = NN(dimensions, activation_funcs, loss_func)
    nn.train(**kwargs)


    ## test regression problem

    num_classes = 1
    tr_X, tr_Y, te_X, te_Y = generate_regression_dataset(n_samples = 100)
    print('===========Regression===========')
    input_dim, n_samples = tr_X.shape

    kwargs = {'Training X': tr_X,
              'Training Y': tr_Y,
              'Test X': te_X,
              'Test Y': te_Y,
              'max_iters': 100,
              'Learning rate': 0.1,
              'record_every': 10,
              'Test loss function name' : 'MSE'
              }

    # input -> hidden -> output
    dimensions = [input_dim, 2, num_classes]

    activation_funcs = {1:Tanh, 2:Identity}
    loss_func = MSELoss
    nn = NN(dimensions, activation_funcs, loss_func)
    nn.train(**kwargs)