# -------------------------------------------------------------------------
'''
    Problem 4: Train an SVM using SMO, on 3 simulated datasets (blobs, circles, and two moons) with 2 kernels (Linear and Gaussian).
    Trained models are saved to file and will be visualized using a jupyter notebook.
'''
from problem1 import *
from problem2 import *
from problem3 import *

from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import pickle
import numpy as np

# --------------------------
def generate_dataset(n_samples, distribution_name):
    if distribution_name == 'blobs':
        X_train, y = make_blobs(n_samples, centers=2, n_features=2, random_state=1)
    elif distribution_name == 'circles':
        X_train, y = make_circles(n_samples, noise=0.01, factor=0.1, random_state=1)
    else:
        X_train, y = make_moons(n_samples, noise=0.01, random_state=1)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train, y)
    X_train_scaled = X_train_scaled.T
    X_train_scaled = np.asmatrix(X_train_scaled)
    y[y == 0] = -1
    y = np.asmatrix(y)
    return X_train_scaled, y

# --------------------------
def train_test_SVM(**kwargs):
    # Instantiate model
    model = SVMModel(kwargs['Training X'],
                     kwargs['Training y'],
                     kwargs['C'],
                     kwargs['kernel'],
                     kwargs['sigma'])

    iter_num, duals, primals, models = train(model, kwargs['max_iters'], kwargs['record_every'])

    for it, d, p in zip(iter_num, duals, primals):
        print('iterations = {}: dual objective value = {}, primal objective value = {}'.format(it, d, p))

    # save your trained model to file
    with open('../data/trained_model_{}_{}.pkl'.format(kwargs['distribution'], kwargs['kernel'].__name__), 'wb') as f:
        pickle.dump(model, f)

    predicted_y = predict(model, kwargs['Test X'])

    # print(np.array(np.asmatrix([[1, -1]])).flatten())
    # accuracy_score(np.array(np.asmatrix([[1, -1]])).flatten(), np.array(np.asmatrix([[1, 1]])).flatten())
    print('Test accuracy = {}'.format(accuracy_score(np.array(kwargs['Test y']).flatten(), np.array(predicted_y).flatten())))

# --------------------------
if __name__ == "__main__":

    C = 1.0
    sigma = 1.0

    n_samples = 100
    for dist in ['blobs', 'circles', 'moons']:
    # for dist in ['moons']:
        print ('Data distribution = {}'.format(dist))
        tr_X, tr_y = generate_dataset(n_samples, dist)
        te_X, te_y = generate_dataset(n_samples, dist)

        for kernel in [linear_kernel, Gaussian_kernel]:
        # for kernel in [Gaussian_kernel]:

            print ('using kernel {}'.format(kernel.__name__))
            kwargs = {'distribution': dist,
                      'Training X': tr_X,
                      'Training y': tr_y,
                      'Test X': te_X,
                      'Test y': te_y,
                      'C': C,
                      'sigma': sigma,
                      'kernel': kernel,
                      'max_iters': 10,
                      'record_every': 1
                      }

            train_test_SVM(**kwargs)
