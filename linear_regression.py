import numpy as np
import matplotlib.pyplot as plt

# Implement code to load a csv file


def MSE(X, y, theta, lambda_):
    # returns the cost and gradient of the given theta,
    # according to the MSE cost function

    # the number of training examples
    m = X.shape[0]

    gradient = np.zeros(theta.shape)

    # this is simply the difference between the real value and the predicted value
    error = X @ theta - y

    # J is kept track of for debugging purposes
    J = 1 / m * (error @ error.T / 2 + lambda_ * theta[1:] @ theta[1:].T)

    gradient[1:] = 1 / m * (X[:, 1:].T @ error + lambda_ * theta[1:])
    gradient[0] = 1 / m * X[:, 0].T @ error
    return J, gradient


def linear_regression(X, y, initial_theta, alpha, lambda_=0.0, iterations=10000):
    # the number of training examples
    m = X.shape[0]

    # feature normalization
    X_norm = X.copy()
    mu = np.zeros(X.shape[1])
    sigma = np.zeros(X.shape[1])

    mu = np.array([np.mean(X_norm[:, i]) for i in range(X_norm.shape[1])])
    sigma = np.array([np.std(X_norm[:, i]) for i in range(X_norm.shape[1])])

    X_norm = (X_norm - mu) / sigma

    # adds the column of ones to the left of the designer matrix
    X_norm = np.concatenate([np.ones((m, 1)), X_norm], axis=1)
    theta = initial_theta

    # performing gradient descent
    for _ in range(iterations):
        _, gradient = MSE(X_norm, y, theta, lambda_)
        theta -= alpha * gradient

    # "denormalizing" theta
    constant_term = - theta[1:] * mu / sigma
    theta[0] += np.sum(constant_term)

    theta[1:] = theta[1:] / sigma

    return theta
