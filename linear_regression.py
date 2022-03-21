import numpy as np
import matplotlib.pyplot as plt

# Implement code to load a csv file

# Implement code for feature scaling


def MSE(X, y, theta, lambda_):
    # returns the cost and gradient of the given theta,
    # according to the MSE cost function

    # the number of training examples
    m = X.shape[0]

    gradient = np.zeros(theta.shape)

    # this is simply the difference between the real value and the predicted value
    error = X @ theta - y

    J = 1 / m * (error @ error.T / 2 + lambda_ * theta[1:] @ theta[1:].T)

    gradient[1:] = 1 / m * (X[:, 1:].T @ error + lambda_ * theta[1:])
    gradient[0] = 1 / m * X[:, 0].T @ error
    return J, gradient


def linear_regression(X, y, initial_theta, alpha, lambda_, iterations):
    # the number of training examples
    m = X.shape[0]

    # adds the column of ones to the left of the designer matrix
    X = np.concatenate([np.ones((m, 1)), X], axis=1)

    theta = initial_theta

    for _ in range(iterations):
        J, gradient = MSE(X, y, theta, lambda_)
        theta -= alpha * gradient

    return theta
