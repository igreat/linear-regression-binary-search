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
    J = 1 / (2*m) * ((error @ error) + lambda_ * theta[1:] @ theta[1:])

    gradient = 1 / m * X.T @ error
    gradient[1:] += 1 / m * (lambda_ * theta[1:])
    return J, gradient


def linear_regression(X, y, initial_theta, alpha, lambda_=0.0, iterations=50000):
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


def polynomial_regression(X, y, initial_theta, p, alpha, lambda_=0.0, iterations=50000):
    # returns the polynomial coefficients that minimize the MSE
    X_p = get_polynomial_features(X, p)
    return linear_regression(X_p, y, initial_theta, alpha, lambda_, iterations)


def get_polynomial_features(X, p):
    # returns the design matrix X with polynomial
    # features added up to degree p
    X_p = np.zeros((X.shape[0], p))
    for i in range(1, p + 1):
        X_p[:, i - 1] = (X**i).ravel()

    return X_p


def plot_graph(theta, min, max):

    number = int((max - min)//0.001)
    x_axis = np.linspace(min, max, number)

    x = np.concatenate([np.ones((x_axis.shape[0], 1)), get_polynomial_features(
        x_axis, theta.shape[0] - 1)], axis=1)

    plt.plot(x_axis, x @ theta)
