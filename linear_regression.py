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


def linear_regression_BS(X, y, initial_theta, iterations=100, epsilon=0.00001):

    # an experiment to see how well a divide and conquer approach works for linear search
    # still has a bunch of bugs and undefined behaviour associated with it that could be fixed
    # conclusion: the gradient descent method is much more reliable and efficient, even for
    #             something as simple as linear regression

    m = X.shape[0]
    X = np.concatenate([np.ones((m, 1)), X], axis=1)
    theta = initial_theta

    initial_direction = get_slope_direction(X, y, theta, epsilon)
    left_bound = np.zeros(theta.shape)
    right_bound = np.zeros(theta.shape)

    # getting the left and right bounds
    if initial_direction:
        theta2 = theta + np.max(y) * 10000
        if get_slope_direction(X, y, theta2, epsilon) != initial_direction:
            right_bound = theta2
            left_bound = theta
    else:
        theta2 = theta - np.max(y) * 10000
        if get_slope_direction(X, y, theta2, epsilon) != initial_direction:
            left_bound = theta2
            right_bound = theta

    for _ in range(iterations):
        mid = (left_bound + right_bound) / 2
        if get_slope_direction(X, y, mid, epsilon):
            left_bound = mid
        else:
            right_bound = mid

    return (right_bound + left_bound) / 2


def polynomial_regression_BS(X, y, initial_theta, p, iterations=100, epsilon=0.00001):
    X_p = get_polynomial_features(X, p)
    return linear_regression_BS(X_p, y, initial_theta, iterations, epsilon)


def get_slope_direction(X, y, theta, epsilon):
    # returns True for right, False for left
    error_left, _ = MSE(X, y, theta - epsilon, 0)
    error_right, _ = MSE(X, y, theta + epsilon, 0)
    return error_left > error_right


def plot_graph(theta, min, max):

    number = int((max - min)//0.001)
    x_axis = np.linspace(min, max, number)

    x = np.concatenate([np.ones((x_axis.shape[0], 1)), get_polynomial_features(
        x_axis, theta.shape[0] - 1)], axis=1)

    plt.plot(x_axis, x @ theta)


#################### SAMPLE TEST EXAMPLE #####################
X = np.arange(10).reshape(-1, 1) + np.random.random((10, 1))*5
y = 2 * np.arange(10)**2 + 5 * np.arange(10) + 7

initial_theta = np.zeros(3)

theta = polynomial_regression_BS(X, y, initial_theta, 2)

min = np.min(X)
max = np.max(X)
plt.plot(X, y, "ro")
plot_graph(theta, min, max)
plt.show()
