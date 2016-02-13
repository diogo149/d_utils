import numpy as np
from scipy.optimize import minimize


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def cost_function(theta, X, y, sample_weight, lambda_=0):
    m = len(y)
    theta = theta.ravel()
    bias, weights = theta[0], theta[1:]  # TODO 1D
    weights = weights.reshape(X.shape[1], y.shape[1])
    h = sigmoid(np.dot(X, weights) + bias)
    J = (-np.dot((y * sample_weight).T, np.log(h))
         - np.dot(((1 - y) * sample_weight).T, np.log(1 - h))) / m
    weights_grad = np.dot(X.T, h - y) / m
    bias_grad = np.dot(np.ones((1, X.shape[0])), h - y) / m
    grad = np.concatenate([bias_grad, weights_grad])

    if lambda_:
        J += lambda_ * np.sum(weights ** 2) / 2 / m

        grad += lambda_ / m * theta.reshape(-1, y.shape[1])
        grad[0] -= lambda_ / m  # for bias

    # TODO use gradient
    # return J, grad.ravel()
    return J


class WeightedLogisticRegression(object):

    """
    a LogisticRegression classifier with sample weights

    TODO this works, but doesn't use gradient information so it's slower
    than it could be
    """

    def __init__(self, C=1.0):
        self.C = C
        self.tol = 0.0001

    def fit(self, X, y, sample_weight=None):
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        # only test for this
        assert y.shape[1] == 1
        if sample_weight is None:
            sample_weight = np.ones(X.shape[0])
        res = minimize(
            cost_function,
            x0=np.zeros((1 + X.shape[1]) * y.shape[1]),
            args=(X,
                  y,
                  sample_weight.reshape(-1, 1),
                  1.0 / self.C),
            method="BFGS",
            # TODO use gradient
            # jac=True,
            tol=self.tol,
        )
        self.bias_ = res.x[0]  # TODO 1D
        self.coef_ = res.x[1:].reshape(X.shape[1], y.shape[1])
        return self

    def predict_proba(self, X):
        probs = sigmoid(np.dot(X, self.coef_))
        return np.hstack([1 - probs, probs])
