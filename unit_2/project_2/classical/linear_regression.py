"""
Ridge-regularized linear regression, used as a naive multiclass baseline
(Notebook 01, section 3).

Linear regression is not a classifier by design: it minimizes squared
error against the *numeric* label, not a class-membership objective. It is
included specifically so the notebooks can show, empirically, why treating
digit labels as numbers is a poor baseline.
"""
import numpy as np


def closed_form(X, Y, lambda_factor):
    """
    Closed-form ridge-regression solution.

    theta = (X^T X + lambda I)^{-1} X^T Y

    Args:
        X - (n, d + 1) NumPy array (n datapoints, d features + bias column)
        Y - (n,) NumPy array of numeric labels (0-9)
        lambda_factor - L2 regularization strength (scalar)

    Returns:
        theta - (d + 1,) NumPy array of regression weights
    """
    n, d = X.shape
    reg_matrix = lambda_factor * np.eye(d)
    reg_matrix[0, 0] = 0  # do not regularize the bias term
    theta = np.linalg.solve(X.T @ X + reg_matrix, X.T @ Y)
    return theta


def compute_test_error_linear(test_x, Y, theta):
    """Rounds the regression output to the nearest valid digit and scores it."""
    test_y_predict = np.round(test_x @ theta)
    test_y_predict[test_y_predict < 0] = 0
    test_y_predict[test_y_predict > 9] = 9
    return 1 - np.mean(test_y_predict == Y)
