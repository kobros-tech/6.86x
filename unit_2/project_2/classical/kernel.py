"""
Polynomial and Gaussian RBF kernels (Notebook 02, sections 4-5).

A kernel K(x, z) computes an inner product in an implicit feature space
phi without ever constructing phi(x) explicitly -- this is the "kernel
trick" referenced throughout the Unit 2 lectures.
"""
import numpy as np


def polynomial_kernel(X, Y, c, p):
    """
    K(x, y) = (x . y + c)^p, computed for every pair of rows in X and Y.

    Args:
        X - (n, d) NumPy array
        Y - (m, d) NumPy array
        c - trade-off coefficient between low- and high-order terms (scalar)
        p - polynomial degree (scalar)

    Returns:
        kernel_matrix - (n, m) NumPy array
    """
    return (X @ Y.T + c) ** p


def rbf_kernel(X, Y, gamma):
    """
    K(x, y) = exp(-gamma * ||x - y||^2), computed for every pair of rows
    in X and Y.

    Args:
        X - (n, d) NumPy array
        Y - (m, d) NumPy array
        gamma - RBF scale parameter (scalar); larger gamma means a
            narrower, more locally sensitive kernel.

    Returns:
        kernel_matrix - (n, m) NumPy array
    """
    X_sq = np.sum(X ** 2, axis=1).reshape(-1, 1)
    Y_sq = np.sum(Y ** 2, axis=1).reshape(1, -1)
    sq_dists = X_sq + Y_sq - 2 * X @ Y.T
    sq_dists = np.maximum(sq_dists, 0)  # guard against tiny negative values
    return np.exp(-gamma * sq_dists)
