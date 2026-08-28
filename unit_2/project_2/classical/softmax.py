"""
Multinomial (Softmax) regression trained with batch gradient descent
(Notebook 01, section 5).

Softmax regression produces one score per digit and converts the scores
into a probability distribution

    p_k = exp(z_k / tau) / sum_j exp(z_j / tau)

rather than only a hard argmax label; the notebook uses this module to
show a full probability vector, not just the predicted class.
"""
import numpy as np
import scipy.sparse as sparse


def augment_feature_vector(X):
    """Prepends a column of ones so the bias term is folded into theta."""
    column_of_ones = np.ones([len(X), 1])
    return np.hstack((column_of_ones, X))


def compute_probabilities(X, theta, temp_parameter):
    """
    Computes p_k = softmax(theta_k . x / temp_parameter) for every class k
    and every row of X.

    Args:
        X - (n, d) NumPy array
        theta - (k, d) NumPy array, row j = parameters for class j
        temp_parameter - softmax temperature (scalar)

    Returns:
        H - (k, n) NumPy array of class probabilities
    """
    z = theta @ X.T / temp_parameter  # (k, n)
    z = z - np.max(z, axis=0)  # numerical stability, does not change softmax
    exp_z = np.exp(z)
    H = exp_z / np.sum(exp_z, axis=0)
    return H


def compute_cost_function(X, Y, theta, lambda_factor, temp_parameter):
    """
    Total cost = average negative log-likelihood of the true labels
    + an L2 regularization term on theta.
    """
    n = X.shape[0]
    k = theta.shape[0]
    H = compute_probabilities(X, theta, temp_parameter)
    clipped = np.clip(H[Y, np.arange(n)], 1e-15, 1.0)
    data_term = -np.mean(np.log(clipped))
    reg_term = (lambda_factor / 2) * np.sum(theta ** 2)
    return data_term + reg_term


def run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter):
    """One step of batch gradient descent on the Softmax cost function."""
    n, d = X.shape
    k = theta.shape[0]
    H = compute_probabilities(X, theta, temp_parameter)  # (k, n)

    Y_one_hot = sparse.coo_matrix(
        (np.ones(n), (Y, np.arange(n))), shape=(k, n)
    ).toarray()

    grad = -(1 / (temp_parameter * n)) * (Y_one_hot - H) @ X + lambda_factor * theta
    theta = theta - alpha * grad
    return theta


def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations):
    """
    Trains Softmax regression with batch gradient descent.

    Returns:
        theta - (k, d) NumPy array of learned parameters
        cost_function_progression - list of cost values, one per iteration
    """
    theta = np.zeros([k, X.shape[1]])
    cost_function_progression = []
    for _ in range(num_iterations):
        cost_function_progression.append(
            compute_cost_function(X, Y, theta, lambda_factor, temp_parameter)
        )
        theta = run_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter)
    return theta, cost_function_progression


def get_classification(X, theta, temp_parameter):
    """Returns the argmax prediction for every row of X."""
    H = compute_probabilities(X, theta, temp_parameter)
    return np.argmax(H, axis=0)


def compute_test_error(X, Y, theta, temp_parameter):
    """Returns the fraction of examples classified incorrectly."""
    assigned_labels = get_classification(X, theta, temp_parameter)
    return 1 - np.mean(assigned_labels == Y)
