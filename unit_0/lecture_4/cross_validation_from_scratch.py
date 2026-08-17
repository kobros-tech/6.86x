"""
Lecture 4 — Cross-Validation From Scratch.

Learning-first implementation built on the Lecture 3 linear SVM objective.
"""

import numpy as np


def score(X, theta, theta0):
    return X @ theta + theta0


def predict(X, theta, theta0):
    return np.where(score(X, theta, theta0) >= 0, 1, -1)


def hinge_loss(z):
    return np.maximum(0.0, 1.0 - z)


def average_hinge_loss(X, y, theta, theta0):
    return np.mean(hinge_loss(y * score(X, theta, theta0)))


def regularization(theta):
    return 0.5 * np.dot(theta, theta)


def objective(X, y, theta, theta0, alpha):
    return average_hinge_loss(X, y, theta, theta0) + alpha * regularization(theta)


def accuracy(X, y, theta, theta0):
    return float(np.mean(predict(X, theta, theta0) == y))


def gradient(X, y, theta, theta0, alpha):
    """Gradient of average hinge loss + alpha/2 * ||theta||^2."""
    z = y * score(X, theta, theta0)
    active = z < 1.0

    # IMPORTANT: divide by the total number of examples, not the number
    # of active hinge-loss examples. This matches the average loss.
    grad_theta = -(y[active, None] * X[active]).sum(axis=0) / len(X)
    grad_theta0 = -y[active].sum() / len(X)

    grad_theta += alpha * theta
    return grad_theta, grad_theta0


def fit_linear_svm(
    X,
    y,
    alpha,
    learning_rate=0.01,
    epochs=1000,
    theta_init=None,
    theta0_init=0.0,
    verbose=False,
):
    """Train the Lecture 3 classifier for one fixed alpha."""
    theta = (
        np.zeros(X.shape[1], dtype=float)
        if theta_init is None
        else np.asarray(theta_init, dtype=float).copy()
    )
    theta0 = float(theta0_init)

    history = {"objective": [], "accuracy": []}

    for epoch in range(epochs):
        J = objective(X, y, theta, theta0, alpha)
        acc = accuracy(X, y, theta, theta0)
        history["objective"].append(J)
        history["accuracy"].append(acc)

        grad_theta, grad_theta0 = gradient(X, y, theta, theta0, alpha)
        theta -= learning_rate * grad_theta
        theta0 -= learning_rate * grad_theta0

        if verbose and (epoch % 100 == 0 or epoch == epochs - 1):
            print(
                f"epoch={epoch:4d} | J={J:.6f} | accuracy={acc:.3f} | "
                f"theta={theta} | theta0={theta0:.4f}"
            )

    return theta, theta0, history


def kfold_indices(n_samples, k=5, seed=42, shuffle=True):
    """Return k non-overlapping validation folds."""
    if k < 2 or k > n_samples:
        raise ValueError("k must satisfy 2 <= k <= n_samples")

    indices = np.arange(n_samples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    return np.array_split(indices, k)


def cross_validate_alpha(
    X,
    y,
    alphas,
    k=5,
    seed=42,
    learning_rate=0.01,
    epochs=1000,
):
    """Evaluate each alpha with K-fold validation and return alpha*."""
    folds = kfold_indices(len(X), k=k, seed=seed)
    results = []

    for alpha in alphas:
        fold_scores = []

        for validation_fold in range(k):
            validation_idx = folds[validation_fold]
            training_idx = np.concatenate(
                [folds[i] for i in range(k) if i != validation_fold]
            )

            theta, theta0, _ = fit_linear_svm(
                X[training_idx],
                y[training_idx],
                alpha=alpha,
                learning_rate=learning_rate,
                epochs=epochs,
            )

            fold_scores.append(
                accuracy(
                    X[validation_idx],
                    y[validation_idx],
                    theta,
                    theta0,
                )
            )

        mean_score = float(np.mean(fold_scores))
        results.append(
            {
                "alpha": float(alpha),
                "fold_scores": fold_scores,
                "mean_validation_accuracy": mean_score,
            }
        )

        print(
            f"alpha={alpha:.5g} | folds={np.round(fold_scores, 3)} | "
            f"mean={mean_score:.3f}"
        )

    best = max(results, key=lambda row: row["mean_validation_accuracy"])
    return results, best["alpha"]


if __name__ == "__main__":
    X = np.array(
        [
            [4, 4], [5, 3], [3, 5],
            [1, 1], [2, 1], [1, 2],
            [4, 1], [2, 5], [2, 2],
            [1, 5], [1, 3], [3, 4], [5, 4],
        ],
        dtype=float,
    )
    y = np.array(
        [1, 1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, 1],
        dtype=float,
    )
    alphas = np.array([1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1])
    results, alpha_star = cross_validate_alpha(X, y, alphas)
    theta, theta0, _ = fit_linear_svm(X, y, alpha_star, verbose=True)
    print("\nSelected alpha*:", alpha_star)
    print("Final training accuracy:", accuracy(X, y, theta, theta0))
