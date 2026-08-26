"""General-purpose matrix factorization utilities for Lecture 7.

The module implements regularized alternating least squares for matrices with
missing entries. The notebooks keep the lecture explanations and experiments,
while the reusable numerical routines live here.
"""

from __future__ import annotations

import numpy as np


def objective(
    Y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    lambda_: float,
) -> float:
    """Return the regularized squared-error objective on observed entries."""
    mask = ~np.isnan(Y)
    residuals = Y[mask] - (U @ V.T)[mask]
    return float(
        0.5 * np.sum(residuals**2)
        + 0.5 * lambda_ * (np.sum(U**2) + np.sum(V**2))
    )


def factorize(
    Y: np.ndarray,
    k: int,
    lambda_: float = 0.05,
    max_iter: int = 500,
    tol: float = 1e-8,
    seed: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Factorize a partially observed matrix using regularized ALS.

    Parameters
    ----------
    Y:
        Rating matrix with missing entries represented by ``np.nan``.
    k:
        Number of latent dimensions.
    lambda_:
        Regularization strength applied to both factor matrices.
    max_iter:
        Maximum number of alternating-minimization iterations.
    tol:
        Stop when the absolute change in the objective is below this value.
    seed:
        Random seed used to initialize the latent factors.

    Returns
    -------
    U, V, history:
        Learned factor matrices and the objective value after each iteration.
    """
    if k < 1:
        raise ValueError("k must be at least 1")
    if lambda_ <= 0:
        raise ValueError("lambda_ must be positive")
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1")
    if tol < 0:
        raise ValueError("tol must be non-negative")

    Y = np.asarray(Y, dtype=float)
    if Y.ndim != 2:
        raise ValueError("Y must be a 2-dimensional matrix")
    if not np.any(~np.isnan(Y)):
        raise ValueError("Y must contain at least one observed rating")

    rng = np.random.default_rng(seed)
    n_users, n_movies = Y.shape
    U = 0.1 * rng.normal(size=(n_users, k))
    V = 0.1 * rng.normal(size=(n_movies, k))
    identity = np.eye(k)
    history: list[float] = []

    for _ in range(max_iter):
        for user in range(n_users):
            observed = ~np.isnan(Y[user])
            A = V[observed].T @ V[observed] + lambda_ * identity
            b = V[observed].T @ Y[user, observed]
            U[user] = np.linalg.solve(A, b)

        for movie in range(n_movies):
            observed = ~np.isnan(Y[:, movie])
            A = U[observed].T @ U[observed] + lambda_ * identity
            b = U[observed].T @ Y[observed, movie]
            V[movie] = np.linalg.solve(A, b)

        current = objective(Y, U, V, lambda_)
        history.append(current)

        if len(history) > 1 and abs(history[-1] - history[-2]) < tol:
            break

    return U, V, np.asarray(history)


def reconstruct(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Return the complete predicted matrix from learned factors."""
    return U @ V.T
