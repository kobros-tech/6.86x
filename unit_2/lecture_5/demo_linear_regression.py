"""Lecture 5 demo: linear regression from scratch.

This program demonstrates the main mechanics of Lecture 5 using only
NumPy and Matplotlib:

- linear prediction;
- squared-error empirical risk;
- analytical gradient;
- gradient descent;
- closed-form least-squares solution;
- visualization of the fitted line and optimization history.

Run with:

    python demo_linear_regression.py
"""

import numpy as np
import matplotlib.pyplot as plt


def predict(X: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Return predictions X theta."""
    return X @ theta


def empirical_risk(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    """Return 1/(2n) times the mean squared prediction error."""
    errors = predict(X, theta) - y
    return float(np.mean(errors**2) / 2.0)


def gradient(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Return the gradient of the squared-error empirical risk."""
    n = X.shape[0]
    errors = predict(X, theta) - y
    return (X.T @ errors) / n


def gradient_descent(
    X: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.05,
    iterations: int = 2_000,
) -> tuple[np.ndarray, list[float]]:
    """Minimize the squared-error objective with batch gradient descent."""
    theta = np.zeros(X.shape[1], dtype=float)
    history: list[float] = []

    for _ in range(iterations):
        history.append(empirical_risk(X, y, theta))
        theta -= learning_rate * gradient(X, y, theta)

    history.append(empirical_risk(X, y, theta))
    return theta, history


def closed_form_solution(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve X^T X theta = X^T y without explicitly computing an inverse."""
    lhs = X.T @ X
    rhs = X.T @ y
    return np.linalg.solve(lhs, rhs)


def main() -> None:
    rng = np.random.default_rng(7)

    # Small noisy dataset generated around y = 2 + 1.5x.
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    noise = rng.normal(0.0, 0.8, size=x.shape)
    y = 2.0 + 1.5 * x + noise

    # Add an intercept feature: x_tilde = [1, x].
    X = np.column_stack((np.ones_like(x), x))

    print("=" * 70)
    print("LECTURE 5 — LINEAR REGRESSION FROM SCRATCH")
    print("=" * 70)
    print("\nDesign matrix shape:", X.shape)
    print("Target shape:", y.shape)
    print("\nModel: y_hat = theta_0 + theta_1 * x")

    # Gradient descent.
    theta_gd, history = gradient_descent(
        X,
        y,
        learning_rate=0.05,
        iterations=2_000,
    )

    # Closed-form least-squares solution.
    theta_closed = closed_form_solution(X, y)

    print("\nGradient-descent solution:")
    print(f"    theta = {theta_gd}")
    print(f"    objective = {empirical_risk(X, y, theta_gd):.6f}")

    print("\nClosed-form solution:")
    print(f"    theta = {theta_closed}")
    print(f"    objective = {empirical_risk(X, y, theta_closed):.6f}")

    print("\nDifference between solutions:")
    print(
        "    ||theta_gd - theta_closed|| = "
        f"{np.linalg.norm(theta_gd - theta_closed):.8f}"
    )

    # Plot the learned regression line.
    x_plot = np.linspace(x.min() - 0.5, x.max() + 0.5, 200)
    X_plot = np.column_stack((np.ones_like(x_plot), x_plot))
    y_plot = predict(X_plot, theta_closed)

    plt.figure(figsize=(8, 5))
    plt.scatter(x, y, label="training data")
    plt.plot(x_plot, y_plot, label="least-squares line")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Lecture 5 — Linear Regression")
    plt.legend()
    plt.tight_layout()

    # Plot the objective during optimization.
    plt.figure(figsize=(8, 5))
    plt.plot(history)
    plt.xlabel("gradient-descent iteration")
    plt.ylabel("empirical risk")
    plt.title("Squared-error objective during gradient descent")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
