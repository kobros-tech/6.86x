import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# DATASET
# ============================================================

X = np.array([
    [4, 4],
    [5, 3],
    [3, 5],
    [1, 1],
    [2, 1],
    [1, 2],
    [4, 1],
    [2, 5],
    [2, 2],
    [1, 5],
    [1, 3],
    [3, 4],
    [5, 4]
])

y = np.array([
    1,
    1,
    1,
    -1,
    -1,
    -1,
    -1,
    1,
    -1,
    1,
    -1,
    1,
    1
])


# ============================================================
# HYPERPARAMETERS
# ============================================================

lambda_reg = 0.1

learning_rate = 0.01

iterations = 500


# ============================================================
# CORE ML FUNCTIONS
# ============================================================

def score(x, theta, theta0):
    """
    f(x) = theta^T x + theta0
    """

    return np.dot(theta, x) + theta0


def agreement(x, yi, theta, theta0):
    """
    z = yi(theta^T x + theta0)

    This is the quantity used by hinge loss.
    """

    return yi * score(x, theta, theta0)


def hinge_loss(z):
    """
    L(z) = max(0, 1-z)
    """

    return max(0, 1 - z)


def average_hinge_loss(X, y, theta, theta0):
    """
    Average training loss.
    """

    losses = []

    for xi, yi in zip(X, y):

        z = agreement(
            xi,
            yi,
            theta,
            theta0
        )

        losses.append(
            hinge_loss(z)
        )

    return np.mean(losses)


def regularization(theta):
    """
    R(theta) = 1/2 ||theta||^2

    = 1/2 (theta1^2 + theta2^2)
    """

    return 0.5 * np.dot(theta, theta)


def objective_function(
        X,
        y,
        theta,
        theta0,
        lambda_reg):

    loss = average_hinge_loss(
        X,
        y,
        theta,
        theta0
    )

    reg = regularization(theta)

    J = loss + lambda_reg * reg

    return J


# ============================================================
# NUMERICAL GRADIENT
# ============================================================

def numerical_gradient(
        X,
        y,
        theta,
        theta0,
        lambda_reg,
        epsilon=0.0001):

    """
    Estimate derivatives numerically.

    dJ/dtheta1
    dJ/dtheta2
    dJ/dtheta0

    using:

        dJ/dx ≈
        [J(x+epsilon) - J(x-epsilon)]
        --------------------------------
                 2 epsilon
    """

    gradient_theta = np.zeros_like(theta)

    # ------------------------------------
    # derivative with respect to theta1
    # ------------------------------------

    for i in range(len(theta)):

        theta_plus = theta.copy()
        theta_minus = theta.copy()

        theta_plus[i] += epsilon
        theta_minus[i] -= epsilon

        J_plus = objective_function(
            X,
            y,
            theta_plus,
            theta0,
            lambda_reg
        )

        J_minus = objective_function(
            X,
            y,
            theta_minus,
            theta0,
            lambda_reg
        )

        gradient_theta[i] = (
            J_plus - J_minus
        ) / (2 * epsilon)

    # ------------------------------------
    # derivative with respect to theta0
    # ------------------------------------

    J_plus = objective_function(
        X,
        y,
        theta,
        theta0 + epsilon,
        lambda_reg
    )

    J_minus = objective_function(
        X,
        y,
        theta,
        theta0 - epsilon,
        lambda_reg
    )

    gradient_theta0 = (
        J_plus - J_minus
    ) / (2 * epsilon)

    return gradient_theta, gradient_theta0


# ============================================================
# GRADIENT DESCENT
# ============================================================

def optimize(
        X,
        y,
        theta,
        theta0,
        lambda_reg,
        learning_rate,
        iterations):

    """
    Gradient descent:

        theta  <- theta  - alpha * dJ/dtheta
        theta0 <- theta0 - alpha * dJ/dtheta0

    """

    theta_history = []

    theta0_history = []

    objective_history = []

    for iteration in range(iterations):

        # ------------------------------------
        # Store current position
        # ------------------------------------

        J = objective_function(
            X,
            y,
            theta,
            theta0,
            lambda_reg
        )

        theta_history.append(theta.copy())

        theta0_history.append(theta0)

        objective_history.append(J)

        # ------------------------------------
        # Calculate gradient
        # ------------------------------------

        grad_theta, grad_theta0 = numerical_gradient(
            X,
            y,
            theta,
            theta0,
            lambda_reg
        )

        # ------------------------------------
        # Move opposite to gradient
        # ------------------------------------

        theta = (
            theta
            - learning_rate * grad_theta
        )

        theta0 = (
            theta0
            - learning_rate * grad_theta0
        )

    return (
        theta,
        theta0,
        np.array(theta_history),
        np.array(theta0_history),
        np.array(objective_history)
    )


# ============================================================
# DECISION BOUNDARY
# ============================================================

def plot_decision_boundary(
        theta,
        theta0,
        label):

    x_values = np.linspace(0, 6, 200)

    if abs(theta[1]) < 1e-10:
        return

    y_values = (
        -(theta[0] * x_values + theta0)
        / theta[1]
    )

    plt.plot(
        x_values,
        y_values,
        linewidth=2,
        label=label
    )


# ============================================================
# PLOT DATASET
# ============================================================

def plot_dataset():

    positive_plotted = False
    negative_plotted = False

    for xi, yi in zip(X, y):

        if yi == 1:

            if not positive_plotted:

                plt.scatter(
                    xi[0],
                    xi[1],
                    marker="o",
                    s=100,
                    label="Positive"
                )

                positive_plotted = True

            else:

                plt.scatter(
                    xi[0],
                    xi[1],
                    marker="o",
                    s=100
                )

        else:

            if not negative_plotted:

                plt.scatter(
                    xi[0],
                    xi[1],
                    marker="x",
                    s=100,
                    label="Negative"
                )

                negative_plotted = True

            else:

                plt.scatter(
                    xi[0],
                    xi[1],
                    marker="x",
                    s=100
                )


# ============================================================
# OBJECTIVE SURFACE
# ============================================================

def plot_objective_surface():

    """
    We visualize:

        J(theta1, theta0)

    while keeping theta2 fixed.

    This lets us see the "valley" of the objective.
    """

    theta2_fixed = 2.0

    theta1_values = np.linspace(
        -2,
        4,
        100
    )

    theta0_values = np.linspace(
        -8,
        2,
        100
    )

    J_values = np.zeros(
        (
            len(theta0_values),
            len(theta1_values)
        )
    )

    for i, theta0 in enumerate(theta0_values):

        for j, theta1 in enumerate(theta1_values):

            theta = np.array([
                theta1,
                theta2_fixed
            ])

            J_values[i, j] = objective_function(
                X,
                y,
                theta,
                theta0,
                lambda_reg
            )

    T1, T0 = np.meshgrid(
        theta1_values,
        theta0_values
    )

    plt.figure(figsize=(8, 6))

    contour = plt.contourf(
        T1,
        T0,
        J_values,
        levels=40
    )

    plt.colorbar(
        contour,
        label="Objective J"
    )

    plt.xlabel(r"$\theta_1$")

    plt.ylabel(r"$\theta_0$")

    plt.title(
        r"Objective Function $J(\theta_1,\theta_0)$"
        "\n"
        r"with $\theta_2 = 2$"
    )

    # --------------------------------------------------------
    # Optimization path
    # --------------------------------------------------------

    plt.plot(
        theta_history[:, 0],
        theta0_history,
        marker="o",
        markersize=2,
        linewidth=1,
        label="Gradient Descent Path"
    )

    # Starting point

    plt.scatter(
        theta_history[0, 0],
        theta0_history[0],
        s=100,
        marker="o",
        label="Start"
    )

    # Final point

    plt.scatter(
        theta_history[-1, 0],
        theta0_history[-1],
        s=120,
        marker="X",
        label="Final"
    )

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "objective_contour.png"
    )


# ============================================================
# OBJECTIVE VS ITERATION
# ============================================================

def plot_objective_history():

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(len(objective_history)),
        objective_history,
        linewidth=2
    )

    plt.xlabel("Iteration")

    plt.ylabel("Objective J")

    plt.title(
        "Objective Function During Optimization"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "objective_history.png"
    )


# ============================================================
# BEFORE / AFTER CLASSIFIER
# ============================================================

def plot_classifiers(
        initial_theta,
        initial_theta0,
        final_theta,
        final_theta0):

    plt.figure(figsize=(7, 7))

    plot_dataset()

    plot_decision_boundary(
        initial_theta,
        initial_theta0,
        "Initial Boundary"
    )

    plot_decision_boundary(
        final_theta,
        final_theta0,
        "Optimized Boundary"
    )

    plt.xlim(0, 6)

    plt.ylim(0, 6)

    plt.xlabel("$x_1$")

    plt.ylabel("$x_2$")

    plt.title(
        "Decision Boundary Before and After Optimization"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "optimized_boundary.png"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Initial parameters
    # --------------------------------------------------------

    initial_theta = np.array([
        0.5,
        0.5
    ])

    initial_theta0 = 0.0

    print("=" * 60)

    print("INITIAL PARAMETERS")

    print("=" * 60)

    print(
        f"theta  = {initial_theta}"
    )

    print(
        f"theta0 = {initial_theta0}"
    )

    initial_J = objective_function(
        X,
        y,
        initial_theta,
        initial_theta0,
        lambda_reg
    )

    print(
        f"Initial J = {initial_J:.4f}"
    )

    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    global theta_history
    global theta0_history
    global objective_history

    (
        final_theta,
        final_theta0,
        theta_history,
        theta0_history,
        objective_history
    ) = optimize(
        X,
        y,
        initial_theta,
        initial_theta0,
        lambda_reg,
        learning_rate,
        iterations
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    final_J = objective_function(
        X,
        y,
        final_theta,
        final_theta0,
        lambda_reg
    )

    print("\n")

    print("=" * 60)

    print("OPTIMIZATION RESULT")

    print("=" * 60)

    print(
        f"Final theta  = {final_theta}"
    )

    print(
        f"Final theta0 = {final_theta0:.4f}"
    )

    print(
        f"Final J      = {final_J:.4f}"
    )

    print(
        f"Iterations   = {iterations}"
    )

    # --------------------------------------------------------
    # Show gradient at beginning
    # --------------------------------------------------------

    initial_gradient, initial_gradient0 = numerical_gradient(
        X,
        y,
        initial_theta,
        initial_theta0,
        lambda_reg
    )

    print("\n")

    print("=" * 60)

    print("CALCULUS AT START")

    print("=" * 60)

    print(
        f"dJ/dtheta1 = {initial_gradient[0]:.4f}"
    )

    print(
        f"dJ/dtheta2 = {initial_gradient[1]:.4f}"
    )

    print(
        f"dJ/dtheta0 = {initial_gradient0:.4f}"
    )

    print("\nGradient:")

    print(
        initial_gradient,
        initial_gradient0
    )

    print(
        "\nGradient descent moves in the opposite direction."
    )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    plot_objective_surface()

    plot_objective_history()

    plot_classifiers(
        initial_theta,
        initial_theta0,
        final_theta,
        final_theta0
    )


if __name__ == "__main__":
    main()

