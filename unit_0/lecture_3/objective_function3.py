import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# LECTURE 3 — OPTIMIZING A LINEAR CLASSIFIER
#
# Objective:
#
#       J(theta, theta0)
#
#       = average hinge loss
#         + lambda/2 * ||theta||^2
#
# Hinge loss:
#
#       L_i = max(0, 1 - y_i(theta^T x_i + theta0))
#
# We optimize J using gradient descent.
#
# We use the augmented-vector representation:
#
#       x_tilde     = [x1, x2, 1]
#       theta_tilde = [theta1, theta2, theta0]
#
# ============================================================


# ============================================================
# Dataset
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
], dtype=float)


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
], dtype=float)


# ============================================================
# Augmented dataset
# ============================================================

X_aug = np.hstack(
    (
        X,
        np.ones((X.shape[0], 1))
    )
)


# ============================================================
# Initial parameters
# ============================================================

# theta = [theta1, theta2]
theta_initial = np.array([1.0, 2.0])

# theta0 = bias
theta0_initial = -5.0


# ============================================================
# Optimization parameters
# ============================================================

learning_rate = 0.01

lambda_reg = 0.1

epochs = 1000


# ============================================================
# Core mathematical functions
# ============================================================

def score(x, theta, theta0):
    """
    Linear classifier:

        f(x) = theta^T x + theta0
    """

    return np.dot(theta, x) + theta0


# ------------------------------------------------------------

def prediction(x, theta, theta0):
    """
    Class prediction.

        +1 if f(x) >= 0
        -1 otherwise
    """

    if score(x, theta, theta0) >= 0:
        return 1

    return -1


# ------------------------------------------------------------

def agreement(x, yi, theta, theta0):
    """
    Agreement:

        z = y_i(theta^T x_i + theta0)
    """

    return yi * score(
        x,
        theta,
        theta0
    )


# ------------------------------------------------------------

def hinge_loss(z):
    """
    Hinge loss:

        L(z) = max(0, 1-z)
    """

    return max(0, 1 - z)


# ============================================================
# Dataset losses
# ============================================================

def all_hinge_losses(X, y, theta, theta0):

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

    return np.array(losses)


# ------------------------------------------------------------

def average_hinge_loss(
        X,
        y,
        theta,
        theta0):

    losses = all_hinge_losses(
        X,
        y,
        theta,
        theta0
    )

    return np.mean(losses)


# ============================================================
# Regularization
# ============================================================

def regularization(theta):
    """
    Regularization:

        1/2 ||theta||^2

    IMPORTANT:
    The bias theta0 is NOT regularized here.
    """

    return 0.5 * np.dot(theta, theta)


# ============================================================
# Objective function
# ============================================================

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

    return loss + lambda_reg * reg


# ============================================================
# Training error
# ============================================================

def training_error(
        X,
        y,
        theta,
        theta0):

    mistakes = 0

    for xi, yi in zip(X, y):

        pred = prediction(
            xi,
            theta,
            theta0
        )

        if pred != yi:
            mistakes += 1

    return mistakes / len(X)


# ============================================================
# Gradient of objective function
# ============================================================

def compute_gradient(
        X,
        y,
        theta,
        theta0,
        lambda_reg):

    n = len(X)

    gradient_theta = np.zeros_like(theta)

    gradient_theta0 = 0.0


    # --------------------------------------------------------
    # Hinge-loss gradient
    # --------------------------------------------------------

    for xi, yi in zip(X, y):

        z = agreement(
            xi,
            yi,
            theta,
            theta0
        )

        # Hinge loss is active when:
        #
        #       y_i(theta^T x_i + theta0) < 1
        #
        if z < 1:

            gradient_theta += -yi * xi

            gradient_theta0 += -yi


    # Average over training examples

    gradient_theta /= n

    gradient_theta0 /= n


    # --------------------------------------------------------
    # Regularization gradient
    #
    # derivative of:
    #
    #       lambda/2 * ||theta||^2
    #
    # is:
    #
    #       lambda * theta
    #
    # --------------------------------------------------------

    gradient_theta += lambda_reg * theta


    return gradient_theta, gradient_theta0


# ============================================================
# Gradient Descent
# ============================================================

def optimize(
        X,
        y,
        theta,
        theta0,
        learning_rate,
        lambda_reg,
        epochs):

    theta = theta.copy()

    theta0 = float(theta0)


    # --------------------------------------------------------
    # History
    # --------------------------------------------------------

    objective_history = []

    hinge_history = []

    regularization_history = []

    error_history = []

    theta1_history = []

    theta2_history = []

    theta0_history = []


    # --------------------------------------------------------
    # Optimization loop
    # --------------------------------------------------------

    for epoch in range(epochs):

        # ----------------------------------------------
        # Calculate current objective
        # ----------------------------------------------

        J = objective_function(
            X,
            y,
            theta,
            theta0,
            lambda_reg
        )

        hinge = average_hinge_loss(
            X,
            y,
            theta,
            theta0
        )

        reg = regularization(theta)

        error = training_error(
            X,
            y,
            theta,
            theta0
        )


        # ----------------------------------------------
        # Save history
        # ----------------------------------------------

        objective_history.append(J)

        hinge_history.append(hinge)

        regularization_history.append(reg)

        error_history.append(error)

        theta1_history.append(theta[0])

        theta2_history.append(theta[1])

        theta0_history.append(theta0)


        # ----------------------------------------------
        # Calculate gradient
        # ----------------------------------------------

        gradient_theta, gradient_theta0 = compute_gradient(
            X,
            y,
            theta,
            theta0,
            lambda_reg
        )


        # ----------------------------------------------
        # Gradient descent update
        #
        # theta_new =
        #       theta_old - learning_rate * gradient
        #
        # ----------------------------------------------

        theta = (
            theta
            - learning_rate * gradient_theta
        )

        theta0 = (
            theta0
            - learning_rate * gradient_theta0
        )


        # ----------------------------------------------
        # Print progress
        # ----------------------------------------------

        if epoch % 100 == 0 or epoch == epochs - 1:

            print(
                f"Epoch {epoch:4d} | "
                f"J = {J:.6f} | "
                f"Hinge = {hinge:.6f} | "
                f"Error = {error:.4f} | "
                f"theta = {theta} | "
                f"theta0 = {theta0:.4f}"
            )


    history = {
        "objective": objective_history,
        "hinge": hinge_history,
        "regularization": regularization_history,
        "error": error_history,
        "theta1": theta1_history,
        "theta2": theta2_history,
        "theta0": theta0_history
    }


    return theta, theta0, history


# ============================================================
# Plot dataset
# ============================================================

def plot_dataset():

    for xi, yi in zip(X, y):

        if yi == 1:

            plt.scatter(
                xi[0],
                xi[1],
                marker="o",
                s=100,
                label="Positive"
            )

        else:

            plt.scatter(
                xi[0],
                xi[1],
                marker="x",
                s=100,
                label="Negative"
            )


# ============================================================
# Plot decision boundary
# ============================================================

def plot_decision_boundary(
        theta,
        theta0,
        label):

    x_values = np.linspace(
        0,
        6,
        200
    )


    # theta1*x + theta2*y + theta0 = 0
    #
    # y =
    # -(theta1*x + theta0) / theta2

    if abs(theta[1]) < 1e-12:

        return


    y_values = -(
        theta[0] * x_values
        + theta0
    ) / theta[1]


    plt.plot(
        x_values,
        y_values,
        linewidth=2,
        label=label
    )


# ============================================================
# Plot 1:
# Decision boundary before and after optimization
# ============================================================

def plot_boundaries(
        initial_theta,
        initial_theta0,
        final_theta,
        final_theta0):

    plt.figure(
        figsize=(8, 7)
    )

    plot_dataset()

    plot_decision_boundary(
        initial_theta,
        initial_theta0,
        "Initial boundary"
    )

    plot_decision_boundary(
        final_theta,
        final_theta0,
        "Optimized boundary"
    )

    plt.xlim(0, 6)

    plt.ylim(0, 6)

    plt.xlabel("x1")

    plt.ylabel("x2")

    plt.title(
        "Decision Boundary Before and After Optimization"
    )

    plt.grid(True)

    handles, labels = (
        plt.gca().get_legend_handles_labels()
    )

    unique = dict(
        zip(labels, handles)
    )

    plt.legend(
        unique.values(),
        unique.keys()
    )

    plt.tight_layout()

    plt.savefig(
        "optimized_boundary.png",
        dpi=150
    )


# ============================================================
# Plot 2:
# Objective J versus iteration
# ============================================================

def plot_objective(history):

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history["objective"],
        linewidth=2
    )

    plt.xlabel(
        "Optimization Iteration"
    )

    plt.ylabel(
        "Objective J"
    )

    plt.title(
        "Objective Function Decreasing During Optimization"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "objective_history.png",
        dpi=150
    )


# ============================================================
# Plot 3:
# Hinge loss and regularization
# ============================================================

def plot_loss_components(history):

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history["hinge"],
        label="Average Hinge Loss",
        linewidth=2
    )

    plt.plot(
        history["regularization"],
        label="Regularization",
        linewidth=2
    )

    plt.xlabel(
        "Optimization Iteration"
    )

    plt.ylabel(
        "Value"
    )

    plt.title(
        "Hinge Loss vs Regularization"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "loss_components.png",
        dpi=150
    )


# ============================================================
# Plot 4:
# Training error
# ============================================================

def plot_training_error(history):

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        history["error"],
        linewidth=2
    )

    plt.xlabel(
        "Optimization Iteration"
    )

    plt.ylabel(
        "Training Error"
    )

    plt.title(
        "Training Error During Optimization"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        "training_error.png",
        dpi=150
    )


# ============================================================
# Plot 5:
# Parameter optimization path
#
# This is particularly useful for understanding calculus.
#
# We watch:
#
#       theta1
#       theta2
#
# move through parameter space.
# ============================================================

def plot_parameter_path(history):

    plt.figure(
        figsize=(8, 7)
    )

    plt.plot(
        history["theta1"],
        history["theta2"],
        linewidth=2
    )

    plt.scatter(
        history["theta1"][0],
        history["theta2"][0],
        s=100,
        marker="o",
        label="Start"
    )

    plt.scatter(
        history["theta1"][-1],
        history["theta2"][-1],
        s=100,
        marker="x",
        label="End"
    )

    plt.xlabel(
        r"$\theta_1$"
    )

    plt.ylabel(
        r"$\theta_2$"
    )

    plt.title(
        "Optimization Path in Parameter Space"
    )

    plt.grid(True)

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "parameter_path.png",
        dpi=150
    )


# ============================================================
# Plot 6:
# 3D optimization path
#
# x-axis = theta1
# y-axis = theta2
# z-axis = theta0
# ============================================================

def plot_3d_parameter_path(history):

    fig = plt.figure(
        figsize=(9, 7)
    )

    ax = fig.add_subplot(
        111,
        projection="3d"
    )

    ax.plot(
        history["theta1"],
        history["theta2"],
        history["theta0"],
        linewidth=2
    )

    ax.scatter(
        history["theta1"][0],
        history["theta2"][0],
        history["theta0"][0],
        s=100,
        marker="o",
        label="Start"
    )

    ax.scatter(
        history["theta1"][-1],
        history["theta2"][-1],
        history["theta0"][-1],
        s=100,
        marker="x",
        label="End"
    )

    ax.set_xlabel(
        r"$\theta_1$"
    )

    ax.set_ylabel(
        r"$\theta_2$"
    )

    ax.set_zlabel(
        r"$\theta_0$"
    )

    ax.set_title(
        "3D Parameter Optimization Path"
    )

    ax.legend()

    plt.tight_layout()

    plt.savefig(
        "3d_parameter_path.png",
        dpi=150
    )


# ============================================================
# Print agreement table
# ============================================================

def print_results(
        theta,
        theta0):

    print("\n")
    print("=" * 105)
    print("FINAL AGREEMENT TABLE")
    print("=" * 105)

    print(
        f"{'x':12}"
        f"{'y':5}"
        f"{'f(x)':10}"
        f"{'z':10}"
        f"{'loss':10}"
        f"{'pred':8}"
    )

    print("-" * 105)


    for xi, yi in zip(X, y):

        f = score(
            xi,
            theta,
            theta0
        )

        z = agreement(
            xi,
            yi,
            theta,
            theta0
        )

        loss = hinge_loss(z)

        pred = prediction(
            xi,
            theta,
            theta0
        )

        print(
            f"{str(xi):12}"
            f"{int(yi):5}"
            f"{f:10.3f}"
            f"{z:10.3f}"
            f"{loss:10.3f}"
            f"{pred:8}"
        )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)

    print(
        "LECTURE 3 — LINEAR CLASSIFIER OPTIMIZATION"
    )

    print("=" * 70)


    # --------------------------------------------------------
    # Initial model
    # --------------------------------------------------------

    initial_theta = theta_initial.copy()

    initial_theta0 = theta0_initial


    initial_J = objective_function(
        X,
        y,
        initial_theta,
        initial_theta0,
        lambda_reg
    )

    initial_error = training_error(
        X,
        y,
        initial_theta,
        initial_theta0
    )


    print("\nINITIAL MODEL")
    print("=" * 70)

    print(
        f"theta  = {initial_theta}"
    )

    print(
        f"theta0 = {initial_theta0:.4f}"
    )

    print(
        f"Objective J = {initial_J:.6f}"
    )

    print(
        f"Training Error = {initial_error:.4f}"
    )


    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("STARTING GRADIENT DESCENT")
    print("=" * 70)


    final_theta, final_theta0, history = optimize(
        X,
        y,
        initial_theta,
        initial_theta0,
        learning_rate,
        lambda_reg,
        epochs
    )


    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    final_J = objective_function(
        X,
        y,
        final_theta,
        final_theta0,
        lambda_reg
    )

    final_hinge = average_hinge_loss(
        X,
        y,
        final_theta,
        final_theta0
    )

    final_reg = regularization(
        final_theta
    )

    final_error = training_error(
        X,
        y,
        final_theta,
        final_theta0
    )


    # --------------------------------------------------------
    # Final model
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL OPTIMIZED MODEL")
    print("=" * 70)

    print(
        f"theta1 = {final_theta[0]:.6f}"
    )

    print(
        f"theta2 = {final_theta[1]:.6f}"
    )

    print(
        f"theta0 = {final_theta0:.6f}"
    )

    print()

    print(
        "Decision Boundary:"
    )

    print(
        f"{final_theta[0]:.6f} * x1 + "
        f"{final_theta[1]:.6f} * x2 + "
        f"{final_theta0:.6f} = 0"
    )

    print()

    print(
        f"Average Hinge Loss = "
        f"{final_hinge:.6f}"
    )

    print(
        f"Regularization     = "
        f"{final_reg:.6f}"
    )

    print(
        f"Lambda             = "
        f"{lambda_reg}"
    )

    print(
        f"Objective J        = "
        f"{final_J:.6f}"
    )

    print(
        f"Training Error     = "
        f"{final_error:.4f}"
    )


    # --------------------------------------------------------
    # Agreement table
    # --------------------------------------------------------

    print_results(
        final_theta,
        final_theta0
    )


    # --------------------------------------------------------
    # Plot everything
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("GENERATING PLOTS")
    print("=" * 70)


    plot_boundaries(
        initial_theta,
        initial_theta0,
        final_theta,
        final_theta0
    )


    plot_objective(
        history
    )


    plot_loss_components(
        history
    )


    plot_training_error(
        history
    )


    plot_parameter_path(
        history
    )


    plot_3d_parameter_path(
        history
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()


"""
======================================================================
LECTURE 3 — LINEAR CLASSIFIER OPTIMIZATION
======================================================================

INITIAL MODEL
======================================================================
theta  = [1. 2.]
theta0 = -5.0000
Objective J = 0.865385
Training Error = 0.3077


======================================================================
STARTING GRADIENT DESCENT
======================================================================
Epoch    0 | J = 0.865385 | Hinge = 0.615385 | Error = 0.3077 | theta = [0.99284615 1.99184615] | theta0 = -5.0031
Epoch  100 | J = 0.152324 | Hinge = 0.040298 | Error = 0.0000 | theta = [0.54552068 1.38967046] | theta0 = -5.2038
Epoch  200 | J = 0.086274 | Hinge = 0.000000 | Error = 0.0000 | theta = [0.52320414 1.20344879] | theta0 = -5.2238
Epoch  300 | J = 0.080609 | Hinge = 0.000000 | Error = 0.0000 | theta = [0.55752307 1.13935025] | theta0 = -5.2062
Epoch  400 | J = 0.079098 | Hinge = 0.000104 | Error = 0.0000 | theta = [0.56335267 1.12643818] | theta0 = -5.1823
Epoch  500 | J = 0.079366 | Hinge = 0.000000 | Error = 0.0000 | theta = [0.5646838  1.12484927] | theta0 = -5.1569
Epoch  600 | J = 0.077899 | Hinge = 0.000000 | Error = 0.0000 | theta = [0.55930289 1.11446748] | theta0 = -5.1338
Epoch  700 | J = 0.077447 | Hinge = 0.000000 | Error = 0.0000 | theta = [0.55570559 1.11221965] | theta0 = -5.1092
Epoch  800 | J = 0.077056 | Hinge = 0.000000 | Error = 0.0000 | theta = [0.55693164 1.1080896 ] | theta0 = -5.0846
Epoch  900 | J = 0.076752 | Hinge = 0.000957 | Error = 0.0000 | theta = [0.55760754 1.10905061] | theta0 = -5.0592
Epoch  999 | J = 0.076278 | Hinge = 0.001065 | Error = 0.0000 | theta = [0.55446202 1.10533794] | theta0 = -5.0354


======================================================================
FINAL OPTIMIZED MODEL
======================================================================
theta1 = 0.554462
theta2 = 1.105338
theta0 = -5.035385

Decision Boundary:
0.554462 * x1 + 1.105338 * x2 + -5.035385 = 0

Average Hinge Loss = 0.000000
Regularization     = 0.764600
Lambda             = 0.1
Objective J        = 0.076460
Training Error     = 0.0000


=========================================================================================================
FINAL AGREEMENT TABLE
=========================================================================================================
x           y    f(x)      z         loss      pred    
---------------------------------------------------------------------------------------------------------
[4. 4.]         1     1.604     1.604     0.000       1
[5. 3.]         1     1.053     1.053     0.000       1
[3. 5.]         1     2.155     2.155     0.000       1
[1. 1.]        -1    -3.376     3.376     0.000      -1
[2. 1.]        -1    -2.821     2.821     0.000      -1
[1. 2.]        -1    -2.270     2.270     0.000      -1
[4. 1.]        -1    -1.712     1.712     0.000      -1
[2. 5.]         1     1.600     1.600     0.000       1
[2. 2.]        -1    -1.716     1.716     0.000      -1
[1. 5.]         1     1.046     1.046     0.000       1
[1. 3.]        -1    -1.165     1.165     0.000      -1
[3. 4.]         1     1.049     1.049     0.000       1
[5. 4.]         1     2.158     2.158     0.000       1
"""
