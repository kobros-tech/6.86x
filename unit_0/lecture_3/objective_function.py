"""
Large Margin Linear Classification

Training data
      |
      v
Linear classifier
θᵀx + θ₀
      |
      v
Agreement
y(θᵀx + θ₀)
      |
      +----------------------+
      |                      |
      v                      v
Hinge loss             Regularization
"Does it fit?"         "Is margin large?"
      |                      |
      +----------+-----------+
                 |
                 v
             Objective
                 |
                 v
          MINIMIZE J(θ, θ₀)
"""


import numpy as np
import matplotlib.pyplot as plt


# ==========================================
# Dataset
# ==========================================

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


# ==========================================
# Parameters
# ==========================================

theta = np.array([1.0, 2.0])

theta0 = -5.0

lambda_reg = 0.1


# ==========================================
# Core Functions
# ==========================================

def score(x, theta, theta0):
    """
    Linear classifier:

        f(x) = θᵀx + θ₀
    """

    return np.dot(theta, x) + theta0


def agreement(x, y, theta, theta0):
    """
    Agreement:

        z = y(θᵀx + θ₀)

    Interpretation:

        z > 0
            correctly classified

        z = 0
            exactly on decision boundary

        z < 0
            incorrectly classified

    For large-margin classification:

        z >= 1
            outside the margin

        0 < z < 1
            inside the margin

        z <= 0
            wrong side of decision boundary
    """

    return y * score(x, theta, theta0)


def hinge_loss(z):
    """
    Hinge loss:

        L(z) = max(0, 1 - z)
    """

    return max(0, 1 - z)


def signed_distance(x, y, theta, theta0):
    """
    Signed distance from the decision boundary:

        y(θᵀx + θ₀) / ||θ||
    """

    norm_theta = np.linalg.norm(theta)

    if norm_theta == 0:
        return 0

    return agreement(
        x,
        y,
        theta,
        theta0
    ) / norm_theta


# ==========================================
# Dataset Metrics
# ==========================================

def average_hinge_loss(
        X,
        y,
        theta,
        theta0):
    """
    Average hinge loss:

        1/n Σ max(0, 1 - yᵢ(θᵀxᵢ + θ₀))
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
    L2 regularization:

        1/2 ||θ||²
    """

    return 0.5 * np.dot(theta, theta)


def objective_function(
        X,
        y,
        theta,
        theta0,
        lambda_reg):
    """
    Objective:

        J(θ, θ₀)
        =
        average hinge loss
        +
        λ/2 ||θ||²
    """

    loss_term = average_hinge_loss(
        X,
        y,
        theta,
        theta0
    )

    reg_term = regularization(theta)

    return (
        loss_term
        + lambda_reg * reg_term
    )


# ==========================================
# Plot Dataset
# ==========================================

def plot_dataset(X, y):

    positive_label_added = False
    negative_label_added = False

    for xi, yi in zip(X, y):

        if yi == 1:

            if not positive_label_added:

                plt.scatter(
                    xi[0],
                    xi[1],
                    marker="o",
                    s=100,
                    label="Positive (+1)"
                )

                positive_label_added = True

            else:

                plt.scatter(
                    xi[0],
                    xi[1],
                    marker="o",
                    s=100
                )

        else:

            if not negative_label_added:

                plt.scatter(
                    xi[0],
                    xi[1],
                    marker="x",
                    s=100,
                    label="Negative (-1)"
                )

                negative_label_added = True

            else:

                plt.scatter(
                    xi[0],
                    xi[1],
                    marker="x",
                    s=100
                )


# ==========================================
# Plot Boundary
# ==========================================

def plot_boundary(
        theta,
        theta0,
        level,
        label,
        linestyle="-"):
    """
    Plot:

        θᵀx + θ₀ = level

    level = 0
        decision boundary

    level = 1
        positive margin boundary

    level = -1
        negative margin boundary
    """

    theta1 = theta[0]
    theta2 = theta[1]

    x_values = np.linspace(0, 6, 200)

    if abs(theta2) > 1e-8:

        y_values = (
            level
            - theta1 * x_values
            - theta0
        ) / theta2

        plt.plot(
            x_values,
            y_values,
            linestyle=linestyle,
            linewidth=2,
            label=label
        )

    else:

        x_value = (
            level - theta0
        ) / theta1

        plt.axvline(
            x_value,
            linestyle=linestyle,
            linewidth=2,
            label=label
        )


# ==========================================
# Plot Classification + Margins
# ==========================================

def plot_classifier(
        X,
        y,
        theta,
        theta0):

    plt.figure(figsize=(8, 8))

    # --------------------------
    # Training examples
    # --------------------------

    plot_dataset(X, y)

    # --------------------------
    # Decision boundary
    # θᵀx + θ₀ = 0
    # --------------------------

    plot_boundary(
        theta,
        theta0,
        level=0,
        label="Decision Boundary",
        linestyle="-"
    )

    # --------------------------
    # Positive margin
    # θᵀx + θ₀ = +1
    # --------------------------

    plot_boundary(
        theta,
        theta0,
        level=1,
        label="Positive Margin (+1)",
        linestyle="--"
    )

    # --------------------------
    # Negative margin
    # θᵀx + θ₀ = -1
    # --------------------------

    plot_boundary(
        theta,
        theta0,
        level=-1,
        label="Negative Margin (-1)",
        linestyle="--"
    )

    # --------------------------
    # Plot formatting
    # --------------------------

    plt.xlim(0, 6)
    plt.ylim(0, 6)

    plt.xlabel("x₁")
    plt.ylabel("x₂")

    plt.title(
        "Large Margin Linear Classifier"
    )

    plt.grid(True)

    plt.legend()

    plt.savefig(
        "large_margin_classifier.png"
    )


# ==========================================
# Plot Hinge Loss
# ==========================================

def plot_hinge_loss():

    z_values = np.linspace(-3, 4, 400)

    loss_values = [
        hinge_loss(z)
        for z in z_values
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(
        z_values,
        loss_values,
        linewidth=2,
        label="Hinge Loss"
    )

    # Margin boundary
    plt.axvline(
        1,
        linestyle="--",
        linewidth=1.5,
        label="z = 1"
    )

    # Zero loss
    plt.axhline(
        0,
        linewidth=1
    )

    plt.xlabel(
        "Agreement z = y(θᵀx + θ₀)"
    )

    plt.ylabel(
        "Hinge Loss"
    )

    plt.title(
        "Hinge Loss"
    )

    plt.grid(True)

    plt.legend()

    plt.savefig(
        "hinge_loss.png"
    )


# ==========================================
# Print Agreement Table
# ==========================================

def print_agreement_table():

    print("\nAgreement Table")

    print("=" * 95)

    print(
        f"{'x':<12}"
        f"{'y':<5}"
        f"{'f(x)':>10}"
        f"{'agreement z':>15}"
        f"{'hinge loss':>15}"
        f"{'distance':>15}"
    )

    print("-" * 95)

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

        distance = signed_distance(
            xi,
            yi,
            theta,
            theta0
        )

        print(
            f"{str(xi):<12}"
            f"{yi:<5}"
            f"{f:>10.2f}"
            f"{z:>15.2f}"
            f"{loss:>15.2f}"
            f"{distance:>15.2f}"
        )


# ==========================================
# Main
# ==========================================

def main():

    # ======================================
    # Parameters
    # ======================================

    print("\n")
    print("=" * 60)
    print("LARGE MARGIN CLASSIFICATION")
    print("=" * 60)

    print(
        f"\nθ  = {theta}"
    )

    print(
        f"θ₀ = {theta0}"
    )

    print(
        f"λ  = {lambda_reg}"
    )

    # ======================================
    # Decision boundary equations
    # ======================================

    print("\n")
    print("=" * 60)
    print("BOUNDARIES")
    print("=" * 60)

    print(
        "\nDecision Boundary:"
    )

    print(
        "θᵀx + θ₀ = 0"
    )

    print(
        f"{theta[0]:.3f}x₁ "
        f"+ {theta[1]:.3f}x₂ "
        f"+ {theta0:.3f} = 0"
    )

    print(
        "\nPositive Margin:"
    )

    print(
        "θᵀx + θ₀ = +1"
    )

    print(
        "\nNegative Margin:"
    )

    print(
        "θᵀx + θ₀ = -1"
    )

    # ======================================
    # Agreement table
    # ======================================

    print_agreement_table()

    # ======================================
    # Calculate objective
    # ======================================

    avg_loss = average_hinge_loss(
        X,
        y,
        theta,
        theta0
    )

    reg = regularization(theta)

    obj = objective_function(
        X,
        y,
        theta,
        theta0,
        lambda_reg
    )

    # ======================================
    # Objective results
    # ======================================

    print("\n")
    print("=" * 60)
    print("OBJECTIVE FUNCTION")
    print("=" * 60)

    print(
        f"\nAverage Hinge Loss = "
        f"{avg_loss:.4f}"
    )

    print(
        f"Regularization     = "
        f"{reg:.4f}"
    )

    print(
        f"λ * Regularization = "
        f"{lambda_reg * reg:.4f}"
    )

    print(
        f"\nObjective J        = "
        f"{obj:.4f}"
    )

    print("=" * 60)

    # ======================================
    # Plot classifier
    # ======================================

    plot_classifier(
        X,
        y,
        theta,
        theta0
    )

    # ======================================
    # Plot hinge loss
    # ======================================

    plot_hinge_loss()


# ==========================================
# Run
# ==========================================

if __name__ == "__main__":
    main()


"""
============================================================
LARGE MARGIN CLASSIFICATION
============================================================

θ  = [1. 2.]
θ₀ = -5.0
λ  = 0.1


============================================================
BOUNDARIES
============================================================

Decision Boundary:
θᵀx + θ₀ = 0
1.000x₁ + 2.000x₂ + -5.000 = 0

Positive Margin:
θᵀx + θ₀ = +1

Negative Margin:
θᵀx + θ₀ = -1

Agreement Table
===============================================================================================
x           y          f(x)    agreement z     hinge loss       distance
-----------------------------------------------------------------------------------------------
[4 4]       1          7.00           7.00           0.00           3.13
[5 3]       1          6.00           6.00           0.00           2.68
[3 5]       1          8.00           8.00           0.00           3.58
[1 1]       -1        -2.00           2.00           0.00           0.89
[2 1]       -1        -1.00           1.00           0.00           0.45
[1 2]       -1         0.00          -0.00           1.00          -0.00
[4 1]       -1         1.00          -1.00           2.00          -0.45
[2 5]       1          7.00           7.00           0.00           3.13
[2 2]       -1         1.00          -1.00           2.00          -0.45
[1 5]       1          6.00           6.00           0.00           2.68
[1 3]       -1         2.00          -2.00           3.00          -0.89
[3 4]       1          6.00           6.00           0.00           2.68
[5 4]       1          8.00           8.00           0.00           3.58


============================================================
OBJECTIVE FUNCTION
============================================================

Average Hinge Loss = 0.6154
Regularization     = 2.5000
λ * Regularization = 0.2500

Objective J        = 0.8654
============================================================
"""
