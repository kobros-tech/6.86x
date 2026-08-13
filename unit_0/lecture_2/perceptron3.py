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
# Helper Functions
# ==========================================

def sign(z):
    """
    Return class label.
    """

    if z > 0:
        return 1

    elif z < 0:
        return -1

    else:
        return 0


def score(x, theta, theta0):
    """
    Calculate the linear score:

        theta^T x + theta0

    where:

        theta  = [w1, w2]
        theta0 = bias
    """

    return np.dot(theta, x) + theta0


def predict(x, theta, theta0):
    """
    Perceptron prediction:

        sign(theta^T x + theta0)
    """

    return sign(
        score(x, theta, theta0)
    )


def agreement(x, y, theta, theta0):
    """
    Calculate the agreement:

        y_i(theta^T x_i + theta0)

    This quantity will become important
    in Lecture 3 when we introduce hinge loss.
    """

    return y * score(
        x,
        theta,
        theta0
    )


def training_error(X, y, theta, theta0):
    """
    Fraction of misclassified points.
    """

    mistakes = 0

    for xi, yi in zip(X, y):

        if predict(xi, theta, theta0) != yi:
            mistakes += 1

    return mistakes / len(X)


# ==========================================
# Perceptron Algorithm
# ==========================================

def perceptron(X, y, epochs=20):

    # --------------------------------------
    # Initialize parameters
    # --------------------------------------

    n_features = X.shape[1]

    theta = np.zeros(n_features)

    theta0 = 0

    history = []

    # ======================================
    # Training
    # ======================================

    for epoch in range(epochs):

        mistakes = 0

        print(
            f"\nEpoch {epoch + 1}"
        )

        print("-" * 60)

        # ----------------------------------
        # Process every training example
        # ----------------------------------

        for xi, yi in zip(X, y):

            # ----------------------------------
            # Linear score
            #
            # theta^T x_i + theta0
            # ----------------------------------

            linear_score = score(
                xi,
                theta,
                theta0
            )

            # ----------------------------------
            # Prediction
            #
            # sign(theta^T x_i + theta0)
            # ----------------------------------

            prediction = sign(
                linear_score
            )

            # ----------------------------------
            # Agreement
            #
            # y_i(theta^T x_i + theta0)
            # ----------------------------------

            z = yi * linear_score

            print(
                f"x={xi}"
                f" y={yi}"
                f" score={linear_score:.3f}"
                f" agreement={z:.3f}"
                f" pred={prediction}"
            )

            # ----------------------------------
            # Perceptron update
            # ----------------------------------

            if prediction != yi:

                # theta <- theta + y_i x_i
                theta = theta + yi * xi

                # theta0 <- theta0 + y_i
                theta0 = theta0 + yi

                mistakes += 1

                print(
                    f"  Mistake:"
                    f" theta={theta}"
                    f" theta0={theta0}"
                )

        # ----------------------------------
        # Training error
        # ----------------------------------

        error = training_error(
            X,
            y,
            theta,
            theta0
        )

        history.append(error)

        print(
            f"Training Error = {error:.3f}"
        )

        # ----------------------------------
        # Stop if no mistakes
        # ----------------------------------

        if mistakes == 0:

            print(
                "\nConverged!"
            )

            break

    return theta, theta0, history


# ==========================================
# Plot Dataset
# ==========================================

def plot_dataset(X, y):

    positive_label = False
    negative_label = False

    for xi, yi in zip(X, y):

        if yi == 1:

            if not positive_label:

                plt.scatter(
                    xi[0],
                    xi[1],
                    marker="o",
                    s=100,
                    label="Positive"
                )

                positive_label = True

            else:

                plt.scatter(
                    xi[0],
                    xi[1],
                    marker="o",
                    s=100
                )

        else:

            if not negative_label:

                plt.scatter(
                    xi[0],
                    xi[1],
                    marker="x",
                    s=100,
                    label="Negative"
                )

                negative_label = True

            else:

                plt.scatter(
                    xi[0],
                    xi[1],
                    marker="x",
                    s=100
                )


# ==========================================
# Plot Decision Boundary
# ==========================================

def plot_decision_boundary(theta, theta0):
    """
    Decision boundary:

        theta^T x + theta0 = 0

    For two dimensions:

        w1*x1 + w2*x2 + theta0 = 0

    Therefore:

        x2 = -(w1*x1 + theta0) / w2
    """

    if abs(theta[1]) < 1e-8:

        return

    x_values = np.linspace(
        0,
        6,
        100
    )

    y_values = -(
        theta[0] * x_values + theta0
    ) / theta[1]

    plt.plot(
        x_values,
        y_values,
        linewidth=2,
        label="Decision Boundary"
    )


# ==========================================
# Main
# ==========================================

def main():

    # ======================================
    # Train Perceptron
    # ======================================

    theta, theta0, history = perceptron(
        X,
        y
    )

    # ======================================
    # Final Parameters
    # ======================================

    print(
        "\n" + "=" * 40
    )

    print(
        "Final Parameters"
    )

    print(
        "=" * 40
    )

    print(
        f"theta = {theta}"
    )

    print(
        f"theta0 = {theta0:.3f}"
    )

    print(
        f"\nw1   = {theta[0]:.3f}"
    )

    print(
        f"w2   = {theta[1]:.3f}"
    )

    print(
        f"bias = {theta0:.3f}"
    )

    # ======================================
    # Decision Boundary
    # ======================================

    print(
        "\nDecision Boundary:"
    )

    print(
        f"{theta[0]:.3f}*x + "
        f"{theta[1]:.3f}*y + "
        f"{theta0:.3f} = 0"
    )

    # ======================================
    # Predictions
    # ======================================

    print(
        "\nPredictions"
    )

    print(
        "=" * 70
    )

    for xi, yi in zip(X, y):

        # ----------------------------------
        # theta^T x_i + theta0
        # ----------------------------------

        linear_score = score(
            xi,
            theta,
            theta0
        )

        # ----------------------------------
        # y_i(theta^T x_i + theta0)
        # ----------------------------------

        z = agreement(
            xi,
            yi,
            theta,
            theta0
        )

        # ----------------------------------
        # Prediction
        # ----------------------------------

        pred = predict(
            xi,
            theta,
            theta0
        )

        print(
            f"x={xi}"
            f" true={yi}"
            f" score={linear_score:.3f}"
            f" agreement={z:.3f}"
            f" pred={pred}"
        )

    # ======================================
    # Plot Data + Decision Boundary
    # ======================================

    plt.figure(
        figsize=(6, 6)
    )

    plot_dataset(
        X,
        y
    )

    plot_decision_boundary(
        theta,
        theta0
    )

    plt.xlim(
        0,
        6
    )

    plt.ylim(
        0,
        6
    )

    plt.grid(
        True
    )

    plt.title(
        "Perceptron with Explicit Bias"
    )

    handles, labels = (
        plt.gca()
        .get_legend_handles_labels()
    )

    unique = dict(
        zip(
            labels,
            handles
        )
    )

    plt.legend(
        unique.values(),
        unique.keys()
    )

    plt.savefig(
        "boundary.png"
    )

    # ======================================
    # Plot Training Error
    # ======================================

    plt.figure(
        figsize=(6, 4)
    )

    plt.plot(
        range(
            1,
            len(history) + 1
        ),
        history,
        marker="o"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Training Error"
    )

    plt.title(
        "Training Error During Learning"
    )

    plt.grid(
        True
    )

    plt.savefig(
        "training_error.png"
    )


# ==========================================
# Run
# ==========================================

if __name__ == "__main__":

    main()

"""
Converged!

========================================
Final Parameters
========================================
theta = [2. 4.]
theta0 = -15.000

w1   = 2.000
w2   = 4.000
bias = -15.000

Decision Boundary:
2.000*x + 4.000*y + -15.000 = 0

Predictions
======================================================================
x=[4 4] true=1 score=9.000 agreement=9.000 pred=1
x=[5 3] true=1 score=7.000 agreement=7.000 pred=1
x=[3 5] true=1 score=11.000 agreement=11.000 pred=1
x=[1 1] true=-1 score=-9.000 agreement=9.000 pred=-1
x=[2 1] true=-1 score=-7.000 agreement=7.000 pred=-1
x=[1 2] true=-1 score=-5.000 agreement=5.000 pred=-1
x=[4 1] true=-1 score=-3.000 agreement=3.000 pred=-1
x=[2 5] true=1 score=9.000 agreement=9.000 pred=1
x=[2 2] true=-1 score=-3.000 agreement=3.000 pred=-1
x=[1 5] true=1 score=7.000 agreement=7.000 pred=1
x=[1 3] true=-1 score=-1.000 agreement=1.000 pred=-1
x=[3 4] true=1 score=7.000 agreement=7.000 pred=1
x=[5 4] true=1 score=11.000 agreement=11.000 pred=1
"""
