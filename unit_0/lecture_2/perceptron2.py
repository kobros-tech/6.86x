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
    [1, 2]
])

y = np.array([
    1,
    1,
    1,
    -1,
    -1,
    -1
])

# Add bias feature (constant 1)
X_aug = np.hstack((X, np.ones((X.shape[0], 1))))

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


def predict(x, theta):
    """
    Perceptron prediction.
    """

    return sign(np.dot(theta, x))


def training_error(X, y, theta):
    """
    Fraction of misclassified points.
    """

    mistakes = 0

    for xi, yi in zip(X, y):

        if predict(xi, theta) != yi:
            mistakes += 1

    return mistakes / len(X)


# ==========================================
# Perceptron Algorithm
# ==========================================

def perceptron(X, y, epochs=20):

    n_features = X.shape[1]

    theta = np.zeros(n_features)

    history = []

    for epoch in range(epochs):

        mistakes = 0

        print(f"\nEpoch {epoch + 1}")
        print("-" * 30)

        for xi, yi in zip(X, y):

            prediction = predict(xi, theta)

            # if prediction != yi:
            if yi * np.dot(theta, xi) <= 0:

                theta = theta + yi * xi

                mistakes += 1

                print(
                    f"Mistake:"
                    f" x={xi}"
                    f" y={yi}"
                    f" -> theta={theta}"
                )

        error = training_error(X, y, theta)

        history.append(error)

        print(f"Training Error = {error:.3f}")

        if mistakes == 0:
            print("\nConverged!")
            break

    return theta, history


# ==========================================
# Plot Dataset
# ==========================================

def plot_dataset(X, y):

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


# ==========================================
# Plot Decision Boundary
# ==========================================

def plot_decision_boundary(theta):
    """
    theta[0] = w1
    theta[1] = w2
    theta[2] = bias
    """

    if abs(theta[1]) < 1e-8:
        return

    x_values = np.linspace(0, 6, 100)

    y_values = -(
        theta[0] * x_values + theta[2]
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

    theta, history = perceptron(X_aug, y)

    print("\n" + "=" * 30)
    print("Final Theta")
    print("=" * 30)

    print(f"w1   = {theta[0]:.3f}")
    print(f"w2   = {theta[1]:.3f}")
    print(f"bias = {theta[2]:.3f}")

    print("\nDecision Boundary:")
    print(
        f"{theta[0]:.3f}*x + "
        f"{theta[1]:.3f}*y + "
        f"{theta[2]:.3f} = 0"
    )

    print("\nPredictions")
    print("=" * 30)

    for xi, yi in zip(X_aug, y):

        pred = predict(xi, theta)

        print(
            f"x={xi[:2]}"
            f" true={yi}"
            f" pred={pred}"
        )

    # --------------------------
    # Plot data + boundary
    # --------------------------

    plt.figure(figsize=(6, 6))

    plot_dataset(X, y)
    plot_decision_boundary(theta)

    plt.xlim(0, 6)
    plt.ylim(0, 6)

    plt.grid(True)
    plt.title("Perceptron with Bias")

    handles, labels = plt.gca().get_legend_handles_labels()
    unique = dict(zip(labels, handles))

    plt.legend(
        unique.values(),
        unique.keys()
    )

    plt.savefig("boundary.png")

    # --------------------------
    # Plot training error
    # --------------------------

    plt.figure(figsize=(6, 4))

    plt.plot(
        range(1, len(history) + 1),
        history,
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Training Error")
    plt.title("Training Error During Learning")

    plt.grid(True)

    plt.savefig("training_error.png")


if __name__ == "__main__":
    main()

"""
This approach succeeds 
as we can displace the theta vector away from the origin

## Epoch 1

Mistake: x=[4. 4. 1.] y=1 -> theta=[4. 4. 1.]

Mistake: x=[1. 1. 1.] y=-1 -> theta=[3. 3. 0.]

Mistake: x=[2. 1. 1.] y=-1 -> theta=[ 1.  2. -1.]

Mistake: x=[1. 2. 1.] y=-1 -> theta=[ 0.  0. -2.]

Training Error = 0.500

## Epoch 2

Mistake: x=[4. 4. 1.] y=1 -> theta=[ 4.  4. -1.]

Mistake: x=[1. 1. 1.] y=-1 -> theta=[ 3.  3. -2.]

Mistake: x=[2. 1. 1.] y=-1 -> theta=[ 1.  2. -3.]

Mistake: x=[1. 2. 1.] y=-1 -> theta=[ 0.  0. -4.]

Training Error = 0.500

## Epoch 3

Mistake: x=[4. 4. 1.] y=1 -> theta=[ 4.  4. -3.]

Mistake: x=[1. 1. 1.] y=-1 -> theta=[ 3.  3. -4.]

Mistake: x=[2. 1. 1.] y=-1 -> theta=[ 1.  2. -5.]

Training Error = 0.000

## Epoch 4

Training Error = 0.000

Converged!

\========================================
Final Theta

\========================================

w1   = 1.000

w2   = 2.000

bias = -5.000

Decision Boundary:
1.000*x + 2.000*y + -5.000 = 0

# Predictions

x=[4. 4.] true=1 pred=1

x=[5. 3.] true=1 pred=1

x=[3. 5.] true=1 pred=1

x=[1. 1.] true=-1 pred=-1

x=[2. 1.] true=-1 pred=-1

x=[1. 2.] true=-1 pred=-1
"""
