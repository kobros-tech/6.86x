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


# ==========================================
# Helper Functions
# ==========================================

def sign(z):
    """
    Return class label.
    """

    return 1 if z > 0 else -1


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

            if prediction != yi:

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

    if abs(theta[1]) < 1e-8:
        return

    x_values = np.linspace(0, 6, 100)

    y_values = -(theta[0] / theta[1]) * x_values

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

    theta, history = perceptron(X, y)

    print("\nFinal Theta")
    print(theta)

    print("\nPredictions")

    for xi, yi in zip(X, y):

        pred = predict(xi, theta)

        print(
            f"x={xi}"
            f" true={yi}"
            f" pred={pred}"
        )

    plt.figure(figsize=(6, 6))

    plot_dataset(X, y)
    plot_decision_boundary(theta)

    plt.xlim(0, 6)
    plt.ylim(0, 6)

    plt.grid(True)
    plt.title("Perceptron Decision Boundary")

    handles, labels = plt.gca().get_legend_handles_labels()

    unique = dict(zip(labels, handles))

    plt.legend(unique.values(), unique.keys())

    plt.show()

    plt.figure(figsize=(6, 4))

    plt.plot(history, marker="o")

    plt.xlabel("Epoch")
    plt.ylabel("Training Error")
    plt.title("Training Error During Learning")

    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    main()

"""
This approach will fail 
as the theta vector is bound to rotate around the origin 
but can not take displacement

Epoch 1

Mistake: x=[4 4] y=1 -> theta=[4. 4.]

Mistake: x=[1 1] y=-1 -> theta=[3. 3.]

Mistake: x=[2 1] y=-1 -> theta=[1. 2.]

Mistake: x=[1 2] y=-1 -> theta=[0. 0.]

Training Error = 0.500

Epoch 2

Mistake: x=[4 4] y=1 -> theta=[4. 4.]

Mistake: x=[1 1] y=-1 -> theta=[3. 3.]

Mistake: x=[2 1] y=-1 -> theta=[1. 2.]

Mistake: x=[1 2] y=-1 -> theta=[0. 0.]

Training Error = 0.500

Epoch 3

Mistake: x=[4 4] y=1 -> theta=[4. 4.]

Mistake: x=[1 1] y=-1 -> theta=[3. 3.]

Mistake: x=[2 1] y=-1 -> theta=[1. 2.]

Mistake: x=[1 2] y=-1 -> theta=[0. 0.]

Training Error = 0.500

Epoch 4

Mistake: x=[4 4] y=1 -> theta=[4. 4.]

Mistake: x=[1 1] y=-1 -> theta=[3. 3.]

Mistake: x=[2 1] y=-1 -> theta=[1. 2.]

Mistake: x=[1 2] y=-1 -> theta=[0. 0.]

Training Error = 0.500

Epoch 5

Mistake: x=[4 4] y=1 -> theta=[4. 4.]

Mistake: x=[1 1] y=-1 -> theta=[3. 3.]

Mistake: x=[2 1] y=-1 -> theta=[1. 2.]

Mistake: x=[1 2] y=-1 -> theta=[0. 0.]

Training Error = 0.500

Epoch 6

Mistake: x=[4 4] y=1 -> theta=[4. 4.]

Mistake: x=[1 1] y=-1 -> theta=[3. 3.]

Mistake: x=[2 1] y=-1 -> theta=[1. 2.]

Mistake: x=[1 2] y=-1 -> theta=[0. 0.]

Training Error = 0.500

Epoch 7

Mistake: x=[4 4] y=1 -> theta=[4. 4.]

Mistake: x=[1 1] y=-1 -> theta=[3. 3.]

Mistake: x=[2 1] y=-1 -> theta=[1. 2.]

Mistake: x=[1 2] y=-1 -> theta=[0. 0.]

Training Error = 0.500

Epoch 8

Mistake: x=[4 4] y=1 -> theta=[4. 4.]

Mistake: x=[1 1] y=-1 -> theta=[3. 3.]

Mistake: x=[2 1] y=-1 -> theta=[1. 2.]

Mistake: x=[1 2] y=-1 -> theta=[0. 0.]

Training Error = 0.500

Epoch 9

Mistake: x=[4 4] y=1 -> theta=[4. 4.]

Mistake: x=[1 1] y=-1 -> theta=[3. 3.]

Mistake: x=[2 1] y=-1 -> theta=[1. 2.]

Mistake: x=[1 2] y=-1 -> theta=[0. 0.]

Training Error = 0.500

Epoch 10

Mistake: x=[4 4] y=1 -> theta=[4. 4.]

Mistake: x=[1 1] y=-1 -> theta=[3. 3.]

Mistake: x=[2 1] y=-1 -> theta=[1. 2.]

Mistake: x=[1 2] y=-1 -> theta=[0. 0.]

Training Error = 0.500

Epoch 11

Mistake: x=[4 4] y=1 -> theta=[4. 4.]

Mistake: x=[1 1] y=-1 -> theta=[3. 3.]

Mistake: x=[2 1] y=-1 -> theta=[1. 2.]

Mistake: x=[1 2] y=-1 -> theta=[0. 0.]

Training Error = 0.500

Epoch 12

Mistake: x=[4 4] y=1 -> theta=[4. 4.]

Mistake: x=[1 1] y=-1 -> theta=[3. 3.]

Mistake: x=[2 1] y=-1 -> theta=[1. 2.]

Mistake: x=[1 2] y=-1 -> theta=[0. 0.]

Training Error = 0.500

Epoch 13

Mistake: x=[4 4] y=1 -> theta=[4. 4.]

Mistake: x=[1 1] y=-1 -> theta=[3. 3.]

Mistake: x=[2 1] y=-1 -> theta=[1. 2.]

Mistake: x=[1 2] y=-1 -> theta=[0. 0.]

Training Error = 0.500

Epoch 14

Mistake: x=[4 4] y=1 -> theta=[4. 4.]

Mistake: x=[1 1] y=-1 -> theta=[3. 3.]

Mistake: x=[2 1] y=-1 -> theta=[1. 2.]

Mistake: x=[1 2] y=-1 -> theta=[0. 0.]

Training Error = 0.500

Epoch 15

Mistake: x=[4 4] y=1 -> theta=[4. 4.]

Mistake: x=[1 1] y=-1 -> theta=[3. 3.]

Mistake: x=[2 1] y=-1 -> theta=[1. 2.]

Mistake: x=[1 2] y=-1 -> theta=[0. 0.]

Training Error = 0.500

Epoch 16

Mistake: x=[4 4] y=1 -> theta=[4. 4.]

Mistake: x=[1 1] y=-1 -> theta=[3. 3.]

Mistake: x=[2 1] y=-1 -> theta=[1. 2.]

Mistake: x=[1 2] y=-1 -> theta=[0. 0.]

Training Error = 0.500

Epoch 17

Mistake: x=[4 4] y=1 -> theta=[4. 4.]

Mistake: x=[1 1] y=-1 -> theta=[3. 3.]

Mistake: x=[2 1] y=-1 -> theta=[1. 2.]

Mistake: x=[1 2] y=-1 -> theta=[0. 0.]

Training Error = 0.500

Epoch 18

Mistake: x=[4 4] y=1 -> theta=[4. 4.]

Mistake: x=[1 1] y=-1 -> theta=[3. 3.]

Mistake: x=[2 1] y=-1 -> theta=[1. 2.]

Mistake: x=[1 2] y=-1 -> theta=[0. 0.]

Training Error = 0.500

Epoch 19

Mistake: x=[4 4] y=1 -> theta=[4. 4.]

Mistake: x=[1 1] y=-1 -> theta=[3. 3.]

Mistake: x=[2 1] y=-1 -> theta=[1. 2.]

Mistake: x=[1 2] y=-1 -> theta=[0. 0.]

Training Error = 0.500

Epoch 20

Mistake: x=[4 4] y=1 -> theta=[4. 4.]

Mistake: x=[1 1] y=-1 -> theta=[3. 3.]

Mistake: x=[2 1] y=-1 -> theta=[1. 2.]

Mistake: x=[1 2] y=-1 -> theta=[0. 0.]

Training Error = 0.500

Final Theta
[0. 0.]

Predictions
x=[4 4] true=1 pred=-1

x=[5 3] true=1 pred=-1

x=[3 5] true=1 pred=-1

x=[1 1] true=-1 pred=-1

x=[2 1] true=-1 pred=-1

x=[1 2] true=-1 pred=-1
"""
