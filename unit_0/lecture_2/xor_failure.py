import numpy as np
import matplotlib.pyplot as plt


# XOR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y = np.array([
    -1,
     1,
     1,
    -1
])


def sign(z):
    return 1 if z > 0 else -1


def classify(x, theta, theta0):
    return sign(np.dot(theta, x) + theta0)


def accuracy(theta, theta0):
    correct = 0

    for xi, yi in zip(X, y):
        prediction = classify(xi, theta, theta0)

        if prediction == yi:
            correct += 1

    return correct / len(X)


# Search many linear classifiers
best_acc = 0
best_theta = None
best_theta0 = None

for theta1 in np.arange(-3, 3.5, 0.5):
    for theta2 in np.arange(-3, 3.5, 0.5):
        for theta0 in np.arange(-3, 3.5, 0.5):

            theta = np.array([theta1, theta2])

            acc = accuracy(theta, theta0)

            if acc > best_acc:
                best_acc = acc
                best_theta = theta.copy()
                best_theta0 = theta0


print("Best accuracy:", best_acc)
print("Best theta:", best_theta)
print("Best theta0:", best_theta0)

print("\nPredictions:")
for xi, yi in zip(X, y):
    pred = classify(xi, best_theta, best_theta0)
    print(
        f"x={xi}, "
        f"true={yi}, "
        f"pred={pred}"
    )


# Visualization
plt.figure(figsize=(6, 6))

for xi, yi in zip(X, y):

    if yi == 1:
        plt.scatter(
            xi[0],
            xi[1],
            marker="o",
            s=120,
            label="Positive"
        )
    else:
        plt.scatter(
            xi[0],
            xi[1],
            marker="x",
            s=120,
            label="Negative"
        )


# Decision boundary
theta1, theta2 = best_theta

if abs(theta2) > 1e-6:

    xs = np.linspace(-0.5, 1.5, 100)

    ys = -(theta1 * xs + best_theta0) / theta2

    plt.plot(xs, ys, linewidth=2)


plt.xlim(-0.5, 1.5)
plt.ylim(-0.5, 1.5)
plt.grid(True)

plt.title(
    f"Best Linear Classifier\nAccuracy={best_acc:.2f}"
)

plt.show()

"""
After running code

Best accuracy: 0.75

Best theta: [-3.   0.5]

Best theta0: 0.0


Predictions:
x=[0 0], true=-1, pred=-1

x=[0 1], true=1, pred=1

x=[1 0], true=1, pred=-1

x=[1 1], true=-1, pred=-1
"""
