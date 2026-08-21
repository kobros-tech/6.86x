"""Small dependency-free demonstration of Project 1's classifiers."""

from linear_classification import accuracy, average_perceptron, pegasos, perceptron


DATA = [
    (1, {0: 1.0, 1: 1.0}),
    (1, {0: 1.0, 2: 1.0}),
    (1, {0: 1.0, 3: 1.0}),
    (-1, {4: 1.0, 1: 1.0}),
    (-1, {4: 1.0, 2: 1.0}),
    (-1, {4: 1.0, 3: 1.0}),
]


def main() -> None:
    models = {
        "Perceptron": perceptron(DATA, epochs=10),
        "Average perceptron": average_perceptron(DATA, epochs=10),
        "Pegasos": pegasos(DATA, lambda_=1e-3, epochs=10, seed=7),
    }

    print(f"Training examples: {len(DATA)}")
    print()

    for name, weights in models.items():
        score = accuracy(weights, DATA)
        print(f"{name:20s} training accuracy: {score:.3f}")


if __name__ == "__main__":
    main()
