"""Small dependency-free demonstration of Project 1."""

from review_analyzer import (
    Review,
    accuracy,
    average_perceptron,
    build_vocabulary,
    pegasos,
    perceptron,
    vectorize,
)


REVIEWS = [
    Review(1, "excellent product and great quality"),
    Review(1, "great value and excellent experience"),
    Review(1, "fast delivery and good quality"),
    Review(-1, "terrible product and poor quality"),
    Review(-1, "bad value and terrible experience"),
    Review(-1, "slow delivery and poor quality"),
]


def main() -> None:
    vocabulary = build_vocabulary(REVIEWS)
    data = vectorize(REVIEWS, vocabulary)

    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Training examples: {len(data)}")
    print()

    models = {
        "Perceptron": perceptron(data, epochs=10),
        "Average perceptron": average_perceptron(data, epochs=10),
        "Pegasos": pegasos(data, lambda_=1e-3, epochs=10, seed=7),
    }

    for name, weights in models.items():
        score = accuracy(weights, data)
        print(f"{name:20s} training accuracy: {score:.3f}")

    print()
    print("Learned vocabulary:")
    for word, index in vocabulary.items():
        print(f"  {word:12s} -> {index}")


if __name__ == "__main__":
    main()
