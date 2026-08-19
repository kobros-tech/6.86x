import unittest

from review_analyzer import (
    Review,
    accuracy,
    average_perceptron,
    build_vocabulary,
    extract_features,
    parse_review_line,
    pegasos,
    perceptron,
    tokenize,
    vectorize,
)


class ReviewAnalyzerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.reviews = [
            Review(1, "great product"),
            Review(1, "excellent product"),
            Review(-1, "bad product"),
            Review(-1, "terrible product"),
        ]
        self.vocabulary = build_vocabulary(self.reviews)
        self.data = vectorize(self.reviews, self.vocabulary)

    def test_tokenize(self) -> None:
        self.assertEqual(tokenize("Great PRODUCT!"), ["great", "product"])

    def test_parse_review_line(self) -> None:
        self.assertEqual(
            parse_review_line("+1\tExcellent product"),
            Review(1, "Excellent product"),
        )

    def test_vocabulary_is_deterministic(self) -> None:
        self.assertEqual(
            list(self.vocabulary), ["bad", "excellent", "great", "product", "terrible"]
        )

    def test_sparse_features(self) -> None:
        features = extract_features("great unknown product", self.vocabulary)
        self.assertEqual(len(features), 2)
        self.assertEqual(features[self.vocabulary["great"]], 1.0)
        self.assertEqual(features[self.vocabulary["product"]], 1.0)

    def test_perceptron_learns_toy_data(self) -> None:
        weights = perceptron(self.data, epochs=10)
        self.assertEqual(accuracy(weights, self.data), 1.0)

    def test_average_perceptron_learns_toy_data(self) -> None:
        weights = average_perceptron(self.data, epochs=10)
        self.assertEqual(accuracy(weights, self.data), 1.0)

    def test_pegasos_learns_toy_data(self) -> None:
        weights = pegasos(self.data, lambda_=1e-3, epochs=20, seed=3)
        self.assertGreaterEqual(accuracy(weights, self.data), 0.75)

    def test_end_to_end(self) -> None:
        # Build the vocabulary from training data only, then evaluate a
        # validation review containing words known from that training set.
        train = self.reviews[:3]
        validation = [Review(-1, "bad product")]
        vocabulary = build_vocabulary(train)
        train_data = vectorize(train, vocabulary)
        validation_data = vectorize(validation, vocabulary)
        weights = perceptron(train_data, epochs=10)
        self.assertEqual(accuracy(weights, validation_data), 1.0)


if __name__ == "__main__":
    unittest.main()
