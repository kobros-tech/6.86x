import math
import unittest

from linear_classification import (
    accuracy,
    average_perceptron,
    pegasos,
    perceptron,
)


class LinearClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = [
            (1, {0: 1.0, 1: 1.0}),
            (1, {0: 1.0, 2: 1.0}),
            (-1, {3: 1.0, 1: 1.0}),
            (-1, {3: 1.0, 2: 1.0}),
        ]

    def test_perceptron_learns_toy_data(self) -> None:
        weights = perceptron(self.data, epochs=10)
        self.assertEqual(accuracy(weights, self.data), 1.0)

    def test_average_perceptron_learns_toy_data(self) -> None:
        weights = average_perceptron(self.data, epochs=10)
        self.assertEqual(accuracy(weights, self.data), 1.0)

    def test_pegasos_learns_toy_data(self) -> None:
        weights = pegasos(self.data, lambda_=1e-3, epochs=20, seed=3)
        self.assertGreaterEqual(accuracy(weights, self.data), 0.75)

    def test_pegasos_is_deterministic_for_fixed_seed(self) -> None:
        first = pegasos(self.data, lambda_=1e-3, epochs=5, batch_size=2, seed=11)
        second = pegasos(self.data, lambda_=1e-3, epochs=5, batch_size=2, seed=11)
        self.assertEqual(first, second)

    def test_pegasos_projection_bounds_weight_norm(self) -> None:
        lambda_ = 1e-3
        weights = pegasos(
            self.data,
            lambda_=lambda_,
            epochs=10,
            batch_size=2,
            seed=7,
        )
        norm = math.sqrt(sum(value * value for value in weights.values()))
        self.assertLessEqual(norm, 1.0 / math.sqrt(lambda_) + 1e-12)

    def test_pegasos_batch_update_uses_full_batch_denominator(self) -> None:
        data = [(1, {0: 1.0}), (1, {})]
        weights = pegasos(data, lambda_=1.0, epochs=1, batch_size=2, seed=0)
        self.assertEqual(weights, {0: 0.5})

    def test_empty_accuracy_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            accuracy({}, [])


if __name__ == "__main__":
    unittest.main()
