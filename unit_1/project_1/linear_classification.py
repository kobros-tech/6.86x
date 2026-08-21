"""General-purpose sparse linear classifiers for Unit 1 Project 1.

The module operates on sparse labeled feature vectors. Text preprocessing and
feature construction are intentionally kept in the project notebooks so the
learning algorithms remain independent of the application domain.
"""

from __future__ import annotations

from collections import defaultdict
import math
import random
from typing import Mapping, Sequence


SparseVector = Mapping[int, float]
LabeledExample = tuple[int, SparseVector]


def dot(weights: SparseVector, features: SparseVector) -> float:
    """Return the dot product of two sparse vectors."""
    if len(weights) <= len(features):
        return sum(value * features.get(index, 0.0) for index, value in weights.items())
    return sum(value * weights.get(index, 0.0) for index, value in features.items())


def predict(weights: SparseVector, features: SparseVector) -> int:
    """Predict +1 or -1 using the sign of the linear score."""
    return 1 if dot(weights, features) >= 0 else -1


def accuracy(
    weights: SparseVector,
    data: Sequence[LabeledExample],
) -> float:
    """Return classification accuracy."""
    if not data:
        raise ValueError("cannot compute accuracy on an empty dataset")
    correct = sum(predict(weights, features) == label for label, features in data)
    return correct / len(data)


def _add_scaled(
    weights: dict[int, float],
    features: SparseVector,
    scale: float,
) -> None:
    for index, value in features.items():
        weights[index] = weights.get(index, 0.0) + scale * value
        if weights[index] == 0.0:
            del weights[index]


def perceptron(
    data: Sequence[LabeledExample],
    epochs: int = 5,
) -> dict[int, float]:
    """Train a sparse perceptron classifier."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    weights: dict[int, float] = {}
    for _ in range(epochs):
        for label, features in data:
            if label * dot(weights, features) <= 0:
                _add_scaled(weights, features, label)
    return weights


def average_perceptron(
    data: Sequence[LabeledExample],
    epochs: int = 5,
) -> dict[int, float]:
    """Train an averaged perceptron with lazy timestamp accumulation."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    weights: dict[int, float] = {}
    totals: defaultdict[int, float] = defaultdict(float)
    timestamps: defaultdict[int, int] = defaultdict(int)
    step = 0

    for _ in range(epochs):
        for label, features in data:
            step += 1

            for index in features:
                totals[index] += (step - timestamps[index]) * weights.get(index, 0.0)
                timestamps[index] = step

            if label * dot(weights, features) <= 0:
                _add_scaled(weights, features, label)

    averaged: dict[int, float] = {}
    for index in set(totals) | set(weights):
        totals[index] += (step - timestamps[index]) * weights.get(index, 0.0)
        value = totals[index] / step
        if value:
            averaged[index] = value
    return averaged


def _project(weights: dict[int, float], radius: float) -> None:
    norm = math.sqrt(sum(value * value for value in weights.values()))
    if norm > radius and norm > 0:
        scale = radius / norm
        for index in list(weights):
            weights[index] *= scale


def pegasos(
    data: Sequence[LabeledExample],
    lambda_: float = 1e-4,
    epochs: int = 5,
    batch_size: int = 1,
    seed: int = 0,
) -> dict[int, float]:
    """Train a linear SVM with the Pegasos stochastic sub-gradient method.

    This implementation uses the mini-batch form of the algorithm. The
    regularization term is applied to every step, while the hinge-loss
    sub-gradient is estimated from the sampled active examples.
    """
    if not data:
        raise ValueError("training data cannot be empty")
    if lambda_ <= 0:
        raise ValueError("lambda_ must be positive")
    if epochs < 1:
        raise ValueError("epochs must be at least 1")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    rng = random.Random(seed)
    weights: dict[int, float] = {}
    step = 0

    for _ in range(epochs):
        indices = list(range(len(data)))
        rng.shuffle(indices)

        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            step += 1
            eta = 1.0 / (lambda_ * step)

            active = [
                data[index]
                for index in batch_indices
                if data[index][0] * dot(weights, data[index][1]) < 1.0
            ]

            shrink = 1.0 - eta * lambda_
            for index in list(weights):
                weights[index] *= shrink
                if abs(weights[index]) < 1e-15:
                    del weights[index]

            if active:
                scale = eta / len(batch_indices)
                for label, features in active:
                    _add_scaled(weights, features, scale * label)

            _project(weights, 1.0 / math.sqrt(lambda_))

    return weights
