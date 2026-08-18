"""Study-oriented implementation of Unit 1 Project 1.

The module contains a small sparse bag-of-words pipeline and three linear
classifiers: perceptron, average perceptron, and Pegasos.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import random
import re
from typing import Iterable, Mapping, Sequence


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")


@dataclass(frozen=True)
class Review:
    """A single binary-sentiment review."""

    label: int
    text: str


def tokenize(text: str) -> list[str]:
    """Return lowercase word tokens from a review."""
    return _TOKEN_RE.findall(text.lower())


def parse_review_line(line: str) -> Review:
    """Parse ``label text`` or ``label<TAB>text`` into a Review."""
    stripped = line.strip()
    if not stripped:
        raise ValueError("review line is empty")

    parts = stripped.split(maxsplit=1)
    if len(parts) != 2:
        raise ValueError("review line must contain a label and review text")

    try:
        label = int(parts[0])
    except ValueError as exc:
        raise ValueError("review label must be +1 or -1") from exc

    if label not in (-1, 1):
        raise ValueError("review label must be +1 or -1")

    return Review(label=label, text=parts[1])


def load_reviews(path: str) -> list[Review]:
    """Load reviews from a text file, ignoring blank lines."""
    reviews: list[Review] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                reviews.append(parse_review_line(line))
            except ValueError as exc:
                raise ValueError(f"invalid review at line {line_number}: {exc}") from exc
    return reviews


def build_vocabulary(reviews: Iterable[Review]) -> dict[str, int]:
    """Build a deterministic word-to-index vocabulary from training reviews."""
    words = {token for review in reviews for token in tokenize(review.text)}
    return {word: index for index, word in enumerate(sorted(words))}


def extract_features(text: str, vocabulary: Mapping[str, int]) -> dict[int, float]:
    """Convert text to a sparse binary bag-of-words feature vector."""
    features: dict[int, float] = {}
    for token in tokenize(text):
        index = vocabulary.get(token)
        if index is not None:
            features[index] = 1.0
    return features


def vectorize(
    reviews: Sequence[Review], vocabulary: Mapping[str, int]
) -> list[tuple[int, dict[int, float]]]:
    """Convert labeled reviews into sparse ``(label, features)`` pairs."""
    return [(review.label, extract_features(review.text, vocabulary)) for review in reviews]


def dot(weights: Mapping[int, float], features: Mapping[int, float]) -> float:
    """Sparse dot product."""
    if len(weights) > len(features):
        weights, features = features, weights
    return sum(weights.get(index, 0.0) * value for index, value in features.items())


def predict(weights: Mapping[int, float], features: Mapping[int, float]) -> int:
    """Predict +1 or -1 using the sign of the linear score."""
    return 1 if dot(weights, features) >= 0 else -1


def accuracy(
    weights: Mapping[int, float], data: Sequence[tuple[int, Mapping[int, float]]]
) -> float:
    """Return classification accuracy."""
    if not data:
        raise ValueError("cannot compute accuracy on an empty dataset")
    correct = sum(predict(weights, features) == label for label, features in data)
    return correct / len(data)


def _add_scaled(
    weights: dict[int, float], features: Mapping[int, float], scale: float
) -> None:
    for index, value in features.items():
        weights[index] = weights.get(index, 0.0) + scale * value
        if weights[index] == 0.0:
            del weights[index]


def perceptron(
    data: Sequence[tuple[int, Mapping[int, float]]],
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
    data: Sequence[tuple[int, Mapping[int, float]]],
    epochs: int = 5,
) -> dict[int, float]:
    """Train an averaged perceptron using an online running average."""
    if epochs < 1:
        raise ValueError("epochs must be at least 1")

    weights: dict[int, float] = {}
    totals: defaultdict[int, float] = defaultdict(float)
    timestamps: defaultdict[int, int] = defaultdict(int)
    step = 0

    for _ in range(epochs):
        for label, features in data:
            step += 1
            if label * dot(weights, features) <= 0:
                _add_scaled(weights, features, label)
            for index in features:
                totals[index] += (step - timestamps[index]) * weights.get(index, 0.0)
                timestamps[index] = step

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
    data: Sequence[tuple[int, Mapping[int, float]]],
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

            # Regularization shrinkage.
            shrink = 1.0 - eta * lambda_
            for index in list(weights):
                weights[index] *= shrink
                if abs(weights[index]) < 1e-15:
                    del weights[index]

            # Estimated hinge-loss sub-gradient contribution.
            active = [
                data[index]
                for index in batch_indices
                if label * dot(weights, features) < 1.0
                for label, features in [data[index]]
            ]
            if active:
                scale = eta / len(active)
                for label, features in active:
                    _add_scaled(weights, features, scale * label)

            _project(weights, 1.0 / math.sqrt(lambda_))

    return weights


def train_and_score(
    train_reviews: Sequence[Review],
    validation_reviews: Sequence[Review],
    algorithm: str = "pegasos",
    **kwargs: object,
) -> tuple[dict[int, float], float]:
    """Build training features, train a classifier, and score validation data."""
    vocabulary = build_vocabulary(train_reviews)
    train_data = vectorize(train_reviews, vocabulary)
    validation_data = vectorize(validation_reviews, vocabulary)

    algorithms = {
        "perceptron": perceptron,
        "average_perceptron": average_perceptron,
        "pegasos": pegasos,
    }
    try:
        trainer = algorithms[algorithm]
    except KeyError as exc:
        raise ValueError(f"unknown algorithm: {algorithm}") from exc

    weights = trainer(train_data, **kwargs)
    return weights, accuracy(weights, validation_data)
