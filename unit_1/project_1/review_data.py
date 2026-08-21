"""Review-specific data preparation for Project 1.

This module deliberately sits between the UCI review dataset and the
application-independent classifiers in ``linear_classification.py``.
"""

from __future__ import annotations

from pathlib import Path
import random
import re
import urllib.request
import zipfile
from typing import Sequence

import numpy as np

DATA_URL = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")

Review = tuple[int, str, str]
SparseVector = dict[int, float]


def tokenize(text: str) -> list[str]:
    """Tokenize a review using the project's simple word pattern."""
    return TOKEN_RE.findall(text.lower())


def load_reviews(project_dir: Path) -> list[Review]:
    """Download, extract, and load the complete 3,000-review dataset."""
    data_dir = project_dir / "data" / "sentiment_labelled_sentences"
    data_dir.mkdir(parents=True, exist_ok=True)
    zip_path = data_dir / "sentiment_labelled_sentences.zip"
    if not zip_path.exists():
        urllib.request.urlretrieve(DATA_URL, zip_path)

    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(data_dir)

    root = next(data_dir.glob("**/sentiment labelled sentences"), data_dir)
    files = {
        "IMDb": root / "imdb_labelled.txt",
        "Amazon": root / "amazon_cells_labelled.txt",
        "Yelp": root / "yelp_labelled.txt",
    }

    reviews: list[Review] = []
    for source, path in files.items():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                text, label = line.rstrip("\n").rsplit("\t", 1)
                reviews.append((1 if label == "1" else -1, text, source))
    return reviews


def stratified_split(
    reviews: Sequence[Review],
    seed: int = 42,
) -> tuple[list[Review], list[Review], list[Review]]:
    """Return reproducible 70/15/15 train/validation/test splits."""
    rng = random.Random(seed)
    groups = {1: [], -1: []}
    for review in reviews:
        groups[review[0]].append(review)

    train: list[Review] = []
    validation: list[Review] = []
    test: list[Review] = []
    for group in groups.values():
        group = group.copy()
        rng.shuffle(group)
        n = len(group)
        train_end = int(0.70 * n)
        validation_end = train_end + int(0.15 * n)
        train.extend(group[:train_end])
        validation.extend(group[train_end:validation_end])
        test.extend(group[validation_end:])

    rng.shuffle(train)
    rng.shuffle(validation)
    rng.shuffle(test)
    return train, validation, test


def build_vocabulary(
    reviews: Sequence[Review],
    min_count: int = 2,
) -> dict[str, int]:
    """Build a deterministic vocabulary from training reviews only."""
    counts: dict[str, int] = {}
    for _, text, _ in reviews:
        for token in set(tokenize(text)):
            counts[token] = counts.get(token, 0) + 1
    words = sorted(word for word, count in counts.items() if count >= min_count)
    return {word: index for index, word in enumerate(words)}


def vectorize(
    reviews: Sequence[Review],
    vocabulary: dict[str, int],
) -> tuple[list[SparseVector], np.ndarray]:
    """Convert reviews to sparse binary vectors using an existing vocabulary."""
    X: list[SparseVector] = []
    y: list[int] = []
    for label, text, _ in reviews:
        features = {
            vocabulary[token]: 1.0
            for token in set(tokenize(text))
            if token in vocabulary
        }
        X.append(features)
        y.append(label)
    return X, np.asarray(y, dtype=int)


def prepare_review_data(
    project_dir: Path,
    seed: int = 42,
    min_count: int = 2,
) -> dict[str, object]:
    """Prepare identical train/validation/test data for all Project 1 notebooks."""
    reviews = load_reviews(project_dir)
    train, validation, test = stratified_split(reviews, seed=seed)
    vocabulary = build_vocabulary(train, min_count=min_count)

    X_train, y_train = vectorize(train, vocabulary)
    X_validation, y_validation = vectorize(validation, vocabulary)
    X_test, y_test = vectorize(test, vocabulary)

    return {
        "reviews": reviews,
        "train_reviews": train,
        "validation_reviews": validation,
        "test_reviews": test,
        "vocabulary": vocabulary,
        "X_train": X_train,
        "y_train": y_train,
        "X_validation": X_validation,
        "y_validation": y_validation,
        "X_test": X_test,
        "y_test": y_test,
    }
