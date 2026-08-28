"""Two-digit MNIST data preparation."""
from pathlib import Path

import numpy as np

from classical.data_utils import get_MNIST_data

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMG_ROWS, IMG_COLS = 42, 28
NUM_CLASSES = 10


def _make_two_digit_images(X, y, n_examples, seed):
    """Create deterministic 42x28 examples by overlapping two MNIST digits."""
    rng = np.random.RandomState(seed)
    n_examples = min(n_examples, len(X) // 2)
    indices = rng.permutation(len(X))[: 2 * n_examples]
    first_indices = indices[:n_examples]
    second_indices = indices[n_examples:]

    first = X[first_indices].reshape(-1, 28, 28).astype(np.float32)
    second = X[second_indices].reshape(-1, 28, 28).astype(np.float32)

    # Place the two 28x28 digits 14 pixels apart.  The overlapping region is
    # blended with a pixel-wise maximum so that neither digit is cropped away.
    images = np.zeros((n_examples, 42, 28), dtype=np.float32)
    images[:, :28, :] = first
    images[:, 14:, :] = np.maximum(images[:, 14:, :], second)

    labels = [y[first_indices], y[second_indices]]
    return images[:, None, :, :], labels


def get_two_digit_data(data_dir: Path = DATA_DIR, use_mini_dataset: bool = True):
    """Load MNIST and construct the two-digit train/test datasets."""
    train_x, train_y, test_x, test_y = get_MNIST_data(data_dir)
    if use_mini_dataset:
        train_count, test_count = 5000, 1000
    else:
        train_count, test_count = len(train_x) // 2, len(test_x) // 2
    X_train, y_train = _make_two_digit_images(train_x, train_y, train_count, 2026)
    X_test, y_test = _make_two_digit_images(test_x, test_y, test_count, 2027)
    return X_train, y_train, X_test, y_test
