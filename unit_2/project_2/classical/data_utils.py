"""
Data loading and plotting utilities shared by the classical-ML notebooks
(01_classical_mnist.ipynb and 02_features_and_kernels.ipynb).

MNIST is fetched from OpenML on demand rather than committed to the
repository. The data loader keeps data acquisition separate from the model
implementations.
"""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from sklearn.datasets import fetch_openml

# data/ is one level above classical/, two_digit/, neural_networks/, notebooks/
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def get_MNIST_data(data_dir: Path = DATA_DIR):
    """
    Downloads/caches MNIST from OpenML and returns flattened train/test data.

    OpenML dataset 554 (`mnist_784`) contains 70,000 examples. The standard
    first 60,000 examples are returned as training data and the final 10,000
    as test data, matching the usual MNIST split.

    Args:
        data_dir: Local cache directory. It is ignored by Git and is used as
            the OpenML cache location.

    Returns:
        train_x - (60000, 784) NumPy array of pixel values in [0, 1]
        train_y - (60000,) NumPy array of integer labels 0-9
        test_x  - (10000, 784) NumPy array of pixel values in [0, 1]
        test_y  - (10000,) NumPy array of integer labels 0-9
    """
    data = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=False,
        parser="auto",
        data_home=str(data_dir),
    )

    X = np.asarray(data.data, dtype=np.float32) / 255.0
    y = np.asarray(data.target, dtype=int)

    return X[:60000], y[:60000], X[60000:], y[60000:]


def plot_images(X, labels=None, max_images=20):
    """Plots a grid of MNIST images (each row of X is a 784-d pixel vector)."""
    if X.ndim == 1:
        X = np.array([X])
    num_images = min(X.shape[0], max_images)
    num_rows = math.floor(math.sqrt(num_images))
    num_cols = math.ceil(num_images / num_rows)
    plt.figure(figsize=(num_cols * 1.2, num_rows * 1.2))
    for i in range(num_images):
        reshaped_image = X[i, :].reshape(28, 28)
        plt.subplot(num_rows, num_cols, i + 1)
        plt.imshow(reshaped_image, cmap=cm.Greys_r)
        plt.axis("off")
        if labels is not None:
            plt.title(str(labels[i]), fontsize=9)
    plt.tight_layout()
    plt.show()


def pick_examples_of(X, Y, labels, total_count):
    """Filters (X, Y) down to rows whose label is in `labels`."""
    bool_arr = None
    for label in labels:
        bool_arr_for_label = Y == label
        bool_arr = bool_arr_for_label if bool_arr is None else (bool_arr | bool_arr_for_label)
    return X[bool_arr][:total_count], Y[bool_arr][:total_count]


def make_train_val_split(X, Y, val_fraction=0.1, seed=12321):
    """
    Splits (X, Y) into a training set and a validation set.

    Used so that hyperparameters (regularization strength, kernel
    parameters, polynomial degree, ...) are chosen with validation data
    only, keeping the test set untouched until the final evaluation
    (README, section 12).
    """
    rng = np.random.RandomState(seed)
    n = X.shape[0]
    permutation = rng.permutation(n)
    val_size = int(n * val_fraction)
    val_idx, train_idx = permutation[:val_size], permutation[val_size:]
    return X[train_idx], Y[train_idx], X[val_idx], Y[val_idx]
