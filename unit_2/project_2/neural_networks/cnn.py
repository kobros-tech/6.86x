"""
Convolutional neural network for single-digit MNIST (Notebook 05).

    image (1x28x28)
        -> Conv(1 -> 8, 3x3) -> ReLU -> MaxPool(2x2)
        -> Conv(8 -> 16, 3x3) -> ReLU -> MaxPool(2x2)
        -> flatten
        -> Linear -> 10 output scores

Kept intentionally small: the goal (per README, section 10) is to
understand why the representation is appropriate for images, not to
maximize accuracy.
"""
import torch.nn as nn

from .train_utils import Flatten


def build_cnn_model(num_classes=10):
    """
    Returns an nn.Sequential CNN. Input is expected as (batch, 1, 28, 28).

    Spatial size after each stage (28x28 input):
        Conv 3x3 (no padding) -> 26x26
        MaxPool 2x2           -> 13x13
        Conv 3x3 (no padding) -> 11x11
        MaxPool 2x2           -> 5x5   (16 channels -> 16*5*5 = 400 features)
    """
    return nn.Sequential(
        nn.Conv2d(1, 8, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(8, 16, kernel_size=3),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        Flatten(),
        nn.Linear(16 * 5 * 5, num_classes),
    )
