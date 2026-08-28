"""
Fully connected (MLP) network for single-digit MNIST (Notebook 04).

    28 x 28 image -> flatten -> 784 -> 128 (ReLU) -> 10 output scores

The flatten step is the important limitation this notebook should
surface: it discards the 2-D spatial layout of the image entirely, which
motivates the convolutional network in Notebook 05.
"""
import torch.nn as nn

from .train_utils import Flatten


def build_fully_connected_model(input_dim=784, hidden_dim=128, num_classes=10):
    """
    Returns an nn.Sequential MLP:
        Flatten -> Linear(input_dim, hidden_dim) -> ReLU -> Linear(hidden_dim, num_classes)

    The final layer outputs raw class scores (logits); F.cross_entropy in
    train_utils.run_epoch applies the softmax internally.
    """
    return nn.Sequential(
        Flatten(),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, num_classes),
    )
