"""
Fully connected network with two output heads for two-digit MNIST
(Notebook 06).

    image (42x28) -> flatten -> shared hidden layers -> two output heads
                                                             |       |
                                                        digit 1   digit 2

A single ten-class output is not enough here: the image encodes two
independent digit labels, so the model needs two separate ten-way
predictions sharing one internal representation.
"""
import torch.nn as nn

from .train_utils import Flatten

IMG_ROWS, IMG_COLS = 42, 28


class MLP(nn.Module):
    def __init__(self, input_dimension=IMG_ROWS * IMG_COLS, hidden_dim=128, num_classes=10):
        super().__init__()
        self.flatten = Flatten()
        self.shared = nn.Sequential(
            nn.Linear(input_dimension, hidden_dim),
            nn.ReLU(),
        )
        self.head_digit1 = nn.Linear(hidden_dim, num_classes)
        self.head_digit2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        xf = self.flatten(x)
        shared_repr = self.shared(xf)
        out_first_digit = self.head_digit1(shared_repr)
        out_second_digit = self.head_digit2(shared_repr)
        return out_first_digit, out_second_digit
