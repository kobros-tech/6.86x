"""
Convolutional network with two output heads for two-digit MNIST
(Notebook 06).

Spatial size after each stage (42x28 input):
    Conv 3x3 (no padding) -> 40x26
    MaxPool 2x2           -> 20x13
    Conv 3x3 (no padding) -> 18x11
    MaxPool 2x2           -> 9x5    (16 channels -> 16*9*5 = 720 features)
"""
import torch.nn as nn

from .train_utils import Flatten


class CNN(nn.Module):
    def __init__(self, input_dimension=None, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.flatten = Flatten()
        self.shared = nn.Sequential(
            nn.Linear(16 * 9 * 5, 128),
            nn.ReLU(),
        )
        self.head_digit1 = nn.Linear(128, num_classes)
        self.head_digit2 = nn.Linear(128, num_classes)

    def forward(self, x):
        features = self.conv(x)
        flat = self.flatten(features)
        shared_repr = self.shared(flat)
        out_first_digit = self.head_digit1(shared_repr)
        out_second_digit = self.head_digit2(shared_repr)
        return out_first_digit, out_second_digit
