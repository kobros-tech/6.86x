"""
Shared PyTorch training utilities for the two-digit MNIST models
(Notebook 06). Adapted from neural_networks/train_utils.py to handle two
label heads and an averaged loss L = (L1 + L2) / 2.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def batchify_data(x_data, y_data, batch_size):
    """
    Splits two-digit data into batches.

    The final batch may be smaller than batch_size so that no examples are
    silently discarded during training, validation, or final evaluation.

    Args:
        x_data - list/array of (1, 42, 28) images
        y_data - [first_digit_labels, second_digit_labels]
    """
    batches = []
    n = len(x_data)
    for i in range(0, n, batch_size):
        batches.append(
            {
                "x": torch.tensor(
                    np.array(x_data[i : i + batch_size]), dtype=torch.float32
                ),
                "y": [
                    torch.tensor(
                        np.array(y_data[0][i : i + batch_size]), dtype=torch.long
                    ),
                    torch.tensor(
                        np.array(y_data[1][i : i + batch_size]), dtype=torch.long
                    ),
                ],
            }
        )
    return batches


def compute_accuracy(predictions, y):
    return np.mean(np.equal(predictions.numpy(), y.numpy()))


def run_epoch(data, model, optimizer):
    """
    Runs one epoch over two-digit batches.

    Returns:
        losses - [avg_loss_digit1, avg_loss_digit2]
        accuracies - [avg_acc_digit1, avg_acc_digit2]
    """
    losses_first, losses_second = [], []
    acc_first, acc_second = [], []
    is_training = optimizer is not None

    for batch in data:
        x, (y1, y2) = batch["x"], batch["y"]
        out1, out2 = model(x)

        pred1 = torch.argmax(out1, dim=1)
        pred2 = torch.argmax(out2, dim=1)
        acc_first.append(compute_accuracy(pred1, y1))
        acc_second.append(compute_accuracy(pred2, y2))

        loss1 = F.cross_entropy(out1, y1)
        loss2 = F.cross_entropy(out2, y2)
        loss = (loss1 + loss2) / 2  # average loss across the two digit heads

        losses_first.append(loss1.data.item())
        losses_second.append(loss2.data.item())

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return (
        [float(np.mean(losses_first)), float(np.mean(losses_second))],
        [float(np.mean(acc_first)), float(np.mean(acc_second))],
    )


def train_model(
    train_data,
    dev_data,
    model,
    lr=0.01,
    momentum=0.9,
    nesterov=False,
    n_epochs=10,
    verbose=True,
):
    """Trains a two-headed model, tracking per-digit loss and accuracy."""
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(1, n_epochs + 1):
        model.train()
        train_loss, train_acc = run_epoch(train_data, model, optimizer)

        model.eval()
        with torch.no_grad():
            val_loss, val_acc = run_epoch(dev_data, model, None)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if verbose:
            print(
                f"Epoch {epoch:2d} | "
                f"train loss1 {train_loss[0]:.4f} loss2 {train_loss[1]:.4f} "
                f"acc1 {train_acc[0]:.4f} acc2 {train_acc[1]:.4f} | "
                f"val loss1 {val_loss[0]:.4f} loss2 {val_loss[1]:.4f} "
                f"acc1 {val_acc[0]:.4f} acc2 {val_acc[1]:.4f}"
            )

    return history


def exact_match_accuracy(model, data):
    """
    Fraction of examples where BOTH digits are predicted correctly
    simultaneously (NOTEBOOK_GUIDE, Notebook 06, required metrics).
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in data:
            x, (y1, y2) = batch["x"], batch["y"]
            out1, out2 = model(x)
            pred1 = torch.argmax(out1, dim=1)
            pred2 = torch.argmax(out2, dim=1)
            both_correct = (pred1 == y1) & (pred2 == y2)
            correct += int(both_correct.sum())
            total += len(y1)
    return correct / total
