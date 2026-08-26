"""
Shared PyTorch training utilities for the single-digit MNIST networks
(Notebook 04 fully connected, Notebook 05 CNN).

Kept separate from the model definitions so both notebooks reuse exactly
the same batching / epoch / training-loop code, and so the notebooks can
focus on explaining architecture rather than boilerplate.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    """Flattens all dimensions except the batch dimension."""

    def forward(self, x):
        return x.view(x.size(0), -1)


def batchify_data(x_data, y_data, batch_size):
    """
    Splits x_data/y_data into a list of {'x': tensor, 'y': tensor} batches.

    The final batch may be smaller than batch_size so that no examples are
    silently discarded during training, validation, or final evaluation.
    """
    batches = []
    for i in range(0, len(x_data), batch_size):
        batches.append(
            {
                "x": torch.tensor(
                    np.array(x_data[i : i + batch_size]), dtype=torch.float32
                ),
                "y": torch.tensor(
                    np.array(y_data[i : i + batch_size]), dtype=torch.long
                ),
            }
        )
    return batches


def compute_accuracy(predictions, y):
    return np.mean(np.equal(predictions.numpy(), y.numpy()))


def run_epoch(data, model, optimizer):
    """
    Runs one epoch over `data` (a list of batches from batchify_data).
    If `optimizer` is provided the model is trained; if it is None the
    model is only evaluated (used with model.eval() on validation/test
    batches).

    Returns:
        avg_loss - average cross-entropy loss over the epoch
        avg_accuracy - average classification accuracy over the epoch
    """
    losses = []
    batch_accuracies = []
    is_training = optimizer is not None

    for batch in data:
        x, y = batch["x"], batch["y"]
        out = model(x)
        predictions = torch.argmax(out, dim=1)
        batch_accuracies.append(compute_accuracy(predictions, y))

        loss = F.cross_entropy(out, y)
        losses.append(loss.data.item())

        if is_training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    avg_loss = float(np.mean(losses))
    avg_accuracy = float(np.mean(batch_accuracies))
    return avg_loss, avg_accuracy


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
    """
    Trains `model` for n_epochs, evaluating on dev_data after every epoch.

    Returns:
        history - dict with 'train_loss', 'train_acc', 'val_loss', 'val_acc'
            lists, one entry per epoch, for plotting learning curves.
    """
    optimizer = torch.optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, nesterov=nesterov
    )

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

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
                f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val loss {val_loss:.4f} acc {val_acc:.4f}"
            )

    return history
