"""
A tiny neural network implemented from first principles: no autograd, no
PyTorch. This module exists to make forward propagation and backprop
concrete before Notebook 04/05 hand the mechanics over to PyTorch.

Architecture:

    2 inputs -> 3 hidden ReLU units -> 1 linear output

Every method here is explicit about the associated derivative so the
notebook can walk through the chain rule step by step (NOTEBOOK_GUIDE,
Notebook 03).
"""
import numpy as np


def rectified_linear_unit(x):
    """ReLU(x) = max(0, x), applied elementwise."""
    return np.maximum(0, x)


def rectified_linear_unit_derivative(x):
    """d/dx ReLU(x): 1 where x > 0, 0 otherwise (subgradient 0 at x = 0)."""
    return (x > 0).astype(float)


def output_layer_activation(x):
    """Linear output activation: returns its input unchanged."""
    return x


def output_layer_activation_derivative(x):
    """Derivative of a linear function is the constant 1."""
    return np.ones_like(x)


class NeuralNetwork:
    """
    A minimal fully connected network with one hidden layer.

    Parameters
    ----------
    input_dim : int
        Number of input features (2 for the toy example in the notebook).
    hidden_dim : int
        Number of hidden ReLU units (3 for the toy example).
    seed : int
        Random seed for weight initialization, for reproducibility.
    """

    def __init__(self, input_dim=2, hidden_dim=3, seed=12321):
        rng = np.random.RandomState(seed)
        # Small random initialization keeps the ReLU units from starting
        # in a dead (always-zero) regime for typical toy inputs.
        self.W1 = rng.normal(scale=0.5, size=(hidden_dim, input_dim))
        self.b1 = np.zeros((hidden_dim, 1))
        self.W2 = rng.normal(scale=0.5, size=(1, hidden_dim))
        self.b2 = np.zeros((1, 1))

    def forward(self, x):
        """
        Forward pass for a single input x (input_dim,) or (input_dim, 1).

        Returns a dict of every intermediate quantity needed for backprop:
        z1 (pre-activation of the hidden layer), a1 (hidden activations),
        z2 (pre-activation of the output), y_hat (network output).
        """
        x = np.asarray(x, dtype=float).reshape(-1, 1)
        z1 = self.W1 @ x + self.b1
        a1 = rectified_linear_unit(z1)
        z2 = self.W2 @ a1 + self.b2
        y_hat = output_layer_activation(z2)
        return {"x": x, "z1": z1, "a1": a1, "z2": z2, "y_hat": y_hat}

    def loss(self, y_hat, y):
        """Squared-error loss: L = 1/2 (y_hat - y)^2."""
        diff = np.asarray(y_hat, dtype=float).reshape(-1) - np.asarray(y, dtype=float).reshape(-1)
        return float(0.5 * diff[0] ** 2)

    def backward(self, cache, y):
        """
        Backpropagates the squared-error loss through the network.

        Returns gradients dW1, db1, dW2, db2 with the same shapes as the
        corresponding parameters, obtained by repeated application of the
        chain rule:

            dL/dy_hat  -> dL/dz2 -> dL/dW2, dL/db2 -> dL/da1
                       -> dL/dz1 -> dL/dW1, dL/db1
        """
        x, z1, a1, z2, y_hat = cache["x"], cache["z1"], cache["a1"], cache["z2"], cache["y_hat"]

        dL_dyhat = y_hat - y  # d/dy_hat of 1/2 (y_hat - y)^2
        dL_dz2 = dL_dyhat * output_layer_activation_derivative(z2)

        dW2 = dL_dz2 @ a1.T
        db2 = dL_dz2

        dL_da1 = self.W2.T @ dL_dz2
        dL_dz1 = dL_da1 * rectified_linear_unit_derivative(z1)

        dW1 = dL_dz1 @ x.T
        db1 = dL_dz1

        return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    def step(self, grads, learning_rate):
        """One gradient-descent update using the gradients from backward()."""
        self.W1 -= learning_rate * grads["dW1"]
        self.b1 -= learning_rate * grads["db1"]
        self.W2 -= learning_rate * grads["dW2"]
        self.b2 -= learning_rate * grads["db2"]

    def train(self, X, Y, learning_rate=0.001, epochs=10):
        """
        Trains on a list/array of (x, y) pairs for a number of epochs.

        Args:
            X - iterable of input vectors, each of length input_dim
            Y - iterable of scalar targets
            learning_rate - gradient-descent step size
            epochs - number of passes over the full dataset

        Returns:
            loss_history - list of per-epoch average loss, for plotting.
        """
        loss_history = []
        for _ in range(epochs):
            epoch_losses = []
            for x, y in zip(X, Y):
                cache = self.forward(x)
                epoch_losses.append(self.loss(cache["y_hat"], y))
                grads = self.backward(cache, y)
                self.step(grads, learning_rate)
            loss_history.append(np.mean(epoch_losses))
        return loss_history

    def predict(self, x):
        return float(np.asarray(self.forward(x)["y_hat"]).ravel()[0])


def numerical_gradient_check(net: NeuralNetwork, x, y, param_name, index, eps=1e-5):
    """
    Approximates d(loss)/d(param[index]) with a centered finite difference
    and compares it against the analytic gradient from backward(). Used
    in the notebook as an optional sanity check on the backprop
    derivation (NOTEBOOK_GUIDE, Notebook 03, step 9).
    """
    param = getattr(net, param_name)

    original_value = param[index]

    param[index] = original_value + eps
    loss_plus = net.loss(net.forward(x)["y_hat"], y)

    param[index] = original_value - eps
    loss_minus = net.loss(net.forward(x)["y_hat"], y)

    param[index] = original_value  # restore
    numeric_grad = (loss_plus - loss_minus) / (2 * eps)

    cache = net.forward(x)
    analytic_grad = net.backward(cache, y)["d" + param_name][index]
    return numeric_grad, analytic_grad
