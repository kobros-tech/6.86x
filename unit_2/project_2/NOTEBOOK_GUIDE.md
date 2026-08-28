# Project 2 — Notebook Guide

This document is the implementation contract for the Project 2 notebooks. The notebooks are study material: each section should connect a mathematical idea to an implementation, experiment, and interpretation.

## Notebook 01 — Classical MNIST

- Introduce MNIST as a ten-class classification problem and explain the 784-dimensional representation.
- Inspect representative images and dataset shapes.
- Compare linear regression, linear SVM, and Softmax regression.
- Explain Softmax probabilities and the final `argmax` prediction.
- Keep the comparison controlled and discuss the limitations of linear decision rules.
- Concepts beyond Units 1–2 (one-vs-rest multiclass SVM, Softmax, temperature, negative log-likelihood) are explained in [`notebooks/notes/01_classical_mnist_CONCEPTS.md`](notebooks/notes/01_classical_mnist_CONCEPTS.md).

## Notebook 02 — Features and Kernels

- Start from raw pixels.
- Explain PCA and dimensionality reduction.
- Demonstrate a small explicit cubic feature map.
- Introduce polynomial and RBF kernels.
- Use validation data for kernel/feature hyperparameters and reserve the test set for final evaluation.
- Concepts beyond Units 1–2 (PCA, the RBF kernel) are explained in [`notebooks/notes/02_features_and_kernels_CONCEPTS.md`](notebooks/notes/02_features_and_kernels_CONCEPTS.md).

## Notebook 03 — Neural Network from Scratch

- Use the small 2-input, 3-hidden-unit ReLU, 1-output network.
- Implement the forward pass, loss, derivatives, backpropagation, and gradient-descent update explicitly.
- Plot training loss and, where useful, verify gradients numerically.
- Do not hide the central calculation behind automatic differentiation.
- Neural networks are covered formally in Unit 3; this notebook is a hands-on preview. Just enough vocabulary to follow the code (ReLU, forward/backpropagation, gradient checking) is in [`notebooks/notes/03_neural_network_from_scratch_CONCEPTS.md`](notebooks/notes/03_neural_network_from_scratch_CONCEPTS.md).

## Notebook 04 — Fully Connected MNIST

- Prepare MNIST tensors and explain the MLP architecture.
- Explain activations, the ten-class output, loss, optimizer, and training loop.
- Track training and validation metrics.
- Evaluate on the untouched test set and inspect representative predictions.
- PyTorch/autograd vocabulary (bridging from the by-hand Notebook 03 implementation) is in [`notebooks/notes/04_mnist_fully_connected_CONCEPTS.md`](notebooks/notes/04_mnist_fully_connected_CONCEPTS.md).

## Notebook 05 — CNN on MNIST

- Explain locality, receptive fields, filter sharing, and pooling before the implementation.
- Track tensor dimensions through the CNN.
- Train and validate the CNN.
- Compare it with the MLP under a clearly stated experimental setup.
- Convolution/pooling vocabulary, and a note disambiguating "kernel" (convolutional filter) from the Lecture 6 kernel-trick sense of the word, is in [`notebooks/notes/05_mnist_cnn_CONCEPTS.md`](notebooks/notes/05_mnist_cnn_CONCEPTS.md).

## Notebook 06 — Two-Digit MNIST

- Explain the two labels and why two output heads are required.
- Compare a shared-representation MLP and CNN.
- Report first-digit accuracy, second-digit accuracy, and exact-match accuracy.
- Explain the combined loss $L=(L_1+L_2)/2$ and why it is averaged rather than summed (summing destabilized CNN training; see the concepts note for the observed effect).
- Multi-head/multi-task vocabulary and the sum-vs-average loss lesson are in [`notebooks/notes/06_two_digit_mnist_CONCEPTS.md`](notebooks/notes/06_two_digit_mnist_CONCEPTS.md).

## Common standards

Every notebook should follow:

```text
Concept → mathematics → implementation → experiment → result → interpretation
```

Avoid unexplained library calls, arbitrary claims of optimal hyperparameters, test-set tuning, and hard-coded numerical results.

Use fixed random seeds where practical and explain important training choices. Numerical results should come from executed cells.

For GitHub-compatible mathematics, use `$...$` for inline equations and `$$...$$` for display equations, with equations outside code blocks.

## Reproducibility

Project data is not committed to Git. MNIST is downloaded from OpenML on demand and cached locally. The two-digit experiment constructs its examples deterministically from MNIST, so a fresh checkout does not depend on repository-local binary archives.
