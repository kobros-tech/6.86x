# Project 2 data

The MNIST data is **not committed to this repository**. The loaders download it from OpenML when it is first needed and reuse the local cache on later runs.

- Dataset: **MNIST / `mnist_784`**
- OpenML dataset ID: **554**
- Source: OpenML
- License: Public
- Dataset page: https://www.openml.org/d/554

The two-digit experiment does not require a second committed dataset. It deterministically constructs two-digit images from the downloaded MNIST images by stacking two 28×28 digits with a 14-pixel overlap, producing 42×28 inputs and two labels.

The generated/cache files remain under `data/` locally and are ignored by Git.
