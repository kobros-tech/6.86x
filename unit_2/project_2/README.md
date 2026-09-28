# Project 2 — MNIST Digit Recognition

Project 2 applies the Unit 2 ideas to handwritten digit recognition using the MNIST dataset. The project starts with classical machine-learning methods and then moves to neural networks, ending with convolutional networks and a two-digit recognition task.

The project is organized as a **study project**: each experiment should connect an implementation to a mathematical idea from the lectures. The code may use NumPy, scikit-learn, and PyTorch where appropriate, but the notebooks should explain what the models are doing rather than treating the libraries as black boxes.

## Dataset

The project uses the standard **MNIST handwritten-digit dataset**. The dataset is **not committed to the repository**. On first use, the classical data loader downloads the `mnist_784` dataset from OpenML and caches it locally for subsequent runs.

The two-digit experiment does not require a separate committed dataset. It constructs its two-digit examples at runtime from the downloaded MNIST data. Downloaded and generated data is stored locally under `data/` and is protected by `data/.gitignore`. See `data/README.md` for the data source and handling details.

The data-loading implementation is kept separate from the model implementations so that the experiments can reuse the same dataset preparation.

## Project goal

Given an MNIST image, predict which digit from `0` through `9` it represents.

An MNIST image is a $28\times28$ grayscale image. After flattening, it can be represented as a vector

$$
x\in\mathbb{R}^{784}.
$$

The project progresses from classical models to learned representations:

```text
MNIST image
    |
    v
784-dimensional pixel vector
    |
    +--> Linear regression
    +--> Linear SVM
    +--> Softmax regression
    |
    +--> PCA / feature mappings
    +--> Polynomial / RBF kernels
    |
    v
Neural networks
    |
    +--> Network from scratch
    +--> Fully connected network
    +--> Convolutional neural network
    |
    v
Two-digit recognition
```

## Learning objectives

By completing the project, you should be able to:

- represent an image as a feature vector;
- compare classical classifiers on the same dataset;
- understand multiclass classification with one-vs-rest SVMs;
- understand multinomial Softmax regression;
- connect Softmax scores to a probability distribution;
- see how feature mappings and kernels change the representation seen by a classifier;
- implement the forward pass of a small neural network;
- derive and implement backpropagation for a simple network;
- train a fully connected neural network on MNIST;
- explain why convolution is useful for images;
- compare a fully connected network with a CNN;
- formulate a prediction problem with two outputs for two-digit MNIST;
- interpret training, validation, and test results without using the test set for model selection.

## Project structure

```text
project_2/
├── README.md
├── NOTEBOOK_GUIDE.md
├── data/
├── notebooks/
│   ├── 01_classical_mnist.ipynb
│   ├── 02_features_and_kernels.ipynb
│   ├── 03_neural_network_from_scratch.ipynb
│   ├── 04_mnist_fully_connected.ipynb
│   ├── 05_mnist_cnn.ipynb
│   └── 06_two_digit_mnist.ipynb
├── classical/
├── neural_networks/
└── two_digit/
```

The important architectural boundary is that **data preparation, reusable models, and experiments remain separate**.

## Connection to Unit 2

The project is the practical synthesis of the Unit 2 material:

```text
Linear regression
      |
      v
Nonlinear classification
      |
      +--> feature mappings
      +--> kernels
      +--> Softmax classification
      |
      v
Neural networks
      |
      +--> forward propagation
      +--> backpropagation
      +--> learned representations
      |
      v
CNNs
      |
      +--> local receptive fields
      +--> shared filters
      +--> spatial structure
```

The project should be studied as a continuation of the lectures, not as an unrelated MNIST coding exercise.

## Notebooks

### 01 — Classical MNIST

`01_classical_mnist.ipynb` compares linear regression, a linear SVM, and multinomial Softmax regression on the same raw-pixel representation. It also establishes the baseline for later experiments.

For multiclass Softmax regression, the model produces one score per digit. If the scores are $z_0,\ldots,z_9$, Softmax converts them into probabilities:

$$
p_k=\frac{\exp(z_k)}{\sum_{j=0}^{9}\exp(z_j)}.
$$

The predicted class is the digit with the largest probability.

### 02 — Features and kernels

`02_features_and_kernels.ipynb` investigates PCA, feature mappings, polynomial kernels, and Gaussian/RBF kernels. The notebook distinguishes explicitly between a feature map $\phi(x)$ and a kernel $K(x,z)$.

For the polynomial kernel used in the project:

$$
K(x,z)=(x^{T}z+c)^p.
$$

If model selection is performed, validation data must be used to choose parameters such as the polynomial degree or RBF scale. The test set remains untouched until final evaluation.

### 03 — Neural network from scratch

`03_neural_network_from_scratch.ipynb` demonstrates a small network with two inputs, three hidden ReLU units, and one linear output. It focuses on forward propagation, loss computation, the chain rule, backpropagation, and gradient descent without hiding the central calculations behind a framework.

### 04 — Fully connected MNIST network

`04_mnist_fully_connected.ipynb` applies neural networks to MNIST. The notebook identifies the tensors, layer dimensions, activations, loss, optimizer, and training loop, and uses training/validation curves to distinguish learning from overfitting.

### 05 — Convolutional neural network

`05_mnist_cnn.ipynb` introduces convolution as an image-oriented representation. It explains locality, shared filters, and pooling before comparing a CNN with the fully connected model.

### 06 — Two-digit MNIST

`06_two_digit_mnist.ipynb` extends the task to an image containing two digits. A shared representation feeds two output heads, one for each digit. The notebook reports per-digit accuracy and an exact-match metric when appropriate.

For two output heads, the total loss can be represented as

$$
L=L_1+L_2,
$$

where $L_1$ and $L_2$ are the losses for the two digits.

## Experimental discipline

The project follows the same experimental safeguards used in Project 1:

```text
training data
      |
      +--> train model
      |
      +--> choose hyperparameters using validation data
      |
      v
retrain selected model
      |
      v
test set
      |
      v
final evaluation
```

Do not use the test set to choose model architecture, regularization, learning rate, number of epochs, feature-map parameters, kernel parameters, or any other hyperparameter.

When experiments are compared, keep the comparison controlled and state what changed.

## What the project should teach

The important outcome is the progression in representation and learning:

1. **Linear models** use a fixed representation and learn a linear decision rule.
2. **Feature mappings and kernels** increase the expressive power available to a classical learner.
3. **Neural networks** learn intermediate representations together with the classifier.
4. **Convolutional networks** build image-specific structure into the representation through locality and shared parameters.
5. **Multi-output networks** show how one input representation can support multiple related classification tasks.

## Relationship to Project 1

Project 1 used product reviews to apply Unit 1 linear-classification ideas. Project 2 follows the same educational pattern but uses images and expands the model family from classical classifiers to fully connected and convolutional neural networks.

The common principle is to separate **data representation**, **learning algorithm**, **model selection**, and **final evaluation**.

## Requirements

The project uses Python 3 and, depending on the notebook, packages such as:

```text
numpy
scipy
matplotlib
scikit-learn
torch
tqdm
```

Install dependencies with:

```bash
pip install -r requirements.txt
```

## Running the project

The notebooks are the primary study interface. Reusable implementations live in the `classical/`, `neural_networks/`, and `two_digit/` Python packages.

Launch Jupyter from the `project_2/` directory:

```bash
jupyter notebook notebooks/
```

To reproduce the notebook outputs, execute the notebooks in order from 01 through 06. Notebooks 04–06 involve PyTorch training loops and will take noticeably longer than Notebooks 01–03.

The Python modules can also be imported directly, for example:

```python
from classical.data_utils import get_MNIST_data
from classical.softmax import softmax_regression, compute_test_error
from neural_networks.fully_connected import build_fully_connected_model
from two_digit.data_utils import get_two_digit_data
```

For repository-wide Markdown and LaTeX conventions, see [MATH_NOTATION.md](../../MATH_NOTATION.md).
