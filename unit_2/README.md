# Unit 2 — Regression, Nonlinear Classification, and Neural Networks

Unit 2 extends the machine-learning foundations from Unit 1 beyond linear classification. It introduces regression, nonlinear decision rules, feature mappings and kernels, collaborative filtering, and neural networks.

The unit progresses from linear regression and regularization to nonlinear classification, then develops matrix factorization and collaborative filtering before bringing these ideas together in Project 2, where classical models and neural networks are applied to handwritten digit recognition.

## Lectures and project

### Lecture 5. Linear Regression

Introduces regression as a prediction problem with real-valued targets. The lecture develops squared-error loss, empirical risk, gradient descent, the closed-form solution, and regularization.

**Study artifact:** `lecture_5/README.md` and the executable Lecture 5 notebook.

### Lecture 6. Nonlinear Classification

Extends linear classification through feature mappings and the kernel trick. The lecture develops the idea of representing nonlinear decision boundaries through a higher-dimensional feature space without explicitly constructing every feature.

**Study artifact:** `lecture_6/README.md` and the executable Lecture 6 notebooks.

### Lecture 7. Matrix Factorization and Collaborative Filtering

Introduces low-rank matrix factorization for recommender systems. The lecture develops latent-factor representations, alternating minimization, regularization, rank selection, and model evaluation with RMSE.

**Study artifacts:** `lecture_7/README.md` and the executable Lecture 7 notebooks.

### Project 2. MNIST Digit Recognition

Applies Unit 2 ideas to handwritten digit recognition. The project begins with classical models and then moves to learned representations:

- linear regression as a baseline;
- linear SVM for multiclass classification;
- multinomial Softmax regression;
- PCA, explicit feature mappings, and polynomial/RBF kernels;
- a neural network implemented from scratch;
- a fully connected PyTorch network;
- a convolutional neural network;
- a two-digit recognition task with multiple output heads.

The project emphasizes the distinction between representation, learning algorithm, model selection, and final evaluation.

**Study artifacts:** `project_2/README.md`, `project_2/NOTEBOOK_GUIDE.md`, reusable Python modules, and the executable notebooks under `project_2/notebooks/`.

## Unit progression

```text
Unit 1
Linear classification
        |
        v
Lecture 5
Linear regression
        |
        +--> squared-error optimization
        +--> gradient descent
        +--> closed-form solution
        +--> regularization
        |
        v
Lecture 6
Nonlinear classification
        |
        +--> feature mappings
        +--> kernel trick
        +--> polynomial / RBF kernels
        |
        v
Lecture 7
Matrix factorization
        |
        +--> low-rank assumption
        +--> latent factors
        +--> alternating minimization
        +--> regularization
        +--> rank selection
        +--> RMSE
        |
        v
Project 2
MNIST digit recognition
        |
        +--> linear regression
        +--> linear SVM
        +--> Softmax regression
        +--> feature mappings and kernels
        +--> neural networks
        +--> CNNs
        +--> multi-output recognition
```

The progression is intentional. Lecture 5 broadens the prediction task from classification to regression. Lecture 6 then shows how a fixed linear model can become more expressive through feature representations and kernels. Lecture 7 changes the structure of the prediction problem again, representing users and items through low-dimensional latent factors. Project 2 uses these ideas as a bridge into neural networks and image recognition.

## Core themes

### 1. Optimization

Unit 2 continues the optimization perspective introduced in Unit 1. Parameters are learned by minimizing an objective, whether that objective is squared error, a classification loss, or a regularized matrix-factorization objective.

### 2. Representation

A major theme is that the representation of the input matters. We move from raw features to transformed features, implicit kernel feature spaces, and learned latent representations.

### 3. Generalization and regularization

Training error alone is not sufficient. Validation data and cross-validation help select hyperparameters, while regularization controls model complexity and can improve performance on unseen data.

### 4. From fixed to learned representations

Feature mappings and kernels give classical algorithms access to richer representations. Neural networks go further by learning intermediate representations from data.

## Documentation convention

Each lecture README is a study guide rather than a copy of the lecture notes. It should explain the mathematics, definitions, intuition, and relationship between the equations and the executable examples.

The notebooks are the executable learning artifacts for lecture and recitation demonstrations. Project 2 uses Python modules for reusable algorithms and data processing, with notebooks providing the presentation and experimentation layer.

### Equation-rendering safeguard

All mathematical READMEs in this unit should follow the repository's conservative GitHub MathJax conventions:

- use `$$` on separate lines for display equations;
- use `$...$` for inline mathematics;
- use explicit `\\` row separators in matrices and multiline environments;
- keep `cases` and `aligned` environments completely inside display blocks;
- avoid malformed LaTeX such as single-backslash matrix row separators;
- verify the **rendered GitHub page** after mathematical edits.

## Relationship to Unit 1

Unit 1 established the basic supervised-learning workflow through linear classification:

- define a prediction problem;
- choose a hypothesis class;
- define an objective or loss;
- optimize parameters;
- evaluate generalization;
- select hyperparameters without using the final test set.

Unit 2 keeps this workflow but broadens the models and representations. Linear regression introduces continuous targets, kernels introduce nonlinear decision boundaries, matrix factorization introduces latent representations, and neural networks learn representations jointly with prediction functions.

## What to remember

The central progression of Unit 2 is:

1. **Regression:** learn parameters by minimizing squared error.
2. **Nonlinear classification:** use feature mappings or kernels to obtain richer decision rules.
3. **Collaborative filtering:** represent users and items with low-dimensional latent factors.
4. **Neural networks:** learn intermediate representations and prediction parameters together.
5. **CNNs:** exploit spatial structure through local connectivity and shared filters.
6. **Model selection:** use validation information for hyperparameters and reserve the test set for final evaluation.
