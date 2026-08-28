# MIT 6.86x — Machine Learning

A study-oriented repository for working through **MIT 6.86x: Machine Learning with Python**.

The repository combines mathematical notes, executable Python/NumPy demonstrations, visualizations, and course projects. The goal is to make the underlying machine-learning ideas understandable and reproducible rather than treating library calls as black boxes.

## Course structure

### Unit 0. Brief Prerequisite Reviews

- **Brief Review of Vectors, Planes, and Optimization**
- **Project 0 Setup, Numpy Exercises, Tutorial on Common Packages**

### Unit 1. Linear Classifiers and Generalizations

- **Lecture 1. Introduction to Machine Learning**
- **Lecture 2. Linear Classifier and Perceptron**
- **Lecture 3. Hinge loss, Margin boundaries and Regularization**
- **Lecture 4. Linear Classification and Generalization**
- **Recitation 1: Tuning the Regularization Hyperparameter by Cross Validation and a Demonstration**
- **Project 1. Automatic Review Analyzer**

### Unit 2. Regression, Nonlinear Classification, and Neural Networks

- **Lecture 5. Linear Regression**
- **Lecture 6. Nonlinear Classification**
- **Lecture 7. Matrix Factorization and Collaborative Filtering**
- **Project 2. MNIST Digit Recognition**

## Learning approach

The repository follows a consistent progression:

1. Introduce the mathematical definition.
2. Work through small examples.
3. Implement the mechanism directly with Python and NumPy.
4. Visualize the geometry or optimization behavior when useful.
5. Connect the implementation to higher-level machine-learning tools.
6. Keep the README files as study references and the notebooks as executable demonstrations.

## Units

| Unit | Topic | Status |
| --- | --- | --- |
| [Unit 0](unit_0/) | Brief Prerequisite Reviews | Reference material |
| [Unit 1](unit_1/) | Linear Classifiers and Generalizations | In progress |
| [Unit 2](unit_2/) | Regression, Nonlinear Classification, and Neural Networks | In progress |

## Mathematical notation

Lecture READMEs use GitHub-compatible MathJax. Display equations are written with `$$` blocks, inline mathematics uses `$...$`, and multiline matrices, `cases`, and `aligned` environments use explicit LaTeX row separators.

When editing mathematical documentation, always inspect the **rendered GitHub page**, not only the raw Markdown source.

## Repository organization

```text
.
├── unit_0/    # Brief Prerequisite Reviews
├── unit_1/    # Linear Classifiers and Generalizations
└── unit_2/    # Regression, Nonlinear Classification, and Neural Networks
```

Each lecture directory is intended to contain its study documentation and executable learning artifacts.

## Purpose

This is a personal study and implementation repository. The examples are written to reinforce the concepts of the course and to make the mathematical steps explicit.
