# MIT 6.86x — Machine Learning

A study-oriented repository for working through **MIT 6.86x: Machine Learning with Python**.

The goal is not simply to reproduce course code. This repository is built as a **learning record and reference**: mathematical ideas are explained, algorithms are implemented with Python and NumPy where appropriate, experiments are visualized, and course projects connect the theory to practical machine-learning problems.

> **Current progress:** Units 0–2 are available. Units 3–5 are planned and will be added as the course progresses.

## Learning path

The repository follows the progression of the course while keeping the mathematical connections visible:

```text
Prerequisites
     |
     v
Unit 0 — Mathematical & Python Foundations
     |
     v
Unit 1 — Linear Classification
     |
     v
Unit 2 — Regression, Nonlinear Classification & Collaborative Filtering
     |
     v
Unit 3 — Neural Networks                         [planned]
     |
     v
Unit 4 — Unsupervised Learning                   [planned]
     |
     v
Unit 5 — Reinforcement Learning                  [planned]
```

The later units are intentionally shown as **planned** rather than as completed course material. The repository will be updated as each unit is studied and implemented.

## Course structure

### Unit 0 — Brief Prerequisite Reviews

Foundational material needed for the machine-learning lectures:

- Vectors, planes, and optimization
- NumPy exercises
- Common Python packages
- Project 0 setup

**Status:** Reference material available

[Explore Unit 0 →](unit_0/)

---

### Unit 1 — Linear Classifiers and Generalizations

Builds the foundations of supervised binary classification and the optimization-based view of learning.

- Introduction to machine learning
- Linear classifiers and decision boundaries
- Perceptron and its convergence
- Hinge loss and margins
- Regularization
- Generalization and model selection
- Cross-validation
- Project 1: **Automatic Review Analyzer**

**Status:** Available / in progress

[Explore Unit 1 →](unit_1/)

---

### Unit 2 — Regression, Nonlinear Classification, and Collaborative Filtering

Extends the ideas from Unit 1 beyond basic linear classification. The unit introduces continuous prediction, nonlinear decision rules, feature mappings and kernels, and low-rank representations for recommender systems.

- Lecture 5: **Linear Regression**
  - Squared-error optimization
  - Gradient descent
  - Closed-form solution
  - Regularization
- Lecture 6: **Nonlinear Classification**
  - Feature mappings
  - Kernel trick
  - Polynomial and RBF kernels
- Lecture 7: **Matrix Factorization and Collaborative Filtering**
  - Low-rank assumption
  - Latent factors
  - Alternating minimization
  - Regularization
  - Rank selection
  - RMSE
- Project 2: **MNIST Digit Recognition**
  - Classical classification methods
  - Softmax regression
  - Feature mappings and kernels
  - Neural-network experiments as a practical bridge toward the later neural-network material

**Status:** Available / in progress

[Explore Unit 2 →](unit_2/)

---

### Unit 3 — Neural Networks *(planned)*

Will develop neural networks in greater depth, including the ideas behind forward propagation, backpropagation, optimization, and learned representations.

**Status:** Not yet added

---

### Unit 4 — Unsupervised Learning *(planned)*

Will introduce learning from data without labeled targets, including the major unsupervised-learning methods covered later in the course.

**Status:** Not yet added

---

### Unit 5 — Reinforcement Learning *(planned)*

Will introduce learning through interaction with an environment, including the core concepts and algorithms used in reinforcement learning.

**Status:** Not yet added

## At a glance

| Unit | Main focus | Status |
| --- | --- | --- |
| [Unit 0](unit_0/) | Prerequisite reviews | Available |
| [Unit 1](unit_1/) | Linear classifiers and generalization | Available / in progress |
| [Unit 2](unit_2/) | Regression, nonlinear classification, matrix factorization | Available / in progress |
| Unit 3 | Neural networks | Planned |
| Unit 4 | Unsupervised learning | Planned |
| Unit 5 | Reinforcement learning | Planned |

## How the repository is organized

Each completed unit follows a similar structure:

```text
unit_N/
├── README.md                 # Unit-level study guide
├── lecture_X/
│   ├── README.md             # Mathematical notes and explanations
│   └── *.ipynb               # Executable demonstrations
└── project_N/
    ├── README.md             # Project guide
    ├── *.py                  # Reusable implementations
    └── *.ipynb               # Experiments and visualizations
```

The exact contents vary by lecture and project, but the principle is consistent: **documentation explains the ideas; notebooks demonstrate them; Python modules contain reusable implementations.**

## Learning approach

The repository follows a consistent study workflow:

1. **Define the problem** and the hypothesis class.
2. **Introduce the mathematics** behind the model or algorithm.
3. **Work through small examples** to make the equations concrete.
4. **Implement the mechanism** directly with Python and NumPy when useful.
5. **Visualize** geometry, optimization, convergence, or model behavior where it improves understanding.
6. **Use higher-level libraries** when they add practical value, while keeping the underlying concepts explicit.
7. **Evaluate generalization** using appropriate validation procedures.
8. **Document the connection** between the mathematical formulation and the executable experiment.

A recurring principle throughout the repository is to distinguish between **representation, model, objective, optimization, model selection, and final evaluation**.

## Mathematical notation

Lecture READMEs use GitHub-compatible MathJax. Display equations use `$$` blocks, inline mathematics uses `$...$`, and multiline equations use explicit LaTeX row separators.

When editing mathematical documentation, the rendered GitHub page should always be checked rather than relying only on the raw Markdown source.

## Current scope

The repository is being developed incrementally rather than pretending that the entire course has already been documented.

**Currently covered:**

- Mathematical and Python prerequisites
- Linear classification
- Perceptron and linear SVM concepts
- Hinge loss, margins, and regularization
- Generalization and cross-validation
- Linear regression
- Nonlinear classification and kernels
- Matrix factorization and collaborative filtering
- MNIST classification experiments

**Coming later:**

- Neural networks as a dedicated unit
- Unsupervised learning
- Reinforcement learning

This separation keeps the root README useful as a **course roadmap** while the unit READMEs provide the detailed study material.

## Purpose

This is a personal study and implementation repository. The examples are written to reinforce the concepts of the course, make mathematical steps explicit, and provide a reproducible record of the learning process.
