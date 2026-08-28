# MIT 6.86x — Machine Learning

A study-oriented repository for working through **MIT 6.86x: Machine Learning with Python**.

The repository combines mathematical study notes, executable Python/NumPy demonstrations, visualizations, and course projects. The goal is to make the underlying machine-learning ideas understandable and reproducible rather than treating library calls as black boxes.

## Course roadmap

The course progresses from mathematical prerequisites through supervised learning, neural networks, unsupervised learning, and reinforcement learning.

| Unit | Main focus | Status |
| --- | --- | --- |
| [Unit 0](unit_0/) | Brief prerequisite reviews | Complete |
| [Unit 1](unit_1/) | Linear classifiers and generalization | Complete |
| [Unit 2](unit_2/) | Regression, nonlinear classification, and collaborative filtering | In progress |
| Unit 3 | Neural networks | Planned |
| Unit 4 | Unsupervised machine learning | Planned |
| Unit 5 | Reinforcement learning | Planned |

## Current progress

The repository currently contains **Units 0–2**. Units 3–5 will be added as the study progresses.

### Unit 0 — Brief Prerequisite Reviews

- Brief review of vectors, planes, and optimization
- Project 0 setup, NumPy exercises, and common Python packages

### Unit 1 — Linear Classifiers and Generalizations

- Introduction to machine learning
- Linear classifiers and the perceptron
- Hinge loss, margins, and regularization
- Generalization and cross-validation
- Project 1: Automatic Review Analyzer

### Unit 2 — Regression, Nonlinear Classification, and Collaborative Filtering

- Lecture 5: Linear Regression
- Lecture 6: Nonlinear Classification
- Lecture 7: Matrix Factorization and Collaborative Filtering
- Project 2: MNIST Digit Recognition

Project 2 currently includes experiments that introduce neural-network methods. The dedicated treatment of neural networks belongs to **Unit 3**, which has not yet been added to the repository.

### Coming later

- **Unit 3:** Neural networks
- **Unit 4:** Unsupervised machine learning
- **Unit 5:** Reinforcement learning

The roadmap will be expanded with the actual lectures and projects when those units are completed rather than guessing their contents in advance.

## Learning approach

The repository follows a consistent progression:

1. Introduce the mathematical definition.
2. Work through small examples.
3. Implement the mechanism directly with Python and NumPy where appropriate.
4. Visualize the geometry or optimization behavior when useful.
5. Connect the implementation to higher-level machine-learning tools.
6. Use the READMEs as study references and the notebooks as executable demonstrations.

The repository-wide rules for writing mathematical notation are documented separately in [MATH_NOTATION.md](MATH_NOTATION.md).

## Repository organization

```text
.
├── MATH_NOTATION.md       # Repository-wide mathematical writing conventions
├── unit_0/                # Brief prerequisite reviews
├── unit_1/                # Linear classifiers and generalization
└── unit_2/                # Regression, nonlinear classification, and collaborative filtering
```

Each lecture directory is intended to contain its study documentation and executable learning artifacts.

## Mathematical notation

Lecture and project READMEs define the symbols needed for their subject matter. General Markdown and LaTeX conventions are maintained centrally in [MATH_NOTATION.md](MATH_NOTATION.md) so that individual study guides do not need to repeat the same rendering checklist.

When editing mathematical documentation, always inspect the **rendered GitHub page**, not only the raw Markdown source.

## Purpose

This is a personal study and implementation repository. The examples are written to reinforce the concepts of the course and to make the mathematical steps explicit.
