# Unit 1 — Linear Classifiers and Generalizations

Unit 1 introduces linear classification and develops the ideas needed to understand how classifiers are trained, regularized, and evaluated for generalization.

The unit progresses from the basic machine-learning problem to linear decision boundaries, the perceptron, hinge loss, margins, regularization, and cross-validation.

## Lectures and recitation

### Lecture 1. Introduction to Machine Learning

Introduces the machine-learning problem, supervised binary classification, hypothesis classes, linear classifiers, decision boundaries, training error, and the perceptron update.

**Study artifact:** `lecture_1/README.md` and the executable Lecture 1 notebook.

### Lecture 2. Linear Classifier and Perceptron

Develops the linear classifier and perceptron more fully, including the augmented-vector representation, perceptron learning, and convergence for linearly separable data.

**Study artifact:** `lecture_2/README.md` and the executable Lecture 2 notebook.

### Lecture 3. Hinge loss, Margin boundaries and Regularization

Introduces hinge loss, margin boundaries, regularization, and the optimization perspective for learning linear classifiers.

**Study artifact:** `lecture_3/README.md` and the executable Lecture 3 notebook.

### Lecture 4. Linear Classification and Generalization

Connects linear classification to generalization and model selection, including validation, cross-validation, and the distinction between parameters and hyperparameters.

**Study artifact:** `lecture_4/README.md` and the executable Lecture 4 notebook.

### Recitation 1: Tuning the Regularization Hyperparameter by Cross Validation and a Demonstration

Provides a practical demonstration of tuning the regularization hyperparameter using cross-validation and connects the procedure to the concepts developed in the lectures.

**Study artifact:** the Recitation 1 notebook and supporting documentation.

## Unit progression

```text
Lecture 1
Introduction to Machine Learning
        ↓
Lecture 2
Linear Classifier and Perceptron
        ↓
Lecture 3
Hinge loss, Margin boundaries and Regularization
        ↓
Lecture 4
Linear Classification and Generalization
        ↓
Recitation 1
Tuning the Regularization Hyperparameter by Cross Validation
```

The progression is intentional: first understand the classification problem and linear hypothesis class, then learn a classifier with the perceptron, introduce an optimization-friendly loss and regularization, and finally study how to select hyperparameters and evaluate generalization.

## Documentation convention

Each lecture README is a study guide rather than a copy of the lecture notes. It should explain the mathematics, definitions, intuition, and relationship between the equations and the executable examples.

The notebooks are the executable learning artifacts. They should expose the calculations and implementations clearly, preferably using Python and NumPy directly before relying on higher-level machine-learning libraries.

### Equation-rendering safeguard

All mathematical READMEs in this unit should follow the repository's conservative GitHub MathJax conventions:

- use `$$` on separate lines for display equations;
- use `$...$` for inline mathematics;
- use explicit `\\` row separators in matrices and multiline environments;
- keep `cases` and `aligned` environments completely inside display blocks;
- avoid malformed LaTeX such as single-backslash matrix row separators;
- verify the **rendered GitHub page** after mathematical edits.

## Unit 0 relationship

Unit 0 provides the prerequisite review material needed before beginning this unit:

- vectors and planes;
- optimization concepts;
- NumPy exercises;
- common Python package tutorials.

Unit 1 then applies those foundations to linear classifiers and the study of generalization.
