# Unit 1 — Linear Classifiers and Generalizations

Unit 1 introduces linear classification and develops the ideas needed to understand how classifiers are trained, regularized, and evaluated for generalization.

The unit progresses from the basic machine-learning problem to linear decision boundaries, the perceptron, hinge loss, margins, regularization, and cross-validation. Project 1 then brings these ideas together in a text-classification application.

## Lectures, recitation, and project

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

### Recitation 4. Cross Validation

Provides a practical demonstration of tuning the regularization hyperparameter using cross-validation and connects the procedure to the concepts developed in Lecture 4.

**Study artifact:** `lecture_4/recitation_4_cross_validation.ipynb`.

### Project 1. Automatic Review Analyzer

Applies Unit 1 to sentiment classification of product reviews. The project builds a sparse bag-of-words representation and implements three linear learning algorithms directly in Python:

- Perceptron;
- Average Perceptron;
- Pegasos, a stochastic sub-gradient method for a regularized linear SVM objective.

The project also demonstrates the separation between training, validation, and final evaluation, and connects the implementation to the Pegasos research paper by Shalev-Shwartz, Singer, Srebro, and Cotter.

**Study artifacts:** `project_1/README.md`, the executable Python implementation and tests, and `project_1/automatic_review_analyzer.ipynb` for experiments and visualizations.

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
Recitation 4
Cross Validation
        ↓
Project 1
Automatic Review Analyzer
        |
        +--> Bag-of-words representation
        +--> Perceptron
        +--> Average Perceptron
        +--> Pegasos / regularized SVM
        +--> Validation and final evaluation
        +--> Research experiments and visualizations
```

The progression is intentional: first understand the classification problem and linear hypothesis class, then learn a classifier with the perceptron, introduce an optimization-friendly loss and regularization, study generalization and hyperparameter selection, and finally apply the complete workflow to a text-classification problem.

## Documentation convention

Each lecture README is a study guide rather than a copy of the lecture notes. It should explain the mathematics, definitions, intuition, and relationship between the equations and the executable examples.

The notebooks are the executable learning artifacts for lecture and recitation demonstrations. Project 1 uses ordinary Python modules for reusable algorithms, data processing, and automated tests, and provides `automatic_review_analyzer.ipynb` as a presentation and experimentation layer for running comparisons and generating visualizations.

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
