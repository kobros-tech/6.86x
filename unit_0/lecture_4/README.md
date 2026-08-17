# Lecture 4 — Regularization, Cross-Validation, and Hyperparameter Selection

This directory contains executable Jupyter notebooks for Lecture 4 and its recitation.

## Main ideas

Lecture 3 optimized the model parameters `theta` for a fixed regularization strength `alpha`. Lecture 4 adds a second level of optimization: selecting `alpha` using validation data.

The workflow is:

```text
training data
    |
    +--> K-fold cross-validation for each candidate alpha
    |        |
    |        +--> train theta on K-1 folds
    |        +--> validate on the held-out fold
    |        +--> repeat K times
    |        +--> average validation score
    |
    +--> choose alpha* with the best mean validation score
             |
             +--> retrain on all training data
                     |
                     +--> evaluate once on untouched test data
```

## Notebooks

- `lecture_4_cross_validation.ipynb` — guided implementation of regularization, K-fold cross-validation, hyperparameter search, and the final model.
- `recitation_4_cross_validation.ipynb` — hands-on implementation exercises and experiments using the same ideas.

Both notebooks are self-contained and generate their plots and logs when executed.

## Important distinction from Lecture 3

We reuse the same linear SVM objective and parameter-optimization idea from Lecture 3. We do **not** invent a new optimizer merely for Lecture 4. The new strategy is **model selection by cross-validation**:

\[
\alpha^* = \arg\max_{\alpha} S(\alpha),
\]

where `S(alpha)` is the mean validation score over the K folds.

## Dataset

The notebooks use scikit-learn's breast-cancer dataset. The features are standardized before training because the linear SVM objective is sensitive to feature scale.

The final test set remains untouched while `alpha` is selected. This avoids using test data to tune the hyperparameter.
