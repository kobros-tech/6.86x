# Lecture 4 — Regularization, Cross-Validation, and Hyperparameter Selection

This directory contains the Lecture 4 and Recitation 4 demo projects as executable Jupyter notebooks. The goal of this README is to make the directory useful as a **standalone study guide**, so you can review the main ideas without returning to the lecture transcript.

---

## 1. What Lecture 4 is about

Lecture 3 focused on learning model parameters such as `theta` from training data.

Lecture 4 introduces a second question:

> **How do we choose the hyperparameters of the learning algorithm?**

In this lecture the main hyperparameter is `alpha`, the strength of regularization.

A useful way to think about the two levels is:

```text
                 MACHINE LEARNING
                       |
             +---------+---------+
             |                   |
       Parameter learning   Hyperparameter selection
             |                   |
          theta               alpha
             |                   |
       optimize objective    cross-validation
```

We therefore **reuse the parameter optimizer from Lecture 3**. Lecture 4 does not require a fundamentally new optimization algorithm. The new strategy is **cross-validation for model selection**.

---

# 2. Parameters vs. hyperparameters

### Parameters

Parameters are learned directly from the training data.

For a linear classifier:

$$
\theta = [\theta_1, \theta_2, \ldots, \theta_d]
$$

The learning algorithm changes `theta` during training to minimize the objective.

### Hyperparameters

Hyperparameters are chosen before/during the model-selection process, rather than being directly learned as part of the parameter optimization.

Examples include:

- regularization strength `alpha`
- learning rate
- number of neighbors in KNN
- tree depth
- SVM kernel parameters

Lecture 4 concentrates on selecting `alpha`.

---

# 3. Regularized objective function

The general idea is to combine two terms:

$$
J(\theta;\alpha)
=
\text{loss}(\theta)
+
\alpha\,\text{regularization}(\theta)
$$

The first term asks:

> How well does the model fit the training examples?

The second term asks:

> How complicated/large is the model?

`alpha` controls the trade-off.

### Small alpha

If

$$
\alpha \approx 0,
$$

regularization has little influence. The model concentrates on minimizing training loss.

This can produce a model that fits the training data extremely well but does not generalize as well to unseen data.

### Large alpha

As `alpha` increases, regularization becomes more important. The model is discouraged from using unnecessarily large parameter values.

If `alpha` becomes too large, the model can become **too constrained** and underfit the data.

So we expect an intermediate value to often work best.

---

# 4. The key trade-off

Think about what happens as `alpha` increases.

### Training performance

At very small `alpha`, the model has considerable freedom to fit the training data.

As regularization becomes stronger:

- training loss generally increases,
- training accuracy can decrease,
- the model becomes less flexible.

Therefore, training performance alone is **not** enough to choose `alpha`.

### Unseen-data performance

For a very small `alpha`, the model may overfit.

Increasing `alpha` can initially improve generalization:

```text
validation accuracy
       ^
       |              /\
       |             /  \
       |            /    \
       |___________/      \____
       +-------------------------> alpha
                         ^
                       alpha*
```

Eventually, excessive regularization causes underfitting, so validation performance falls again.

The ideal value is therefore approximately:

$$
\alpha^*
=
\arg\max_{\alpha} S(\alpha),
$$

where `S(alpha)` is the validation score obtained for that value of `alpha`.

If using an objective/error rather than accuracy, the equivalent selection rule is to **minimize** the validation objective:

$$
\alpha^*
=
\arg\min_{\alpha} J_{validation}(\alpha).
$$

---

# 5. Why we cannot choose alpha using the test set

Suppose we have:

```text
                 All available data
                        |
                +-------+-------+
                |               |
             Training          Test
                |               |
         choose alpha       final evaluation
```

The test set is supposed to represent **unseen data**.

If we repeatedly try different `alpha` values and select the one that performs best on the test set, then the test set has influenced our model-selection decision.

It is no longer a genuinely untouched test set.

Therefore:

> **Use training data for learning and validation/model selection. Save the test set for the final evaluation.**

But this creates a problem: where do we get validation data from if we only have a training set?

The answer is **cross-validation**.

---

# 6. Validation set

The simplest approach is to split the training data into two parts:

```text
Training data
     |
     +------------------+
     |                  |
   train              validation
     |                  |
 learn theta       evaluate alpha
```

For a candidate `alpha`:

1. Train `theta` using the training portion.
2. Evaluate the trained model on the validation portion.
3. Record the validation score.
4. Try another `alpha`.
5. Select the best `alpha`.

The disadvantage is that the result depends strongly on one particular split.

Cross-validation reduces this dependence.

---

# 7. K-fold cross-validation

Suppose we choose:

$$
K=5.
$$

We divide the training data into five folds:

```text
+---------+---------+---------+---------+---------+
| Fold 1  | Fold 2  | Fold 3  | Fold 4  | Fold 5  |
+---------+---------+---------+---------+---------+
```

For one candidate `alpha`, we train and validate five times.

### Fold 1

```text
TRAIN TRAIN TRAIN TRAIN VALIDATION
```

### Fold 2

```text
TRAIN TRAIN TRAIN VALIDATION TRAIN
```

### Fold 3

```text
TRAIN TRAIN VALIDATION TRAIN TRAIN
```

### Fold 4

```text
TRAIN VALIDATION TRAIN TRAIN TRAIN
```

### Fold 5

```text
VALIDATION TRAIN TRAIN TRAIN TRAIN
```

Every example gets an opportunity to be in the validation set, while the remaining examples are used for training.

---

# 8. Computing the cross-validation score

For a candidate `alpha`, suppose the five validation accuracies are:

$$
S_1(\alpha),S_2(\alpha),S_3(\alpha),S_4(\alpha),S_5(\alpha).
$$

The mean cross-validation score is:

$$
S(\alpha)
=
\frac{1}{K}
\sum_{k=1}^{K}S_k(\alpha).
$$

For `K = 5`:

$$
S(\alpha)
=
\frac{S_1+S_2+S_3+S_4+S_5}{5}.
$$

This gives us one performance estimate for that particular value of `alpha`.

We repeat the entire K-fold process for every candidate value:

$$
\alpha_1,\alpha_2,\ldots,\alpha_M.
$$

Then choose:

$$
\boxed{\alpha^*=\arg\max_{\alpha}S(\alpha)}
$$

when the score is accuracy (or another metric where larger is better).

---

# 9. The complete cross-validation algorithm

The complete procedure is:

```text
Choose candidate alphas
        |
        v
alpha_1, alpha_2, ..., alpha_M
        |
        v
For each alpha:
        |
        +--> split training data into K folds
        |
        +--> for each fold:
        |       |
        |       +--> train theta on K-1 folds
        |       +--> evaluate on remaining fold
        |       +--> record validation score
        |
        +--> average K validation scores
        |
        v
S(alpha)
        |
        v
Choose alpha* with maximum S(alpha)
        |
        v
Train final model using alpha*
        |
        v
Use all training data
        |
        v
Evaluate ONCE on untouched test data
```

This is the central workflow of Lecture 4.

---

# 10. Why cross-validation works

Each validation fold behaves like a small unseen dataset.

Instead of asking:

> "Which alpha works best on this one particular validation split?"

we ask:

> "Which alpha consistently works well across several different validation splits?"

This makes the hyperparameter selection less dependent on a single arbitrary split.

There is also a useful conceptual distinction:

- **training performance** tells us how well the model fits data it was trained on;
- **validation performance** helps us select the model/hyperparameters;
- **test performance** estimates how the final selected model performs on genuinely unseen data.

---

# 11. Lecture example: breast-cancer classification

The practical lecture example uses the breast-cancer dataset available through scikit-learn.

The labels represent two classes:

- benign
- malignant

The dataset contains multiple numerical tumor attributes, such as measurements related to tumor radius and texture.

The full feature vector contains many attributes. The lecture visualizes the first two features to give an intuitive picture of the classification problem.

The model itself can use all available features.

---

# 12. Feature scaling

The lecture standardizes the input features to approximately:

$$
\text{mean}=0,
\qquad
\text{variance}=1.
$$

This is especially important for models involving parameter magnitudes and regularization because features measured on very different scales can otherwise affect optimization and the regularization trade-off unevenly.

A robust implementation should fit the scaler using the training portion of each cross-validation fold rather than allowing validation/test information to influence preprocessing. In scikit-learn, a `Pipeline` is a convenient way to enforce this safely.

---

# 13. The lecture's SVM model

The practical example uses scikit-learn's stochastic-gradient-based linear classifier:

```python
SGDClassifier(
    loss="hinge",
    penalty="l2",
    alpha=alpha
)
```

Conceptually:

- `loss="hinge"` gives the SVM-style hinge loss;
- `penalty="l2"` applies L2 regularization to the parameters;
- `alpha` controls the regularization strength.

The model is trained separately for each candidate `alpha`.

Then K-fold cross-validation measures how well that choice generalizes to held-out folds.

---

# 14. Searching over alpha

We normally do not magically calculate the perfect `alpha` analytically.

Instead, we define a candidate grid such as:

```text
alpha = 0.001
alpha = 0.01
alpha = 0.02
alpha = 0.03
...
alpha = 1.0
```

Then evaluate every candidate using cross-validation.

For each candidate:

$$
\alpha_i
\rightarrow
\text{train K models}
\rightarrow
\text{K validation scores}
\rightarrow
\text{mean score }S(\alpha_i).
$$

Finally:

$$
\alpha^*=\arg\max_i S(\alpha_i).
$$

This is a **hyperparameter search**.

The lecture example finds an `alpha*` around `0.04` for its particular setup. That number is **not universal**; it depends on the data, preprocessing, model, random state, candidate grid, and cross-validation procedure.

---

# 15. After alpha* is selected

Finding `alpha*` is not the end of training.

Once the best hyperparameter has been selected, we train the final model using that value:

```python
model = SGDClassifier(
    loss="hinge",
    penalty="l2",
    alpha=alpha_star
)

model.fit(X_train, y_train)
```

The final model can then be evaluated on the untouched test set.

The important rule is:

> **Do not use the test set to decide alpha*.**

---

# 16. Decision boundary

For a linear classifier, the decision function has the form:

$$
\theta^T x + b = 0.
$$

In two dimensions this becomes a line:

$$
\theta_1x_1+\theta_2x_2+b=0.
$$

The lecture plots the decision boundary using the first two attributes so that we can visually inspect the classifier.

Remember that the plotted 2D line is only a visualization. The actual model can be trained using all features:

$$
\theta^T x+b=0
$$

in the full feature space.

---

# 17. Important conceptual distinction: optimization vs. model selection

This is one of the most important ideas to remember.

### Inner problem — optimize parameters

For a fixed `alpha`, find good parameters:

$$
\theta^*(\alpha)
=
\arg\min_{\theta} J(\theta;\alpha).
$$

This is the optimization problem discussed in the earlier material.

### Outer problem — select hyperparameter

Use cross-validation to decide which `alpha` is best:

$$
\alpha^*
=
\arg\max_{\alpha} S(\alpha).
$$

So Lecture 4 effectively introduces a two-level process:

```text
                 choose alpha
                     |
                     v
             +---------------+
             | fixed alpha   |
             |               |
             | optimize theta|
             +---------------+
                     |
                     v
              validation score
                     |
                     +----> compare with other alphas
```

This is why **we do not need a completely new optimizer in Lecture 4**. We are adding a model-selection layer around the existing training procedure.

---

# 18. Common mistakes

### Mistake 1 — Choosing alpha using the test set

Wrong:

```text
try alpha values -> test each one -> choose best alpha
```

Correct:

```text
training data -> cross-validation -> choose alpha*
                                      |
                                      v
                              final test evaluation
```

### Mistake 2 — Looking only at training accuracy

A model can have excellent training accuracy and poor generalization.

Hyperparameters should be selected using validation performance, not training performance.

### Mistake 3 — Thinking larger alpha is always better

More regularization is not automatically better. Too little regularization can overfit; too much can underfit.

### Mistake 4 — Treating alpha* = 0.04 as a universal constant

It is only the best candidate for the particular experiment.

### Mistake 5 — Data leakage during preprocessing

If feature scaling is calculated using validation/test examples before cross-validation, information from the held-out data can leak into training.

Use a pipeline so preprocessing is fitted only on the training portion of each fold.

### Mistake 6 — Confusing validation with the final test set

A validation fold is temporarily held out to help make decisions. The final test set should remain untouched until the model-selection process is complete.

---

# 19. What to remember for an exam

If you remember only the following, you have the core of Lecture 4:

1. **Parameters** such as `theta` are learned by the training algorithm.
2. **Hyperparameters** such as `alpha` control how the algorithm/model is trained.
3. `alpha` controls the strength of regularization.
4. Small `alpha` can allow overfitting.
5. Very large `alpha` can cause underfitting.
6. The best `alpha` should be selected using validation data, not the test set.
7. **K-fold cross-validation** repeatedly trains on `K-1` folds and validates on the remaining fold.
8. Average the validation scores across folds.
9. Try multiple candidate `alpha` values.
10. Choose

$$
\boxed{\alpha^*=\arg\max_{\alpha}S(\alpha)}
$$

when higher validation score is better.

11. After choosing `alpha*`, retrain the final model using the training data.
12. Evaluate the final model on the untouched test set.

---

# 20. Lecture 4 demo notebooks

The notebooks in this directory turn these ideas into executable experiments:

- **Lecture notebook** — follows the lecture's practical SVM/cross-validation example.
- **Recitation notebook** — reinforces the same concepts with hands-on implementation and experiments.

Run the notebooks from top to bottom to reproduce the calculations, logs, and figures.

---

# 21. One-page mental model

```text
REGULARIZED LEARNING

For a chosen alpha:

    minimize

        J(theta; alpha)
        = loss(theta) + alpha * regularization(theta)

    over theta

            |
            v
      trained parameters theta*

            |
            v
       validation score

Repeat for many alpha values

            |
            v

      alpha* = argmax S(alpha)

            |
            v

   retrain final model with alpha*

            |
            v

      untouched test evaluation
```

The key idea is simple:

> **Lecture 3 asks: "What parameters should the model learn?"**
>
> **Lecture 4 asks: "How should we choose the hyperparameter that controls that learning?"**
>
> The answer is **cross-validation**.
