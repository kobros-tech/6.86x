# Lecture 3 — Linear Classification and Optimization

Lecture 3 is intentionally organized as **one executable notebook + one study README**.

- `lecture_3_linear_svm_from_scratch.ipynb` — the complete learning implementation
- `README.md` — the mathematical study guide

There are **no separate `.py` files and no committed `.png` files** for Lecture 3. All code, logs, and figures are produced inside the notebook.

## Learning workflow

The notebook follows the same order we study the material:

1. linear classifier and decision boundary
2. signed margin
3. hinge loss
4. L2 regularization
5. complete objective
6. gradient derivation
7. gradient descent from scratch with NumPy
8. optimization logs and plots
9. experiments with different values of `alpha`
10. scikit-learn comparison only after understanding the implementation

---

## 1. Linear classifier

For an input vector $x$, the classifier score is

$$
f(x)=\theta^T x+\theta_0.
$$

Prediction is based on the sign of the score:

$$
\hat{y}=\begin{cases}
+1 & \text{if } f(x)\ge 0,\\
-1 & \text{if } f(x)<0.
\end{cases}
$$

The decision boundary is

$$
\theta^T x+\theta_0=0.
$$

In two dimensions this is a line. In higher dimensions it is a hyperplane.

---

## 2. Signed margin

For training example $(x_i,y_i)$ with $y_i\in\{-1,+1\}$:

$$
z_i=y_i\left(\theta^T x_i+\theta_0\right).
$$

The sign tells us whether the example is correctly classified, while the magnitude measures how confidently it lies on the correct side of the boundary.

The SVM margin requirement is

$$
z_i\ge 1.
$$

---

## 3. Hinge loss

The hinge loss for one example is

$$
L_i=\max\left(0,1-z_i\right).
$$

Therefore:

- if $z_i\ge 1$, the loss is zero;
- if $z_i<1$, the loss is positive;
- a correctly classified point can still have positive loss if it is inside the desired margin.

The average hinge loss is

$$
L(\theta,\theta_0)=\frac{1}{n}\sum_{i=1}^{n}\max\left(0,1-y_i\left(\theta^T x_i+\theta_0\right)\right).
$$

---

## 4. L2 regularization

We penalize large weights with

$$
R(\theta)=\frac{1}{2}\lVert\theta\rVert_2^2.
$$

The bias is not regularized in our implementation.

The hyperparameter `alpha` controls the strength of this penalty.

- small `alpha` → prioritize fitting the training data;
- large `alpha` → place more pressure on small weights.

---

## 5. Complete objective

For a fixed value of `alpha`, the training objective is

$$
J(\theta,\theta_0;\alpha)=L(\theta,\theta_0)+\alpha R(\theta).
$$

Expanded:

$$
J(\theta,\theta_0;\alpha)
=\frac{1}{n}\sum_{i=1}^{n}\max\left(0,1-y_i\left(\theta^T x_i+\theta_0\right)\right)
+\frac{\alpha}{2}\lVert\theta\rVert_2^2.
$$

The first term asks:

> How well does the model fit the training examples while respecting the margin?

The second term asks:

> How large are the model weights?

Training minimizes their combined value.

---

## 6. Gradient

For an active example, where $z_i<1$:

$$
\nabla_{\theta}L_i=-y_i x_i.
$$

For the bias:

$$
\frac{\partial L_i}{\partial\theta_0}=-y_i.
$$

Including the L2 penalty gives

$$
\nabla_{\theta}J
=-\frac{1}{n}\sum_{i:z_i<1}y_i x_i+\alpha\theta.
$$

and

$$
\frac{\partial J}{\partial\theta_0}
=-\frac{1}{n}\sum_{i:z_i<1}y_i.
$$

Notice that regularization affects $\theta$ but not $\theta_0$.

---

## 7. Gradient descent

With learning rate $\eta$:

$$
\theta\leftarrow\theta-\eta\nabla_{\theta}J
$$

and

$$
\theta_0\leftarrow\theta_0-\eta\frac{\partial J}{\partial\theta_0}.
$$

The notebook implements these updates directly rather than hiding them behind a machine-learning library.

During training we record:

- objective value;
- hinge-loss component;
- regularization component;
- training error;
- parameter values.

These values are plotted directly by the notebook.

---

## 8. The inner optimization problem

For each **fixed** value of `alpha`, the optimizer finds the best model parameters:

$$
\left(\theta^{*}(\alpha),\theta_0^{*}(\alpha)\right)
=\arg\min_{\theta,\theta_0}J(\theta,\theta_0;\alpha).
$$

This is the key connection to Lecture 4.

Lecture 3 answers:

> Given `alpha`, what are the best $\theta$ and $\theta_0$?

It does **not** yet answer:

> Which `alpha` should we choose?

---

## 9. Why the best training objective is not enough

Increasing `alpha` changes the balance between fitting the training examples and keeping the weights small.

A very small value may allow a model to fit the training data too aggressively. A very large value may regularize too strongly and cause underfitting.

Therefore the value of `alpha` should be selected using data that was not used to fit that particular model.

That is the motivation for cross-validation in Lecture 4.

---

## 10. Lecture 3 → Lecture 4

Lecture 3 solves the **inner optimization**:

$$
\text{fixed }\alpha
\quad\Longrightarrow\quad
\text{optimize }\theta,\theta_0.
$$

Lecture 4 introduces the **outer model-selection problem**. For candidate values of `alpha`, cross-validation measures validation performance and selects the best one:

$$
\alpha^{*}=\arg\max_{\alpha}S(\alpha).
$$

Conceptually:

```text
candidate alpha values
        |
        v
optimize theta, theta0
        |
        v
measure validation performance
        |
        v
repeat for every candidate alpha
        |
        v
choose alpha*
        |
        v
train the final model
```

So Lecture 4 **builds on** the optimization mechanism from Lecture 3; it does not replace it with an unrelated training strategy.

---

## 11. From scratch first, library second

The notebook deliberately implements the important mechanics with NumPy first:

- score
- prediction
- margin
- hinge loss
- regularization
- objective
- gradient
- gradient descent
- training history
- visualization

Only after that do we compare with scikit-learn's `SGDClassifier` using hinge loss and L2 regularization.

The purpose of the library comparison is not to replace the implementation. It is to connect the mathematics we studied to a production-quality machine-learning API.

---

## 12. What to study from the notebook

When working through the notebook, make sure you can explain these relationships without looking them up:

$$
f(x)=\theta^T x+\theta_0
$$

$$
z_i=y_i f(x_i)
$$

$$
L_i=\max(0,1-z_i)
$$

$$
J=L+\alpha R
$$

$$
\left(\theta^{*}(\alpha),\theta_0^{*}(\alpha)\right)
=\arg\min_{\theta,\theta_0}J(\theta,\theta_0;\alpha)
$$

and finally, in Lecture 4,

$$
\alpha^{*}=\arg\max_{\alpha}S(\alpha).
$$

The last two equations are especially important: the first is the **inner parameter optimization**, while the second is the **outer hyperparameter selection**.

---

## Running

Open `lecture_3_linear_svm_from_scratch.ipynb` in Jupyter or VS Code and run the cells from top to bottom.

The notebook is self-contained: code, calculations, training logs, and plots are generated when the notebook is executed. Nothing needs to be loaded from a `.py` or `.png` file.
