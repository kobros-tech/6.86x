# Lecture 3 — Linear Classification and Optimization

Lecture 3 is implemented as a **single executable Jupyter notebook** plus this study README. There are no separate Python source files or image assets for the lecture.

Main notebook:

- [`lecture_3_linear_svm_from_scratch.ipynb`](./lecture_3_linear_svm_from_scratch.ipynb)

The workflow is learning-first:

1. understand the mathematics;
2. implement the classifier and optimizer ourselves with NumPy;
3. inspect intermediate values, logs, and plots directly in the notebook;
4. experiment with the regularization strength;
5. only then compare with scikit-learn.

---

## 1. Linear classifier

For an input vector $x$, a linear classifier computes

$$
f(x)=\theta^T x+\theta_0.
$$

The prediction is determined by the sign of the score:

$$
\hat y=
\begin{cases}
+1 & f(x)\ge0,\\
-1 & f(x)<0.
\end{cases}
$$

The decision boundary is

$$
\theta^T x+\theta_0=0.
$$

In two dimensions this is a line; in higher dimensions it is a hyperplane.

---

## 2. Margin

For training example $(x_i,y_i)$, where $y_i\in\{-1,+1\}$, the signed margin is

$$
z_i=y_i(\theta^T x_i+\theta_0).
$$

If the example is classified correctly, its margin is positive. A larger positive margin means the example is farther from the decision boundary in score space.

---

## 3. Hinge loss

The hinge loss is

$$
L_i=\max(0,1-z_i).
$$

Therefore:

- $z_i\ge1$: zero loss;
- $z_i<1$: positive loss;
- even a correctly classified example can have nonzero loss if it is inside the desired margin.

The average hinge loss is

$$
L(\theta,\theta_0)=\frac{1}{n}\sum_{i=1}^{n}\max\left(0,1-y_i(\theta^T x_i+\theta_0)\right).
$$

---

## 4. L2 regularization

We add an L2 penalty to discourage unnecessarily large weights:

$$
R(\theta)=\frac{1}{2}\|\theta\|_2^2.
$$

The bias $\theta_0$ is not regularized in our implementation.

The hyperparameter $\alpha$ controls how strongly regularization affects the objective.

---

## 5. Complete objective

For a fixed $\alpha$:

$$
J(\theta,\theta_0;\alpha)=L(\theta,\theta_0)+\alpha R(\theta).
$$

Equivalently,

$$
J(\theta,\theta_0;\alpha)
=\frac{1}{n}\sum_{i=1}^{n}\max\left(0,1-y_i(\theta^T x_i+\theta_0)\right)
+\frac{\alpha}{2}\|\theta\|_2^2.
$$

The first term asks:

> How well does the model fit the training examples while respecting the margin?

The second term asks:

> How large are the model weights?

The optimizer balances these two goals.

---

## 6. Gradient

For an active example, meaning $z_i<1$:

$$
\nabla_\theta L_i=-y_i x_i
$$

and

$$
\frac{\partial L_i}{\partial\theta_0}=-y_i.
$$

After averaging the active examples and adding the L2 gradient:

$$
\nabla_\theta J
=-\frac{1}{n}\sum_{i:z_i<1}y_i x_i+\alpha\theta.
$$

For the bias:

$$
\frac{\partial J}{\partial\theta_0}
=-\frac{1}{n}\sum_{i:z_i<1}y_i.
$$

The regularization term affects $\theta$ but not $\theta_0$.

---

## 7. Gradient descent

We update the parameters in the direction that decreases the objective:

$$
\theta\leftarrow\theta-\eta\nabla_\theta J
$$

and

$$
\theta_0\leftarrow\theta_0-\eta\frac{\partial J}{\partial\theta_0}.
$$

Here $\eta$ is the learning rate.

The notebook records the objective, hinge loss, regularization, training error, and parameter values during optimization.

---

## 8. What the notebook visualizes

The notebook shows:

- the training data;
- the hinge-loss function;
- the decision boundary before and after optimization;
- the objective during gradient descent;
- hinge loss and regularization separately;
- training error;
- the parameter path through $(\theta_1,\theta_2)$ space;
- the effect of different $\alpha$ values;
- a final scikit-learn comparison.

Training error is **not** the same as the objective: hinge loss also penalizes insufficient margin.

---

## 9. The inner optimization problem

For every fixed $\alpha$, Lecture 3 solves

$$
\left(\theta^*(\alpha),\theta_0^*(\alpha)\right)
=\arg\min_{\theta,\theta_0}J(\theta,\theta_0;\alpha).
$$

The important point is that $\alpha$ is fixed during this optimization. The optimizer finds the best model parameters for that particular regularization strength.

This creates the next question:

> How should we choose $\alpha$?

The value that minimizes training objective is not necessarily the value that generalizes best to unseen data.

---

## 10. Lecture 3 → Lecture 4

Lecture 3 solves the **inner optimization**:

$$
\text{fixed }\alpha
\quad\Longrightarrow\quad
\text{optimize }\theta,\theta_0.
$$

Lecture 4 adds the **outer model-selection problem**. We evaluate candidate values of $\alpha$ using cross-validation and select the one with the best validation score:

$$
\alpha^*=\arg\max_{\alpha}S(\alpha).
$$

So the complete conceptual workflow is:

```text
candidate alpha
      |
      v
optimize theta, theta0
      |
      v
measure validation performance
      |
      v
repeat for other alphas
      |
      v
choose alpha*
      |
      v
retrain final model
```

Lecture 4 therefore builds on the optimization mechanism from Lecture 3 rather than replacing it with a completely different training strategy.

---

## 11. External library comparison

Only after the from-scratch implementation do we introduce scikit-learn:

```python
from sklearn.linear_model import SGDClassifier

model = SGDClassifier(
    loss="hinge",
    penalty="l2",
    alpha=0.1,
)
```

The purpose is to connect the mathematics and algorithm we studied to a practical, tested machine-learning library.

**From scratch:** understand the mechanism.

**Library:** use an established implementation in real applications.

---

## 12. Key things to remember

- A linear classifier produces a score from a weighted sum of features.
- The decision boundary is where the score is zero.
- The signed margin is $y_i f(x_i)$.
- Hinge loss penalizes margins below $1$.
- L2 regularization discourages large weights.
- $\alpha$ controls regularization strength.
- The objective is what we optimize during training.
- Training error and objective value are different measurements.
- Gradient descent updates parameters opposite to the gradient.
- Lecture 3 optimizes $\theta$ and $\theta_0$ for fixed $\alpha$.
- Lecture 4 uses cross-validation to choose $\alpha$.

---

## Running the notebook

Open `lecture_3_linear_svm_from_scratch.ipynb` in Jupyter or VS Code and run the cells from top to bottom.

Everything needed for the Lecture 3 implementation, calculations, logs, and figures is contained in the notebook itself.
