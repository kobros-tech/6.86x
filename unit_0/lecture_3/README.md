# Lecture 3 — Linear Classification and Optimization

This directory contains the Lecture 3 learning material as **executable code plus study documentation**.

The main notebook is:

- [`lecture_3_linear_svm_from_scratch.ipynb`](./lecture_3_linear_svm_from_scratch.ipynb)

The learning workflow is intentional:

1. understand the mathematics;
2. implement the classifier and optimizer ourselves with NumPy;
3. inspect intermediate values, logs, and plots;
4. experiment with the parameters;
5. only then compare with an external implementation from scikit-learn.

---

## 1. Linear classifier

For an input vector $x$, a linear classifier computes a score

$$
f(x) = \theta^T x + \theta_0
$$

and predicts

$$
\hat y = \begin{cases}
+1 & f(x) \ge 0 \\
-1 & f(x) < 0.
\end{cases}
$$

The decision boundary is the set of points for which

$$
\theta^T x + \theta_0 = 0.
$$

In two dimensions this is a line. In higher dimensions it becomes a hyperplane.

---

## 2. Margin

For a training example $(x_i,y_i)$, with $y_i \in \{-1,+1\}$, define the signed margin

$$
z_i = y_i(\theta^T x_i + \theta_0).
$$

The sign tells us whether the example is classified correctly, while the magnitude tells us how far the example is from the decision boundary in score space.

A positive example with a positive score has positive margin. A negative example with a negative score also has positive margin because the two signs multiply to a positive value.

---

## 3. Hinge loss

The hinge loss is

$$
L_i = \max(0,1-z_i).
$$

Therefore:

- if $z_i \ge 1$, the loss is zero;
- if $z_i < 1$, the example contributes loss;
- examples inside the desired margin receive a penalty even when they are classified correctly.

This is important: an SVM is not simply trying to classify every training point correctly. It also tries to obtain a useful margin.

The average hinge loss is

$$
L(\theta,\theta_0) = \frac{1}{n}\sum_{i=1}^{n}\max\left(0,1-y_i(\theta^T x_i+\theta_0)\right).
$$

---

## 4. Regularization

We add an L2 regularization term to discourage unnecessarily large weights:

$$
R(\theta) = \frac{1}{2}\|\theta\|_2^2.
$$

The bias is not regularized in our implementation.

The regularization strength is controlled by the hyperparameter $\alpha$.

---

## 5. Complete objective function

For a fixed $\alpha$, our optimization problem is

$$
J(\theta,\theta_0;\alpha) = L(\theta,\theta_0) + \alpha R(\theta).
$$

Equivalently,

$$
J(\theta,\theta_0;\alpha) = \frac{1}{n}\sum_{i=1}^{n}\max\left(0,1-y_i(\theta^T x_i+\theta_0)\right) + \frac{\alpha}{2}\|\theta\|_2^2.
$$

The first term asks:

> How well does the model fit the training examples while respecting the margin?

The second term asks:

> How large are the model weights?

The optimizer must balance both objectives.

---

## 6. Gradient

For an example whose margin is below $1$,

$$
y_i(\theta^T x_i+\theta_0)<1,
$$

the hinge-loss contribution has gradient

$$
\nabla_\theta L_i = -y_i x_i
$$

and

$$
\frac{\partial L_i}{\partial \theta_0}=-y_i.
$$

After averaging the active hinge-loss gradients over all examples and adding the L2 contribution, we obtain

$$
\nabla_\theta J = -\frac{1}{n}\sum_{i:z_i<1}y_i x_i + \alpha\theta
$$

and

$$
\frac{\partial J}{\partial\theta_0} = -\frac{1}{n}\sum_{i:z_i<1}y_i.
$$

Notice that the regularization term affects $\theta$ but not $\theta_0$ in our formulation.

---

## 7. Gradient descent

We update the parameters in the direction that decreases the objective:

$$
\theta \leftarrow \theta - \eta\nabla_\theta J
$$

and

$$
\theta_0 \leftarrow \theta_0 - \eta\frac{\partial J}{\partial\theta_0},
$$

where $\eta$ is the learning rate.

The notebook records the objective, hinge loss, training error, and parameter values at every iteration so that the optimization process can be inspected rather than treated as a black box.

---

## 8. What the plots show

The notebook produces several useful views of the algorithm:

### Decision boundary

Shows the initial boundary and the boundary obtained after optimization.

### Objective history

Shows whether gradient descent is actually reducing $J$.

### Hinge loss and regularization

Separates the two components of the objective so we can see their competing effects.

### Training error

Shows classification error during optimization. This is related to the objective, but it is **not the same quantity** as the objective because hinge loss also cares about the margin.

### Parameter path

Shows how $(\theta_1,\theta_2)$ moves through parameter space during gradient descent.

---

## 9. Why the optimizer is not enough

For Lecture 3, $\alpha$ is treated as a fixed hyperparameter.

For every chosen $\alpha$, we solve an inner optimization problem:

$$
(\theta^*(\alpha),\theta_0^*(\alpha)) = \arg\min_{\theta,\theta_0} J(\theta,\theta_0;\alpha).
$$

But this creates a new question:

> How should we choose $\alpha$?

Choosing the value that gives the smallest training objective is not necessarily the same as choosing the value that generalizes best to unseen data.

That is the motivation for Lecture 4.

---

## 10. Lecture 3 → Lecture 4

Lecture 3 gives us the **inner optimization**:

$$
\text{fixed }\alpha \quad\Longrightarrow\quad \text{optimize }\theta,\theta_0.
$$

Lecture 4 adds an **outer model-selection loop**:

$$
\alpha^* = \arg\max_\alpha S(\alpha),
$$

where $S(\alpha)$ is the cross-validation score.

So the overall learning workflow becomes:

```text
candidate alpha
      |
      v
optimize theta for that alpha
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

This is why Lecture 4 can reuse the optimization code developed in Lecture 3 rather than introducing a completely different training mechanism.

---

## 11. External library comparison

Only after the from-scratch implementation, the notebook introduces:

```python
from sklearn.linear_model import SGDClassifier

model = SGDClassifier(
    loss="hinge",
    penalty="l2",
    alpha=0.1,
)
```

The purpose is not to replace our implementation. It is to connect the mathematics and algorithms we studied to a practical machine-learning library.

The important distinction is:

**from scratch:** understand and implement the mechanism;

**library:** use a tested implementation when building real applications.

---

## 12. Key things to remember

- A linear classifier produces a score using a weighted sum of features.
- The decision boundary is where the score equals zero.
- The signed margin is $y_i f(x_i)$.
- Hinge loss penalizes margins below $1$.
- L2 regularization discourages large weights.
- $\alpha$ controls the strength of regularization.
- The objective is the quantity optimized during training.
- Training error and objective value are different measurements.
- Gradient descent updates parameters opposite to the gradient.
- Lecture 3 solves the optimization problem for a fixed $\alpha$.
- Lecture 4 uses cross-validation to choose $\alpha$.

---

## Running the notebook

Open `lecture_3_linear_svm_from_scratch.ipynb` in Jupyter or VS Code and run the cells from top to bottom.

The notebook is intentionally self-contained so that the mathematics, implementation, logs, and figures can be studied in one place.
