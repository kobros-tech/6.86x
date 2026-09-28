# Lecture 5 — Linear Regression

This directory contains the Lecture 5 study guide and an executable demonstration of linear regression. The README follows the study-oriented structure used throughout the repository: introduce the mathematical idea, explain the intuition, connect the equations to the learning algorithm, and then reinforce the ideas with a small NumPy implementation.

---

## 1. What Lecture 5 is about

Unit 1 focused on **classification**: learning a function that predicts a class label.

Lecture 5 changes the prediction task. In **regression**, the target can be any real-valued number.

The central question is:

> **How can we learn a function that predicts a continuous value from a feature vector?**

The lecture develops linear regression through four connected ideas:

1. define a linear prediction function;
2. measure prediction error with squared loss;
3. learn the parameters by minimizing empirical risk;
4. study both gradient-based learning and the closed-form solution.

The lecture also discusses the effect of limited/noisy data and the role of regularization in controlling model complexity.

---

## 2. From classification to regression

In Unit 1, a linear classifier produced a score and converted it into a class decision.

For regression, the output itself is the prediction:

$$
f(x;\theta)=\theta^{T} x.
$$

Here:

- $x$ is the feature vector;
- $\theta$ is the parameter vector;
- $\theta^{T} x$ is a real-valued prediction.

The lecture initially writes the model with an intercept:

$$
f(x;\theta)=\theta_0+\sum_{j=1}^{d}\theta_jx_j.
$$

For convenience, the intercept can be represented as an additional feature whose value is $1$. This lets us use the compact vector form above.

### Important point

**Linear** refers to the model being linear in its parameters. We can still construct useful features from the original input before applying a linear model.

---

## 3. What does a good prediction mean?

For a training example $(x_i,y_i)$, the model predicts

$$
\hat y_i=\theta^{T} x_i.
$$

The prediction error is

$$
\hat y_i-y_i.
$$

A learning algorithm needs a numerical way to measure how bad this error is.

Lecture 5 uses **squared error**:

$$
L_i(\theta)=\frac{1}{2}\left(\theta^{T} x_i-y_i\right)^2.
$$

The factor $\frac{1}{2}$ is convenient when differentiating; it does not change which parameter vector minimizes the objective.

Squaring has two useful effects:

- positive and negative errors do not cancel each other;
- large errors receive disproportionately larger penalties.

---

## 4. Empirical risk

For $n$ training examples, we combine the individual losses into an empirical objective:

$$
J(\theta)=\frac{1}{2n}\sum_{i=1}^{n}\left(\theta^{T} x_i-y_i\right)^2.
$$

The learning problem is therefore

$$
\theta^*=\arg\min_{\theta}J(\theta).
$$

This is the same optimization pattern we encountered in Unit 1:

```text
training data
     |
     v
choose a model family
     |
     v
measure the error
     |
     v
build an objective
     |
     v
optimize the parameters
     |
     v
learn theta*
```

The major difference is the **loss function and prediction task**.

---

## 5. The geometry of one-dimensional linear regression

With one feature and an intercept, the model is

$$
\hat y=\theta_0+\theta_1x.
$$

The training data are points $(x_i,y_i)$ in a plane. Learning linear regression means finding the line whose predictions give the smallest squared-error objective.

Conceptually:

```text
y
^
|                 *
|            *        *
|       *
|   *
|       --------------------  learned line
|
+------------------------------> x
```

The demo in this directory makes this geometry visible and shows how the learned line changes while gradient descent is running.

---

## 6. Matrix representation

Put the training examples into a design matrix:

$$
X=
\begin{bmatrix}
 x_1^{T} \\
 x_2^{T} \\
 \vdots \\
 x_n^{T}
\end{bmatrix}
$$

and put the target values into a vector:

$$
y=
\begin{bmatrix}
 y_1 \\
 y_2 \\
 \vdots \\
 y_n
\end{bmatrix}.
$$

All predictions can then be written together as

$$
\hat y=X\theta.
$$

The objective becomes

$$
J(\theta)=\frac{1}{2n}\left\|X\theta-y\right\|^2.
$$

This form is important because it lets us derive the gradient and the closed-form solution cleanly.

---

## 7. Gradient of the squared-error objective

For

$$
J(\theta)=\frac{1}{2n}\left\|X\theta-y\right\|^2,
$$

the gradient with respect to $\theta$ is

$$
\nabla J(\theta)=\frac{1}{n}X^{T}(X\theta-y).
$$

The interpretation is useful:

- $X\theta-y$ contains the current prediction errors;
- $X^{T}$ maps those errors back onto the parameter directions;
- the result tells us how the objective changes if we change each parameter.

To **minimize** the objective, we move in the opposite direction of the gradient.

---

## 8. Gradient descent

Starting with some initial parameter vector $\theta$, gradient descent repeatedly applies

$$
\theta\leftarrow\theta-\eta\nabla J(\theta),
$$

where $\eta$ is the learning rate.

Substituting the gradient gives

$$
\theta\leftarrow
\theta-\frac{\eta}{n}X^{T}(X\theta-y).
$$

The learning loop is therefore:

```text
initialize theta
       |
       v
compute predictions X theta
       |
       v
compute errors X theta - y
       |
       v
compute gradient
       |
       v
move theta against the gradient
       |
       v
repeat
```

### Connection to Unit 1

This should look familiar from the optimization work in Unit 1. The important new observation is that **the size and direction of the regression error continuously influence the update**.

---

## 9. Why the gradient update makes sense

Suppose the prediction is too large for some training examples. Their errors contribute one direction to the gradient.

Suppose the prediction is too small. Their errors contribute in the opposite direction.

The gradient combines the effects of all training examples and tells us which parameter changes would increase the objective.

Therefore, subtracting the gradient moves the parameters toward a lower squared-error objective.

This is fundamentally different from the simple Perceptron update, where the update is triggered by a classification mistake.

---

## 10. Closed-form solution

Because the squared-error objective is quadratic in $\theta$, we can also solve the optimization problem analytically.

At an optimum, the gradient is zero:

$$
\nabla J(\theta)=0.
$$

Therefore,

$$
X^{T}(X\theta-y)=0.
$$

Rearranging gives the normal equations:

$$
X^{T}X\theta=X^{T}y.
$$

When $X^{T}X$ is invertible, the solution can be written as

$$
\theta^*=(X^{T}X)^{-1}X^{T}y.
$$

This is the **closed-form linear regression solution**.

### Important numerical point

The formula is useful for understanding the mathematics, but practical numerical code should generally avoid explicitly computing a matrix inverse. Solving the corresponding linear system is more numerically appropriate.

The demo therefore compares gradient descent with a linear-system solution rather than relying on an explicit inverse.

---

## 11. Gradient descent vs. closed form

The two approaches solve the same underlying least-squares problem in different ways.

| Method | Main idea | Main characteristic |
|---|---|---|
| Gradient descent | Iteratively improve $\theta$ | Requires a learning rate and iterations |
| Closed form | Solve the normal equations | Direct solution when the system is well behaved |

For a small dataset, the closed-form solution is convenient and gives us an excellent reference for checking our gradient-descent implementation.

For larger problems, iterative optimization can be more practical because explicitly forming and solving the required matrix system can become expensive.

---

## 12. Generalization and insufficient data

A central lesson of the lecture is that fitting the training data is not the whole machine-learning problem.

The lecture distinguishes two broad situations:

### Structural limitation

The chosen function family may be unable to represent the relationship in the data.

### Estimation limitation

The function family may be suitable, but the available data may be insufficient or noisy for reliable parameter estimation.

This distinction connects Lecture 5 to the generalization ideas from Unit 1.

A model can minimize training error without necessarily giving the best predictions on unseen examples.

---

## 13. Regularization

When the data are insufficient or noisy, controlling the complexity of the learned parameters can improve generalization.

The general regularized objective has the form

$$
J_{\mathrm{reg}}(\theta)=J(\theta)+\lambda R(\theta),
$$

where $\lambda$ controls the strength of the regularization term.

For an L2-style penalty, the idea is to discourage unnecessarily large parameter values.

The important conceptual trade-off is:

```text
weak regularization
        |
        v
more freedom to fit training data
        |
        +---- possible overfitting

strong regularization
        |
        v
more restriction on parameters
        |
        +---- possible underfitting
```

The exact regularized objective and conventions should be kept consistent with the formulation used in the MIT lecture material.

---

## 14. Feature representation and linearity

A linear model does not necessarily mean that the original relationship between raw variables must look like a straight line.

We can construct features first.

For example, a one-dimensional input $z$ could be represented using features such as

$$
\phi(z)=
\begin{bmatrix}
1 \\
z \\
z^2
\end{bmatrix}.
$$

A linear model in this feature space becomes

$$
\hat y=\theta^{T}\phi(z).
$$

The model is still linear in $\theta$, even though the prediction as a function of the original variable $z$ can be nonlinear.

This idea will become particularly important as Unit 2 moves from linear methods toward **nonlinear classification**.

---

## 15. Demo: Linear Regression From Scratch

The accompanying demo uses only **NumPy and Matplotlib** to make the learning mechanism explicit.

It demonstrates:

1. a small noisy regression dataset;
2. a linear prediction function;
3. squared-error empirical risk;
4. the analytical gradient;
5. gradient descent from scratch;
6. the closed-form solution;
7. comparison of the two learned parameter vectors;
8. visualization of the data and fitted line;
9. visualization of the objective decreasing during training.

Run:

```bash
python demo_linear_regression.py
```

The important goal is not to obtain a line with a library call. It is to see the complete chain:

```text
data
  |
  v
prediction = X theta
  |
  v
error = prediction - y
  |
  v
squared-error objective
  |
  v
gradient
  |
  v
theta <- theta - eta * gradient
  |
  v
repeat until convergence
```

The closed-form solution is then used as an independent check of the iterative result.

---

## 16. What to observe in the demo

When running the program, pay attention to four things.

### A. The initial line

The starting parameter vector generally gives a poor fit.

### B. The objective

As gradient descent progresses, the squared-error objective should generally decrease when the learning rate is appropriate.

### C. The learned line

The line moves toward a position that balances the errors across all training examples.

### D. Agreement with the closed-form solution

After sufficient iterations, the gradient-descent parameters should be close to the closed-form least-squares solution.

This gives us a useful experimental test:

> **Does our derivative and gradient-descent implementation actually solve the objective we derived mathematically?**

---

## 17. Common mistakes

### Mistake 1 — Forgetting the intercept

A model of the form

$$
\hat y=\theta_1x
$$

is forced through the origin. If an intercept is required, represent it explicitly, for example by adding a constant feature equal to $1$.

### Mistake 2 — Using the wrong gradient sign

For minimization:

$$
\theta\leftarrow\theta-\eta\nabla J(\theta).
$$

We move **against** the gradient.

### Mistake 3 — Confusing prediction with error

Prediction:

$$
\hat y=X\theta.
$$

Error:

$$
X\theta-y.
$$

They are not the same quantity.

### Mistake 4 — Forgetting the transpose

The matrix gradient is

$$
\nabla J(\theta)=\frac{1}{n}X^{T}(X\theta-y).
$$

The transpose is what maps the example-wise errors back into parameter space.

### Mistake 5 — Expecting gradient descent to converge with any learning rate

A learning rate that is too large can cause unstable updates or divergence. A learning rate that is too small can make learning unnecessarily slow.

### Mistake 6 — Explicitly computing a matrix inverse in numerical code

Although

$$
\theta^*=(X^{T}X)^{-1}X^{T}y
$$

is the useful mathematical expression, numerical implementations should generally solve the linear system directly.

---

## 18. What to remember for an exam

If you remember only the following, you have the core of Lecture 5:

1. Regression predicts a continuous real-valued target.
2. A linear regression prediction can be written as

$$
\hat y=\theta^{T} x.
$$

3. For squared loss, the empirical objective is

$$
J(\theta)=\frac{1}{2n}\sum_{i=1}^{n}\left(\theta^{T} x_i-y_i\right)^2.
$$

4. In matrix form,

$$
J(\theta)=\frac{1}{2n}\left\|X\theta-y\right\|^2.
$$

5. Its gradient is

$$
\nabla J(\theta)=\frac{1}{n}X^{T}(X\theta-y).
$$

6. Gradient descent minimizes the objective using

$$
\theta\leftarrow\theta-\eta\nabla J(\theta).
$$

7. Setting the gradient to zero gives the normal equations:

$$
X^{T}X\theta=X^{T}y.
$$

8. When the system is invertible, the closed-form solution is

$$
\theta^*=(X^{T}X)^{-1}X^{T}y.
$$

9. Gradient descent and the closed-form method are two different ways of solving the same least-squares problem.
10. Generalization depends not only on fitting the training data, but also on the suitability of the model and the amount/noise of available data.
11. Regularization can control parameter magnitude and help address estimation problems.
12. A linear model can still use nonlinear feature representations.

---

## 19. Connection to the rest of Unit 2

Lecture 5 establishes the regression and optimization foundation for the next parts of Unit 2.

```text
Unit 1
Linear classification
       |
       v
Lecture 5
Linear regression
       |
       +--> squared-error optimization
       +--> gradient descent
       +--> closed-form solution
       +--> regularization
       |
       v
Unit 2
Nonlinear classification
       |
       v
Collaborative filtering
```

The most important transition is that we are no longer restricted to predicting class labels. We are learning how a parameterized function can predict **real-valued quantities**, optimize a continuous objective, and generalize beyond the training examples.

---

## 20. Repository convention

This README is a study guide rather than a transcription of the MIT lecture. The executable demo is deliberately small so that every mathematical quantity can be traced directly to the Python implementation.

For repository-wide Markdown and LaTeX conventions, see [MATH_NOTATION.md](../../MATH_NOTATION.md).
