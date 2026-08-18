# Lecture 2 — Linear Classification and the Perceptron

Lecture 2 is organized as **one executable notebook + one study README**.

- `lecture_2_perceptron.ipynb` — the complete learning implementation
- `README.md` — the mathematical study guide

The notebook is the primary learning artifact. Open it in Jupyter or VS Code and run it from top to bottom. It contains the explanations, NumPy implementations, training logs, and plots.

---

## 1. Linear classifier

For an input vector $x$, a linear classifier computes a score:

$$
f(x)=\theta^T x+\theta_0
$$

Prediction is based on the sign of the score:

$$
\hat{y}=\begin{cases}
+1 & \text{if } f(x)\ge 0,\\
-1 & \text{if } f(x)<0
\end{cases}
$$

The decision boundary is

$$
\theta^T x+\theta_0=0
$$

In two dimensions this is a straight line. In higher dimensions it is a hyperplane.

The bias $\theta_0$ allows the decision boundary to move away from the origin.

---

## 2. Agreement

For a training example $(x_i,y_i)$ with label $y_i\in\{-1,+1\}$, define the agreement:

$$
z_i=y_i\left(\theta^T x_i+\theta_0\right)
$$

The sign tells us whether the example is correctly classified:

- $z_i>0$ — correctly classified;
- $z_i<0$ — incorrectly classified;
- larger positive $z_i$ — stronger agreement with the correct class.

Agreement becomes especially important in Lecture 3 when we introduce the signed margin and hinge loss.

---

## 3. Classification error

Training error is the fraction of training examples for which the predicted label differs from the true label.

For predictions $\hat{y}_i$ and labels $y_i$:

$$
\text{error}=\frac{1}{n}\sum_{i=1}^{n}\mathbf{1}\left[\hat{y}_i\ne y_i\right]
$$

The notebook calculates this directly so that the relationship between predictions and error is visible rather than hidden behind a library function.

---

## 4. XOR and the limitation of linear models

XOR is not linearly separable. No single straight decision boundary can correctly separate its classes.

This gives an important lesson:

> If the data cannot be separated by the chosen model family, changing the parameters cannot make a linear model solve the problem.

The limitation comes from the **model family**, not from a particular choice of parameters.

---

## 5. Perceptron learning

The perceptron starts with parameters and examines training examples one at a time.

If an example is misclassified, the perceptron updates the weight vector:

$$
\theta\leftarrow\theta+y_i x_i
$$

and the bias:

$$
\theta_0\leftarrow\theta_0+y_i
$$

Why does the update have this direction?

The current example contributes the signed quantity

$$
y_i\left(\theta^T x_i+\theta_0\right)
$$

When the example is misclassified, this quantity is negative. Adding $y_i x_i$ to $\theta$ and $y_i$ to $\theta_0$ increases the example's agreement:

$$
y_i\left((\theta+y_i x_i)^T x_i+(\theta_0+y_i)\right)
$$

The update therefore moves the classifier in a direction intended to improve the current example's classification.

---

## 6. Perceptron convergence

For linearly separable training data, the perceptron convergence theorem guarantees that the algorithm eventually finds a separating hyperplane under the standard perceptron assumptions.

For data that are not linearly separable, such as XOR, the perceptron does not have a separating solution to converge to.

This is another way to see the distinction between:

- changing parameters inside a model family; and
- changing the model family itself.

---

## 7. Training from scratch

The notebook implements the perceptron directly with NumPy/Python.

The learning loop is conceptually:

```text
initialize theta and theta_0
        |
        v
visit each training example
        |
        v
compute the score
        |
        v
check the predicted label
        |
        +---- correct ----> keep parameters
        |
        +---- wrong ------> update theta and theta_0
        |
        v
repeat for additional epochs
```

For each epoch, the notebook records the training error so that we can inspect how learning changes the classifier.

---

## 8. Decision boundary

For two features, the decision boundary is

$$
\theta_1x_1+\theta_2x_2+\theta_0=0
$$

The notebook plots the training examples together with this boundary.

The plot gives a geometric interpretation of the classifier:

- one side corresponds to positive scores;
- the other side corresponds to negative scores;
- points on the boundary have score zero.

---

## 9. Training-error plot

The notebook records the classification error after each training epoch.

This lets us see whether the perceptron is learning a separating boundary for the chosen dataset.

For linearly separable data, the error can eventually reach zero. For non-linearly-separable data, zero training error cannot be achieved by a single linear classifier.

---

## 10. Augmented-vector formulation

The bias can be incorporated into the feature vector by adding a constant feature equal to $1$.

Define

$$
\tilde{x}=\begin{bmatrix}
x\\
1
\end{bmatrix}
$$

and the augmented parameter vector

$$
\tilde{\theta}=\begin{bmatrix}
\theta\\
\theta_0
\end{bmatrix}
$$

Then the classifier becomes

$$
f(x)=\tilde{\theta}^T\tilde{x}
$$

because

$$
\tilde{\theta}^T\tilde{x}
=
\begin{bmatrix}
\theta^T & \theta_0
\end{bmatrix}
\begin{bmatrix}
x\\
1
\end{bmatrix}
=\theta^T x+\theta_0
$$

The bias has not disappeared. It is now represented by the final coordinate of the augmented vectors.

---

## 11. Augmented-vector example

Suppose

$$
x=\begin{bmatrix}
4\\
4
\end{bmatrix},
\qquad
\theta=\begin{bmatrix}
1\\
2
\end{bmatrix},
\qquad
\theta_0=-5
$$

Then

$$
\tilde{x}=\begin{bmatrix}
4\\
4\\
1
\end{bmatrix},
\qquad
\tilde{\theta}=\begin{bmatrix}
1\\
2\\
-5
\end{bmatrix}
$$

and therefore

$$
\tilde{\theta}^T\tilde{x}
=1(4)+2(4)-5(1)
=7
$$

So the classifier score is $7$, which is positive. The corresponding prediction is therefore $+1$.

---

## 12. From Lecture 2 to Lecture 3

Lecture 2 focuses on learning a separating linear classifier with the perceptron.

Lecture 3 develops a more general optimization view using loss and regularization.

A typical Lecture 3 objective is

$$
J(\theta,\theta_0;\alpha)
=L(\theta,\theta_0)+\alpha R(\theta)
$$

The parameter $\alpha$ controls the trade-off between fitting the examples and regularizing the model.

The conceptual progression is:

```text
Lecture 2
linear classifier
      |
      v
perceptron learning
      |
      v
Lecture 3
loss + regularization + optimization
```

---

## 13. From Lecture 3 to Lecture 4

Once $\alpha$ is introduced in Lecture 3, we need a principled way to choose it.

Lecture 4 introduces validation and cross-validation for hyperparameter selection.

The progression is:

```text
Lecture 2 → linear classifier and perceptron
Lecture 3 → loss, regularization, and optimization
Lecture 4 → hyperparameter selection and cross-validation
```

Lecture 3 solves the parameter-optimization problem for a fixed $\alpha$:

$$
(\theta^*(\alpha),\theta_0^*(\alpha))
=\arg\min_{\theta,\theta_0}J(\theta,\theta_0;\alpha)
$$

Lecture 4 then considers the outer model-selection problem:

$$
\alpha^*=\arg\max_{\alpha}S(\alpha)
$$

when $S(\alpha)$ is a validation score where larger is better.

---

## 14. Implementation philosophy

This repository follows the same learning order throughout the lectures:

1. Understand the mathematical operation.
2. Implement it directly with NumPy/Python.
3. Inspect intermediate values, logs, and plots.
4. Only then compare with an external machine-learning library when appropriate.

The Lecture 2 notebook therefore does **not** depend on scikit-learn for the perceptron itself.

The goal is to understand what the algorithm is doing before using a higher-level implementation.

---

## 15. What to study from the notebook

When working through the notebook, make sure you can explain these relationships without looking them up:

$$
f(x)=\theta^T x+\theta_0
$$

$$
z_i=y_i f(x_i)
$$

$$
\hat{y}=\operatorname{sign}(f(x))
$$

$$
\theta\leftarrow\theta+y_i x_i
$$

$$
\theta_0\leftarrow\theta_0+y_i
$$

and, with the augmented representation,

$$
f(x)=\tilde{\theta}^T\tilde{x}
$$

You should also understand the connection to the next lectures:

$$
\text{Lecture 2}
\longrightarrow
\text{Lecture 3}
\longrightarrow
\text{Lecture 4}
$$

```text
linear classification
        ↓
perceptron learning
        ↓
loss + regularization + optimization
        ↓
cross-validation + hyperparameter selection
```

---

## Running

Open `lecture_2_perceptron.ipynb` in Jupyter or VS Code and run the cells from top to bottom.

The notebook is self-contained: code, calculations, training logs, and plots are generated when the notebook is executed.
