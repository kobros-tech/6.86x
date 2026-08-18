# Lecture 2 — Linear Classification and the Perceptron

The lecture notebook is the primary learning artifact:

- [`lecture_2_perceptron.ipynb`](./lecture_2_perceptron.ipynb)

Open the notebook in Jupyter and run it from top to bottom. It contains the explanations, code, printed training log, and plots.

## What this lecture teaches

### 1. Linear classifier

A linear classifier assigns a score to an input vector:

$$
f(x)=\theta^T x+\theta_0
$$

The prediction is determined by the sign of the score.

The decision boundary is:

$$
\theta^T x+\theta_0=0
$$

For two features this is a straight line. The bias $\theta_0$ allows the line to move away from the origin.

### 2. Agreement

For a training example $(x_i,y_i)$ with labels $y_i\in\{-1,+1\}$, define:

$$
z_i=y_i\left(\theta^T x_i+\theta_0\right)
$$

This tells us which side of the boundary the example occupies.

- $z_i>0$: correctly classified
- $z_i<0$: incorrectly classified
- larger positive $z_i$: stronger agreement with the correct class

Agreement becomes important in Lecture 3 when we introduce hinge loss.

### 3. Classification error

Training error is the fraction of examples for which the predicted label differs from the true label.

The notebook calculates this directly instead of hiding it behind a library function.

### 4. XOR and the limitation of linear models

XOR is not linearly separable. No single straight decision boundary can correctly separate its classes.

This is an important lesson: if the data cannot be separated by the model family, changing the parameters cannot solve the problem.

### 5. Perceptron learning

The perceptron starts with parameters and examines training examples one at a time. If an example is misclassified, it updates:

$$
\theta\leftarrow\theta+y_i x_i
$$

and the bias:

$$
\theta_0\leftarrow\theta_0+y_i
$$

The update changes the classifier in a direction that improves the current example's agreement.

### 6. Augmented-vector formulation

We can incorporate the bias into the feature vector by adding a constant feature equal to 1:

$$
\tilde{x}=\begin{bmatrix}x\\1\end{bmatrix}
$$

and define the augmented parameter vector as:

$$
\tilde{\theta}=\begin{bmatrix}\theta\\\theta_0\end{bmatrix}
$$

Then the classifier becomes:

$$
f(x)=\tilde{\theta}^T\tilde{x}
$$

because:

$$
\tilde{\theta}^T\tilde{x}
=
\begin{bmatrix}\theta^T & \theta_0\end{bmatrix}
\begin{bmatrix}x\\1\end{bmatrix}
=\theta^T x+\theta_0
$$

For example, if:

$$
x=\begin{bmatrix}4\\4\end{bmatrix},\qquad
\theta=\begin{bmatrix}1\\2\end{bmatrix},\qquad
\theta_0=-5
$$

then:

$$
\tilde{x}=\begin{bmatrix}4\\4\\1\end{bmatrix},\qquad
\tilde{\theta}=\begin{bmatrix}1\\2\\-5\end{bmatrix}
$$

and therefore:

$$
\tilde{\theta}^T\tilde{x}=1(4)+2(4)-5(1)=7
$$

The bias has not disappeared. It is now represented by the final coordinate of the augmented vectors.

## From Lecture 2 to Lecture 3

Lecture 2 asks us to learn a separating linear classifier. Lecture 3 develops a more general optimization view using loss and regularization.

A typical Lecture 3 objective is:

$$
J(\theta,\theta_0;\alpha)=L(\theta,\theta_0)+\alpha R(\theta)
$$

The parameter $\alpha$ controls the trade-off between fitting the examples and regularizing the model.

## From Lecture 3 to Lecture 4

Once $\alpha$ is introduced, we need a principled way to choose it. Lecture 4 introduces validation and cross-validation for that purpose.

So the progression is:

**Lecture 2 → linear classifier and perceptron**  
**Lecture 3 → loss, regularization, and optimization**  
**Lecture 4 → hyperparameter selection and cross-validation**

## Implementation philosophy

This repository uses the following learning order:

1. Understand the mathematical operation.
2. Implement it directly with NumPy/Python.
3. Inspect the intermediate values and plots.
4. Only then use an external library when appropriate.

The Lecture 2 notebook therefore does **not** depend on scikit-learn for the perceptron itself.

## Notebook contents

1. Linear classifier
2. Score and prediction
3. Agreement
4. Classification error
5. XOR failure
6. Perceptron update
7. Training from scratch
8. Decision-boundary plot
9. Training-error plot
10. Augmented-vector formulation
11. Connection to Lectures 3 and 4
