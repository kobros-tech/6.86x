# Lecture 6 — Nonlinear Classification

This directory contains the Lecture 6 study guide and executable demonstrations for **nonlinear classification**.

---

## 9. Kernel Perceptron

The lecture derives this idea for the Perceptron.

In feature space, the ordinary Perceptron update is

$$
\theta\leftarrow\theta+y_i\phi(x_i)
$$

whenever example $i$ is misclassified.

Starting from $\theta=0$, the final parameter vector can be written as

$$
\theta=\sum_{j=1}^{n}\alpha_j y_j \phi(x_j)
$$

where $\alpha_j$ counts how many times the Perceptron has updated on training example $j$.

Now consider a new example $x$. Its score is obtained by substituting the expansion of $\theta$ into the inner product:

$$
\theta^T\phi(x)
=
\left(\sum_{j=1}^{n}\alpha_j y_j \phi(x_j)\right)^T\phi(x)
$$

By linearity of the inner product,

$$
\theta^T\phi(x)
=
\sum_{j=1}^{n}\alpha_j y_j \phi(x_j)^T\phi(x)
$$

Using the kernel definition,

$$
\theta^T\phi(x)
=
\sum_{j=1}^{n}\alpha_j y_j K(x_j,x)
$$

So we no longer need to store the high-dimensional $\theta$ explicitly.

### Kernel Perceptron update

Initialize

$$
\alpha_j=0\qquad\text{for all }j
$$

For training example $i$, compute the score

$$
s_i=\sum_{j=1}^{n}\alpha_j y_j K(x_j,x_i)
$$

If

$$
y_i s_i\le 0
$$

make a mistake and update

$$
\alpha_i\leftarrow\alpha_i+1
$$

The classifier can therefore be implemented entirely in terms of the kernel function and the training examples.
